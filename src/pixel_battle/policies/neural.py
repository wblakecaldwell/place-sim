"""NeuralRedditPolicy: drives all pixel placements using RedditCitizenNet with single-pass cross-faction GPU batching."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from pixel_battle.canvas import CanvasState
from pixel_battle.factions import Faction
from pixel_battle.ml.model import (
    ARCH_ARTIST,
    ARCH_BUILDER,
    ARCH_DEFENDER,
    ARCH_FREE,
    ARCH_VOID,
    NO_GOAL_IDX,
    RedditCitizenNet,
)
from pixel_battle.palette import BLACK_IDX, WHITE_IDX
from pixel_battle.policies.base import PlacementBatch
from pixel_battle.policies.stochastic import StochasticPolicy

log = logging.getLogger(__name__)


class NeuralRedditPolicy(StochasticPolicy):
    """Drop-in neural PixelPolicy powered by RedditCitizenNet on PyTorch MPS.

    Batches ALL ready builders and defenders across up to 240+ active factions into ONE combined
    GPU forward pass per Tick (REQ-ML-008) while tracking 20% One-Shot session turnover (REQ-SIM-014).
    """

    def __init__(
        self,
        model_path: Path,
        num_users: int,
        width: int,
        height: int,
        base_cooldown: int,
        device: torch.device,
        seed: int = 42,
        temp_pos: float = 0.95,
        temp_col: float = 0.85,
    ) -> None:
        super().__init__(
            num_users=num_users,
            width=width,
            height=height,
            base_cooldown=base_cooldown,
            device=device,
            seed=seed,
        )
        self.model_path = Path(model_path)
        self.temp_pos = temp_pos
        self.temp_col = temp_col

        ckpt = torch.load(self.model_path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        hidden_dim = int(cfg.get("hidden_dim", 64))
        self.patch_size = int(cfg.get("patch_size", 17))
        self.patch_radius = self.patch_size // 2
        self.use_prev_color = bool(cfg.get("use_prev_color", False))

        self.net = RedditCitizenNet(
            hidden_dim=hidden_dim,
            patch_size=self.patch_size,
            use_prev_color=self.use_prev_color,
        ).to(device)
        self.net.load_state_dict(ckpt["model_state"])
        self.net.eval()

        offs = torch.arange(-self.patch_radius, self.patch_radius + 1, dtype=torch.int32, device=device)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        self._off_y = oy
        self._off_x = ox

        g = torch.Generator(device="cpu").manual_seed(seed + 13)
        r = torch.rand(num_users, generator=g)
        arch = torch.full((num_users,), ARCH_BUILDER, dtype=torch.long)
        arch = torch.where(r >= 0.42, torch.tensor(ARCH_FREE, dtype=torch.long), arch)
        arch = torch.where(r >= 0.68, torch.tensor(ARCH_ARTIST, dtype=torch.long), arch)
        arch = torch.where(r >= 0.84, torch.tensor(ARCH_DEFENDER, dtype=torch.long), arch)
        arch = torch.where(r >= 0.94, torch.tensor(ARCH_VOID, dtype=torch.long), arch)
        self.citizen_arch = arch.to(device)
        self._cur_tick = 0

        log.info(
            "Loaded NeuralRedditPolicy (%dx%d, prev=%s, h=%d) from %s (metrics=%s) on %s",
            self.patch_size,
            self.patch_size,
            self.use_prev_color,
            hidden_dim,
            self.model_path,
            ckpt.get("metrics", {}),
            device,
        )

    def _extract_canvas_patches(
        self,
        tick: int,
        canvas: CanvasState,
        cx: torch.Tensor,
        cy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Vectorized [B, P, P] extraction of canvas colors, prev_colors, and normalized recency on GPU."""
        R = self.patch_radius
        pad_c = F.pad(canvas.colors.to(torch.long), (R, R, R, R), value=WHITE_IDX)
        age = (int(tick) - canvas.last_edit_tick).clamp(min=0, max=600).to(torch.float32)
        rec_grid = torch.exp(-age / 35.0)
        pad_r = F.pad(rec_grid, (R, R, R, R), value=0.0)

        gy = (cy[:, None, None] + R + self._off_y[None, :, :]).clamp(0, self.height + 2 * R - 1)
        gx = (cx[:, None, None] + R + self._off_x[None, :, :]).clamp(0, self.width + 2 * R - 1)

        prev_p: torch.Tensor | None = None
        if self.use_prev_color:
            pad_prev = F.pad(canvas.prev_colors.to(torch.long), (R, R, R, R), value=WHITE_IDX)
            prev_p = pad_prev[gy, gx]

        return pad_c[gy, gx], prev_p, pad_r[gy, gx]

    def _extract_goal_patches(
        self,
        fac: Faction,
        cx: torch.Tensor,
        cy: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized [B, P, P] extraction of faction blueprint aligned to (cx, cy), NO_GOAL outside."""
        gy = cy[:, None, None] + self._off_y[None, :, :] - fac.y0
        gx = cx[:, None, None] + self._off_x[None, :, :] - fac.x0
        in_bounds = (gy >= 0) & (gy < fac.h) & (gx >= 0) & (gx < fac.w)
        sy = gy.clamp(0, fac.h - 1)
        sx = gx.clamp(0, fac.w - 1)
        bp_vals = fac.blueprint[sy, sx].to(torch.long)
        return torch.where(
            in_bounds & (bp_vals >= 0),
            bp_vals,
            torch.full_like(bp_vals, NO_GOAL_IDX),
        )

    def _run_net_batched(
        self,
        cur_p: torch.Tensor,
        prev_p: torch.Tensor | None,
        rec_p: torch.Tensor,
        goal_p: torch.Tensor,
        arch: torch.Tensor,
        temp_pos: float,
        temp_col: float,
        valid_pos_mask: torch.Tensor | None = None,
        max_chunk: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cur_p.shape[0]
        if B <= max_chunk:
            return self.net.sample_actions(
                cur_p,
                rec_p,
                goal_p,
                arch,
                prev_patch=prev_p,
                temp_pos=temp_pos,
                temp_col=temp_col,
                valid_pos_mask=valid_pos_mask,
            )
        dys, dxs, cols = [], [], []
        for s in range(0, B, max_chunk):
            e = min(B, s + max_chunk)
            pp = prev_p[s:e] if prev_p is not None else None
            vm = valid_pos_mask[s:e] if valid_pos_mask is not None else None
            dy, dx, c = self.net.sample_actions(
                cur_p[s:e],
                rec_p[s:e],
                goal_p[s:e],
                arch[s:e],
                prev_patch=pp,
                temp_pos=temp_pos,
                temp_col=temp_col,
                valid_pos_mask=vm,
            )
            dys.append(dy)
            dxs.append(dx)
            cols.append(c)
        return torch.cat(dys), torch.cat(dxs), torch.cat(cols)

    def step(
        self,
        tick: int,
        canvas: CanvasState,
        factions: list[Faction],
        shared_borders: list[tuple[int, int, tuple[int, int, int, int]]],
    ) -> PlacementBatch:
        """Single-pass cross-faction batched neural step for 200+ active factions."""
        self._cur_tick = tick
        dev = self.device
        P = self.patch_size
        R = self.patch_radius
        j_lim = max(4, R - 3)

        self.cooldown -= 1
        ready_mask = self.cooldown <= 0
        ready_ids = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)

        if ready_ids.numel() == 0:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
            return PlacementBatch(
                xs=empty_i32,
                ys=empty_i32,
                colors=torch.empty(0, dtype=torch.uint8, device=dev),
                factions=torch.empty(0, dtype=torch.int16, device=dev),
                user_ids=empty_i32,
            )

        lo = max(2, int(self.base_cooldown * 0.65))
        hi = max(lo + 2, int(self.base_cooldown * 1.35))
        self.cooldown[ready_ids] = torch.randint(lo, hi, (ready_ids.numel(),), dtype=torch.int32, device=dev)

        out_xs: list[torch.Tensor] = []
        out_ys: list[torch.Tensor] = []
        out_cols: list[torch.Tensor] = []
        out_facs: list[torch.Tensor] = []
        out_uids: list[torch.Tensor] = []
        placed_slots_list: list[torch.Tensor] = []

        ready_fac = self.user_factions[ready_ids]

        # 1. Free agents (including one-shot citizens) in one batched pass
        free_u = ready_ids[ready_fac == -1]
        if free_u.numel() > 0:
            fx, fy, fc = self._step_free_agents(free_u, canvas)
            out_xs.append(fx)
            out_ys.append(fy)
            out_cols.append(fc)
            out_facs.append(torch.full((free_u.numel(),), -1, dtype=torch.int16, device=dev))
            out_uids.append(self.slot_user_ids[free_u].to(torch.int64))
            placed_slots_list.append(free_u)

        # 2. Gather ready members across active factions in O(N_ready) without per-faction GPU sync stalls
        fac_by_id = {f.id: f for f in factions if f.alive}
        borders_by_fac: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
        for fa, fb, rect in shared_borders:
            borders_by_fac.setdefault(fa, []).append((fb, rect))
            borders_by_fac.setdefault(fb, []).append((fa, rect))

        # Precompute recency grid once per tick instead of once per faction
        age_full = (int(tick) - canvas.last_edit_tick).clamp(min=0, max=200).to(torch.float32)
        rec_full = torch.exp(-age_full / 28.0)

        # Accumulators for a single combined neural forward pass across all factions
        bat_cx: list[torch.Tensor] = []
        bat_cy: list[torch.Tensor] = []
        bat_goal: list[torch.Tensor] = []
        bat_arch: list[torch.Tensor] = []
        bat_fac_id: list[torch.Tensor] = []
        bat_slots: list[torch.Tensor] = []

        org_mask = ready_fac >= 0
        org_u = ready_ids[org_mask]
        if org_u.numel() > 0:
            org_f = ready_fac[org_mask]
            order = torch.argsort(org_f)
            org_u = org_u[order]
            org_f = org_f[order]
            uniq_f, counts = torch.unique_consecutive(org_f, return_counts=True)
            uniq_f_list = uniq_f.detach().cpu().tolist()
            counts_list = counts.detach().cpu().tolist()

            offset = 0
            for fid_val, k in zip(uniq_f_list, counts_list):
                u_idx = org_u[offset : offset + k]
                offset += k
                fac = fac_by_id.get(int(fid_val))
                if fac is None:
                    self.user_factions[u_idx] = -1
                    continue

                if fac.is_void:
                    vx, vy, vc = self._step_void_faction(fac, k, canvas)
                    n_out = int(vx.numel())
                    if n_out > 0:
                        used = u_idx[:n_out]
                        self.pos_x[used] = vx
                        self.pos_y[used] = vy
                        out_xs.append(vx)
                        out_ys.append(vy)
                        out_cols.append(vc)
                        out_facs.append(torch.full((n_out,), fac.id, dtype=torch.int16, device=dev))
                        out_uids.append(self.slot_user_ids[used].to(torch.int64))
                        placed_slots_list.append(used)
                    continue

                if fac.art_mode == "clean_space":
                    vx, vy, vc = StochasticPolicy._step_art_faction(
                        self, tick, fac, k, canvas, fac_by_id, borders_by_fac.get(fac.id, [])
                    )
                    n_out = int(vx.numel())
                    if n_out > 0:
                        used = u_idx[:n_out]
                        self.pos_x[used] = vx
                        self.pos_y[used] = vy
                        out_xs.append(vx)
                        out_ys.append(vy)
                        out_cols.append(vc)
                        out_facs.append(torch.full((n_out,), fac.id, dtype=torch.int16, device=dev))
                        out_uids.append(self.slot_user_ids[used].to(torch.int64))
                        placed_slots_list.append(used)
                    continue

                # Standard/flag/sprite/painting/ribbon/fractal faction (avoid .item() GPU syncs)
                x0, y0, x1, y1 = fac.x0, fac.y0, fac.x1, fac.y1
                h, w = fac.h, fac.w

                rw = fac.role_weights
                w_build = rw.get("builder", 0.6)
                w_def = rw.get("defender", 0.25)
                w_exp = rw.get("expander", 0.15)
                total_w = max(1e-4, w_build + w_def + w_exp)
                k_build = min(k, int(round(k * (w_build / total_w))))
                k_def = min(k - k_build, int(round(k * (w_def / total_w))))
                k_exp = max(0, k - k_build - k_def)

                slot_cursor = 0
                k_neural = k_build + k_def
                # When integrity >= 99.5%, redirect builders to ally defense or expansion
                if k_neural > 0 and fac.integrity < 0.995:
                    live = canvas.colors[y0:y1, x0:x1].to(torch.int16)
                    bp = fac.blueprint
                    valid = bp >= 0
                    mm_f = (valid & (live != bp)).to(torch.float32)
                    rec_w = rec_full[y0:y1, x0:x1]
                    # Small positive epsilon on valid pixels avoids .any().item() / .sum().item() MPS stalls
                    flat = (mm_f * (0.35 + 1.5 * rec_w + 0.6 * fac.edge_weight) + valid.to(torch.float32) * 1e-5 + 1e-9).flatten()
                    idx = torch.multinomial(flat, k_neural, replacement=True)
                    ay = (idx // w).to(torch.int32) + y0
                    ax = (idx % w).to(torch.int32) + x0
                    eff_j = min(j_lim, max(2, min(w, h) // 2))
                    cx = (ax + torch.randint(-eff_j, eff_j + 1, (k_neural,), dtype=torch.int32, device=dev)).clamp(0, self.width - 1)
                    cy = (ay + torch.randint(-eff_j, eff_j + 1, (k_neural,), dtype=torch.int32, device=dev)).clamp(0, self.height - 1)

                    goal_p = self._extract_goal_patches(fac, cx, cy)
                    arch = torch.empty(k_neural, dtype=torch.long, device=dev)
                    arch[:k_build] = ARCH_BUILDER
                    arch[k_build:] = ARCH_DEFENDER

                    bat_cx.append(cx)
                    bat_cy.append(cy)
                    bat_goal.append(goal_p)
                    bat_arch.append(arch)
                    bat_fac_id.append(torch.full((k_neural,), fac.id, dtype=torch.int16, device=dev))
                    bat_slots.append(u_idx[slot_cursor : slot_cursor + k_neural])
                    slot_cursor += k_neural
                elif k_neural > 0:
                    helped = False
                    if k_def > 0:
                        for ally_id in fac.allies:
                            ally = fac_by_id.get(ally_id)
                            if ally is not None and (not ally.is_void) and ally.integrity < 0.90:
                                ax, ay, ac = self._assist_ally(ally, k_def, canvas)
                                n_a = int(ax.numel())
                                if n_a > 0:
                                    used = u_idx[slot_cursor : slot_cursor + n_a]
                                    slot_cursor += n_a
                                    self.pos_x[used] = ax
                                    self.pos_y[used] = ay
                                    out_xs.append(ax)
                                    out_ys.append(ay)
                                    out_cols.append(ac)
                                    out_facs.append(torch.full((n_a,), fac.id, dtype=torch.int16, device=dev))
                                    out_uids.append(self.slot_user_ids[used].to(torch.int64))
                                    placed_slots_list.append(used)
                                    helped = True
                                    break
                    k_exp += k_build + (0 if helped else k_def)

                if k_exp > 0 and slot_cursor < k:
                    k_exp_eff = min(k_exp, k - slot_cursor)
                    ex, ey, ec = self._step_expanders(
                        fac, k_exp_eff, canvas, fac_by_id, borders_by_fac.get(fac.id, [])
                    )
                    n_e = int(ex.numel())
                    if n_e > 0:
                        used = u_idx[slot_cursor : slot_cursor + n_e]
                        self.pos_x[used] = ex
                        self.pos_y[used] = ey
                        out_xs.append(ex)
                        out_ys.append(ey)
                        out_cols.append(ec)
                        out_facs.append(torch.full((n_e,), fac.id, dtype=torch.int16, device=dev))
                        out_uids.append(self.slot_user_ids[used].to(torch.int64))
                        placed_slots_list.append(used)

        # 3. Run one combined GPU neural forward pass for all queued faction builders/defenders
        if bat_cx:
            all_cx = torch.cat(bat_cx)
            all_cy = torch.cat(bat_cy)
            all_goal = torch.cat(bat_goal)
            all_arch = torch.cat(bat_arch)
            all_fid = torch.cat(bat_fac_id)
            all_slots = torch.cat(bat_slots)

            cur_p, prev_p, rec_p = self._extract_canvas_patches(tick, canvas, all_cx, all_cy)
            mm_pos = ((all_goal < NO_GOAL_IDX) & (cur_p != all_goal)).flatten(1)
            any_bp = (all_goal < NO_GOAL_IDX).flatten(1)
            has_mm = mm_pos.any(dim=1, keepdim=True)
            valid_pos = torch.where(has_mm, mm_pos, any_bp)
            has_valid = valid_pos.any(dim=1)
            valid_pos[~has_valid, R * P + R] = True

            dy, dx, cols = self._run_net_batched(
                cur_p,
                prev_p,
                rec_p,
                all_goal,
                all_arch,
                temp_pos=0.75,
                temp_col=0.55,
                valid_pos_mask=valid_pos,
            )
            px = (all_cx + dx).clamp(0, self.width - 1)
            py = (all_cy + dy).clamp(0, self.height - 1)
            self.pos_x[all_slots] = px
            self.pos_y[all_slots] = py
            out_xs.append(px)
            out_ys.append(py)
            out_cols.append(cols)
            out_facs.append(all_fid)
            out_uids.append(self.slot_user_ids[all_slots].to(torch.int64))
            placed_slots_list.append(all_slots)

        if not out_xs:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
            return PlacementBatch(
                xs=empty_i32,
                ys=empty_i32,
                colors=torch.empty(0, dtype=torch.uint8, device=dev),
                factions=torch.empty(0, dtype=torch.int16, device=dev),
                user_ids=empty_i32,
            )

        batch = PlacementBatch(
            xs=torch.cat(out_xs),
            ys=torch.cat(out_ys),
            colors=torch.cat(out_cols),
            factions=torch.cat(out_facs),
            user_ids=torch.cat(out_uids),
        )
        self._consume_and_recycle_sessions(torch.cat(placed_slots_list))
        return batch

    def _step_free_agents(
        self, u_ids: torch.Tensor, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(u_ids.numel())
        dev = self.device
        W, H = self.width, self.height

        jump_mask = torch.rand(n, device=dev) < 0.14
        cx = self.pos_x[u_ids].clone()
        cy = self.pos_y[u_ids].clone()

        n_jump = int(jump_mask.sum().item())
        if n_jump > 0:
            corner_mask = torch.rand(n_jump, device=dev) < 0.25
            pool_w = max(1, W // 16)
            coarse = F.avg_pool2d(
                canvas.activity.unsqueeze(0).unsqueeze(0), kernel_size=16, stride=16
            ).flatten()
            cell_idx = torch.multinomial(coarse + 0.12, n_jump, replacement=True)
            jx = ((cell_idx % pool_w) * 16 + torch.randint(0, 16, (n_jump,), device=dev)).to(torch.int32)
            jy = ((cell_idx // pool_w) * 16 + torch.randint(0, 16, (n_jump,), device=dev)).to(torch.int32)
            n_c = int(corner_mask.sum().item())
            if n_c > 0:
                which = torch.randint(0, 4, (n_c,), device=dev)
                edge_off_x = torch.randint(0, 42, (n_c,), dtype=torch.int32, device=dev)
                edge_off_y = torch.randint(0, 42, (n_c,), dtype=torch.int32, device=dev)
                ex = torch.where((which & 1) == 0, edge_off_x, (W - 1) - edge_off_x)
                ey = torch.where((which & 2) == 0, edge_off_y, (H - 1) - edge_off_y)
                jx[corner_mask] = ex
                jy[corner_mask] = ey
            cx[jump_mask] = jx
            cy[jump_mask] = jy

        cx = (cx + torch.randint(-3, 4, (n,), dtype=torch.int32, device=dev)).clamp(0, W - 1)
        cy = (cy + torch.randint(-3, 4, (n,), dtype=torch.int32, device=dev)).clamp(0, H - 1)

        cur_p, prev_p, rec_p = self._extract_canvas_patches(self._cur_tick, canvas, cx, cy)
        goal_p = torch.full_like(cur_p, NO_GOAL_IDX)
        arch = self.citizen_arch[u_ids]

        dy, dx, cols = self._run_net_batched(
            cur_p, prev_p, rec_p, goal_p, arch, temp_pos=self.temp_pos, temp_col=self.temp_col
        )

        blank_patch = (cur_p == WHITE_IDX).all(dim=(1, 2))
        seed_blank = blank_patch & (torch.rand(n, device=dev) < 0.45)
        cols = torch.where(seed_blank, self.pref_color[u_ids], cols)

        px = (cx + dx).clamp(0, W - 1)
        py = (cy + dy).clamp(0, H - 1)
        self.pos_x[u_ids] = px
        self.pos_y[u_ids] = py
        return px, py, cols

    def _step_void_faction(
        self, fac: Faction, k: int, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drive 4 Void variants: neural VOID tendrils + structured Black Hole / Fractal / Void Monster blueprints."""
        dev = self.device
        x0, y0, x1, y1 = fac.x0, fac.y0, fac.x1, fac.y1
        h, w = y1 - y0, x1 - x0

        xs_parts: list[torch.Tensor] = []
        ys_parts: list[torch.Tensor] = []
        cs_parts: list[torch.Tensor] = []
        k_tendril = k

        if fac.void_variant in ("black_hole", "fractal", "monster") and fac.blueprint.shape == (h, w):
            bp = fac.blueprint
            live = canvas.colors[y0:y1, x0:x1].to(torch.int16)
            mm = (bp >= 0) & (live != bp)
            k_bp = int(round(k * 0.68))
            if k_bp > 0 and bool(mm.any().item()):
                flat = (mm.to(torch.float32) * (0.25 + fac.edge_weight)).flatten()
                idx = torch.multinomial(flat, k_bp, replacement=True)
                ly = idx // w
                lx = idx % w
                xs_parts.append((lx + x0).to(torch.int32))
                ys_parts.append((ly + y0).to(torch.int32))
                cs_parts.append(bp[ly, lx].to(torch.uint8))
                k_tendril = k - k_bp

        if k_tendril > 0:
            live = canvas.colors[y0:y1, x0:x1]
            is_black = live == BLACK_IDX
            if not bool(is_black.any().item()):
                cx = torch.full((k_tendril,), (x0 + x1) // 2, dtype=torch.int32, device=dev) + torch.randint(-3, 4, (k_tendril,), dtype=torch.int32, device=dev)
                cy = torch.full((k_tendril,), (y0 + y1) // 2, dtype=torch.int32, device=dev) + torch.randint(-3, 4, (k_tendril,), dtype=torch.int32, device=dev)
                xs_parts.append(cx.clamp(0, self.width - 1))
                ys_parts.append(cy.clamp(0, self.height - 1))
                cs_parts.append(torch.full((k_tendril,), BLACK_IDX, dtype=torch.uint8, device=dev))
            else:
                bf = is_black.to(torch.float32)
                nbr = torch.zeros_like(bf)
                if h > 1:
                    nbr[1:, :] += bf[:-1, :]
                    nbr[:-1, :] += bf[1:, :]
                if w > 1:
                    nbr[:, 1:] += bf[:, :-1]
                    nbr[:, :-1] += bf[:, 1:]
                frontier = ((nbr > 0) & (~is_black)).to(torch.float32) + 0.01
                idx = torch.multinomial(frontier.flatten(), k_tendril, replacement=True)
                cy = (idx // w).to(torch.int32) + y0
                cx = (idx % w).to(torch.int32) + x0

                cur_p, prev_p, rec_p = self._extract_canvas_patches(self._cur_tick, canvas, cx, cy)
                goal_p = torch.full_like(cur_p, NO_GOAL_IDX)
                arch = torch.full((k_tendril,), ARCH_VOID, dtype=torch.long, device=dev)
                dy, dx, cols = self._run_net_batched(cur_p, prev_p, rec_p, goal_p, arch, temp_pos=0.90, temp_col=0.65)
                xs_parts.append((cx + dx).clamp(x0, x1 - 1))
                ys_parts.append((cy + dy).clamp(y0, y1 - 1))
                cs_parts.append(cols)

        return torch.cat(xs_parts), torch.cat(ys_parts), torch.cat(cs_parts)
