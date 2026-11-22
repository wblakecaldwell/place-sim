"""GPU-batched PyTorch MPS stochastic r/place citizen policy with real r/place population turnover (20% One-Shot)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pixel_battle.canvas import CanvasState
from pixel_battle.factions import Faction
from pixel_battle.palette import (
    BLACK_IDX,
    BURGUNDY_IDX,
    DARK_RED_IDX,
    NUM_COLORS,
    PINK_IDX,
    RED_IDX,
    WHITE_IDX,
)
from pixel_battle.policies.base import PlacementBatch

# Per-tier member capacity so Micro sprites stay nimble while Macro flags & Ribbons field armies
_TIER_MEMBER_CAP: dict[str, int] = {
    "micro": 135,
    "medium": 360,
    "macro": 950,
    "ribbon": 850,
}


class StochasticPolicy:
    """Vectorized GPU citizen simulator with dynamic session click budgets (20% One-Shot Citizens)."""

    def __init__(
        self,
        num_users: int,
        width: int,
        height: int,
        base_cooldown: int,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.num_users = num_users
        self.width = width
        self.height = height
        self.base_cooldown = max(4, base_cooldown)
        self.device = device

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.pos_x = torch.randint(0, width, (num_users,), generator=g, dtype=torch.int32).to(device)
        self.pos_y = torch.randint(0, height, (num_users,), generator=g, dtype=torch.int32).to(device)
        self.cooldown = torch.randint(1, self.base_cooldown + 1, (num_users,), generator=g, dtype=torch.int32).to(device)
        self.user_factions = torch.full((num_users,), -1, dtype=torch.int16, device=device)

        # Dynamic r/place session turnover (REQ-SIM-013, REQ-SIM-014):
        # Each GPU online slot holds a unique monotonic 32-bit user_id and heavy-tailed click budget.
        self.slot_user_ids = torch.arange(1, num_users + 1, dtype=torch.int64, device=device)
        self.next_user_id: int = num_users + 1
        init_budgets = self._sample_click_budgets(num_users, gen=g)
        self.initial_clicks = init_budgets.clone()
        self.remaining_clicks = init_budgets.clone()
        self.total_unique_users: int = num_users
        self.total_completed_sessions: int = 0
        self.total_one_shot_sessions: int = 0

        # Aesthetic preference: spatially correlated color basins + individual variety
        n_basins = 12
        bx = torch.randint(16, max(17, width - 16), (n_basins,), generator=g, dtype=torch.float32)
        by = torch.randint(16, max(17, height - 16), (n_basins,), generator=g, dtype=torch.float32)
        bcol = torch.randint(1, NUM_COLORS, (n_basins,), generator=g, dtype=torch.uint8)
        px_f = self.pos_x.cpu().to(torch.float32).unsqueeze(1)
        py_f = self.pos_y.cpu().to(torch.float32).unsqueeze(1)
        d2 = (px_f - bx.unsqueeze(0)) ** 2 + (py_f - by.unsqueeze(0)) ** 2
        nearest_basin = d2.argmin(dim=1)
        basin_pref = bcol[nearest_basin]
        rand_pref = torch.randint(1, NUM_COLORS, (num_users,), generator=g, dtype=torch.uint8)
        use_basin = torch.rand(num_users, generator=g) < 0.65
        self.pref_color = torch.where(use_basin, basin_pref, rand_pref).to(device)

        # Citizen archetype traits:
        # 0: Schelling conformist (44%), 1: Striper (22%), 2: Lattice (14%), 3: Doodler (13%), 4: Void-curious (7%)
        r = torch.rand(num_users, generator=g)
        traits = torch.zeros(num_users, dtype=torch.uint8)
        traits = torch.where(r >= 0.44, torch.tensor(1, dtype=torch.uint8), traits)
        traits = torch.where(r >= 0.66, torch.tensor(2, dtype=torch.uint8), traits)
        traits = torch.where(r >= 0.80, torch.tensor(3, dtype=torch.uint8), traits)
        traits = torch.where(r >= 0.93, torch.tensor(4, dtype=torch.uint8), traits)
        self.trait = traits.to(device)

        self.loyalty = (0.25 + 0.70 * torch.rand(num_users, generator=g)).to(device)

        # Tiny 7x7 heart motif mask for allied border seams
        heart = torch.tensor(
            [
                [0, 1, 1, 0, 1, 1, 0],
                [1, 2, 2, 1, 2, 2, 1],
                [1, 2, 2, 2, 2, 2, 1],
                [1, 2, 2, 2, 2, 2, 1],
                [0, 1, 2, 2, 2, 1, 0],
                [0, 0, 1, 2, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=torch.uint8,
            device=device,
        )
        self._heart_mask = heart

    def _sample_click_budgets(self, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """Sample real r/place heavy-tailed click budgets: 20% One-Shot(1), 50% casual(2-12), 24% regular(13-70), 6% veteran(71-350)."""
        if gen is not None:
            u = torch.rand(n, generator=gen)
            v = torch.rand(n, generator=gen) ** 1.55
            dev = torch.device("cpu")
        else:
            dev = self.device
            u = torch.rand(n, device=dev)
            v = torch.rand(n, device=dev) ** 1.55

        casual = (2 + (v * 11.0).to(torch.int32)).clamp(2, 12)
        regular = (13 + (v * 58.0).to(torch.int32)).clamp(13, 70)
        veteran = (71 + (v * 280.0).to(torch.int32)).clamp(71, 350)

        out = torch.ones(n, dtype=torch.int32, device=dev)
        out = torch.where(u >= 0.20, casual, out)
        out = torch.where(u >= 0.70, regular, out)
        out = torch.where(u >= 0.94, veteran, out)
        return out.to(self.device)

    def _consume_and_recycle_sessions(self, placed_slot_ids: torch.Tensor) -> None:
        """Decrement click budgets for slots that placed a pixel; recycle expired slots into brand-new unique user_ids."""
        if placed_slot_ids.numel() == 0:
            return
        dev = self.device
        # Deduplicate just in case (normally unique within a single tick)
        self.remaining_clicks[placed_slot_ids] -= 1
        done_mask = self.remaining_clicks[placed_slot_ids] <= 0
        if not bool(done_mask.any().item()):
            return
        done_slots = placed_slot_ids[done_mask]
        n_done = int(done_slots.numel())
        n_one_shot = int((self.initial_clicks[done_slots] == 1).sum().item())
        self.total_completed_sessions += n_done
        self.total_one_shot_sessions += n_one_shot

        # Assign brand-new unique user IDs & fresh click budgets
        new_ids = torch.arange(
            self.next_user_id,
            self.next_user_id + n_done,
            dtype=torch.int64,
            device=dev,
        )
        self.next_user_id += n_done
        self.total_unique_users += n_done

        new_budgets = self._sample_click_budgets(n_done)
        self.slot_user_ids[done_slots] = new_ids
        self.initial_clicks[done_slots] = new_budgets
        self.remaining_clicks[done_slots] = new_budgets
        self.user_factions[done_slots] = -1
        self.pos_x[done_slots] = torch.randint(0, self.width, (n_done,), dtype=torch.int32, device=dev)
        self.pos_y[done_slots] = torch.randint(0, self.height, (n_done,), dtype=torch.int32, device=dev)
        self.pref_color[done_slots] = torch.randint(1, NUM_COLORS, (n_done,), dtype=torch.uint8, device=dev)
        self.loyalty[done_slots] = 0.25 + 0.70 * torch.rand(n_done, device=dev)

    def get_population_stats(self) -> dict[str, float | int]:
        one_shot_pct = (
            100.0 * self.total_one_shot_sessions / self.total_completed_sessions
            if self.total_completed_sessions > 0
            else 20.0
        )
        return {
            "online_slots": self.num_users,
            "total_unique_users": self.total_unique_users,
            "completed_sessions": self.total_completed_sessions,
            "one_shot_sessions": self.total_one_shot_sessions,
            "one_shot_pct": round(one_shot_pct, 2),
        }

    def step(
        self,
        tick: int,
        canvas: CanvasState,
        factions: list[Faction],
        shared_borders: list[tuple[int, int, tuple[int, int, int, int]]],
    ) -> PlacementBatch:
        self.cooldown -= 1
        ready_mask = self.cooldown <= 0
        ready_ids = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)

        if ready_ids.numel() == 0:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=self.device)
            return PlacementBatch(
                xs=empty_i32,
                ys=empty_i32,
                colors=torch.empty(0, dtype=torch.uint8, device=self.device),
                factions=torch.empty(0, dtype=torch.int16, device=self.device),
                user_ids=empty_i32,
            )

        # Reset cooldowns with human-like jitter
        lo = max(2, int(self.base_cooldown * 0.65))
        hi = max(lo + 2, int(self.base_cooldown * 1.35))
        self.cooldown[ready_ids] = torch.randint(
            lo, hi, (ready_ids.numel(),), dtype=torch.int32, device=self.device
        )

        out_xs: list[torch.Tensor] = []
        out_ys: list[torch.Tensor] = []
        out_cols: list[torch.Tensor] = []
        out_facs: list[torch.Tensor] = []
        out_uids: list[torch.Tensor] = []
        placed_slots_list: list[torch.Tensor] = []

        ready_fac = self.user_factions[ready_ids]

        # 1. Free agents (faction_id == -1)
        free_u = ready_ids[ready_fac == -1]
        if free_u.numel() > 0:
            fx, fy, fc = self._step_free_agents(free_u, canvas)
            out_xs.append(fx)
            out_ys.append(fy)
            out_cols.append(fc)
            out_facs.append(torch.full((free_u.numel(),), -1, dtype=torch.int16, device=self.device))
            out_uids.append(self.slot_user_ids[free_u].to(torch.int64))
            placed_slots_list.append(free_u)

        # Index active factions and allied borders
        fac_by_id = {f.id: f for f in factions if f.alive}
        borders_by_fac: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
        for fa, fb, rect in shared_borders:
            borders_by_fac.setdefault(fa, []).append((fb, rect))
            borders_by_fac.setdefault(fb, []).append((fa, rect))

        # O(N_ready) grouping of faction members so 200+ factions don't scan ready_fac 200 times
        org_mask = ready_fac >= 0
        if bool(org_mask.any().item()):
            org_u = ready_ids[org_mask]
            org_f = ready_fac[org_mask]
            order = torch.argsort(org_f)
            org_u = org_u[order]
            org_f = org_f[order]
            uniq_f, counts = torch.unique_consecutive(org_f, return_counts=True)
            uniq_f_list = uniq_f.detach().cpu().tolist()
            counts_list = counts.detach().cpu().tolist()

            offset = 0
            for fid_val, cnt in zip(uniq_f_list, counts_list):
                u_idx = org_u[offset : offset + cnt]
                offset += cnt
                fac = fac_by_id.get(int(fid_val))
                if fac is None:
                    self.user_factions[u_idx] = -1
                    continue

                if fac.is_void:
                    vx, vy, vc = self._step_void_faction(fac, cnt, canvas)
                else:
                    vx, vy, vc = self._step_art_faction(
                        tick, fac, cnt, canvas, fac_by_id, borders_by_fac.get(fac.id, [])
                    )

                n_out = int(vx.numel())
                if n_out > 0:
                    used_slots = u_idx[:n_out]
                    self.pos_x[used_slots] = vx
                    self.pos_y[used_slots] = vy
                    out_xs.append(vx)
                    out_ys.append(vy)
                    out_cols.append(vc)
                    out_facs.append(torch.full((n_out,), fac.id, dtype=torch.int16, device=self.device))
                    out_uids.append(self.slot_user_ids[used_slots].to(torch.int64))
                    placed_slots_list.append(used_slots)

        if not out_xs:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=self.device)
            return PlacementBatch(
                xs=empty_i32,
                ys=empty_i32,
                colors=torch.empty(0, dtype=torch.uint8, device=self.device),
                factions=torch.empty(0, dtype=torch.int16, device=self.device),
                user_ids=empty_i32,
            )

        batch = PlacementBatch(
            xs=torch.cat(out_xs),
            ys=torch.cat(out_ys),
            colors=torch.cat(out_cols),
            factions=torch.cat(out_facs),
            user_ids=torch.cat(out_uids),
        )
        # Recycle finished sessions after emitting their user_id into the batch
        self._consume_and_recycle_sessions(torch.cat(placed_slots_list))
        return batch

    def _step_free_agents(
        self, u_ids: torch.Tensor, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(u_ids.numel())
        dev = self.device
        W, H = self.width, self.height

        # 12% jump toward trending activity hotspots or flush corners
        jump_mask = torch.rand(n, device=dev) < 0.12
        x = self.pos_x[u_ids].clone()
        y = self.pos_y[u_ids].clone()

        n_jump = int(jump_mask.sum().item())
        if n_jump > 0:
            pool_w = max(1, W // 16)
            coarse = F.avg_pool2d(
                canvas.activity.unsqueeze(0).unsqueeze(0), kernel_size=16, stride=16
            ).flatten()
            probs = coarse + 0.08
            cell_idx = torch.multinomial(probs, n_jump, replacement=True)
            cy = (cell_idx // pool_w) * 16 + torch.randint(0, 16, (n_jump,), device=dev)
            cx = (cell_idx % pool_w) * 16 + torch.randint(0, 16, (n_jump,), device=dev)
            x[jump_mask] = cx.to(torch.int32)
            y[jump_mask] = cy.to(torch.int32)

        dx = torch.randint(-2, 3, (n,), dtype=torch.int32, device=dev)
        dy = torch.randint(-2, 3, (n,), dtype=torch.int32, device=dev)
        x = (x + dx).clamp(0, W - 1)
        y = (y + dy).clamp(0, H - 1)

        # Safe clamped 8-neighborhood read (allows roaming all the way to 0 and W-1/H-1)
        xm = (x - 1).clamp(0, W - 1)
        xp = (x + 1).clamp(0, W - 1)
        ym = (y - 1).clamp(0, H - 1)
        yp = (y + 1).clamp(0, H - 1)

        c_l = canvas.colors[y, xm]
        c_r = canvas.colors[y, xp]
        c_u = canvas.colors[ym, x]
        c_d = canvas.colors[yp, x]
        c_ul = canvas.colors[ym, xm]
        c_ur = canvas.colors[ym, xp]
        c_dl = canvas.colors[yp, xm]
        c_dr = canvas.colors[yp, xp]

        nbrs = torch.stack([c_l, c_r, c_u, c_d, c_ul, c_ur, c_dl, c_dr], dim=1)
        non_white = nbrs != WHITE_IDX
        has_nw = non_white.any(dim=1)
        same_as = (nbrs.unsqueeze(2) == nbrs.unsqueeze(1)) & non_white.unsqueeze(2)
        freq = same_as.sum(dim=2).to(torch.float32) * non_white.to(torch.float32)
        nbr_score = freq + 0.35 * torch.rand((n, 8), device=dev) * non_white.to(torch.float32)
        best_nbr_idx = nbr_score.argmax(dim=1)
        nbr_col = nbrs[torch.arange(n, device=dev), best_nbr_idx]

        pref = self.pref_color[u_ids]
        traits = self.trait[u_ids]
        colors = pref.clone()

        m0 = (traits == 0) & has_nw & (torch.rand(n, device=dev) < 0.86)
        colors = torch.where(m0, nbr_col, colors)

        horiz_nw = (c_l != WHITE_IDX) | (c_r != WHITE_IDX)
        horiz_col = torch.where(c_l != WHITE_IDX, c_l, c_r)
        m1 = (traits == 1) & horiz_nw & (torch.rand(n, device=dev) < 0.78)
        colors = torch.where(m1, horiz_col, colors)
        x = torch.where(traits == 1, (x + torch.randint(-2, 3, (n,), dtype=torch.int32, device=dev)).clamp(0, W - 1), x)

        m2 = traits == 2
        parity = ((x + y) & 1) == 0
        alt_col = torch.where(has_nw, nbr_col, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev))
        colors = torch.where(m2 & (~parity), alt_col, colors)

        m3 = (traits == 3) & (torch.rand(n, device=dev) < 0.30)
        rand_col = torch.randint(1, NUM_COLORS, (n,), dtype=torch.uint8, device=dev)
        colors = torch.where(m3, rand_col, colors)

        touches_black = (nbrs == BLACK_IDX).any(dim=1)
        m4_spread = (traits == 4) & touches_black
        m4_red = m4_spread & (torch.rand(n, device=dev) < 0.14)
        colors = torch.where(m4_spread, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev), colors)
        colors = torch.where(m4_red, torch.tensor(DARK_RED_IDX, dtype=torch.uint8, device=dev), colors)
        m4_seed = (traits == 4) & (~touches_black) & (torch.rand(n, device=dev) < 0.25)
        colors = torch.where(m4_seed, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev), colors)

        self.pos_x[u_ids] = x
        self.pos_y[u_ids] = y
        return x, y, colors

    def _step_art_faction(
        self,
        tick: int,
        fac: Faction,
        k: int,
        canvas: CanvasState,
        fac_by_id: dict[int, Faction],
        my_borders: list[tuple[int, tuple[int, int, int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        x0, y0, x1, y1 = fac.x0, fac.y0, fac.x1, fac.y1
        h, w = fac.h, fac.w

        live = canvas.colors[y0:y1, x0:x1].to(torch.int16)
        bp = fac.blueprint
        valid = bp >= 0
        mismatch = valid & (live != bp)
        correct = valid & (live == bp)

        adj_correct = torch.zeros_like(correct, dtype=torch.float32)
        cf = correct.to(torch.float32)
        if h > 1:
            adj_correct[1:, :] += cf[:-1, :]
            adj_correct[:-1, :] += cf[1:, :]
        if w > 1:
            adj_correct[:, 1:] += cf[:, :-1]
            adj_correct[:, :-1] += cf[:, 1:]
        adj_correct = torch.clamp(adj_correct, 0.0, 2.0)

        age = (int(tick) - canvas.last_edit_tick[y0:y1, x0:x1]).clamp(min=0, max=240).to(torch.float32)
        recency = torch.exp(-age / 28.0)

        rw = fac.role_weights
        w_build = rw.get("builder", 0.6)
        w_def = rw.get("defender", 0.25)
        w_exp = rw.get("expander", 0.15)
        total_w = max(1e-4, w_build + w_def + w_exp)
        k_build = min(k, int(round(k * (w_build / total_w))))
        k_def = min(k - k_build, int(round(k * (w_def / total_w))))
        k_exp = max(0, k - k_build - k_def)

        xs_list: list[torch.Tensor] = []
        ys_list: list[torch.Tensor] = []
        cs_list: list[torch.Tensor] = []

        has_mismatch = bool(mismatch.any().item())
        mm_f = mismatch.to(torch.float32)

        if k_build > 0 and has_mismatch:
            build_prob = mm_f * (0.20 + fac.edge_weight + 1.6 * adj_correct)
            flat = build_prob.flatten()
            if float(flat.sum().item()) > 0:
                idx = torch.multinomial(flat, k_build, replacement=True)
                ly = idx // w
                lx = idx % w
                cols = bp[ly, lx].to(torch.uint8)
                # 2.0% human misclick noise (skipped for clean_space purists who strictly erase to white)
                if fac.art_mode != "clean_space":
                    misclick = torch.rand(k_build, device=dev) < 0.02
                    if misclick.any():
                        cols = torch.where(
                            misclick,
                            torch.full_like(cols, fac.accent_color % NUM_COLORS),
                            cols,
                        )
                xs_list.append((lx + x0).to(torch.int32))
                ys_list.append((ly + y0).to(torch.int32))
                cs_list.append(cols)
        elif k_build > 0:
            k_exp += k_build

        if k_def > 0:
            if has_mismatch:
                def_prob = mm_f * (0.12 + 6.5 * recency + 0.7 * fac.edge_weight)
                flat = def_prob.flatten()
                if float(flat.sum().item()) > 0:
                    idx = torch.multinomial(flat, k_def, replacement=True)
                    ly = idx // w
                    lx = idx % w
                    xs_list.append((lx + x0).to(torch.int32))
                    ys_list.append((ly + y0).to(torch.int32))
                    cs_list.append(bp[ly, lx].to(torch.uint8))
            else:
                helped = False
                for ally_id in fac.allies:
                    ally = fac_by_id.get(ally_id)
                    if ally is not None and (not ally.is_void) and ally.integrity < 0.90:
                        ax, ay, ac = self._assist_ally(ally, k_def, canvas)
                        if ax.numel() > 0:
                            xs_list.append(ax)
                            ys_list.append(ay)
                            cs_list.append(ac)
                            helped = True
                            break
                if not helped:
                    k_exp += k_def

        if k_exp > 0:
            ex, ey, ec = self._step_expanders(fac, k_exp, canvas, fac_by_id, my_borders)
            if ex.numel() > 0:
                xs_list.append(ex)
                ys_list.append(ey)
                cs_list.append(ec)

        if not xs_list:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
            return empty_i32, empty_i32, torch.empty(0, dtype=torch.uint8, device=dev)
        return torch.cat(xs_list), torch.cat(ys_list), torch.cat(cs_list)

    def _assist_ally(
        self, ally: Faction, k: int, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        live = canvas.colors[ally.y0 : ally.y1, ally.x0 : ally.x1].to(torch.int16)
        bp = ally.blueprint
        mm = (bp >= 0) & (live != bp)
        flat = mm.to(torch.float32).flatten()
        if float(flat.sum().item()) <= 0:
            empty = torch.empty(0, dtype=torch.int32, device=self.device)
            return empty, empty, torch.empty(0, dtype=torch.uint8, device=self.device)
        idx = torch.multinomial(flat, k, replacement=True)
        ly = idx // ally.w
        lx = idx % ally.w
        return (lx + ally.x0).to(torch.int32), (ly + ally.y0).to(torch.int32), bp[ly, lx].to(torch.uint8)

    def _step_expanders(
        self,
        fac: Faction,
        k: int,
        canvas: CanvasState,
        fac_by_id: dict[int, Faction],
        my_borders: list[tuple[int, tuple[int, int, int, int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        W, H = self.width, self.height

        # 1. Active Raid on a rival neighbor
        if fac.raid_target_id is not None and fac.raid_target_id in fac_by_id:
            target = fac_by_id[fac.raid_target_id]
            rx = torch.randint(target.x0, max(target.x0 + 1, target.x1), (k,), dtype=torch.int32, device=dev)
            ry = torch.randint(target.y0, max(target.y0 + 1, target.y1), (k,), dtype=torch.int32, device=dev)
            pull = torch.rand(k, device=dev) < 0.60
            cx_i = int(max(target.x0, min(target.x1 - 1, round(fac.cx))))
            cy_i = int(max(target.y0, min(target.y1 - 1, round(fac.cy))))
            rx = torch.where(pull, ((rx + cx_i) // 2).clamp(0, W - 1), rx)
            ry = torch.where(pull, ((ry + cy_i) // 2).clamp(0, H - 1), ry)
            col_pick = torch.where(
                torch.rand(k, device=dev) < 0.72,
                torch.tensor(fac.primary_color % NUM_COLORS, dtype=torch.uint8, device=dev),
                torch.tensor(fac.secondary_color % NUM_COLORS, dtype=torch.uint8, device=dev),
            )
            return rx, ry, col_pick

        # 2. Allied border hearts
        if my_borders and k >= 2:
            k_heart = max(1, int(k * 0.45))
            k_rest = k - k_heart
            ally_id, (bx0, by0, bx1, by1) = my_borders[0]
            ally = fac_by_id.get(ally_id)
            heart_col = PINK_IDX if (fac.id + ally_id) % 2 == 0 else RED_IDX
            hx = torch.randint(bx0, max(bx0 + 1, bx1), (k_heart,), dtype=torch.int32, device=dev).clamp(0, W - 1)
            hy = torch.randint(by0, max(by0 + 1, by1), (k_heart,), dtype=torch.int32, device=dev).clamp(0, H - 1)
            pat = self._heart_mask[(hy - by0) % 7, (hx - bx0) % 7]
            bg_col = (
                fac.primary_color % NUM_COLORS
                if ally is None
                else (fac.primary_color if (fac.id % 2 == 0) else ally.primary_color) % NUM_COLORS
            )
            hc = torch.full((k_heart,), bg_col, dtype=torch.uint8, device=dev)
            hc = torch.where(pat == 1, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev), hc)
            hc = torch.where(pat == 2, torch.tensor(heart_col, dtype=torch.uint8, device=dev), hc)

            if k_rest == 0:
                return hx, hy, hc
            ox, oy, oc = self._expand_perimeter(fac, k_rest)
            return torch.cat([hx, ox]), torch.cat([hy, oy]), torch.cat([hc, oc])

        return self._expand_perimeter(fac, k)

    def _expand_perimeter(
        self, fac: Faction, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Organic territory growth hugging the silhouette or clean-space/ribbon frontier."""
        dev = self.device
        W, H = self.width, self.height

        # Clean-space purists keep erasing 1-2px outside their blank enclave cleanly in WHITE_IDX
        if fac.art_mode == "clean_space":
            rx = torch.randint(max(0, fac.x0 - 2), min(W, fac.x1 + 2), (k,), dtype=torch.int32, device=dev)
            ry = torch.randint(max(0, fac.y0 - 2), min(H, fac.y1 + 2), (k,), dtype=torch.int32, device=dev)
            return rx, ry, torch.full((k,), WHITE_IDX, dtype=torch.uint8, device=dev)

        # Keep micro-sprites, paintings, comic strips, and cartoon memes crisp inside their outline
        if fac.size_tier == "micro" or fac.art_mode in ("painting", "comic_strip", "cartoon_meme"):
            rx = torch.randint(fac.x0, max(fac.x0 + 1, fac.x1), (k,), dtype=torch.int32, device=dev)
            ry = torch.randint(fac.y0, max(fac.y0 + 1, fac.y1), (k,), dtype=torch.int32, device=dev)
            bp_c = fac.blueprint[(ry - fac.y0).clamp(0, fac.h - 1), (rx - fac.x0).clamp(0, fac.w - 1)]
            valid_m = bp_c >= 0
            return rx[valid_m], ry[valid_m], bp_c[valid_m].to(torch.uint8)

        # Transverse ribbons push directly past their advancing tips
        if fac.art_mode == "ribbon" and fac.ribbon_axis in ("h", "v"):
            if fac.ribbon_axis == "h":
                tip = torch.where(
                    torch.rand(k, device=dev) < 0.5,
                    (torch.full((k,), fac.x1, dtype=torch.int32, device=dev) + torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                    (torch.full((k,), fac.x0 - 1, dtype=torch.int32, device=dev) - torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                ).clamp(0, W - 1)
                ry = torch.randint(fac.y0, max(fac.y0 + 1, fac.y1), (k,), dtype=torch.int32, device=dev)
                col_idx = fac.blueprint[(ry - fac.y0).clamp(0, fac.h - 1), torch.zeros(k, dtype=torch.long, device=dev)].clamp(0, NUM_COLORS - 1).to(torch.uint8)
                return tip, ry, col_idx
            else:
                tip = torch.where(
                    torch.rand(k, device=dev) < 0.5,
                    (torch.full((k,), fac.y1, dtype=torch.int32, device=dev) + torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                    (torch.full((k,), fac.y0 - 1, dtype=torch.int32, device=dev) - torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                ).clamp(0, H - 1)
                rx = torch.randint(fac.x0, max(fac.x0 + 1, fac.x1), (k,), dtype=torch.int32, device=dev)
                col_idx = fac.blueprint[torch.zeros(k, dtype=torch.long, device=dev), (rx - fac.x0).clamp(0, fac.w - 1)].clamp(0, NUM_COLORS - 1).to(torch.uint8)
                return rx, tip, col_idx

        # Run silhouette halo expansion only on transparent cutouts to avoid .item() stalls on solid flags
        if fac.art_mode in ("diffusion", "procedural", "banner", "fractal", "black_hole", "void_monster"):
            bp = fac.blueprint
            valid = bp >= 0
            vf = valid.to(torch.float32)
            dil = torch.zeros_like(vf)
            h, w = fac.h, fac.w
            if h > 1:
                dil[1:, :] += vf[:-1, :]
                dil[:-1, :] += vf[1:, :]
            if w > 1:
                dil[:, 1:] += vf[:, :-1]
                dil[:, :-1] += vf[:, 1:]
            halo = (dil > 0) & (~valid)
            h_idx = torch.nonzero(halo, as_tuple=False)
            if h_idx.shape[0] > 0:
                pick = torch.randint(0, h_idx.shape[0], (k,), device=dev)
                ly = h_idx[pick, 0] + torch.randint(-2, 3, (k,), device=dev)
                lx = h_idx[pick, 1] + torch.randint(-2, 3, (k,), device=dev)
                edx, edy = fac.expand_dir
                if edx != 0 or edy != 0:
                    lx = lx + int(edx) * torch.randint(0, 4, (k,), device=dev)
                    ly = ly + int(edy) * torch.randint(0, 4, (k,), device=dev)
                rx = (lx + fac.x0).to(torch.int32).clamp(0, W - 1)
                ry = (ly + fac.y0).to(torch.int32).clamp(0, H - 1)
                r = torch.rand(k, device=dev)
                cols = torch.full((k,), fac.primary_color % NUM_COLORS, dtype=torch.uint8, device=dev)
                cols = torch.where(r > 0.65, torch.tensor(fac.secondary_color % NUM_COLORS, dtype=torch.uint8, device=dev), cols)
                cols = torch.where(r > 0.90, torch.tensor(fac.accent_color % NUM_COLORS, dtype=torch.uint8, device=dev), cols)
                return rx, ry, cols

        edx, edy = fac.expand_dir
        if edx == 0 and edy == 0:
            ang = torch.rand(k, device=dev) * (2.0 * math.pi)
            rad = (max(fac.w, fac.h) * 0.48) + torch.rand(k, device=dev) * 5.0
            rx = (fac.cx + rad * torch.cos(ang)).to(torch.int32).clamp(0, W - 1)
            ry = (fac.cy + rad * torch.sin(ang)).to(torch.int32).clamp(0, H - 1)
        else:
            rx = torch.randint(fac.x0, max(fac.x0 + 1, fac.x1), (k,), dtype=torch.int32, device=dev)
            ry = torch.randint(fac.y0, max(fac.y0 + 1, fac.y1), (k,), dtype=torch.int32, device=dev)
            drift = torch.randint(1, 5, (k,), dtype=torch.int32, device=dev)
            if edx > 0:
                rx = (torch.full_like(rx, fac.x1 - 1) + drift).clamp(0, W - 1)
            elif edx < 0:
                rx = (torch.full_like(rx, fac.x0) - drift).clamp(0, W - 1)
            if edy > 0:
                ry = (torch.full_like(ry, fac.y1 - 1) + drift).clamp(0, H - 1)
            elif edy < 0:
                ry = (torch.full_like(ry, fac.y0) - drift).clamp(0, H - 1)

        r = torch.rand(k, device=dev)
        cols = torch.where(
            r < 0.70,
            torch.tensor(fac.primary_color % NUM_COLORS, dtype=torch.uint8, device=dev),
            torch.tensor(fac.secondary_color % NUM_COLORS, dtype=torch.uint8, device=dev),
        )
        return rx, ry, cols

    def _step_void_faction(
        self, fac: Faction, k: int, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drive 4 Void variants: creeping tendrils, Black Hole, Void Fractal, or Abyssal Monster."""
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
            is_void_pix = (live == BLACK_IDX) | (live == BURGUNDY_IDX)
            if not bool(is_void_pix.any().item()):
                cx = torch.full((k_tendril,), (x0 + x1) // 2, dtype=torch.int32, device=dev) + torch.randint(-3, 4, (k_tendril,), dtype=torch.int32, device=dev)
                cy = torch.full((k_tendril,), (y0 + y1) // 2, dtype=torch.int32, device=dev) + torch.randint(-3, 4, (k_tendril,), dtype=torch.int32, device=dev)
                xs_parts.append(cx.clamp(0, self.width - 1))
                ys_parts.append(cy.clamp(0, self.height - 1))
                cs_parts.append(torch.full((k_tendril,), BLACK_IDX, dtype=torch.uint8, device=dev))
            else:
                vf = is_void_pix.to(torch.float32)
                nbr_count = torch.zeros_like(vf)
                if h > 1:
                    nbr_count[1:, :] += vf[:-1, :]
                    nbr_count[:-1, :] += vf[1:, :]
                if w > 1:
                    nbr_count[:, 1:] += vf[:, :-1]
                    nbr_count[:, :-1] += vf[:, 1:]
                if h > 1 and w > 1:
                    nbr_count[1:, 1:] += 0.45 * vf[:-1, :-1]
                    nbr_count[:-1, :-1] += 0.45 * vf[1:, 1:]
                    nbr_count[1:, :-1] += 0.45 * vf[:-1, 1:]
                    nbr_count[:-1, 1:] += 0.45 * vf[1:, :-1]

                frontier = (nbr_count > 0) & (~is_void_pix)
                non_white_bonus = (live != WHITE_IDX).to(torch.float32) * 2.2 + 0.5
                tendril_shape = torch.where(nbr_count <= 2.1, torch.tensor(2.5, device=dev), torch.tensor(0.9, device=dev))
                weights = frontier.to(torch.float32) * non_white_bonus * tendril_shape + 0.002
                idx = torch.multinomial(weights.flatten(), k_tendril, replacement=True)
                ly = idx // w
                lx = idx % w
                r = torch.rand(k_tendril, device=dev)
                cols = torch.full((k_tendril,), BLACK_IDX, dtype=torch.uint8, device=dev)
                cols = torch.where(r > 0.87, torch.tensor(BURGUNDY_IDX, dtype=torch.uint8, device=dev), cols)
                cols = torch.where(r > 0.95, torch.tensor(DARK_RED_IDX, dtype=torch.uint8, device=dev), cols)
                xs_parts.append((lx + x0).to(torch.int32))
                ys_parts.append((ly + y0).to(torch.int32))
                cs_parts.append(cols)

        return torch.cat(xs_parts), torch.cat(ys_parts), torch.cat(cs_parts)

    def recruit_initial_members(self, faction: Faction, target_count: int) -> int:
        """Pull nearby and color-aligned multi-click citizens into a new faction."""
        not_one_shot = self.initial_clicks > 1
        if faction.is_void:
            pool_mask = not_one_shot & ((self.user_factions == -1) | (self.loyalty < 0.40))
            free_ids = torch.nonzero(pool_mask, as_tuple=False).squeeze(-1)
        else:
            free_mask = not_one_shot & (self.user_factions == -1)
            free_ids = torch.nonzero(free_mask, as_tuple=False).squeeze(-1)
        if free_ids.numel() == 0:
            return 0

        dx = self.pos_x[free_ids].to(torch.float32) - faction.cx
        dy = self.pos_y[free_ids].to(torch.float32) - faction.cy
        dist = torch.sqrt(dx * dx + dy * dy)

        if faction.is_void:
            trait_bonus = (self.trait[free_ids] == 4).to(torch.float32) * 3.0 + (self.trait[free_ids] == 3).to(torch.float32) * 1.5
            score = torch.exp(-dist / 120.0) + trait_bonus
        else:
            color_match = (
                (self.pref_color[free_ids] == faction.primary_color).to(torch.float32) * 1.8
                + (self.pref_color[free_ids] == faction.secondary_color).to(torch.float32) * 1.1
            )
            score = torch.exp(-dist / 85.0) * 2.5 + color_match + 0.05

        take = min(target_count, max(1, int(free_ids.numel() * 0.25)))
        chosen_rel = torch.multinomial(score, take, replacement=False)
        chosen = free_ids[chosen_rel]
        self.user_factions[chosen] = faction.id
        self.pos_x[chosen] = torch.randint(faction.x0, max(faction.x0 + 1, faction.x1), (take,), dtype=torch.int32, device=self.device)
        self.pos_y[chosen] = torch.randint(faction.y0, max(faction.y0 + 1, faction.y1), (take,), dtype=torch.int32, device=self.device)
        return take

    def social_step(self, factions: list[Faction], canvas: CanvasState) -> None:
        """Area-proportional recruitment of multi-click free agents + defection from losing factions."""
        alive = [f for f in factions if f.alive]
        if not alive:
            return
        dev = self.device

        # Keep >= 16% of online slots as free agents (including all One-Shot tourists)
        eligible_mask = (self.user_factions == -1) & (self.initial_clicks > 1)
        free_ids = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        min_free_eligible = int(self.num_users * 0.12)

        if free_ids.numel() > min_free_eligible:
            recruit_budget = min(int((free_ids.numel() - min_free_eligible) * 0.14), 480)
            if recruit_budget > 0:
                perm = torch.randperm(free_ids.numel(), device=dev)[:recruit_budget]
                candidates = free_ids[perm]
                w_vals = []
                for f in alive:
                    cap = 1500 if f.is_void else _TIER_MEMBER_CAP.get(f.size_tier, 360)
                    headroom = max(0.04, 1.0 - (f.member_count / max(1.0, float(cap))))
                    area_w = math.sqrt(max(100.0, float(f.w * f.h))) / 40.0
                    wt = max(0.05, f.hype * (0.45 + f.morale) * area_w * headroom * (1.5 if f.is_void else 1.0))
                    w_vals.append(wt)
                weights = torch.tensor(w_vals, dtype=torch.float32, device=dev)
                fac_choices = torch.multinomial(weights, recruit_budget, replacement=True)
                fac_ids_t = torch.tensor([f.id for f in alive], dtype=torch.int16, device=dev)
                self.user_factions[candidates] = fac_ids_t[fac_choices]

        # Defection: low-morale / crushed factions lose disloyal members back to free agents
        for f in alive:
            if f.morale < 0.52:
                m_idx = torch.nonzero(self.user_factions == f.id, as_tuple=False).squeeze(-1)
                if m_idx.numel() == 0:
                    continue
                churn_rate = 0.04 if f.morale > 0.35 else 0.11
                leave = (torch.rand(m_idx.numel(), device=dev) < churn_rate) & (
                    self.loyalty[m_idx] < (0.85 - f.morale * 0.4)
                )
                if leave.any():
                    self.user_factions[m_idx[leave]] = -1

    def release_faction_members(self, faction_id: int) -> None:
        mask = self.user_factions == faction_id
        self.user_factions[mask] = -1
