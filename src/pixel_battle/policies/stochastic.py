"""GPU-batched PyTorch MPS stochastic r/place citizen policy with real r/place population turnover (20% One-Shot)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pixel_battle.canvas import CanvasState
from pixel_battle.factions import Faction, timezone_awake_factor
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
        target_organized_share: float = 0.80,
        free_agent_clustering: float = 0.65,
        targeted_chaos_only: bool = False,
    ) -> None:
        self.num_users = num_users
        self.width = width
        self.height = height
        self.active_w = width
        self.active_h = height
        self.virtual_hour_utc: float = 12.0
        self.enable_timezones: bool = True
        self.base_cooldown = max(4, base_cooldown)
        self.device = device
        self.target_organized_share = max(0.05, min(0.95, float(target_organized_share)))
        self.free_agent_clustering = max(0.0, min(1.0, float(free_agent_clustering)))
        self.targeted_chaos_only = bool(targeted_chaos_only)
        self._chaos_squads: list[dict[str, int]] = []
        self._chaos_rng_state: int = seed + 9091

        g = torch.Generator(device="cpu").manual_seed(seed)
        # Initialize citizens around Schelling focal hubs (center + 4 corners + 4 mid-canvas hubs)
        # so empty white space stays clean instead of carpeting the blank canvas with TV static!
        hubs_x = torch.tensor(
            [width // 2, 16, max(16, width - 17), 16, max(16, width - 17), width // 2, width // 2, width // 4, (3 * width) // 4],
            dtype=torch.int32,
        )
        hubs_y = torch.tensor(
            [height // 2, 16, 16, max(16, height - 17), max(16, height - 17), height // 4, (3 * height) // 4, height // 2, height // 2],
            dtype=torch.int32,
        )
        h_pick = torch.randint(0, len(hubs_x), (num_users,), generator=g)
        spread = max(12, min(42, min(width, height) // 14))
        jx = torch.randint(-spread, spread + 1, (num_users,), generator=g, dtype=torch.int32)
        jy = torch.randint(-spread, spread + 1, (num_users,), generator=g, dtype=torch.int32)
        self.pos_x = (hubs_x[h_pick] + jx).clamp(0, width - 1).to(device)
        self.pos_y = (hubs_y[h_pick] + jy).clamp(0, height - 1).to(device)
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
        basin_prob = 0.10 + 0.82 * self.free_agent_clustering
        use_basin = torch.rand(num_users, generator=g) < basin_prob
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
        """Sample 1:1 r/place 2022 heavy-tailed click budgets (~15.5 mean px/user; 20% exact One-Shot)."""
        if gen is not None:
            u = torch.rand(n, generator=gen)
            v = torch.rand(n, generator=gen) ** 1.55
            dev = torch.device("cpu")
        else:
            dev = self.device
            u = torch.rand(n, device=dev)
            v = torch.rand(n, device=dev) ** 1.55

        # 20% One-Shot (1), 54% casual (2-9), 21% regular (10-42), 5% veteran (45-240) -> ~15.5 mean
        casual = (2 + (v * 8.0).to(torch.int32)).clamp(2, 9)
        regular = (10 + (v * 33.0).to(torch.int32)).clamp(10, 42)
        veteran = (45 + (v * 196.0).to(torch.int32)).clamp(45, 240)

        out = torch.ones(n, dtype=torch.int32, device=dev)
        out = torch.where(u >= 0.20, casual, out)
        out = torch.where(u >= 0.74, regular, out)
        out = torch.where(u >= 0.95, veteran, out)
        return out.to(self.device)

    def _random_allowed_colors(self, n: int, canvas: CanvasState, non_white: bool = True) -> torch.Tensor:
        cols = [c for c in canvas.allowed_colors if (not non_white or c != WHITE_IDX)]
        if not cols:
            return torch.full((n,), WHITE_IDX, dtype=torch.uint8, device=self.device)
        pool = torch.tensor(cols, dtype=torch.uint8, device=self.device)
        idx = torch.randint(0, len(cols), (n,), device=self.device)
        return pool[idx]

    def _clamp_colors_to_allowed(self, cols: torch.Tensor, canvas: CanvasState) -> torch.Tensor:
        if canvas.palette_tier >= 4:
            return torch.full_like(cols, WHITE_IDX)
        if canvas.palette_tier == 3:
            return cols
        allowed = canvas.allowed_color_mask[cols.to(torch.long)]
        if bool(allowed.all().item()):
            return cols
        repl = self._random_allowed_colors(int(cols.numel()), canvas, non_white=True)
        return torch.where(allowed, cols, repl)

    def _step_whiteout(
        self,
        ready_ids: torch.Tensor,
        canvas: CanvasState,
    ) -> PlacementBatch:
        """Act 4 Whiteout Finale: every ready citizen places pure white, targeting remaining colored pixels."""
        dev = self.device
        n = int(ready_ids.numel())
        W, H = canvas.active_w, canvas.active_h
        # Coarse non-white density map so citizens sweep colored art cleanly back to pure white
        stride = max(8, min(16, min(W, H) // 32))
        non_white = (canvas.colors[:H, :W] != WHITE_IDX).to(torch.float32)
        coarse = F.avg_pool2d(
            non_white.unsqueeze(0).unsqueeze(0), kernel_size=stride, stride=stride
        ).squeeze(0).squeeze(0)
        ch, cw = coarse.shape
        weights = (coarse + 0.015).flatten()
        cell_idx = torch.multinomial(weights, n, replacement=True)
        px = ((cell_idx % cw) * stride + torch.randint(0, stride, (n,), device=dev)).clamp(0, W - 1).to(torch.int32)
        py = ((cell_idx // cw) * stride + torch.randint(0, stride, (n,), device=dev)).clamp(0, H - 1).to(torch.int32)
        self.pos_x[ready_ids] = px
        self.pos_y[ready_ids] = py
        batch = PlacementBatch(
            xs=px,
            ys=py,
            colors=torch.full((n,), WHITE_IDX, dtype=torch.uint8, device=dev),
            factions=torch.full((n,), -1, dtype=torch.int16, device=dev),
            user_ids=self.slot_user_ids[ready_ids].to(torch.int64),
        )
        self._consume_and_recycle_sessions(ready_ids, canvas=canvas)
        return batch

    def _sample_focal_positions(
        self, n: int, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (x, y) coordinates drawn to existing painted clusters, active borders, and Schelling hubs (leaving empty white clean)."""
        dev = self.device
        W, H = canvas.active_w, canvas.active_h
        stride = 16
        painted = (canvas.colors[:H, :W] != WHITE_IDX).to(torch.float32)
        act_view = canvas.activity[:H, :W]
        signal = act_view * 1.6 + painted * 2.4
        coarse = F.avg_pool2d(
            signal.unsqueeze(0).unsqueeze(0), kernel_size=stride, stride=stride
        ).squeeze(0).squeeze(0)
        gh, gw = coarse.shape
        # Anchor subtle Schelling focal points (center + 4 corners) when canvas is freshly blank
        coarse = coarse.clone()
        coarse[gh // 2, gw // 2] += 0.45
        coarse[0, 0] += 0.35
        coarse[0, gw - 1] += 0.35
        coarse[gh - 1, 0] += 0.35
        coarse[gh - 1, gw - 1] += 0.35
        # Tiny uniform floor (0.0015) so untouched empty white space stays crisp and clean
        probs = (coarse + 0.0015).flatten()
        cell_idx = torch.multinomial(probs, n, replacement=True)
        cx = ((cell_idx % gw) * stride + torch.randint(0, stride, (n,), device=dev)).clamp(0, W - 1).to(torch.int32)
        cy = ((cell_idx // gw) * stride + torch.randint(0, stride, (n,), device=dev)).clamp(0, H - 1).to(torch.int32)
        return cx, cy

    def _consume_and_recycle_sessions(
        self,
        placed_slot_ids: torch.Tensor,
        canvas: CanvasState | None = None,
    ) -> None:
        """Decrement click budgets for slots that placed a pixel; recycle expired slots into brand-new unique user_ids."""
        if placed_slot_ids.numel() == 0:
            return
        dev = self.device
        self.remaining_clicks[placed_slot_ids] -= 1
        done_mask = self.remaining_clicks[placed_slot_ids] <= 0
        if not bool(done_mask.any().item()):
            return
        done_slots = placed_slot_ids[done_mask]
        n_done = int(done_slots.numel())
        n_one_shot = int((self.initial_clicks[done_slots] == 1).sum().item())
        self.total_completed_sessions += n_done
        self.total_one_shot_sessions += n_one_shot

        new_ids = torch.arange(
            self.next_user_id,
            self.next_user_id + n_done,
            dtype=torch.int64,
            device=dev,
        )
        self.next_user_id += n_done
        self.total_unique_users += n_done

        aw = canvas.active_w if canvas is not None else self.active_w
        ah = canvas.active_h if canvas is not None else self.active_h
        new_budgets = self._sample_click_budgets(n_done)
        self.slot_user_ids[done_slots] = new_ids
        self.initial_clicks[done_slots] = new_budgets
        self.remaining_clicks[done_slots] = new_budgets
        self.user_factions[done_slots] = -1
        if canvas is not None:
            fx, fy = self._sample_focal_positions(n_done, canvas)
            self.pos_x[done_slots] = fx
            self.pos_y[done_slots] = fy
            self.pref_color[done_slots] = self._random_allowed_colors(n_done, canvas, non_white=True)
        else:
            self.pos_x[done_slots] = torch.randint(0, aw, (n_done,), dtype=torch.int32, device=dev)
            self.pos_y[done_slots] = torch.randint(0, ah, (n_done,), dtype=torch.int32, device=dev)
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
        self.active_w = canvas.active_w
        self.active_h = canvas.active_h
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

        # Act 4 Whiteout Finale: lock every placement to pure white sweeping remaining colored pixels
        if canvas.is_whiteout:
            return self._step_whiteout(ready_ids, canvas)

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
            if self.targeted_chaos_only:
                fx, fy, fc, used_free = self._step_targeted_chaos(tick, free_u, canvas, factions)
                if fx.numel() > 0:
                    fc = self._clamp_colors_to_allowed(fc, canvas)
                    out_xs.append(fx.clamp(0, canvas.active_w - 1))
                    out_ys.append(fy.clamp(0, canvas.active_h - 1))
                    out_cols.append(fc)
                    out_facs.append(torch.full((fx.numel(),), -1, dtype=torch.int16, device=self.device))
                    out_uids.append(self.slot_user_ids[used_free].to(torch.int64))
                    placed_slots_list.append(used_free)
            else:
                fx, fy, fc = self._step_free_agents(free_u, canvas)
                fc = self._clamp_colors_to_allowed(fc, canvas)
                out_xs.append(fx.clamp(0, canvas.active_w - 1))
                out_ys.append(fy.clamp(0, canvas.active_h - 1))
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

        # O(N_ready) grouping of faction members
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

                # Apply regional timezone sleep/wake modulation
                eff_cnt = cnt
                if self.enable_timezones and not fac.is_void and not fac.is_streamer_raid:
                    awake = timezone_awake_factor(fac.timezone_region, self.virtual_hour_utc)
                    if awake < 0.98:
                        eff_cnt = max(0, int(round(cnt * awake)))
                        if eff_cnt == 0:
                            continue
                        u_idx = u_idx[:eff_cnt]

                if fac.is_void:
                    vx, vy, vc = self._step_void_faction(fac, eff_cnt, canvas)
                else:
                    vx, vy, vc = self._step_art_faction(
                        tick, fac, eff_cnt, canvas, fac_by_id, borders_by_fac.get(fac.id, [])
                    )

                n_out = int(vx.numel())
                if n_out > 0:
                    vc = self._clamp_colors_to_allowed(vc, canvas)
                    vx = vx.clamp(0, canvas.active_w - 1)
                    vy = vy.clamp(0, canvas.active_h - 1)
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
        self._consume_and_recycle_sessions(torch.cat(placed_slots_list), canvas=canvas)
        return batch

    def _step_targeted_chaos(
        self,
        tick: int,
        u_ids: torch.Tensor,
        canvas: CanvasState,
        factions: list[Faction],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Targeted chaos mode: zero empty-space noise; chaotic players attack active artworks via random griefers & mini-raid squads."""
        dev = self.device
        W, H = canvas.active_w, canvas.active_h
        empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
        empty_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
        empty_idx = torch.empty(0, dtype=u_ids.dtype, device=dev)

        alive_art = [f for f in factions if f.alive and not f.is_void and f.w >= 8 and f.h >= 8]
        if not alive_art:
            return empty_i32, empty_i32, empty_u8, empty_idx

        n_raw = int(u_ids.numel())
        # Cap per-tick chaotic attack volume so griefers & mini-mobs create lively repair battles without drowning out builders
        max_chaos = max(120, int(self.num_users * 0.015))
        n = min(n_raw, max_chaos)
        used = u_ids[:n]

        # Maintain 4-6 rotating small-scale coordinated Chaos Squads (40-150 players swarming tight patches on active art)
        alive_by_id = {f.id: f for f in alive_art}
        self._chaos_squads = [
            sq for sq in self._chaos_squads
            if sq["end_tick"] > tick and sq["fac_id"] in alive_by_id
        ]
        target_squads = min(6, max(1, len(alive_art) // 10))
        vandal_pal = [BLACK_IDX, DARK_RED_IDX, RED_IDX, PINK_IDX, 6, 11, 19, 21]
        while len(self._chaos_squads) < target_squads:
            self._chaos_rng_state = (self._chaos_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            tgt = alive_art[self._chaos_rng_state % len(alive_art)]
            pw = max(6, min(22, tgt.w // 2))
            ph = max(6, min(22, tgt.h // 2))
            self._chaos_rng_state = (self._chaos_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            ox = self._chaos_rng_state % max(1, tgt.w - pw + 1)
            self._chaos_rng_state = (self._chaos_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            oy = self._chaos_rng_state % max(1, tgt.h - ph + 1)
            self._chaos_rng_state = (self._chaos_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            c_main = vandal_pal[self._chaos_rng_state % len(vandal_pal)]
            c_acc = vandal_pal[(self._chaos_rng_state // 7) % len(vandal_pal)]
            dur = 55 + (self._chaos_rng_state % 45)
            self._chaos_squads.append(
                {
                    "fac_id": tgt.id,
                    "x0": tgt.x0 + ox,
                    "y0": tgt.y0 + oy,
                    "x1": min(W, tgt.x0 + ox + pw),
                    "y1": min(H, tgt.y0 + oy + ph),
                    "c_main": c_main,
                    "c_acc": c_acc,
                    "end_tick": tick + dur,
                }
            )

        xs_parts: list[torch.Tensor] = []
        ys_parts: list[torch.Tensor] = []
        cs_parts: list[torch.Tensor] = []
        slots_parts: list[torch.Tensor] = []

        # ~35% of chaotic players form tight coordinated Chaos Squad mini-raids inside active artworks
        n_squad_tot = int(round(n * 0.35)) if self._chaos_squads else 0
        cursor = 0
        if n_squad_tot > 0 and self._chaos_squads:
            per_sq = max(1, n_squad_tot // len(self._chaos_squads))
            for sq in self._chaos_squads:
                take = min(per_sq, n - cursor)
                if take <= 0:
                    break
                sx = torch.randint(sq["x0"], max(sq["x0"] + 1, sq["x1"]), (take,), dtype=torch.int32, device=dev)
                sy = torch.randint(sq["y0"], max(sq["y0"] + 1, sq["y1"]), (take,), dtype=torch.int32, device=dev)
                # Only vandalize pixels actually owned by/painted inside an artwork (keep exterior white untouched!)
                tgt_f = alive_by_id.get(sq["fac_id"])
                if tgt_f is not None:
                    ly = (sy - tgt_f.y0).clamp(0, tgt_f.h - 1)
                    lx = (sx - tgt_f.x0).clamp(0, tgt_f.w - 1)
                    in_art = tgt_f.blueprint[ly, lx] >= 0
                    sx = sx[in_art]
                    sy = sy[in_art]
                    sub_slots = used[cursor : cursor + take][in_art]
                else:
                    sub_slots = used[cursor : cursor + take]
                cursor += take
                m_cnt = int(sx.numel())
                if m_cnt > 0:
                    sc = torch.where(
                        torch.rand(m_cnt, device=dev) < 0.78,
                        torch.full((m_cnt,), sq["c_main"], dtype=torch.uint8, device=dev),
                        torch.full((m_cnt,), sq["c_acc"], dtype=torch.uint8, device=dev),
                    )
                    xs_parts.append(sx)
                    ys_parts.append(sy)
                    cs_parts.append(sc)
                    slots_parts.append(sub_slots)

        # ~65% of chaotic players act as random griefers/vandals picking pixels strictly inside active faction artworks
        n_grief = n - cursor
        if n_grief > 0:
            # Vectorized sampling across up to 64 sampled active artworks per tick (zero empty-space leaks)
            sample_facs = alive_art if len(alive_art) <= 64 else [
                alive_art[(tick * 17 + i * 31) % len(alive_art)] for i in range(64)
            ]
            x0s = torch.tensor([f.x0 for f in sample_facs], dtype=torch.int32, device=dev)
            y0s = torch.tensor([f.y0 for f in sample_facs], dtype=torch.int32, device=dev)
            ws = torch.tensor([max(1, f.w) for f in sample_facs], dtype=torch.int32, device=dev)
            hs = torch.tensor([max(1, f.h) for f in sample_facs], dtype=torch.int32, device=dev)

            f_pick = torch.randint(0, len(sample_facs), (n_grief,), device=dev)
            gx = (x0s[f_pick] + (torch.rand(n_grief, device=dev) * ws[f_pick].to(torch.float32)).to(torch.int32)).clamp(0, W - 1)
            gy = (y0s[f_pick] + (torch.rand(n_grief, device=dev) * hs[f_pick].to(torch.float32)).to(torch.int32)).clamp(0, H - 1)
            # Restrict vandals strictly to already-painted faction pixels (never blank white canvas!)
            on_art = canvas.owner[gy, gx] >= 0
            gx = gx[on_art]
            gy = gy[on_art]
            g_slots = used[cursor : cursor + n_grief][on_art]
            g_cnt = int(gx.numel())
            if g_cnt > 0:
                gc = self._random_allowed_colors(g_cnt, canvas, non_white=True)
                # 35% mischief black dots (classic r/place eye/face griefing)
                gc = torch.where(
                    torch.rand(g_cnt, device=dev) < 0.35,
                    torch.full((g_cnt,), BLACK_IDX, dtype=torch.uint8, device=dev),
                    gc,
                )
                xs_parts.append(gx)
                ys_parts.append(gy)
                cs_parts.append(gc)
                slots_parts.append(g_slots)

        if not xs_parts:
            return empty_i32, empty_i32, empty_u8, empty_idx
        out_x = torch.cat(xs_parts)
        out_y = torch.cat(ys_parts)
        out_c = torch.cat(cs_parts)
        out_s = torch.cat(slots_parts)
        self.pos_x[out_s] = out_x
        self.pos_y[out_s] = out_y
        return out_x, out_y, out_c, out_s

    def _step_free_agents(
        self, u_ids: torch.Tensor, canvas: CanvasState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = int(u_ids.numel())
        dev = self.device
        W, H = canvas.active_w, canvas.active_h

        c = self.free_agent_clustering
        x = self.pos_x[u_ids].clamp(0, W - 1).clone()
        y = self.pos_y[u_ids].clamp(0, H - 1).clone()

        # Chaotic players love claiming blank white spots (while still gravitating toward active art edges!)
        on_white = canvas.colors[y, x] == WHITE_IDX
        r_jump = torch.rand(n, device=dev)
        jump_mask = ((~on_white) & (r_jump < 0.22)) | (on_white & (r_jump < 0.25))
        jx, jy = self._sample_focal_positions(n, canvas)
        # Half of jumpers probe a random canvas spot to claim remaining blank white pockets
        rx = torch.randint(0, W, (n,), dtype=torch.int32, device=dev)
        ry = torch.randint(0, H, (n,), dtype=torch.int32, device=dev)
        use_white = jump_mask & (r_jump < 0.11) & (canvas.colors[ry, rx] == WHITE_IDX)
        x = torch.where(use_white, rx, torch.where(jump_mask, jx, x))
        y = torch.where(use_white, ry, torch.where(jump_mask, jy, y))

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

        p_conform = 0.28 + 0.66 * c
        m0 = (traits == 0) & has_nw & (torch.rand(n, device=dev) < p_conform)
        colors = torch.where(m0, nbr_col, colors)

        horiz_nw = (c_l != WHITE_IDX) | (c_r != WHITE_IDX)
        horiz_col = torch.where(c_l != WHITE_IDX, c_l, c_r)
        p_stripe = 0.25 + 0.60 * c
        m1 = (traits == 1) & horiz_nw & (torch.rand(n, device=dev) < p_stripe)
        colors = torch.where(m1, horiz_col, colors)
        x = torch.where(traits == 1, (x + torch.randint(-2, 3, (n,), dtype=torch.int32, device=dev)).clamp(0, W - 1), x)

        m2 = traits == 2
        parity = ((x + y) & 1) == 0
        alt_col = torch.where(has_nw, nbr_col, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev))
        colors = torch.where(m2 & (~parity), alt_col, colors)

        p_doodle = max(0.06, 0.38 - 0.28 * c)
        m3 = (traits == 3) & has_nw & (torch.rand(n, device=dev) < p_doodle)
        rand_col = self._random_allowed_colors(n, canvas)
        colors = torch.where(m3, rand_col, colors)

        touches_black = (nbrs == BLACK_IDX).any(dim=1)
        m4_spread = (traits == 4) & touches_black
        m4_red = m4_spread & (torch.rand(n, device=dev) < 0.14)
        colors = torch.where(m4_spread, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev), colors)
        colors = torch.where(m4_red, torch.tensor(DARK_RED_IDX, dtype=torch.uint8, device=dev), colors)
        m4_seed = (traits == 4) & (~touches_black) & (torch.rand(n, device=dev) < 0.25)
        colors = torch.where(m4_seed, torch.tensor(BLACK_IDX, dtype=torch.uint8, device=dev), colors)

        # Never paint pure white onto a blank white pixel when claiming whitespace (pure GPU op, zero MPS sync!)
        still_white = (colors == WHITE_IDX) & (canvas.colors[y, x] == WHITE_IDX)
        colors = torch.where(still_white, torch.clamp(pref, min=1, max=NUM_COLORS - 1), colors)

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
        W, H = canvas.active_w, canvas.active_h

        # 1. Active Raid on a rival neighbor: push a coherent solid front strictly along opaque pixels on the shared side
        if fac.raid_target_id is not None and fac.raid_target_id in fac_by_id:
            target = fac_by_id[fac.raid_target_id]
            dx_c = target.cx - fac.cx
            dy_c = target.cy - fac.cy
            # Concentrate attack along the ~35% edge strip facing the attacker so borders push cleanly instead of bbox confetti
            if abs(dx_c) >= abs(dy_c):
                band_w = max(4, int(round(target.w * 0.38)))
                bx0 = target.x0 if dx_c >= 0 else max(target.x0, target.x1 - band_w)
                bx1 = min(target.x1, target.x0 + band_w) if dx_c >= 0 else target.x1
                by0, by1 = target.y0, target.y1
            else:
                band_h = max(4, int(round(target.h * 0.38)))
                by0 = target.y0 if dy_c >= 0 else max(target.y0, target.y1 - band_h)
                by1 = min(target.y1, target.y0 + band_h) if dy_c >= 0 else target.y1
                bx0, bx1 = target.x0, target.x1

            # Oversample 2x then filter strictly to opaque target pixels (blueprint >= 0) — zero empty-space confetti!
            n_try = max(k * 2, 4)
            rx = torch.randint(bx0, max(bx0 + 1, bx1), (n_try,), dtype=torch.int32, device=dev).clamp(0, W - 1)
            ry = torch.randint(by0, max(by0 + 1, by1), (n_try,), dtype=torch.int32, device=dev).clamp(0, H - 1)
            ly = (ry - target.y0).clamp(0, max(0, target.h - 1))
            lx = (rx - target.x0).clamp(0, max(0, target.w - 1))
            on_target = target.blueprint[ly, lx] >= 0
            rx = rx[on_target][:k]
            ry = ry[on_target][:k]
            m_hit = int(rx.numel())
            if m_hit == 0:
                empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
                return empty_i32, empty_i32, torch.empty(0, dtype=torch.uint8, device=dev)
            col_pick = torch.where(
                torch.rand(m_hit, device=dev) < 0.88,
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
            ox, oy, oc = self._expand_perimeter(fac, k_rest, canvas)
            return torch.cat([hx, ox]), torch.cat([hy, oy]), torch.cat([hc, oc])

        return self._expand_perimeter(fac, k, canvas)

    def _expand_perimeter(
        self, fac: Faction, k: int, canvas: CanvasState | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Organic territory growth hugging the silhouette or clean-space/ribbon frontier."""
        dev = self.device
        W = canvas.active_w if canvas is not None else self.active_w
        H = canvas.active_h if canvas is not None else self.active_h

        # Clean-space purists keep erasing 1-2px outside their blank enclave cleanly in WHITE_IDX
        if fac.art_mode == "clean_space":
            rx = torch.randint(max(0, fac.x0 - 2), min(W, fac.x1 + 2), (k,), dtype=torch.int32, device=dev)
            ry = torch.randint(max(0, fac.y0 - 2), min(H, fac.y1 + 2), (k,), dtype=torch.int32, device=dev)
            return rx, ry, torch.full((k,), WHITE_IDX, dtype=torch.uint8, device=dev)

        # Keep micro-sprites, paintings, comic strips, cartoon memes, and targeted-chaos artworks crisp inside their outline
        if (
            self.targeted_chaos_only
            or fac.size_tier == "micro"
            or fac.art_mode in ("painting", "comic_strip", "flux_comic", "cartoon_meme", "niche_badge")
        ):
            if fac.art_mode != "ribbon":
                rx = torch.randint(fac.x0, max(fac.x0 + 1, fac.x1), (k,), dtype=torch.int32, device=dev)
                ry = torch.randint(fac.y0, max(fac.y0 + 1, fac.y1), (k,), dtype=torch.int32, device=dev)
                bp_c = fac.blueprint[(ry - fac.y0).clamp(0, fac.h - 1), (rx - fac.x0).clamp(0, fac.w - 1)]
                valid_m = bp_c >= 0
                return rx[valid_m], ry[valid_m], bp_c[valid_m].to(torch.uint8)

        # Transverse ribbons push directly past their advancing tips (never fall through to circular rings!)
        if fac.art_mode == "ribbon":
            if fac.ribbon_axis == "h":
                tip = torch.where(
                    torch.rand(k, device=dev) < 0.5,
                    (torch.full((k,), fac.x1, dtype=torch.int32, device=dev) + torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                    (torch.full((k,), fac.x0 - 1, dtype=torch.int32, device=dev) - torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                ).clamp(0, W - 1)
                ry = torch.randint(fac.y0, max(fac.y0 + 1, fac.y1), (k,), dtype=torch.int32, device=dev)
                col_idx = fac.blueprint[(ry - fac.y0).clamp(0, fac.h - 1), torch.zeros(k, dtype=torch.long, device=dev)].clamp(0, NUM_COLORS - 1).to(torch.uint8)
                return tip, ry, col_idx
            if fac.ribbon_axis == "v":
                tip = torch.where(
                    torch.rand(k, device=dev) < 0.5,
                    (torch.full((k,), fac.y1, dtype=torch.int32, device=dev) + torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                    (torch.full((k,), fac.y0 - 1, dtype=torch.int32, device=dev) - torch.randint(0, 6, (k,), dtype=torch.int32, device=dev)),
                ).clamp(0, H - 1)
                rx = torch.randint(fac.x0, max(fac.x0 + 1, fac.x1), (k,), dtype=torch.int32, device=dev)
                col_idx = fac.blueprint[torch.zeros(k, dtype=torch.long, device=dev), (rx - fac.x0).clamp(0, fac.w - 1)].clamp(0, NUM_COLORS - 1).to(torch.uint8)
                return rx, tip, col_idx
            empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
            return empty_i32, empty_i32, torch.empty(0, dtype=torch.uint8, device=dev)

        # Run silhouette halo expansion using cached halo coords (zero per-tick MPS torch.nonzero stalls)
        if fac.art_mode in ("flux_art", "ai_code", "streamer_raid", "diffusion", "procedural", "banner", "fractal", "black_hole", "void_monster"):
            h_idx = fac._halo_xy_cache
            if h_idx is None:
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
                fac._halo_xy_cache = h_idx
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
                xs_parts.append(cx.clamp(0, canvas.active_w - 1))
                ys_parts.append(cy.clamp(0, canvas.active_h - 1))
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

    def mobilize_flash_raid(self, faction: Faction, target_count: int) -> int:
        """Rally a temporary streamer flash-raid surge from free agents and low-loyalty citizens."""
        not_one_shot = self.initial_clicks > 1
        pool_mask = not_one_shot & ((self.user_factions == -1) | (self.loyalty < 0.55))
        pool = torch.nonzero(pool_mask, as_tuple=False).squeeze(-1)
        if pool.numel() == 0:
            return 0
        take = min(target_count, max(1, int(pool.numel() * 0.22)))
        perm = torch.randperm(pool.numel(), device=self.device)[:take]
        chosen = pool[perm]
        self.user_factions[chosen] = faction.id
        self.pos_x[chosen] = torch.randint(faction.x0, max(faction.x0 + 1, faction.x1), (take,), dtype=torch.int32, device=self.device)
        self.pos_y[chosen] = torch.randint(faction.y0, max(faction.y0 + 1, faction.y1), (take,), dtype=torch.int32, device=self.device)
        # Wake raiders immediately for dramatic blitz speed
        self.cooldown[chosen] = torch.randint(1, max(3, self.base_cooldown // 2), (take,), dtype=torch.int32, device=self.device)
        return take

    def social_step(self, factions: list[Faction], canvas: CanvasState) -> None:
        """Area-proportional recruitment of multi-click free agents + burnout defection from fading factions."""
        alive = [f for f in factions if f.alive]
        if not alive:
            return
        dev = self.device

        # Track target_organized_share against continuous session turnover (keeping one-shot tourists free)
        eligible_mask = (self.user_factions == -1) & (self.initial_clicks > 1)
        free_ids = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        current_org = int((self.user_factions >= 0).sum().item())

        # Keep remaining (1 - target_organized_share) slots free (one-shot slots already account for ~2% online)
        min_free_frac = max(0.05, 1.0 - self.target_organized_share - 0.02)
        min_free_eligible = int(self.num_users * min_free_frac)

        # Compute per-faction capacity scaled smoothly so active factions can sustain target_organized_share
        target_org_slots = int(self.num_users * self.target_organized_share)
        n_alive = len(alive)
        per_fac_share = int((target_org_slots * 1.45) / max(1, n_alive))
        caps: list[int] = []
        total_headroom = 0
        for f in alive:
            if f.is_void or f.is_streamer_raid:
                cap = 3500
            else:
                base_cap = _TIER_MEMBER_CAP.get(f.size_tier, 360)
                tier_mult = 0.55 if f.size_tier == "micro" else (1.15 if f.size_tier == "medium" else 2.0)
                cap = max(base_cap, min(2600, int(per_fac_share * tier_mult)))
            caps.append(cap)
            total_headroom += max(0, cap - f.member_count)

        desired_org = min(target_org_slots, current_org + total_headroom)
        deficit = max(0, desired_org - current_org)

        if free_ids.numel() > min_free_eligible and deficit > 0:
            avail_free = int(free_ids.numel()) - min_free_eligible
            # Smoothly recruit enough citizens each social step to hold target_organized_share against session churn
            max_step_budget = max(640, int(self.num_users * 0.14))
            recruit_budget = min(avail_free, max(int(deficit * 0.65), min(deficit, max_step_budget)))
            if recruit_budget > 0:
                perm = torch.randperm(free_ids.numel(), device=dev)[:recruit_budget]
                candidates = free_ids[perm]
                w_vals = []
                for f, cap in zip(alive, caps):
                    headroom = max(0.03, 1.0 - (f.member_count / max(1.0, float(cap))))
                    area_w = math.sqrt(max(100.0, float(f.w * f.h))) / 40.0
                    awake = timezone_awake_factor(f.timezone_region, self.virtual_hour_utc) if self.enable_timezones else 1.0
                    wt = max(0.02, f.hype * (0.40 + f.morale) * area_w * headroom * awake * (1.8 if f.is_void else 1.0))
                    w_vals.append(wt)
                weights = torch.tensor(w_vals, dtype=torch.float32, device=dev)
                fac_choices = torch.multinomial(weights, recruit_budget, replacement=True)
                fac_ids_t = torch.tensor([f.id for f in alive], dtype=torch.int16, device=dev)
                self.user_factions[candidates] = fac_ids_t[fac_choices]

        # Vectorized defection for low-morale or burning-out factions (scales cleanly to 1,200+ factions)
        weak_ids = [f.id for f in alive if (not f.is_streamer_raid) and (f.morale < 0.52 or f.hype < 0.35)]
        if weak_ids:
            max_id = max(f.id for f in alive) + 1
            churn_lut = torch.zeros(max_id, dtype=torch.float32, device=dev)
            loy_lut = torch.zeros(max_id, dtype=torch.float32, device=dev)
            for f in alive:
                if f.is_streamer_raid:
                    continue
                if f.morale < 0.52 or f.hype < 0.35:
                    burn_extra = 0.08 if f.hype < 0.35 else 0.0
                    churn_lut[f.id] = (0.04 if f.morale > 0.35 else 0.12) + burn_extra
                    loy_lut[f.id] = 0.88 - f.morale * 0.35
            org_m = self.user_factions >= 0
            org_idx = torch.nonzero(org_m, as_tuple=False).squeeze(-1)
            if org_idx.numel() > 0:
                fids = self.user_factions[org_idx].to(torch.long).clamp(0, max_id - 1)
                rates = churn_lut[fids]
                thresh = loy_lut[fids]
                leave = (rates > 0.0) & (torch.rand(org_idx.numel(), device=dev) < rates) & (self.loyalty[org_idx] < thresh)
                if bool(leave.any().item()):
                    self.user_factions[org_idx[leave]] = -1

    def release_faction_members(self, faction_id: int) -> None:
        mask = self.user_factions == faction_id
        self.user_factions[mask] = -1
