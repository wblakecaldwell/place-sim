"""Spatial multi-scale cluster, corner/edge nucleation, cameo nesting, ribbon track, and Void detectors."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from pixel_battle.canvas import CanvasState
from pixel_battle.factions import Faction
from pixel_battle.palette import BLACK_IDX, WHITE_IDX


@dataclass
class ClusterCandidate:
    x0: int
    y0: int
    w: int
    h: int
    score: float
    density: float
    dominant_colors: list[int]
    is_corner: bool = False
    is_edge: bool = False
    is_cameo: bool = False


def _build_occupancy_mask(
    H: int,
    W: int,
    factions: list[Faction],
    exclude_host_flags: bool = False,
) -> np.ndarray:
    """Mark non-transparent silhouette pixels as occupied (optionally leaving wide flags/ribbons open for cameos)."""
    occupied = np.zeros((H, W), dtype=bool)
    for f in factions:
        if not f.alive:
            continue
        if exclude_host_flags and (f.category in ("flag", "ribbon")) and max(f.w, f.h) >= 44 and not f.is_cameo:
            continue
        if f.is_void:
            occupied[f.y0 : f.y1, f.x0 : f.x1] = True
        else:
            mask = (f.blueprint >= 0).detach().cpu().numpy()
            mh, mw = mask.shape
            y1 = min(H, f.y0 + mh)
            x1 = min(W, f.x0 + mw)
            occupied[f.y0 : y1, f.x0 : x1] |= mask[: y1 - f.y0, : x1 - f.x0]
    return occupied


def find_pareidolia_candidates(
    canvas: CanvasState,
    factions: list[Faction],
    window: int = 56,
    stride: int = 28,
    min_density: float = 0.08,
    top_k: int = 2,
    allow_cameo: bool = False,
    rng: random.Random | None = None,
    prefer_region: tuple[int, int, int, int] | None = None,
) -> list[ClusterCandidate]:
    """Scan unclaimed regions inside the active viewport (flush corners, edges, tight micro-gaps, optional cameos)."""
    H, W = canvas.active_h, canvas.active_w
    colors = canvas.colors[:H, :W].detach().cpu().numpy()
    activity = canvas.activity[:H, :W].detach().cpu().numpy()
    window = max(12, min(window, min(H, W)))
    # Scale stride gently on 2000x2000 so discovery scans stay sub-20ms
    min_stride = max(6, int(max(W, H) // 64))
    stride = max(min_stride, stride)

    occupied = _build_occupancy_mask(H, W, factions, exclude_host_flags=False)
    rx0, ry0, rx1, ry1 = prefer_region if prefer_region is not None else (0, 0, W, H)
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(W, rx1), min(H, ry1)
    if rx1 - rx0 < window or ry1 - ry0 < window:
        rx0, ry0, rx1, ry1 = 0, 0, W, H

    xs = sorted(set(list(range(rx0, max(rx0 + 1, rx1 - window), stride)) + [max(rx0, rx1 - window)]))
    ys = sorted(set(list(range(ry0, max(ry0 + 1, ry1 - window), stride)) + [max(ry0, ry1 - window)]))

    alive_edge_count = sum(
        1 for f in factions if f.alive and not f.is_void and (f.x0 == 0 or f.y0 == 0 or f.x1 >= W or f.y1 >= H)
    )
    max_occ = 0.22 if window >= 34 else 0.16

    candidates: list[ClusterCandidate] = []
    for y0 in ys:
        for x0 in xs:
            occ_frac = float(occupied[y0 : y0 + window, x0 : x0 + window].mean())
            if occ_frac > max_occ:
                continue

            patch = colors[y0 : y0 + window, x0 : x0 + window]
            non_white = patch != WHITE_IDX
            density = float(non_white.mean())
            at_left_right = (x0 == 0) or (x0 + window >= W)
            at_top_bottom = (y0 == 0) or (y0 + window >= H)
            is_corner = at_left_right and at_top_bottom
            is_edge = at_left_right or at_top_bottom

            eff_min = min_density * (0.55 if is_corner else (0.75 if is_edge else 1.0))
            if window <= 28:
                # Micro sprites can colonize fresh gaps even with very light pioneer dust
                eff_min = min(eff_min, 0.02)
            if density < eff_min:
                continue

            painted = patch[non_white]
            counts = np.bincount(painted, minlength=32)
            counts[WHITE_IDX] = 0
            top_cols = np.argsort(counts)[::-1]
            top3_share = float(counts[top_cols[:3]].sum() / max(1, len(painted)))
            act_score = float(activity[y0 : y0 + window, x0 : x0 + window].mean())

            if is_corner and window >= 36 and alive_edge_count < 6:
                wall_bonus = 0.25
            elif is_edge and alive_edge_count < 10:
                wall_bonus = 0.08
            else:
                wall_bonus = 0.0

            jitter = (rng.random() * 0.25) if rng is not None else 0.0
            score = (
                density * 1.2
                + top3_share * 0.9
                + min(act_score, 3.0) * 0.40
                + wall_bonus
                + jitter
                - occ_frac * 2.4
            )
            dom = [int(c) for c in top_cols[:3] if counts[c] > 0]
            if not dom:
                dom = [15, 4]
            candidates.append(
                ClusterCandidate(
                    x0=x0,
                    y0=y0,
                    w=window,
                    h=window,
                    score=score,
                    density=density,
                    dominant_colors=dom,
                    is_corner=is_corner,
                    is_edge=is_edge,
                    is_cameo=False,
                )
            )

    # Optional Cameo Micro Factions sitting right on top of wide flags or ribbons (like r/place Ness/Kirby)
    if allow_cameo and window <= 26:
        cameo_occ = _build_occupancy_mask(H, W, factions, exclude_host_flags=True)
        hosts = [
            f for f in factions
            if f.alive and not f.is_void and not f.is_cameo
            and f.category in ("flag", "ribbon")
            and f.w >= window + 8 and f.h >= min(window, 14)
        ]
        _rng = rng or random.Random(42)
        _rng.shuffle(hosts)
        for host in hosts[:8]:
            for _ in range(4):
                cx0 = _rng.randint(host.x0 + 3, max(host.x0 + 3, host.x1 - window - 3))
                cy0_hi = max(host.y0, host.y1 - window)
                cy0 = _rng.randint(host.y0, cy0_hi) if cy0_hi > host.y0 else max(0, min(H - window, host.y0))
                if float(cameo_occ[cy0 : cy0 + window, cx0 : cx0 + window].mean()) < 0.08:
                    candidates.append(
                        ClusterCandidate(
                            x0=cx0,
                            y0=cy0,
                            w=window,
                            h=window,
                            score=1.45 + _rng.random() * 0.45,
                            density=0.85,
                            dominant_colors=[host.primary_color, host.secondary_color],
                            is_corner=False,
                            is_edge=False,
                            is_cameo=True,
                        )
                    )
                    break

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected: list[ClusterCandidate] = []
    min_sep = max(12, int(window * 0.85))
    for cand in candidates:
        too_close = False
        for s in selected:
            if abs(cand.x0 - s.x0) < min_sep and abs(cand.y0 - s.y0) < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append(cand)
            if len(selected) >= top_k:
                break
    return selected


def find_ribbon_track(
    canvas: CanvasState,
    factions: list[Faction],
    axis: str,
    thickness: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Pick a starting (x0, y0, init_w, init_h) for a progressive Transverse Ribbon inside the active viewport."""
    H, W = canvas.active_h, canvas.active_w
    existing_h = [f.y0 for f in factions if f.alive and f.art_mode == "ribbon" and f.ribbon_axis == "h"]
    existing_v = [f.x0 for f in factions if f.alive and f.art_mode == "ribbon" and f.ribbon_axis == "v"]

    init_len = rng.choice([56, 68, 84])
    if axis == "h":
        best_y = rng.randint(24, max(25, H - thickness - 24))
        for _ in range(25):
            cand_y = rng.randint(16, max(17, H - thickness - 16))
            if all(abs(cand_y - ey) >= 48 for ey in existing_h):
                best_y = cand_y
                break
        start_left = rng.random() < 0.5
        x0 = 0 if start_left else (W - init_len)
        return x0, best_y, init_len, thickness

    if axis == "v":
        best_x = rng.randint(24, max(25, W - thickness - 24))
        for _ in range(25):
            cand_x = rng.randint(16, max(17, W - thickness - 16))
            if all(abs(cand_x - ex) >= 48 for ex in existing_v):
                best_x = cand_x
                break
        start_top = rng.random() < 0.5
        y0 = 0 if start_top else (H - init_len)
        return best_x, y0, thickness, init_len

    # Diagonal ('d1' or 'd2') starting from a corner or edge
    init_diag = rng.choice([72, 88])
    if axis == "d1":
        if rng.random() < 0.5:
            off_x = rng.randint(0, max(1, W - init_diag - 60))
            return off_x, 0, init_diag, init_diag
        off_y = rng.randint(0, max(1, H - init_diag - 60))
        return 0, off_y, init_diag, init_diag
    else:
        off_x = rng.randint(0, max(1, W - init_diag - 60))
        return off_x, max(0, H - init_diag), init_diag, init_diag


def update_faction_metrics(
    canvas: CanvasState,
    factions: list[Faction],
    user_factions: torch.Tensor,
) -> None:
    """Recompute each faction's member count, blueprint integrity, owned pixels, and churn without per-faction GPU stalls."""
    if not factions:
        return
    valid_u = user_factions[user_factions >= 0]
    max_id = max(f.id for f in factions) + 1
    counts = (
        torch.bincount(valid_u.to(torch.int64), minlength=max_id).detach().cpu().numpy()
        if valid_u.numel() > 0
        else np.zeros(max_id, dtype=np.int64)
    )
    owner_grid = canvas.owner[: canvas.active_h, : canvas.active_w]
    valid_o = owner_grid[owner_grid >= 0]
    owned_counts = (
        torch.bincount(valid_o.to(torch.int64), minlength=max_id).detach().cpu().numpy()
        if valid_o.numel() > 0
        else np.zeros(max_id, dtype=np.int64)
    )

    # Single bulk GPU->CPU transfer avoids O(N_factions) MPS pipeline syncs at 1,200+ factions
    colors_np = canvas.colors[: canvas.active_h, : canvas.active_w].detach().cpu().numpy().astype(np.int16, copy=False)
    act_np = canvas.activity[: canvas.active_h, : canvas.active_w].detach().cpu().numpy()
    ah, aw = colors_np.shape

    for f in factions:
        if not f.alive:
            f.member_count = 0
            continue
        f.member_count = int(counts[f.id]) if f.id < len(counts) else 0
        f.owned_pixels = int(owned_counts[f.id]) if f.id < len(owned_counts) else 0

        y0, y1 = max(0, f.y0), min(ah, f.y1)
        x0, x1 = max(0, f.x0), min(aw, f.x1)
        if y1 <= y0 or x1 <= x0:
            f.integrity = 0.0
            continue
        crop_colors = colors_np[y0:y1, x0:x1]
        f.recent_churn = float(act_np[y0:y1, x0:x1].mean())

        if f.is_void and f.void_variant == "tendrils":
            f.integrity = float((crop_colors == BLACK_IDX).mean())
        else:
            bp_np = getattr(f, "_bp_cpu_cache", None)
            if bp_np is None or bp_np.shape != (f.h, f.w):
                bp_np = f.blueprint.detach().cpu().numpy()
                f._bp_cpu_cache = bp_np  # type: ignore[attr-defined]
            bh, bw = crop_colors.shape
            bp_sub = bp_np[:bh, :bw]
            valid = bp_sub >= 0
            total_valid = int(valid.sum())
            if total_valid > 0:
                matched = int(((crop_colors == bp_sub) & valid).sum())
                f.integrity = matched / total_valid
            else:
                f.integrity = 0.0

        target_morale = min(1.0, max(0.1, 0.35 + 0.65 * f.integrity))
        f.morale = 0.7 * f.morale + 0.3 * target_morale


def find_open_exodus_spot(
    canvas: CanvasState,
    factions: list[Faction],
    w: int,
    h: int,
    exclude_faction_id: int = -1,
    prefer_region: tuple[int, int, int, int] | None = None,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Find a clean, least-contested (x0, y0) rectangle distributed across the active viewport (not stuck at y=0)."""
    H, W = canvas.active_h, canvas.active_w
    occupied = np.zeros((H, W), dtype=np.float32)
    for f in factions:
        if not f.alive or f.id == exclude_faction_id:
            continue
        occupied[max(0, f.y0) : min(H, f.y1), max(0, f.x0) : min(W, f.x1)] += 1.0

    rx0, ry0, rx1, ry1 = prefer_region if prefer_region is not None else (0, 0, W, H)
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(W, rx1), min(H, ry1)
    if rx1 - rx0 < w or ry1 - ry0 < h:
        rx0, ry0, rx1, ry1 = 0, 0, W, H

    _rng = rng or random.Random()
    best_score = 1e9
    best_pos = (rx0, ry0)
    # Fast randomized candidate sampling spreads spawns evenly across the full height & width
    max_x = max(rx0, rx1 - w)
    max_y = max(ry0, ry1 - h)
    for _ in range(48):
        x0 = _rng.randint(rx0, max_x) if max_x > rx0 else rx0
        y0 = _rng.randint(ry0, max_y) if max_y > ry0 else ry0
        occ = float(occupied[y0 : y0 + h, x0 : x0 + w].mean())
        score = occ * 10.0 + _rng.random() * 0.15
        if score < best_score:
            best_score = score
            best_pos = (x0, y0)
            if occ == 0.0:
                break
    return best_pos


def find_neighbor_spawn_spot(
    canvas: CanvasState,
    factions: list[Faction],
    w: int,
    h: int,
    rng: random.Random | None = None,
    prefer_region: tuple[int, int, int, int] | None = None,
    max_overlap: float = 0.14,
) -> tuple[int, int] | None:
    """Pack a new artwork snugly next to an existing alive artwork (0-2px seam) so r/place neighborhoods grow organically."""
    H, W = canvas.active_h, canvas.active_w
    rx0, ry0, rx1, ry1 = prefer_region if prefer_region is not None else (0, 0, W, H)
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(W, rx1), min(H, ry1)
    if rx1 - rx0 < w or ry1 - ry0 < h:
        rx0, ry0, rx1, ry1 = 0, 0, W, H

    anchors = [
        f for f in factions
        if f.alive and not f.is_void and f.art_mode != "ribbon"
        and f.x1 > rx0 and f.x0 < rx1 and f.y1 > ry0 and f.y0 < ry1
    ]
    if not anchors:
        return None

    occupied = np.zeros((H, W), dtype=bool)
    for f in factions:
        if not f.alive:
            continue
        occupied[max(0, f.y0) : min(H, f.y1), max(0, f.x0) : min(W, f.x1)] = True

    _rng = rng or random.Random()
    _rng.shuffle(anchors)
    best: tuple[int, int] | None = None
    best_occ = max_overlap
    for anchor in anchors[:24]:
        gap = _rng.randint(0, 2)
        sides = [
            (anchor.x1 + gap, anchor.y0 + _rng.randint(-h // 3, max(0, anchor.h - h // 2))),
            (anchor.x0 - w - gap, anchor.y0 + _rng.randint(-h // 3, max(0, anchor.h - h // 2))),
            (anchor.x0 + _rng.randint(-w // 3, max(0, anchor.w - w // 2)), anchor.y1 + gap),
            (anchor.x0 + _rng.randint(-w // 3, max(0, anchor.w - w // 2)), anchor.y0 - h - gap),
        ]
        _rng.shuffle(sides)
        for cx, cy in sides:
            cx = max(rx0, min(rx1 - w, cx))
            cy = max(ry0, min(ry1 - h, cy))
            occ = float(occupied[cy : cy + h, cx : cx + w].mean())
            if occ < best_occ:
                best_occ = occ
                best = (cx, cy)
                if occ == 0.0:
                    return best
    return best


def find_whitespace_niche(
    canvas: CanvasState,
    factions: list[Faction],
    rng: random.Random | None = None,
    min_dim: int = 10,
    max_w: int = 38,
    max_h: int = 34,
) -> tuple[int, int, int, int, np.ndarray] | None:
    """Locate an unclaimed pixel-level whitespace niche (x0, y0, w, h, local_free_mask) of organic varied size."""
    H, W = canvas.active_h, canvas.active_w
    claimed = np.zeros((H, W), dtype=bool)
    for f in factions:
        if not f.alive:
            continue
        y0, y1 = max(0, f.y0), min(H, f.y1)
        x0, x1 = max(0, f.x0), min(W, f.x1)
        if y1 <= y0 or x1 <= x0:
            continue
        bp_np = getattr(f, "_bp_cpu_cache", None)
        if bp_np is None or bp_np.shape != (f.h, f.w):
            bp_np = f.blueprint.detach().cpu().numpy()
            f._bp_cpu_cache = bp_np  # type: ignore[attr-defined]
        sub = bp_np[: y1 - y0, : x1 - x0]
        claimed[y0:y1, x0:x1] |= sub >= 0

    free = ~claimed
    free_ys, free_xs = np.where(free)
    n_free = len(free_ys)
    if n_free < min_dim * min_dim:
        return None

    _rng = rng or random.Random()
    best: tuple[int, int, int, int, np.ndarray] | None = None
    best_score = -1.0

    # Sample varied niche target aspect ratios so every niche faction has a distinct organic size!
    for _ in range(36):
        pick = _rng.randint(0, n_free - 1)
        sy = int(free_ys[pick])
        sx = int(free_xs[pick])
        cap_w = _rng.randint(min_dim + 2, max(min_dim + 2, max_w))
        cap_h = _rng.randint(min_dim + 2, max(min_dim + 2, max_h))

        x0 = max(0, sx - 3)
        x1 = min(W, sx + 4)
        y0 = max(0, sy - 3)
        y1 = min(H, sy + 4)

        # Greedily grow outward into contiguous unclaimed whitespace in all 4 directions
        for _step in range(14):
            grew = False
            if (x1 - x0) < cap_w:
                if x0 > 0 and float(free[y0:y1, max(0, x0 - 2) : x0].mean()) >= 0.45:
                    x0 = max(0, x0 - 2)
                    grew = True
                if x1 < W and (x1 - x0) < cap_w and float(free[y0:y1, x1 : min(W, x1 + 2)].mean()) >= 0.45:
                    x1 = min(W, x1 + 2)
                    grew = True
            if (y1 - y0) < cap_h:
                if y0 > 0 and float(free[max(0, y0 - 2) : y0, x0:x1].mean()) >= 0.45:
                    y0 = max(0, y0 - 2)
                    grew = True
                if y1 < H and (y1 - y0) < cap_h and float(free[y1 : min(H, y1 + 2), x0:x1].mean()) >= 0.45:
                    y1 = min(H, y1 + 2)
                    grew = True
            if not grew:
                break

        nw, nh = x1 - x0, y1 - y0
        if nw < min_dim or nh < min_dim:
            continue
        loc = free[y0:y1, x0:x1]
        free_px = int(loc.sum())
        if free_px < max(55, int(nw * nh * 0.42)):
            continue
        # Favor substantial unclaimed pockets AND corners/edges so perimeter gaps vanish fast
        edge_bonus = 1.25 if (x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2) else 1.0
        score = float(free_px) * edge_bonus * (0.85 + 0.30 * _rng.random())
        if score > best_score:
            best_score = score
            best = (x0, y0, nw, nh, loc.copy())

    return best


def find_void_spawn_spot(canvas: CanvasState, factions: list[Faction], size: int = 84) -> tuple[int, int]:
    """Pick a contested border or busy art edge inside the active viewport for a Void eruption."""
    H, W = canvas.active_h, canvas.active_w
    act = canvas.activity[:H, :W].detach().cpu().numpy()
    best_score = -1.0
    best_pos = (max(0, W // 2 - size // 2), max(0, H // 2 - size // 2))
    step = max(28, int(max(W, H) // 36))
    for y0 in range(0, max(1, H - size + 1), step):
        for x0 in range(0, max(1, W - size + 1), step):
            s = float(act[y0 : y0 + size, x0 : x0 + size].mean())
            if s > best_score:
                best_score = s
                best_pos = (x0, y0)
    return best_pos
