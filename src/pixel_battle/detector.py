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
) -> list[ClusterCandidate]:
    """Scan unclaimed regions (including flush corners, edges, tight micro-gaps, and optional flag cameos)."""
    colors = canvas.colors_numpy()
    activity = canvas.activity_numpy()
    H, W = colors.shape
    window = max(12, min(window, min(H, W)))
    stride = max(6, stride)

    occupied = _build_occupancy_mask(H, W, factions, exclude_host_flags=False)
    xs = sorted(set(list(range(0, max(1, W - window), stride)) + [max(0, W - window)]))
    ys = sorted(set(list(range(0, max(1, H - window), stride)) + [max(0, H - window)]))

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

            if is_corner and window >= 36:
                wall_bonus = 0.65
            elif is_edge and alive_edge_count < 16:
                wall_bonus = 0.22
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
    """Pick a starting (x0, y0, init_w, init_h) for a progressive Transverse Ribbon avoiding existing tracks."""
    H, W = canvas.height, canvas.width
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
        off = rng.randint(0, max(1, W - init_diag - 60))
        if rng.random() < 0.5:
            return off, 0, init_diag, init_diag
        return 0, off, init_diag, init_diag
    else:
        off = rng.randint(0, max(1, W - init_diag - 60))
        return off, max(0, H - init_diag), init_diag, init_diag


def update_faction_metrics(
    canvas: CanvasState,
    factions: list[Faction],
    user_factions: torch.Tensor,
) -> None:
    """Recompute each faction's member count, blueprint integrity, owned pixels, and churn."""
    if not factions:
        return
    valid_u = user_factions[user_factions >= 0]
    max_id = max(f.id for f in factions) + 1
    counts = (
        torch.bincount(valid_u.to(torch.int64), minlength=max_id).detach().cpu().numpy()
        if valid_u.numel() > 0
        else np.zeros(max_id, dtype=np.int64)
    )
    owner_grid = canvas.owner

    # Vectorized owner pixel count (avoids 200 full-grid equality scans)
    valid_o = owner_grid[owner_grid >= 0]
    owned_counts = (
        torch.bincount(valid_o.to(torch.int64), minlength=max_id).detach().cpu().numpy()
        if valid_o.numel() > 0
        else np.zeros(max_id, dtype=np.int64)
    )

    for f in factions:
        if not f.alive:
            f.member_count = 0
            continue
        f.member_count = int(counts[f.id]) if f.id < len(counts) else 0
        f.owned_pixels = int(owned_counts[f.id]) if f.id < len(owned_counts) else 0

        crop_colors = canvas.colors[f.y0 : f.y1, f.x0 : f.x1].to(torch.int16)
        crop_act = canvas.activity[f.y0 : f.y1, f.x0 : f.x1]
        f.recent_churn = float(crop_act.mean().item())

        if f.is_void and f.void_variant == "tendrils":
            black_frac = float((crop_colors == BLACK_IDX).to(torch.float32).mean().item())
            f.integrity = black_frac
        else:
            valid = f.blueprint >= 0
            total_valid = int(valid.sum().item())
            if total_valid > 0 and crop_colors.shape == f.blueprint.shape:
                matched = int(((crop_colors == f.blueprint) & valid).sum().item())
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
) -> tuple[int, int]:
    """Find the least-contested (x0, y0) rectangle (including flush edges/corners) for exodus."""
    H, W = canvas.height, canvas.width
    occupied = np.zeros((H, W), dtype=np.float32)
    for f in factions:
        if not f.alive or f.id == exclude_faction_id:
            continue
        occupied[f.y0 : f.y1, f.x0 : f.x1] += 1.0

    best_score = 1e9
    best_pos = (0, 0)
    step = 16 if max(w, h) <= 32 else 24
    xs = sorted(set(list(range(0, max(1, W - w), step)) + [max(0, W - w)]))
    ys = sorted(set(list(range(0, max(1, H - h), step)) + [max(0, H - h)]))
    act = canvas.activity_numpy()
    for y0 in ys:
        for x0 in xs:
            occ = float(occupied[y0 : y0 + h, x0 : x0 + w].mean())
            churn = float(act[y0 : y0 + h, x0 : x0 + w].mean())
            score = occ * 10.0 + churn * 0.5
            if score < best_score:
                best_score = score
                best_pos = (x0, y0)
    return best_pos


def find_void_spawn_spot(canvas: CanvasState, factions: list[Faction], size: int = 84) -> tuple[int, int]:
    """Pick a contested border or busy art edge for a Void / Black Hole / Monster eruption."""
    H, W = canvas.height, canvas.width
    act = canvas.activity_numpy()
    best_score = -1.0
    best_pos = (W // 2 - size // 2, H // 2 - size // 2)
    for y0 in range(0, max(1, H - size + 1), 28):
        for x0 in range(0, max(1, W - size + 1), 28):
            s = float(act[y0 : y0 + size, x0 : x0 + size].mean())
            if s > best_score:
                best_score = s
                best_pos = (x0, y0)
    return best_pos
