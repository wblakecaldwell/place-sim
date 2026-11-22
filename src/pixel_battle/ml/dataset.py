"""Download, chronologically replay, and extract 25x25 temporal hindsight patches from Reddit r/place 2022."""

from __future__ import annotations

import gzip
import logging
import subprocess
from pathlib import Path

import numpy as np

from pixel_battle.ml.model import (
    ARCH_ARTIST,
    ARCH_BUILDER,
    ARCH_DEFENDER,
    ARCH_VOID,
    NO_GOAL_IDX,
    PATCH_RADIUS,
    PATCH_SIZE,
)
from pixel_battle.palette import (
    BLACK_IDX,
    BURGUNDY_IDX,
    COLOR_HEX,
    DARK_RED_IDX,
    WHITE_IDX,
    rgb_to_palette_indices,
)

log = logging.getLogger(__name__)

# 6 contiguous Day 1 chunks: Hours 0 -> 12 nonstop from blank canvas (~12M ground-truth placements)
EARLY_CHUNKS = [10, 6, 9, 14, 15, 17]
# 2 contiguous Day 3 peak chunks: all 32 colors unlocked, intense faction wars & repairs (~3.8M placements)
PEAK_CHUNKS = [36, 37]
DEFAULT_CHUNKS = EARLY_CHUNKS + PEAK_CHUNKS

REDDIT_URL_TEMPLATE = (
    "https://placedata.reddit.com/data/canvas-history/2022_place_canvas_history-{idx:012d}.csv.gzip"
)

HEX_TO_IDX: dict[str, int] = {hx.upper(): i for i, hx in enumerate(COLOR_HEX)}


def _hex_to_idx(hx: str) -> int:
    key = hx.strip().upper()
    if key in HEX_TO_IDX:
        return HEX_TO_IDX[key]
    raw = key.lstrip("#")
    if len(raw) == 6:
        rgb = np.array([[[int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)]]], dtype=np.uint8)
        idx = int(rgb_to_palette_indices(rgb)[0, 0])
        HEX_TO_IDX[key] = idx
        return idx
    return BLACK_IDX


def download_reddit_chunks(data_dir: Path, chunk_ids: list[int] | None = None) -> list[Path]:
    """Download official Reddit r/place 2022 chunks into data_dir if not already cached."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = chunk_ids or DEFAULT_CHUNKS
    paths: list[Path] = []
    for cid in ids:
        out = data_dir / f"chunk_{cid:02d}.csv.gzip"
        if not out.exists() or out.stat().st_size < 10_000_000:
            url = REDDIT_URL_TEMPLATE.format(idx=cid)
            log.info("Downloading official Reddit r/place 2022 chunk %02d from %s...", cid, url)
            subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(out), url], check=True)
        paths.append(out)
    return paths


def parse_chunk_fast(gz_path: Path, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Reddit r/place 2022 .csv.gzip file into chronologically sorted compact arrays."""
    log.info("Parsing %s...", gz_path.name)
    ts_list: list[int] = []
    x_list: list[int] = []
    y_list: list[int] = []
    c_list: list[int] = []

    hex_map = HEX_TO_IDX
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as fh:
        fh.readline()  # skip header
        for line in fh:
            p1 = line.find(",")
            if p1 < 19:
                continue
            p2 = line.find(",", p1 + 1)
            if p2 < 0:
                continue
            p3 = line.find(",", p2 + 1)
            if p3 < 0:
                continue

            coord_str = line[p3 + 1 :].strip().strip('"')
            c_comma = coord_str.find(",")
            if c_comma < 0:
                continue
            if coord_str.find(",", c_comma + 1) >= 0:
                continue
            try:
                x = int(coord_str[:c_comma])
                y = int(coord_str[c_comma + 1 :])
            except ValueError:
                continue
            if not (0 <= x < 2000 and 0 <= y < 2000):
                continue

            ts_str = line[:p1]
            try:
                day = int(ts_str[8:10])
                hr = int(ts_str[11:13])
                mn = int(ts_str[14:16])
                sc = int(ts_str[17:19])
                ms = 0
                if len(ts_str) > 20 and ts_str[19] == ".":
                    end_ms = ts_str.find(" ", 20)
                    ms_part = ts_str[20:end_ms] if end_ms > 20 else ts_str[20:23]
                    ms = int(ms_part.ljust(3, "0")[:3])
                ts_ms = (((day * 24 + hr) * 60 + mn) * 60 + sc) * 1000 + ms
            except ValueError:
                continue

            hx = line[p2 + 1 : p3].strip().upper()
            col = hex_map.get(hx)
            if col is None:
                col = _hex_to_idx(hx)

            ts_list.append(ts_ms)
            x_list.append(x)
            y_list.append(y)
            c_list.append(col)
            if max_rows is not None and len(ts_list) >= max_rows:
                break

    ts_arr = np.array(ts_list, dtype=np.int64)
    xs_arr = np.array(x_list, dtype=np.uint16)
    ys_arr = np.array(y_list, dtype=np.uint16)
    cs_arr = np.array(c_list, dtype=np.uint8)

    order = np.argsort(ts_arr, kind="mergesort")
    log.info("Parsed & sorted %d valid placements from %s.", len(order), gz_path.name)
    return ts_arr[order], xs_arr[order], ys_arr[order], cs_arr[order]


def _load_and_merge_chunks(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tss, xss, yss, css = [], [], [], []
    for p in paths:
        t, x, y, c = parse_chunk_fast(p)
        tss.append(t)
        xss.append(x)
        yss.append(y)
        css.append(c)
    ts_m = np.concatenate(tss)
    x_m = np.concatenate(xss)
    y_m = np.concatenate(yss)
    c_m = np.concatenate(css)
    order = np.argsort(ts_m, kind="mergesort")
    log.info("Merged %d chunks -> %d chronological placements.", len(paths), len(order))
    return ts_m[order], x_m[order], y_m[order], c_m[order]


def _extract_from_stream(
    ts_arr: np.ndarray,
    xs_arr: np.ndarray,
    ys_arr: np.ndarray,
    cs_arr: np.ndarray,
    target_samples: int,
    patch_size: int = PATCH_SIZE,
    burn_in: int = 0,
    horizon_events: int = 380_000,
    min_patch_density: float = 0.0,
    void_boost: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Two-pointer chronological hindsight replay extracting PxP temporal training patches."""
    rng = rng or np.random.default_rng(42)
    n_total = len(ts_arr)
    usable = n_total - burn_in - horizon_events - 1
    if usable <= 1000:
        raise ValueError(f"Stream too short ({n_total}) for burn_in={burn_in}, horizon={horizon_events}")

    H, W = 2000, 2000
    P = patch_size
    R = P // 2
    max_jitter = max(4, R - 3)  # e.g. -9..+9 inside 25x25

    cur_canvas = np.full((H, W), WHITE_IDX, dtype=np.uint8)
    prev_canvas = np.full((H, W), WHITE_IDX, dtype=np.uint8)
    last_ts = np.full((H, W), -10**12, dtype=np.int64)
    fut_canvas = np.full((H, W), WHITE_IDX, dtype=np.uint8)

    # Pre-advance fut_canvas ahead of cur_canvas by horizon_events
    fut_ptr = 0
    while fut_ptr < min(n_total, burn_in + horizon_events):
        fut_canvas[ys_arr[fut_ptr], xs_arr[fut_ptr]] = cs_arr[fut_ptr]
        fut_ptr += 1

    # Burn-in cur_canvas (preserving pre-flip prev_canvas cleanly)
    for i in range(burn_in):
        x, y, c = int(xs_arr[i]), int(ys_arr[i]), int(cs_arr[i])
        old = cur_canvas[y, x]
        if c != old:
            prev_canvas[y, x] = old
        cur_canvas[y, x] = c
        last_ts[y, x] = ts_arr[i]

    base_prob = min(0.95, (target_samples * (1.25 if min_patch_density > 0 else 1.05)) / usable)

    cur_patches = np.empty((target_samples, P, P), dtype=np.uint8)
    prev_patches = np.empty((target_samples, P, P), dtype=np.uint8)
    rec_patches = np.empty((target_samples, P, P), dtype=np.uint8)
    goal_patches = np.empty((target_samples, P, P), dtype=np.uint8)
    archetypes = np.empty(target_samples, dtype=np.uint8)
    target_pos = np.empty(target_samples, dtype=np.uint16)
    target_col = np.empty(target_samples, dtype=np.uint8)

    collected = 0
    void_colors = (BLACK_IDX, BURGUNDY_IDX, DARK_RED_IDX)

    for i in range(burn_in, n_total - horizon_events):
        if fut_ptr < n_total:
            fut_canvas[ys_arr[fut_ptr], xs_arr[fut_ptr]] = cs_arr[fut_ptr]
            fut_ptr += 1

        x = int(xs_arr[i])
        y = int(ys_arr[i])
        c_new = int(cs_arr[i])
        ts_now = int(ts_arr[i])
        c_old = int(cur_canvas[y, x])

        if collected < target_samples:
            # Fast Void check so we can oversample real Void tendril placements 4x
            is_void_cand = (c_new in void_colors) and (c_old != BLACK_IDX) and (c_old != WHITE_IDX)
            p_sample = min(1.0, base_prob * void_boost) if is_void_cand else base_prob
            if rng.random() < p_sample:
                jx = int(rng.integers(-max_jitter, max_jitter + 1))
                jy = int(rng.integers(-max_jitter, max_jitter + 1))
                cx = x + jx
                cy = y + jy
                if R <= cx < W - R and R <= cy < H - R:
                    c_patch = cur_canvas[cy - R : cy + R + 1, cx - R : cx + R + 1]
                    if min_patch_density <= 0.0 or float((c_patch != WHITE_IDX).mean()) >= min_patch_density:
                        c_prev = int(prev_canvas[y, x])
                        age_ms = ts_now - int(last_ts[y, x])
                        n3 = cur_canvas[max(0, y - 1) : min(H, y + 2), max(0, x - 1) : min(W, x + 2)]
                        n4_same = (
                            int(cur_canvas[y, max(0, x - 1)] == c_new)
                            + int(cur_canvas[y, min(W - 1, x + 1)] == c_new)
                            + int(cur_canvas[max(0, y - 1), x] == c_new)
                            + int(cur_canvas[min(H - 1, y + 1), x] == c_new)
                        )
                        touches_black = bool((n3 == BLACK_IDX).any())

                        if (c_new == c_prev and c_new != c_old and age_ms < 240_000) or (
                            c_new != c_old and n4_same >= 3
                        ):
                            arch = ARCH_DEFENDER
                        elif is_void_cand and touches_black:
                            arch = ARCH_VOID
                        elif bool((n3 == c_new).any()):
                            arch = ARCH_BUILDER
                        else:
                            arch = ARCH_ARTIST

                        ts_patch = last_ts[cy - R : cy + R + 1, cx - R : cx + R + 1]
                        dt_sec = np.clip((ts_now - ts_patch) * 1e-3, 0.0, 3600.0)
                        rec_u8 = np.clip(np.exp(-dt_sec / 180.0) * 255.0, 0, 255).astype(np.uint8)
                        p_patch = prev_canvas[cy - R : cy + R + 1, cx - R : cx + R + 1]

                        goal_aligned = (c_new == int(fut_canvas[y, x])) and (arch != ARCH_VOID)
                        if goal_aligned and rng.random() < 0.48:
                            g_patch = fut_canvas[cy - R : cy + R + 1, cx - R : cx + R + 1].copy()
                            if rng.random() < 0.18:
                                cut = int(rng.integers(3, 9))
                                side = int(rng.integers(0, 4))
                                if side == 0:
                                    g_patch[:cut, :] = NO_GOAL_IDX
                                elif side == 1:
                                    g_patch[-cut:, :] = NO_GOAL_IDX
                                elif side == 2:
                                    g_patch[:, :cut] = NO_GOAL_IDX
                                else:
                                    g_patch[:, -cut:] = NO_GOAL_IDX
                                g_patch[R - jy, R - jx] = c_new
                        else:
                            g_patch = np.full((P, P), NO_GOAL_IDX, dtype=np.uint8)

                        cur_patches[collected] = c_patch
                        prev_patches[collected] = p_patch
                        rec_patches[collected] = rec_u8
                        goal_patches[collected] = g_patch
                        archetypes[collected] = arch
                        target_pos[collected] = (R - jy) * P + (R - jx)
                        target_col[collected] = c_new
                        collected += 1

        # Apply placement and update prev_canvas only when pixel color changes
        if c_new != c_old:
            prev_canvas[y, x] = c_old
        cur_canvas[y, x] = c_new
        last_ts[y, x] = ts_now

    log.info("Extracted %d / %d training samples from stream (%dx%d).", collected, target_samples, P, P)
    return {
        "cur_patches": cur_patches[:collected],
        "prev_patches": prev_patches[:collected],
        "recency": rec_patches[:collected],
        "goal_patches": goal_patches[:collected],
        "archetypes": archetypes[:collected],
        "target_pos": target_pos[:collected],
        "target_col": target_col[:collected],
    }


def build_reddit_training_dataset(
    data_dir: Path = Path("data/reddit_place"),
    out_npz: Path = Path("data/reddit_place/train_dataset_v2.npz"),
    early_samples: int = 320_000,
    peak_samples: int = 160_000,
    patch_size: int = PATCH_SIZE,
    seed: int = 42,
) -> Path:
    """Download contiguous r/place 2022 chunks and build balanced PxP temporal hindsight .npz."""
    data_dir = Path(data_dir)
    out_npz = Path(out_npz)
    if out_npz.exists():
        log.info("Using cached dataset at %s", out_npz)
        return out_npz

    early_paths = download_reddit_chunks(data_dir, EARLY_CHUNKS)
    peak_paths = download_reddit_chunks(data_dir, PEAK_CHUNKS)
    rng = np.random.default_rng(seed)

    # 1. Contiguous Day 1 Hours 0 -> 12 (~12M exact ground-truth placements from blank canvas)
    ts_e, x_e, y_e, c_e = _load_and_merge_chunks(early_paths)
    d_early = _extract_from_stream(
        ts_e,
        x_e,
        y_e,
        c_e,
        target_samples=early_samples,
        patch_size=patch_size,
        burn_in=20_000,
        horizon_events=380_000,
        min_patch_density=0.04,
        void_boost=1.0,
        rng=rng,
    )
    del ts_e, x_e, y_e, c_e

    # 2. Contiguous Day 3 Peak (~4.0M placements, all 32 colors unlocked; burn in first 1.2M)
    ts_p, x_p, y_p, c_p = _load_and_merge_chunks(peak_paths)
    d_peak = _extract_from_stream(
        ts_p,
        x_p,
        y_p,
        c_p,
        target_samples=peak_samples,
        patch_size=patch_size,
        burn_in=1_200_000,
        horizon_events=300_000,
        min_patch_density=0.25,
        void_boost=1.0,
        rng=rng,
    )
    del ts_p, x_p, y_p, c_p

    merged = {k: np.concatenate([d_early[k], d_peak[k]], axis=0) for k in d_early}
    del d_early, d_peak

    perm = rng.permutation(len(merged["target_col"]))
    merged = {k: v[perm] for k, v in merged.items()}

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **merged)
    n = len(merged["target_col"])
    arch_counts = np.bincount(merged["archetypes"], minlength=5)
    R = patch_size // 2
    goal_frac = float((merged["goal_patches"][:, R, R] < NO_GOAL_IDX).mean())
    log.info(
        "Saved %d Reddit r/place v2 samples (%dx%d) -> %s (goal_frac=%.1f%%, archetypes=%s)",
        n,
        patch_size,
        patch_size,
        out_npz,
        goal_frac * 100.0,
        arch_counts.tolist(),
    )
    return out_npz
