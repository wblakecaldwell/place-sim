"""Official r/place 32-color palette with fast perceptual CIE-LAB quantization."""

from __future__ import annotations

import numpy as np

# Index 0 is White so a zero-initialized canvas starts cleanly blank white.
PALETTE_SPEC: list[tuple[str, str]] = [
    ("white", "#FFFFFF"),          # 0
    ("black", "#000000"),          # 1
    ("burgundy", "#6D001A"),       # 2
    ("dark_red", "#BE0039"),       # 3
    ("red", "#FF4500"),            # 4
    ("orange", "#FFA800"),         # 5
    ("yellow", "#FFD635"),         # 6
    ("pale_yellow", "#FFF8B8"),    # 7
    ("dark_green", "#00A368"),     # 8
    ("green", "#00CC78"),          # 9
    ("light_green", "#7EED56"),    # 10
    ("dark_teal", "#00756F"),      # 11
    ("teal", "#009EAA"),           # 12
    ("light_teal", "#00CCC0"),     # 13
    ("dark_blue", "#2450A4"),      # 14
    ("blue", "#3690EA"),           # 15
    ("light_blue", "#51E9F4"),     # 16
    ("indigo", "#493AC1"),         # 17
    ("periwinkle", "#6A5CFF"),     # 18
    ("lavender", "#94B3FF"),       # 19
    ("dark_purple", "#811E9F"),    # 20
    ("purple", "#B44AC0"),         # 21
    ("pale_purple", "#E4ABFF"),    # 22
    ("magenta", "#DE107F"),        # 23
    ("pink", "#FF3881"),           # 24
    ("light_pink", "#FF99AA"),     # 25
    ("dark_brown", "#6D482F"),     # 26
    ("brown", "#9C6926"),          # 27
    ("beige", "#FFB470"),          # 28
    ("dark_gray", "#515252"),      # 29
    ("gray", "#898D90"),           # 30
    ("light_gray", "#D4D7D9"),     # 31
]

WHITE_IDX = 0
BLACK_IDX = 1
BURGUNDY_IDX = 2
DARK_RED_IDX = 3
RED_IDX = 4
ORANGE_IDX = 5
YELLOW_IDX = 6
GREEN_IDX = 9
BLUE_IDX = 15
DARK_BLUE_IDX = 14
PURPLE_IDX = 21
PINK_IDX = 24

NUM_COLORS = len(PALETTE_SPEC)
COLOR_NAMES: list[str] = [name for name, _ in PALETTE_SPEC]
COLOR_HEX: list[str] = [hx for _, hx in PALETTE_SPEC]
NAME_TO_IDX: dict[str, int] = {name: i for i, (name, _) in enumerate(PALETTE_SPEC)}

# Official r/place 2022 progressive palette unlock tiers:
# Tier 1 (Day 1): 16 initial colors
PALETTE_TIER_1_16: tuple[int, ...] = (
    0, 1, 4, 5, 6, 8, 10, 14, 15, 16, 20, 21, 25, 27, 30, 31,
)
# Tier 2 (Day 2): 24 colors (+ dark_red, green, dark_teal, teal, indigo, periwinkle, pink, dark_gray)
PALETTE_TIER_2_24: tuple[int, ...] = tuple(
    sorted(set(PALETTE_TIER_1_16) | {3, 9, 11, 12, 17, 18, 24, 29})
)
# Tier 3 (Day 3): all 32 colors
PALETTE_TIER_3_32: tuple[int, ...] = tuple(range(NUM_COLORS))
# Tier 4 (Whiteout Finale): pure white only
PALETTE_TIER_4_WHITEOUT: tuple[int, ...] = (WHITE_IDX,)


def get_tier_colors(tier: int) -> tuple[int, ...]:
    """Return the unlocked palette indices for r/place 2022 palette tier 1..4."""
    if tier <= 1:
        return PALETTE_TIER_1_16
    if tier == 2:
        return PALETTE_TIER_2_24
    if tier == 3:
        return PALETTE_TIER_3_32
    return PALETTE_TIER_4_WHITEOUT


def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


PALETTE_RGB: np.ndarray = np.array(
    [_hex_to_rgb(hx) for _, hx in PALETTE_SPEC], dtype=np.uint8
)  # [32, 3]


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 or float [..., 3] sRGB to CIE-LAB [..., 3] float32."""
    arr = rgb.astype(np.float32) / 255.0
    # sRGB -> linear RGB
    mask = arr > 0.04045
    lin = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)

    # Linear RGB -> XYZ (D65)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    eps = 0.008856
    kappa = 903.3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return np.stack([L, a, b_ch], axis=-1)


PALETTE_LAB: np.ndarray = _rgb_to_lab(PALETTE_RGB)  # [32, 3]


def rgb_to_palette_indices(
    rgb: np.ndarray,
    allowed_colors: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """Map an [H, W, 3] uint8 RGB image to nearest allowed palette indices via CIE-LAB."""
    lab = _rgb_to_lab(rgb)  # [H, W, 3]
    weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    if allowed_colors is None or len(allowed_colors) >= NUM_COLORS:
        diff = lab[..., None, :] - PALETTE_LAB[None, None, :, :]
        dist2 = np.sum((diff * weights) ** 2, axis=-1)
        return np.argmin(dist2, axis=-1).astype(np.uint8)
    allowed_arr = np.asarray(sorted(set(int(c) % NUM_COLORS for c in allowed_colors)), dtype=np.uint8)
    sub_lab = PALETTE_LAB[allowed_arr]  # [K, 3]
    diff = lab[..., None, :] - sub_lab[None, None, :, :]
    dist2 = np.sum((diff * weights) ** 2, axis=-1)
    nearest = np.argmin(dist2, axis=-1)
    return allowed_arr[nearest]


def remap_indices_to_allowed(
    grid: np.ndarray,
    allowed_colors: tuple[int, ...] | list[int] | None,
) -> np.ndarray:
    """Remap an int16/uint8 palette grid (preserving -1 transparency) into allowed_colors via CIE-LAB."""
    if allowed_colors is None or len(allowed_colors) >= NUM_COLORS:
        return grid
    allowed_arr = np.asarray(sorted(set(int(c) % NUM_COLORS for c in allowed_colors)), dtype=np.int16)
    if allowed_arr.size == 0:
        return grid
    # Precompute 32-entry lookup table
    weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    diff = PALETTE_LAB[:, None, :] - PALETTE_LAB[allowed_arr][None, :, :]
    dist2 = np.sum((diff * weights) ** 2, axis=-1)
    lut = allowed_arr[np.argmin(dist2, axis=-1)]
    out = grid.copy()
    valid = out >= 0
    if valid.any():
        out[valid] = lut[np.clip(out[valid], 0, NUM_COLORS - 1)]
    return out


def palette_to_rgb(indices: np.ndarray) -> np.ndarray:
    """Convert [H, W] uint8 palette index grid to [H, W, 3] uint8 RGB image."""
    return PALETTE_RGB[indices]


def resolve_color(name_or_idx: str | int, default: int = BLUE_IDX) -> int:
    """Resolve a palette color by name, fuzzy substring, or integer index."""
    if isinstance(name_or_idx, int):
        return int(name_or_idx) % NUM_COLORS
    key = str(name_or_idx).strip().lower().replace(" ", "_").replace("-", "_")
    if key in NAME_TO_IDX:
        return NAME_TO_IDX[key]
    for name, idx in NAME_TO_IDX.items():
        if key in name or name in key:
            return idx
    return default
