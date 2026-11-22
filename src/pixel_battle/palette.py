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


def rgb_to_palette_indices(rgb: np.ndarray) -> np.ndarray:
    """Map an [H, W, 3] uint8 RGB image to nearest 32-color palette indices via CIE-LAB."""
    lab = _rgb_to_lab(rgb)  # [H, W, 3]
    diff = lab[..., None, :] - PALETTE_LAB[None, None, :, :]  # [H, W, 32, 3]
    # Slightly boost chroma importance (a, b) so vibrant hues don't wash to gray
    weights = np.array([0.85, 1.15, 1.15], dtype=np.float32)
    dist2 = np.sum((diff * weights) ** 2, axis=-1)  # [H, W, 32]
    return np.argmin(dist2, axis=-1).astype(np.uint8)


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
