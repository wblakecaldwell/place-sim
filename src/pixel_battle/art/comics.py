"""Procedural pixel-art short comic strips (2/3/4 panels) and expressive cartoon/meme portraits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    BURGUNDY_IDX,
    DARK_BLUE_IDX,
    DARK_RED_IDX,
    GREEN_IDX,
    NAME_TO_IDX,
    ORANGE_IDX,
    PINK_IDX,
    PURPLE_IDX,
    RED_IDX,
    WHITE_IDX,
    YELLOW_IDX,
)

LIGHT_BLUE = NAME_TO_IDX["light_blue"]
LIGHT_GREEN = NAME_TO_IDX["light_green"]
DARK_GREEN = NAME_TO_IDX["dark_green"]
LIGHT_PINK = NAME_TO_IDX["light_pink"]
PALE_YELLOW = NAME_TO_IDX["pale_yellow"]
INDIGO = NAME_TO_IDX["indigo"]
DARK_PURPLE = NAME_TO_IDX["dark_purple"]
MAGENTA = NAME_TO_IDX["magenta"]
BEIGE = NAME_TO_IDX["beige"]
BROWN = NAME_TO_IDX["brown"]
DARK_BROWN = NAME_TO_IDX["dark_brown"]
GRAY = NAME_TO_IDX["gray"]
DARK_GRAY = NAME_TO_IDX["dark_gray"]
LIGHT_GRAY = NAME_TO_IDX["light_gray"]
TEAL = NAME_TO_IDX["teal"]
LIGHT_TEAL = NAME_TO_IDX["light_teal"]
LAVENDER = NAME_TO_IDX["lavender"]
PERIWINKLE = NAME_TO_IDX["periwinkle"]


# ---------------------------------------------------------------------------
# Compact 3x5 pixel micro-font for comic speech bubbles & meme captions
# ---------------------------------------------------------------------------

_MINI_FONT: dict[str, tuple[str, ...]] = {
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("#.#", "##.", "###", ".##", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", ".##", "..#"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "?": ("##.", "..#", ".#.", "...", ".#."),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "0": (".#.", "#.#", "#.#", "#.#", ".#."),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("##.", "..#", ".#.", "#..", "###"),
    "3": ("##.", "..#", ".#.", "..#", "##."),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    "6": (".##", "#..", "###", "#.#", ".#."),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": (".#.", "#.#", ".#.", "#.#", ".#."),
    "9": (".#.", "#.#", ".##", "..#", "##."),
    " ": ("...", "...", "...", "...", "..."),
}


def _draw_mini_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: int = BLACK_IDX,
    outline: int | None = None,
) -> None:
    h, w = img.shape
    cx = x
    for ch in text.upper():
        glyph = _MINI_FONT.get(ch)
        if glyph is None:
            cx += 3
            continue
        if outline is not None:
            for gy, row in enumerate(glyph):
                for gx, c in enumerate(row):
                    if c == "#":
                        for oy in (-1, 0, 1):
                            for ox in (-1, 0, 1):
                                py, px = y + gy + oy, cx + gx + ox
                                if 0 <= py < h and 0 <= px < w:
                                    img[py, px] = outline
        for gy, row in enumerate(glyph):
            for gx, c in enumerate(row):
                if c == "#":
                    py, px = y + gy, cx + gx
                    if 0 <= py < h and 0 <= px < w:
                        img[py, px] = color
        cx += 4


def _draw_bubble(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    fill: int = WHITE_IDX,
    border: int = BLACK_IDX,
    text_col: int = BLACK_IDX,
    tail_dir: str = "down",
) -> None:
    """Draw a tiny crisp comic speech bubble with a pointer tail."""
    h, w = img.shape
    tw = max(4, len(text) * 4 - 1)
    bw = tw + 4
    bh = 9
    x0 = max(1, min(w - bw - 1, cx - bw // 2))
    y0 = max(1, min(h - bh - 2, cy - bh // 2))
    x1 = min(w - 1, x0 + bw)
    y1 = min(h - 1, y0 + bh)
    img[y0:y1, x0:x1] = fill
    img[y0, x0 + 1 : x1 - 1] = border
    img[y1 - 1, x0 + 1 : x1 - 1] = border
    img[y0 + 1 : y1 - 1, x0] = border
    img[y0 + 1 : y1 - 1, x1 - 1] = border
    # Tail notch
    tx = max(x0 + 2, min(x1 - 3, cx))
    if tail_dir == "down" and y1 + 1 < h:
        img[y1 - 1, tx] = fill
        img[y1, tx] = border
        if tx + 1 < w:
            img[y1, tx + 1] = border
    elif tail_dir == "left" and x0 - 1 >= 0:
        ty = (y0 + y1) // 2
        img[ty, x0] = fill
        img[ty, x0 - 1] = border
    _draw_mini_text(img, text, x0 + 2, y0 + 2, color=text_col)


def _fill_ellipse(
    img: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    fill: int,
    outline: int | None = BLACK_IDX,
) -> None:
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    d = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - cy) / max(1.0, ry)) ** 2
    if outline is not None:
        d_out = ((xx - cx) / max(1.0, rx + 1.0)) ** 2 + ((yy - cy) / max(1.0, ry + 1.0)) ** 2
        img[d_out <= 1.0] = outline
    img[d <= 1.0] = fill


def _fill_rect(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill: int,
    outline: int | None = None,
) -> None:
    h, w = img.shape
    xa, xb = max(0, x0), min(w, x1)
    ya, yb = max(0, y0), min(h, y1)
    if xb <= xa or yb <= ya:
        return
    img[ya:yb, xa:xb] = fill
    if outline is not None:
        img[ya, xa:xb] = outline
        img[yb - 1, xa:xb] = outline
        img[ya:yb, xa] = outline
        img[ya:yb, xb - 1] = outline


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int, thickness: int = 1) -> None:
    h, w = img.shape
    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) * 2))
    xs = np.linspace(x0, x1, steps)
    ys = np.linspace(y0, y1, steps)
    r = max(0, thickness // 2)
    for xf, yf in zip(xs, ys):
        ix, iy = int(round(xf)), int(round(yf))
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                if 0 <= iy + oy < h and 0 <= ix + ox < w:
                    img[iy + oy, ix + ox] = color


def _add_cutout_outline(bp: np.ndarray, outline_col: int = BLACK_IDX) -> np.ndarray:
    h, w = bp.shape
    fg = bp >= 0
    dil = fg.copy()
    dil[1:, :] |= fg[:-1, :]
    dil[:-1, :] |= fg[1:, :]
    dil[:, 1:] |= fg[:, :-1]
    dil[:, :-1] |= fg[:, 1:]
    out = bp.copy()
    out[dil & ~fg] = outline_col
    return out


def _resize_nn(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    sh, sw = arr.shape
    if sh == target_h and sw == target_w:
        return arr.copy()
    yi = np.clip((np.arange(target_h) * sh) // target_h, 0, sh - 1)
    xi = np.clip((np.arange(target_w) * sw) // target_w, 0, sw - 1)
    return arr[np.ix_(yi, xi)].astype(np.int16, copy=True)


# ---------------------------------------------------------------------------
# 1. Short Multi-Panel Comic Strips (2, 3, and 4 panels)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComicSpec:
    key: str
    title: str
    layout: str  # "3h" (3x1 horizontal), "2v" (2-panel stacked), "2x2" (4-panel grid)
    slogan: str
    primary: int
    secondary: int
    accent: int


COMIC_CATALOG: list[ComicSpec] = [
    ComicSpec("this_is_fine", "This Is Fine", "3h", "Everything is totally fine.", ORANGE_IDX, YELLOW_IDX, RED_IDX),
    ComicSpec("cat_vs_laser", "Cat vs Laser", "3h", "Must catch the red dot.", ORANGE_IDX, LIGHT_BLUE, RED_IDX),
    ComicSpec("coffee_coder", "Coffee Power", "3h", "Before and after espresso.", PURPLE_IDX, YELLOW_IDX, LIGHT_TEAL),
    ComicSpec("stonks_trip", "Stonks Rollercoaster", "3h", "Line only goes up... right?", GREEN_IDX, RED_IDX, BLUE_IDX),
    ComicSpec("ufo_cow", "Cow Abduction", "3h", "Beam up the pasture.", DARK_BLUE_IDX, LIGHT_GREEN, WHITE_IDX),
    ComicSpec("frog_tea", "Frog Sips Tea", "3h", "None of my business.", GREEN_IDX, LIGHT_BLUE, YELLOW_IDX),
    ComicSpec("void_knocks", "Void At The Door", "3h", "Defend the cozy pixel home.", LIGHT_BLUE, BLACK_IDX, MAGENTA),
    ComicSpec("bonk_shiba", "Shiba Bonk", "3h", "Straight to pixel jail.", YELLOW_IDX, ORANGE_IDX, RED_IDX),
    ComicSpec("penguin_ice", "Cool Penguin", "3h", "Style recovery on ice.", LIGHT_BLUE, BLUE_IDX, ORANGE_IDX),
    ComicSpec("duck_knife", "Duck Found A Blade", "3h", "Peace was never an option.", LIGHT_GREEN, WHITE_IDX, YELLOW_IDX),
    ComicSpec("drake_choice", "Pixel Choice", "2v", "Skip the mess, pick the art.", ORANGE_IDX, RED_IDX, YELLOW_IDX),
    ComicSpec("galaxy_brain", "Galaxy Brain", "2x2", "Ascending four tiers of pixel IQ.", INDIGO, MAGENTA, YELLOW_IDX),
]

COMIC_BY_KEY: dict[str, ComicSpec] = {c.key: c for c in COMIC_CATALOG}
COMIC_KEYS: list[str] = [c.key for c in COMIC_CATALOG]


def _draw_cartoon_dog(p: np.ndarray, cx: int, cy: int, scale: float = 1.0, hat: bool = True, happy: bool = True) -> None:
    r = max(3, int(round(5 * scale)))
    # Body
    _fill_rect(p, cx - r + 1, cy + r - 1, cx + r, cy + r + int(5 * scale), YELLOW_IDX, BLACK_IDX)
    # Ears
    _fill_ellipse(p, cx - r, cy, max(1.5, 1.8 * scale), max(2.2, 3.0 * scale), ORANGE_IDX, BLACK_IDX)
    _fill_ellipse(p, cx + r, cy, max(1.5, 1.8 * scale), max(2.2, 3.0 * scale), ORANGE_IDX, BLACK_IDX)
    # Head
    _fill_ellipse(p, cx, cy, r, r - 0.5, YELLOW_IDX, BLACK_IDX)
    # Snout
    _fill_ellipse(p, cx, cy + 2, max(2, r - 2), max(1.5, 2.2 * scale), PALE_YELLOW, None)
    if 0 <= cy + 1 < p.shape[0] and 0 <= cx < p.shape[1]:
        p[cy + 1, cx] = BLACK_IDX
    # Eyes
    ey = max(0, cy - 1)
    for ex in (cx - 2, cx + 2):
        if 0 <= ey < p.shape[0] and 0 <= ex < p.shape[1]:
            p[ey, ex] = BLACK_IDX
    # Smile or panic
    if happy and cy + 3 < p.shape[0]:
        _draw_line(p, cx - 2, cy + 3, cx + 2, cy + 3, BLACK_IDX)
    elif cy + 3 < p.shape[0]:
        _fill_ellipse(p, cx, cy + 3, 1.5, 1.5, RED_IDX, BLACK_IDX)
    # Tiny bowler hat
    if hat:
        hy = cy - r
        _fill_rect(p, cx - 3, hy - 3, cx + 4, hy, DARK_GRAY, BLACK_IDX)
        _draw_line(p, cx - 4, hy, cx + 4, hy, BLACK_IDX)


def _draw_flames(p: np.ndarray, intensity: int) -> None:
    h, w = p.shape
    if intensity <= 0:
        return
    base_h = int(h * (0.35 if intensity == 1 else 0.72))
    for x in range(w):
        wave = int(2.5 * math.sin(x * 0.65 + intensity) + 1.5 * math.cos(x * 1.1))
        fh = max(2, min(h - 2, base_h + wave))
        y_top = h - fh
        p[y_top:h, x] = RED_IDX
        if y_top + 2 < h:
            p[y_top + 2 : h, x] = ORANGE_IDX
        if y_top + 4 < h:
            p[y_top + 4 : h, x] = YELLOW_IDX


def _draw_cartoon_cat(p: np.ndarray, cx: int, cy: int, col: int = ORANGE_IDX, dizzy: bool = False) -> None:
    # Pointy ears
    _draw_line(p, cx - 4, cy - 3, cx - 2, cy - 6, BLACK_IDX)
    _draw_line(p, cx - 2, cy - 6, cx - 1, cy - 3, BLACK_IDX)
    _draw_line(p, cx + 4, cy - 3, cx + 2, cy - 6, BLACK_IDX)
    _draw_line(p, cx + 2, cy - 6, cx + 1, cy - 3, BLACK_IDX)
    _fill_rect(p, cx - 3, cy - 5, cx - 1, cy - 3, col)
    _fill_rect(p, cx + 2, cy - 5, cx + 4, cy - 3, col)
    # Head
    _fill_ellipse(p, cx, cy, 5.0, 4.2, col, BLACK_IDX)
    # Stripes on forehead
    if 0 <= cy - 3 < p.shape[0] and 0 <= cx < p.shape[1]:
        p[cy - 3, cx] = BROWN
    # Eyes
    for ex in (cx - 2, cx + 2):
        if 0 <= cy - 1 < p.shape[0] and 0 <= ex < p.shape[1]:
            p[cy - 1, ex] = MAGENTA if dizzy else BLACK_IDX
            if dizzy and 0 <= ex + 1 < p.shape[1]:
                p[cy - 1, ex + 1] = WHITE_IDX
    # Pink nose & whiskers
    if 0 <= cy + 1 < p.shape[0] and 0 <= cx < p.shape[1]:
        p[cy + 1, cx] = PINK_IDX
    _draw_line(p, cx - 6, cy + 1, cx - 4, cy + 1, BLACK_IDX)
    _draw_line(p, cx + 4, cy + 1, cx + 6, cy + 1, BLACK_IDX)


def _draw_cartoon_frog(p: np.ndarray, cx: int, cy: int, smug: bool = False) -> None:
    # Wide frog head + eye bumps
    _fill_ellipse(p, cx - 3, cy - 4, 2.5, 2.5, GREEN_IDX, BLACK_IDX)
    _fill_ellipse(p, cx + 3, cy - 4, 2.5, 2.5, GREEN_IDX, BLACK_IDX)
    _fill_ellipse(p, cx, cy, 6.2, 4.2, GREEN_IDX, BLACK_IDX)
    _fill_ellipse(p, cx, cy + 1, 4.5, 2.0, LIGHT_GREEN, None)
    # Eyes
    for ex in (cx - 3, cx + 3):
        if 0 <= cy - 4 < p.shape[0] and 0 <= ex < p.shape[1]:
            p[cy - 4, ex] = WHITE_IDX
            if 0 <= ex + (1 if smug else 0) < p.shape[1]:
                p[cy - 4, ex + (1 if smug else 0)] = BLACK_IDX
        if smug and 0 <= cy - 5 < p.shape[0] and 0 <= ex - 1 < p.shape[1]:
            _draw_line(p, ex - 1, cy - 5, ex + 1, cy - 5, DARK_GREEN)
    # Red cartoon lips
    _draw_line(p, cx - 3, cy + 2, cx + 3, cy + 2, RED_IDX)
    if smug and cy + 1 < p.shape[0] and cx + 4 < p.shape[1]:
        p[cy + 1, cx + 4] = RED_IDX


def _draw_comic_panel_content(key: str, panel_idx: int, ph: int, pw: int) -> np.ndarray:
    """Render one crisp cartoon panel [ph, pw] for a comic strip."""
    p = np.full((ph, pw), WHITE_IDX, dtype=np.int16)
    cx, cy = pw // 2, ph // 2

    if key == "this_is_fine":
        p[:, :] = PALE_YELLOW if panel_idx == 0 else (ORANGE_IDX if panel_idx == 1 else RED_IDX)
        # Table
        _fill_rect(p, 0, int(ph * 0.72), pw, ph, BROWN, BLACK_IDX)
        if panel_idx >= 1:
            _draw_flames(p, panel_idx)
        _draw_cartoon_dog(p, cx - 2, int(ph * 0.48), scale=0.95, hat=True, happy=True)
        # Coffee mug on table
        _fill_rect(p, cx + 5, int(ph * 0.60), cx + 9, int(ph * 0.74), WHITE_IDX, BLACK_IDX)
        if panel_idx == 0:
            _draw_bubble(p, "OK", cx + 4, 6, fill=WHITE_IDX)
        elif panel_idx == 1:
            _draw_bubble(p, "HOT", cx + 4, 6, fill=PALE_YELLOW)
        else:
            _draw_bubble(p, "FINE", cx, 6, fill=WHITE_IDX)

    elif key == "cat_vs_laser":
        p[: int(ph * 0.68), :] = LIGHT_BLUE
        p[int(ph * 0.68) :, :] = BEIGE
        _draw_line(p, 0, int(ph * 0.68), pw - 1, int(ph * 0.68), BLACK_IDX)
        if panel_idx == 0:
            _draw_cartoon_cat(p, int(pw * 0.32), int(ph * 0.56))
            _fill_ellipse(p, int(pw * 0.78), int(ph * 0.80), 1.6, 1.6, RED_IDX, DARK_RED_IDX)
            _draw_bubble(p, "?", int(pw * 0.40), 6)
        elif panel_idx == 1:
            # Speedlines & flying cat
            _draw_line(p, 2, int(ph * 0.45), int(pw * 0.35), int(ph * 0.45), WHITE_IDX)
            _draw_line(p, 3, int(ph * 0.58), int(pw * 0.32), int(ph * 0.58), WHITE_IDX)
            _draw_cartoon_cat(p, int(pw * 0.58), int(ph * 0.36))
            _fill_ellipse(p, int(pw * 0.84), int(ph * 0.82), 1.6, 1.6, RED_IDX, DARK_RED_IDX)
        else:
            # Dizzy cat with red dot on nose
            _draw_cartoon_cat(p, cx, int(ph * 0.62), dizzy=True)
            if 0 <= int(ph * 0.62) + 1 < ph:
                p[int(ph * 0.62) + 1, cx] = RED_IDX
            _draw_bubble(p, "BONK", cx, 6, fill=PALE_YELLOW)

    elif key == "coffee_coder":
        bg = (DARK_PURPLE, TEAL, INDIGO)[panel_idx]
        p[:, :] = bg
        # Desk & monitor
        _fill_rect(p, 0, int(ph * 0.74), pw, ph, DARK_GRAY, BLACK_IDX)
        _fill_rect(p, int(pw * 0.64), int(ph * 0.48), pw - 2, int(ph * 0.74), LIGHT_BLUE if panel_idx == 2 else GRAY, BLACK_IDX)
        # Character head
        hx, hy = int(pw * 0.34), int(ph * 0.52)
        _fill_rect(p, hx - 4, hy + 4, hx + 5, int(ph * 0.74), BLUE_IDX if panel_idx < 2 else ORANGE_IDX, BLACK_IDX)
        _fill_ellipse(p, hx, hy, 4.5, 4.5, BEIGE, BLACK_IDX)
        if panel_idx == 0:
            # Sleepy purple eye bags
            _draw_line(p, hx - 2, hy, hx - 1, hy, PURPLE_IDX)
            _draw_line(p, hx + 1, hy, hx + 2, hy, PURPLE_IDX)
            _draw_bubble(p, "ZZZ", cx + 3, 6)
        elif panel_idx == 1:
            # Big coffee mug + steam
            _fill_rect(p, hx + 4, hy + 1, hx + 10, hy + 7, WHITE_IDX, BLACK_IDX)
            _draw_line(p, hx + 6, hy - 3, hx + 6, hy - 1, WHITE_IDX)
            _draw_line(p, hx + 8, hy - 4, hx + 8, hy - 1, WHITE_IDX)
            _draw_bubble(p, "SIP", cx + 3, 6, fill=PALE_YELLOW)
        else:
            # Glowing wide eyes + yellow lightning rays
            for angle in range(0, 360, 45):
                rx = int(hx + math.cos(math.radians(angle)) * 11)
                ry = int(hy + math.sin(math.radians(angle)) * 10)
                _draw_line(p, hx, hy, rx, ry, YELLOW_IDX)
            _fill_ellipse(p, hx, hy, 4.5, 4.5, BEIGE, BLACK_IDX)
            _fill_rect(p, hx - 3, hy - 2, hx - 1, hy + 1, WHITE_IDX)
            _fill_rect(p, hx + 1, hy - 2, hx + 3, hy + 1, WHITE_IDX)
            p[hy - 1, hx - 2] = LIGHT_TEAL
            p[hy - 1, hx + 2] = LIGHT_TEAL
            _draw_bubble(p, "GO!", cx + 2, 6, fill=YELLOW_IDX)

    elif key == "stonks_trip":
        p[:, :] = DARK_BLUE_IDX if panel_idx < 2 else BURGUNDY_IDX
        # Grid lines
        for gy in range(4, ph, 6):
            p[gy, :] = INDIGO if panel_idx < 2 else DARK_RED_IDX
        if panel_idx == 0:
            _draw_line(p, 2, ph - 4, pw - 3, 6, LIGHT_GREEN, thickness=2)
            _draw_bubble(p, "UP!", cx - 3, 7, fill=LIGHT_GREEN)
        elif panel_idx == 1:
            _draw_line(p, 2, ph - 5, cx, 7, LIGHT_GREEN, thickness=2)
            _draw_line(p, cx, 7, pw - 3, ph - 3, RED_IDX, thickness=2)
            _draw_bubble(p, "UH", cx, 6, fill=PALE_YELLOW)
        else:
            _draw_line(p, 2, 7, pw - 3, ph - 4, RED_IDX, thickness=2)
            # Suit guy face peeking in panic
            _fill_ellipse(p, int(pw * 0.30), int(ph * 0.65), 4.0, 4.5, LIGHT_GRAY, BLACK_IDX)
            _draw_bubble(p, "OOF", cx + 3, 7, fill=WHITE_IDX, text_col=RED_IDX)

    elif key == "ufo_cow":
        p[:, :] = DARK_BLUE_IDX
        p[int(ph * 0.74) :, :] = DARK_GREEN
        # Stars
        for sx, sy in ((4, 4), (pw - 5, 5), (cx + 5, 3), (7, 9)):
            if 0 <= sy < ph and 0 <= sx < pw:
                p[sy, sx] = PALE_YELLOW
        # Tractor beam in panel 1 & 2
        if panel_idx >= 1:
            for y in range(8, int(ph * 0.80)):
                half_w = int(2 + (y - 8) * 0.32)
                xa = max(1, cx - half_w)
                xb = min(pw - 1, cx + half_w)
                p[y, xa:xb] = LIGHT_GREEN if (y % 2 == 0) else LIGHT_TEAL
        # UFO saucer in panels 1 & 2
        if panel_idx >= 1:
            _fill_ellipse(p, cx, 6, 4.0, 2.5, LIGHT_BLUE, BLACK_IDX)
            _fill_ellipse(p, cx, 8, 8.0, 2.2, LIGHT_GRAY, BLACK_IDX)
            p[8, max(0, cx - 4)] = RED_IDX
            p[8, cx] = YELLOW_IDX
            p[8, min(pw - 1, cx + 4)] = RED_IDX
        # Spotted cow rising across panels
        cow_y = int(ph * 0.68) if panel_idx == 0 else (int(ph * 0.56) if panel_idx == 1 else int(ph * 0.38))
        _fill_rect(p, cx - 5, cow_y - 3, cx + 5, cow_y + 3, WHITE_IDX, BLACK_IDX)
        _fill_rect(p, cx - 3, cow_y - 2, cx, cow_y + 1, BLACK_IDX)
        _fill_rect(p, cx + 2, cow_y - 1, cx + 4, cow_y + 1, BLACK_IDX)
        _fill_rect(p, cx + 4, cow_y - 2, cx + 7, cow_y + 1, LIGHT_PINK, BLACK_IDX)
        if panel_idx == 2:
            _draw_bubble(p, "MOO?", cx, ph - 6, fill=WHITE_IDX)

    elif key == "frog_tea":
        p[:, :] = LIGHT_BLUE
        _fill_rect(p, 0, int(ph * 0.74), pw, ph, BEIGE, BLACK_IDX)
        _draw_cartoon_frog(p, cx - 1, int(ph * 0.50), smug=(panel_idx >= 1))
        # Teacup lifting up
        cup_y = int(ph * 0.64) if panel_idx == 0 else int(ph * 0.56)
        _fill_rect(p, cx + 4, cup_y, cx + 9, cup_y + 5, WHITE_IDX, BLACK_IDX)
        if panel_idx == 2:
            _draw_bubble(p, "TEA", cx, 6, fill=PALE_YELLOW)
        elif panel_idx == 1:
            _draw_bubble(p, "SIP", cx, 6, fill=WHITE_IDX)

    elif key == "void_knocks":
        p[:, :] = LIGHT_BLUE if panel_idx != 1 else DARK_PURPLE
        p[int(ph * 0.72) :, :] = GREEN_IDX
        # Cozy pixel cottage
        _fill_rect(p, cx - 6, int(ph * 0.48), cx + 6, int(ph * 0.74), PALE_YELLOW, BLACK_IDX)
        for ry in range(5):
            _draw_line(p, cx - 7 + ry, int(ph * 0.48) - ry, cx + 7 - ry, int(ph * 0.48) - ry, RED_IDX)
        _fill_rect(p, cx - 2, int(ph * 0.58), cx + 2, int(ph * 0.74), BROWN, BLACK_IDX)
        if panel_idx == 0:
            _fill_ellipse(p, pw - 6, 6, 3.0, 3.0, YELLOW_IDX, ORANGE_IDX)
        elif panel_idx == 1:
            # Creeping Void monster over roof
            _fill_ellipse(p, cx, 7, 8.0, 5.0, BLACK_IDX, DARK_PURPLE)
            p[7, max(0, cx - 3)] = RED_IDX
            p[7, min(pw - 1, cx + 3)] = RED_IDX
            _draw_bubble(p, "EEK!", cx, ph - 6, fill=WHITE_IDX, text_col=RED_IDX)
        else:
            # Rainbow beam blasting up
            for bi, bc in enumerate((RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX)):
                _draw_line(p, cx - 3 + bi, int(ph * 0.42), cx - 5 + bi * 2, 2, bc)
            _draw_bubble(p, "NOPE", cx, 6, fill=YELLOW_IDX)

    elif key == "bonk_shiba":
        p[:, :] = PALE_YELLOW if panel_idx < 2 else LIGHT_PINK
        p[int(ph * 0.76) :, :] = BEIGE
        if panel_idx == 0:
            _draw_cartoon_dog(p, int(pw * 0.36), int(ph * 0.58), scale=0.85, hat=False, happy=True)
            _fill_ellipse(p, int(pw * 0.76), int(ph * 0.72), 3.0, 2.0, RED_IDX, BLACK_IDX)
            _draw_bubble(p, "HEH", cx, 6)
        elif panel_idx == 1:
            _draw_cartoon_dog(p, int(pw * 0.36), int(ph * 0.62), scale=0.85, hat=False, happy=False)
            # Yellow bat swinging down + red burst
            _draw_line(p, int(pw * 0.85), 5, int(pw * 0.44), int(ph * 0.46), ORANGE_IDX, thickness=3)
            _draw_bubble(p, "BONK", cx + 2, 7, fill=YELLOW_IDX, text_col=RED_IDX)
        else:
            # Squished flat doge
            _fill_ellipse(p, cx, int(ph * 0.70), 8.0, 3.0, YELLOW_IDX, BLACK_IDX)
            p[int(ph * 0.69), cx - 3] = BLACK_IDX
            p[int(ph * 0.69), cx + 3] = BLACK_IDX
            # Dizzy stars
            for sx in (cx - 5, cx, cx + 5):
                p[int(ph * 0.44), max(1, min(pw - 2, sx))] = YELLOW_IDX
            _draw_bubble(p, "OW!", cx, 6, fill=WHITE_IDX)

    elif key == "penguin_ice":
        p[: int(ph * 0.66), :] = LIGHT_BLUE
        p[int(ph * 0.66) :, :] = WHITE_IDX
        _draw_line(p, 0, int(ph * 0.66), pw - 1, int(ph * 0.66), BLUE_IDX)
        if panel_idx == 0:
            # Standing penguin with cool shades
            _fill_ellipse(p, cx, int(ph * 0.52), 4.5, 6.0, BLACK_IDX, BLACK_IDX)
            _fill_ellipse(p, cx, int(ph * 0.56), 2.8, 4.0, WHITE_IDX, None)
            _fill_rect(p, cx - 3, int(ph * 0.40), cx + 4, int(ph * 0.44), DARK_BLUE_IDX, BLACK_IDX)
            p[int(ph * 0.46), cx] = ORANGE_IDX
            _draw_bubble(p, "COOL", cx, 6)
        elif panel_idx == 1:
            # Upside-down slipping penguin
            _fill_ellipse(p, cx, int(ph * 0.48), 6.0, 4.0, BLACK_IDX, BLACK_IDX)
            _fill_ellipse(p, cx, int(ph * 0.48), 3.8, 2.2, WHITE_IDX, None)
            _draw_line(p, cx - 4, int(ph * 0.32), cx - 2, int(ph * 0.40), ORANGE_IDX, thickness=2)
            _draw_line(p, cx + 4, int(ph * 0.32), cx + 2, int(ph * 0.40), ORANGE_IDX, thickness=2)
            _draw_bubble(p, "WHOA", cx, 6, fill=PALE_YELLOW)
        else:
            # Belly-sliding with fish
            _fill_ellipse(p, int(pw * 0.42), int(ph * 0.66), 6.5, 3.2, BLACK_IDX, BLACK_IDX)
            _fill_ellipse(p, int(pw * 0.42), int(ph * 0.69), 4.5, 1.5, WHITE_IDX, None)
            _fill_ellipse(p, int(pw * 0.78), int(ph * 0.66), 3.0, 1.8, ORANGE_IDX, BLACK_IDX)
            _draw_bubble(p, "NICE", cx, 6, fill=LIGHT_GREEN)

    elif key == "duck_knife":
        p[: int(ph * 0.66), :] = LIGHT_BLUE
        p[int(ph * 0.66) :, :] = LIGHT_GREEN
        # White cartoon duck
        dx = cx if panel_idx != 2 else int(pw * 0.56)
        dy = int(ph * 0.56)
        _fill_ellipse(p, dx - 2, dy + 2, 5.2, 3.2, WHITE_IDX, BLACK_IDX)
        _fill_ellipse(p, dx + 2, dy - 2, 3.2, 3.0, WHITE_IDX, BLACK_IDX)
        _fill_rect(p, dx + 4, dy - 2, dx + 8, dy, ORANGE_IDX, BLACK_IDX)
        p[dy - 3, min(pw - 1, dx + 3)] = BLACK_IDX
        if panel_idx == 1:
            # Shiny pixel knife on grass
            _draw_line(p, int(pw * 0.74), int(ph * 0.76), int(pw * 0.88), int(ph * 0.70), LIGHT_GRAY, thickness=2)
            _draw_bubble(p, "OOH", cx, 6)
        elif panel_idx == 2:
            # Angry brow + knife in beak
            _draw_line(p, dx + 1, dy - 5, dx + 4, dy - 4, BLACK_IDX)
            _draw_line(p, dx + 7, dy - 1, min(pw - 2, dx + 13), dy - 4, LIGHT_GRAY, thickness=2)
            _draw_bubble(p, "RUN!", cx, 6, fill=YELLOW_IDX, text_col=RED_IDX)
        else:
            _draw_bubble(p, "HONK", cx, 6)

    elif key == "drake_choice":
        # 2-panel stacked: panel 0 = NO (left orange jacket guy + right red X), panel 1 = YES (smiling + gold heart)
        p[:, : pw // 2] = ORANGE_IDX
        p[:, pw // 2 :] = WHITE_IDX
        _draw_line(p, pw // 2, 0, pw // 2, ph - 1, BLACK_IDX)
        # Cartoon character on left half
        lx, ly = pw // 4, int(ph * 0.56)
        _fill_rect(p, lx - 6, ly + 3, lx + 7, ph, DARK_RED_IDX if panel_idx == 0 else BLUE_IDX, BLACK_IDX)
        _fill_ellipse(p, lx, ly - 1, 5.0, 5.2, BROWN, BLACK_IDX)
        rx, ry = (3 * pw) // 4, ph // 2
        if panel_idx == 0:
            # Hand raised in rejection + red X on right
            _fill_rect(p, lx + 4, ly - 1, lx + 8, ly + 4, BROWN, BLACK_IDX)
            _draw_line(p, rx - 5, ry - 5, rx + 5, ry + 5, RED_IDX, thickness=2)
            _draw_line(p, rx + 5, ry - 5, rx - 5, ry + 5, RED_IDX, thickness=2)
            _draw_mini_text(p, "NO", rx - 4, 2, color=RED_IDX)
        else:
            # Big happy smile + glowing pink pixel heart on right
            _draw_line(p, lx - 2, ly + 1, lx + 2, ly + 1, WHITE_IDX)
            hy = min(ph - 5, ry + 3)
            _fill_ellipse(p, rx - 2, hy - 1, 2.8, 2.5, PINK_IDX, BLACK_IDX)
            _fill_ellipse(p, rx + 2, hy - 1, 2.8, 2.5, PINK_IDX, BLACK_IDX)
            _fill_ellipse(p, rx, hy + 2, 3.6, 2.5, PINK_IDX, None)
            _draw_mini_text(p, "YES!", rx - 7, 1, color=DARK_GREEN)

    elif key == "galaxy_brain":
        # 4-panel escalating cosmic enlightenment!
        bgs = (DARK_GRAY, DARK_BLUE_IDX, DARK_PURPLE, INDIGO)
        glows = (LIGHT_PINK, LIGHT_BLUE, MAGENTA, YELLOW_IDX)
        p[:, :] = bgs[panel_idx % 4]
        hx, hy = cx, int(ph * 0.54)
        if panel_idx >= 2:
            # Cosmic rays radiating from brain
            ray_col = LIGHT_TEAL if panel_idx == 2 else YELLOW_IDX
            for a in range(0, 360, 30):
                rx = int(hx + math.cos(math.radians(a)) * (pw * 0.48))
                ry = int(hy + math.sin(math.radians(a)) * (ph * 0.48))
                _draw_line(p, hx, hy, rx, ry, ray_col)
        # Head silhouette
        _fill_ellipse(p, hx, hy, 5.5, 6.2, BLACK_IDX, WHITE_IDX if panel_idx >= 2 else GRAY)
        # Glowing brain size grows each panel!
        br = 2.0 + panel_idx * 1.1
        _fill_ellipse(p, hx, hy - 1, br, br * 0.85, glows[panel_idx % 4], WHITE_IDX if panel_idx == 3 else None)
        if panel_idx == 3:
            # Laser eyes
            _draw_line(p, hx + 3, hy + 1, pw - 1, hy - 2, YELLOW_IDX, thickness=2)
            _draw_line(p, hx + 3, hy + 1, pw - 1, hy + 4, MAGENTA, thickness=1)
        labels = ("10 IQ", "100", "BIG", "GOD")
        _draw_mini_text(p, labels[panel_idx % 4], 2, 2, color=WHITE_IDX, outline=BLACK_IDX)

    return p


def generate_comic_strip(
    comic_key: str | None,
    target_w: int,
    target_h: int,
    seed: int = 42,
) -> tuple[np.ndarray, ComicSpec]:
    """Generate a crisp multi-panel comic strip [target_h, target_w] with black borders & white gutters."""
    spec = COMIC_BY_KEY.get(comic_key or "", COMIC_CATALOG[seed % len(COMIC_CATALOG)])
    bp = np.full((target_h, target_w), WHITE_IDX, dtype=np.int16)
    # Crisp outer black comic border
    bp[0, :] = BLACK_IDX
    bp[-1, :] = BLACK_IDX
    bp[:, 0] = BLACK_IDX
    bp[:, -1] = BLACK_IDX

    gutter = 2 if min(target_w, target_h) >= 30 else 1
    pad = 2

    if spec.layout == "3h":
        n_pan = 3
        inner_w = target_w - 2 * pad - gutter * (n_pan - 1)
        pw = max(14, inner_w // n_pan)
        ph = max(16, target_h - 2 * pad)
        x = pad
        for idx in range(n_pan):
            w_this = (target_w - pad - x) if idx == n_pan - 1 else pw
            if w_this < 8:
                break
            panel = _draw_comic_panel_content(spec.key, idx, max(12, ph - 2), max(12, w_this - 2))
            # Draw black panel box + content
            bp[pad : pad + ph, x : x + w_this] = BLACK_IDX
            bp[pad + 1 : pad + ph - 1, x + 1 : x + w_this - 1] = _resize_nn(panel, ph - 2, w_this - 2)
            x += w_this + gutter

    elif spec.layout == "2v":
        n_pan = 2
        inner_h = target_h - 2 * pad - gutter
        ph = max(12, inner_h // n_pan)
        pw = max(18, target_w - 2 * pad)
        y = pad
        for idx in range(n_pan):
            h_this = (target_h - pad - y) if idx == n_pan - 1 else ph
            if h_this < 8:
                break
            panel = _draw_comic_panel_content(spec.key, idx, max(10, h_this - 2), max(16, pw - 2))
            bp[y : y + h_this, pad : pad + pw] = BLACK_IDX
            bp[y + 1 : y + h_this - 1, pad + 1 : pad + pw - 1] = _resize_nn(panel, h_this - 2, pw - 2)
            y += h_this + gutter

    else:  # "2x2"
        pw = max(12, (target_w - 2 * pad - gutter) // 2)
        ph = max(12, (target_h - 2 * pad - gutter) // 2)
        coords = [
            (pad, pad),
            (pad + pw + gutter, pad),
            (pad, pad + ph + gutter),
            (pad + pw + gutter, pad + ph + gutter),
        ]
        for idx, (px0, py0) in enumerate(coords):
            w_this = (target_w - pad - px0) if idx % 2 == 1 else pw
            h_this = (target_h - pad - py0) if idx >= 2 else ph
            if w_this < 8 or h_this < 8:
                continue
            panel = _draw_comic_panel_content(spec.key, idx, max(10, h_this - 2), max(10, w_this - 2))
            bp[py0 : py0 + h_this, px0 : px0 + w_this] = BLACK_IDX
            bp[py0 + 1 : py0 + h_this - 1, px0 + 1 : px0 + w_this - 1] = _resize_nn(panel, h_this - 2, w_this - 2)

    return bp, spec


# ---------------------------------------------------------------------------
# 2. Medium & Macro Pixel Cartoons and Classic Memes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CartoonMemeSpec:
    key: str
    title: str
    category: str  # "meme" | "cartoon"
    slogan: str
    primary: int
    secondary: int
    accent: int


CARTOON_MEME_CATALOG: list[CartoonMemeSpec] = [
    CartoonMemeSpec("doge_portrait", "Much Doge", "meme", "Very pixel, much canvas, wow.", YELLOW_IDX, BEIGE, ORANGE_IDX),
    CartoonMemeSpec("pepe_smug", "Smug Frog", "meme", "Feels good man.", GREEN_IDX, LIGHT_GREEN, RED_IDX),
    CartoonMemeSpec("nyan_cat", "Nyan Cat", "meme", "Infinite pop-tart rainbow trail.", PINK_IDX, LIGHT_BLUE, YELLOW_IDX),
    CartoonMemeSpec("stonks_man", "Stonks Man", "meme", "Pixel portfolio to the moon.", BLUE_IDX, LIGHT_GREEN, ORANGE_IDX),
    CartoonMemeSpec("deal_with_it", "Deal With It", "meme", "Drop the pixel shades.", YELLOW_IDX, BLACK_IDX, LIGHT_BLUE),
    CartoonMemeSpec("surprised_mouse", "Shocked Mouse", "meme", "Wait, they painted over us?!", YELLOW_IDX, RED_IDX, ORANGE_IDX),
    CartoonMemeSpec("troll_grin", "Mischief Grin", "meme", "A tiny bit of pixel trolling.", WHITE_IDX, BLACK_IDX, RED_IDX),
    CartoonMemeSpec("capybara_chill", "Chill Capybara", "cartoon", "Unbothered in the pixel hot spring.", BROWN, ORANGE_IDX, LIGHT_BLUE),
    CartoonMemeSpec("wizard_frog", "Wizard Frog", "cartoon", "Casting color spells across the grid.", GREEN_IDX, INDIGO, YELLOW_IDX),
    CartoonMemeSpec("rubber_duck", "Giant Duck", "cartoon", "Quack squad holds the pond.", YELLOW_IDX, ORANGE_IDX, LIGHT_BLUE),
    CartoonMemeSpec("cartoon_robot", "Retro Bot", "cartoon", "Beep boop, painting with precision.", LIGHT_TEAL, ORANGE_IDX, YELLOW_IDX),
    CartoonMemeSpec("taco_cat", "Taco Cat", "cartoon", "Spelled backwards it is still Taco Cat.", YELLOW_IDX, LIGHT_GREEN, ORANGE_IDX),
    CartoonMemeSpec("gigachad_bust", "Pixel Chad", "meme", "Average pixel-art enjoyer.", LIGHT_GRAY, DARK_GRAY, WHITE_IDX),
    CartoonMemeSpec("scream_possum", "Chaos Possum", "meme", "Scream first, ask questions later.", LIGHT_GRAY, PINK_IDX, BLACK_IDX),
]

CARTOON_MEME_BY_KEY: dict[str, CartoonMemeSpec] = {c.key: c for c in CARTOON_MEME_CATALOG}
CARTOON_MEME_KEYS: list[str] = [c.key for c in CARTOON_MEME_CATALOG]


def generate_cartoon_meme(
    meme_key: str | None,
    target_w: int,
    target_h: int,
    seed: int = 42,
) -> tuple[np.ndarray, CartoonMemeSpec]:
    """Generate a crisp pixel-art cartoon character or classic meme portrait [target_h, target_w]."""
    spec = CARTOON_MEME_BY_KEY.get(meme_key or "", CARTOON_MEME_CATALOG[seed % len(CARTOON_MEME_CATALOG)])
    # Draw on a crisp canonical 44x44 (or 56x36 for wide Nyan Cat) canvas, then scale cleanly
    if spec.key == "nyan_cat":
        ch, cw = 36, 56
    else:
        ch, cw = 44, 44

    bp = np.full((ch, cw), -1, dtype=np.int16)
    cx, cy = cw // 2, ch // 2

    if spec.key == "doge_portrait":
        # Pointed Shiba ears
        for dy in range(11):
            w_ear = max(1, (11 - dy) // 2)
            _fill_rect(bp, 9 + dy // 3, 4 + dy, 9 + dy // 3 + w_ear + 3, 5 + dy, ORANGE_IDX)
            _fill_rect(bp, cw - 13 - dy // 3 - w_ear, 4 + dy, cw - 10 - dy // 3, 5 + dy, ORANGE_IDX)
        # Warm golden Shiba face + cream muzzle/cheeks
        _fill_ellipse(bp, cx, cy + 3, 15.0, 14.5, YELLOW_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 6, 13.0, 10.5, PALE_YELLOW, None)
        _fill_ellipse(bp, cx, cy - 3, 12.5, 8.5, ORANGE_IDX, None)
        _fill_ellipse(bp, cx, cy + 1, 10.5, 7.5, YELLOW_IDX, None)
        _fill_ellipse(bp, cx, cy + 8, 7.0, 5.0, WHITE_IDX, None)
        # Sideways skeptical Doge eyes
        for ex in (cx - 6, cx + 5):
            _fill_ellipse(bp, ex, cy - 1, 2.5, 2.0, WHITE_IDX, BLACK_IDX)
            bp[cy - 1, ex - 1] = BLACK_IDX
            bp[cy, ex - 1] = BLACK_IDX
        # Dark snout & subtle smirk
        _fill_ellipse(bp, cx, cy + 6, 2.6, 1.8, BLACK_IDX, None)
        _draw_line(bp, cx - 3, cy + 10, cx + 3, cy + 10, BLACK_IDX)
        bp[cy + 9, cx + 4] = BLACK_IDX
        # Classic comic caption
        _draw_mini_text(bp, "WOW", 3, ch - 7, color=MAGENTA, outline=WHITE_IDX)
        _draw_mini_text(bp, "SUCH", cw - 18, 2, color=LIGHT_BLUE, outline=BLACK_IDX)

    elif spec.key == "pepe_smug":
        # Rounded eye domes + wide frog face cutout
        _fill_ellipse(bp, cx - 7, cy - 8, 6.2, 5.5, GREEN_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx + 6, cy - 8, 6.2, 5.5, GREEN_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 3, 16.5, 12.5, GREEN_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 5, 14.0, 9.0, LIGHT_GREEN, None)
        # Blue shirt collar at bottom
        _fill_rect(bp, cx - 10, ch - 8, cx + 11, ch - 1, BLUE_IDX, BLACK_IDX)
        # Heavy-lidded white eyes looking sideways
        for ex in (cx - 7, cx + 6):
            _fill_ellipse(bp, ex, cy - 7, 4.2, 2.8, WHITE_IDX, BLACK_IDX)
            _fill_rect(bp, ex - 4, cy - 11, ex + 5, cy - 8, GREEN_IDX)
            _draw_line(bp, ex - 4, cy - 8, ex + 4, cy - 8, BLACK_IDX)
            _fill_rect(bp, ex + 1, cy - 7, ex + 3, cy - 5, BLACK_IDX)
        # Signature thick red cartoon lips with smug curl
        _fill_rect(bp, cx - 9, cy + 4, cx + 10, cy + 8, RED_IDX, BLACK_IDX)
        _draw_line(bp, cx - 8, cy + 6, cx + 9, cy + 6, DARK_RED_IDX)
        bp[cy + 3, cx + 10] = RED_IDX
        if seed % 3 == 0:
            # Optional cool sunglasses variant
            _fill_rect(bp, cx - 12, cy - 9, cx - 2, cy - 5, BLACK_IDX)
            _fill_rect(bp, cx + 1, cy - 9, cx + 11, cy - 5, BLACK_IDX)
            _draw_line(bp, cx - 2, cy - 7, cx + 1, cy - 7, BLACK_IDX)
            bp[cy - 8, cx - 10] = WHITE_IDX
            bp[cy - 8, cx + 3] = WHITE_IDX

    elif spec.key == "nyan_cat":
        # Starry dark-blue badge sky
        bp[:, :] = DARK_BLUE_IDX
        bp[0, :] = BLACK_IDX
        bp[-1, :] = BLACK_IDX
        bp[:, 0] = BLACK_IDX
        bp[:, -1] = BLACK_IDX
        for sx, sy in ((6, 5), (15, 4), (44, 5), (50, 12), (8, 30), (25, 31), (48, 29)):
            bp[sy, sx] = WHITE_IDX
            if sx + 1 < cw:
                bp[sy, sx + 1] = PALE_YELLOW
        # 6-color wavy rainbow trail on the left
        rb_cols = (RED_IDX, ORANGE_IDX, YELLOW_IDX, LIGHT_GREEN, LIGHT_BLUE, PURPLE_IDX)
        for x in range(2, 28):
            shift = 1 if ((x // 4) % 2 == 0) else 0
            for ci, rc in enumerate(rb_cols):
                y0 = 10 + ci * 3 + shift
                bp[y0 : y0 + 3, x] = rc
        # Pop-tart body (tan crust + pink frosting + magenta sprinkles)
        _fill_rect(bp, 22, 9, 42, 26, BEIGE, BLACK_IDX)
        _fill_rect(bp, 24, 11, 40, 24, LIGHT_PINK, PINK_IDX)
        for px, py in ((27, 13), (33, 14), (29, 19), (36, 18), (31, 22), (37, 22)):
            bp[py, px] = MAGENTA
        # Cute gray cat head + ears + rosy cheeks
        _fill_rect(bp, 37, 11, 41, 15, GRAY, BLACK_IDX)
        _fill_rect(bp, 45, 11, 49, 15, GRAY, BLACK_IDX)
        _fill_ellipse(bp, 43, 19, 6.5, 5.5, GRAY, BLACK_IDX)
        bp[18, 40] = BLACK_IDX
        bp[18, 46] = BLACK_IDX
        bp[20, 39] = PINK_IDX
        bp[20, 47] = PINK_IDX
        _draw_line(bp, 41, 21, 45, 21, BLACK_IDX)
        # Little gray paws
        _fill_rect(bp, 23, 26, 27, 29, GRAY, BLACK_IDX)
        _fill_rect(bp, 35, 26, 39, 29, GRAY, BLACK_IDX)

    elif spec.key == "stonks_man":
        bp[:, :] = DARK_BLUE_IDX
        bp[0, :] = BLACK_IDX
        bp[-1, :] = BLACK_IDX
        bp[:, 0] = BLACK_IDX
        bp[:, -1] = BLACK_IDX
        # Background stock grid + soaring bright green arrow
        for gy in range(6, ch, 8):
            bp[gy, 1:-1] = INDIGO
        _draw_line(bp, 4, ch - 8, 16, ch - 14, GREEN_IDX, thickness=2)
        _draw_line(bp, 16, ch - 14, 24, ch - 10, GREEN_IDX, thickness=2)
        _draw_line(bp, 24, ch - 10, cw - 6, 8, LIGHT_GREEN, thickness=3)
        _draw_line(bp, cw - 11, 8, cw - 5, 8, LIGHT_GREEN, thickness=2)
        _draw_line(bp, cw - 6, 8, cw - 6, 14, LIGHT_GREEN, thickness=2)
        # Suit + orange tie + smooth pale mannequin head
        _fill_rect(bp, 6, 27, 26, ch - 1, BLACK_IDX, DARK_GRAY)
        _fill_rect(bp, 13, 28, 19, ch - 1, WHITE_IDX)
        _fill_rect(bp, 15, 29, 17, ch - 1, ORANGE_IDX)
        _fill_ellipse(bp, 16, 17, 7.2, 9.2, LIGHT_GRAY, BLACK_IDX)
        _fill_ellipse(bp, 17, 16, 5.5, 7.5, WHITE_IDX, None)
        bp[15, 13] = DARK_BLUE_IDX
        bp[15, 19] = DARK_BLUE_IDX
        _draw_line(bp, 14, 21, 18, 21, DARK_GRAY)
        _draw_mini_text(bp, "STONKS", 13, ch - 7, color=YELLOW_IDX, outline=BLACK_IDX)

    elif spec.key == "deal_with_it":
        # Smiling bright cartoon yellow face with iconic black-and-white pixelated shades!
        _fill_ellipse(bp, cx, cy, 16.5, 16.5, YELLOW_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx - 4, cy - 5, 7.0, 5.0, PALE_YELLOW, None)
        # Iconic thick pixelated sunglasses
        _fill_rect(bp, cx - 15, cy - 6, cx + 16, cy - 4, BLACK_IDX)
        _fill_rect(bp, cx - 13, cy - 5, cx - 2, cy + 2, BLACK_IDX)
        _fill_rect(bp, cx + 2, cy - 5, cx + 13, cy + 2, BLACK_IDX)
        # White diagonal pixel reflections
        for rx in (cx - 11, cx + 4):
            bp[cy - 4, rx] = WHITE_IDX
            bp[cy - 3, rx + 1] = WHITE_IDX
            bp[cy - 2, rx + 2] = WHITE_IDX
            bp[cy - 4, rx + 3] = WHITE_IDX
            bp[cy - 3, rx + 4] = WHITE_IDX
        # Confident cartoon smirk
        _draw_line(bp, cx - 6, cy + 8, cx + 5, cy + 8, BLACK_IDX, thickness=2)
        _draw_line(bp, cx + 5, cy + 8, cx + 8, cy + 5, BLACK_IDX, thickness=2)
        _draw_mini_text(bp, "DEAL", 13, ch - 7, color=WHITE_IDX, outline=BLACK_IDX)

    elif spec.key == "surprised_mouse":
        # Long yellow ears with crisp black tips
        _draw_line(bp, cx - 8, cy - 7, cx - 16, 4, YELLOW_IDX, thickness=4)
        _draw_line(bp, cx - 13, 7, cx - 16, 4, BLACK_IDX, thickness=4)
        _draw_line(bp, cx + 8, cy - 7, cx + 16, 4, YELLOW_IDX, thickness=4)
        _draw_line(bp, cx + 13, 7, cx + 16, 4, BLACK_IDX, thickness=4)
        # Plump yellow cartoon head & shoulders
        _fill_ellipse(bp, cx, cy + 4, 14.5, 13.5, YELLOW_IDX, BLACK_IDX)
        # Round black eyes with tiny white dots
        _fill_ellipse(bp, cx - 6, cy + 1, 2.5, 2.5, BLACK_IDX, None)
        _fill_ellipse(bp, cx + 6, cy + 1, 2.5, 2.5, BLACK_IDX, None)
        bp[cy, cx - 7] = WHITE_IDX
        bp[cy, cx + 5] = WHITE_IDX
        # Bright red round cheeks
        _fill_ellipse(bp, cx - 10, cy + 6, 3.0, 3.0, RED_IDX, None)
        _fill_ellipse(bp, cx + 10, cy + 6, 3.0, 3.0, RED_IDX, None)
        # Iconic shocked open "O" mouth
        _fill_ellipse(bp, cx, cy + 8, 3.2, 4.2, DARK_RED_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 9, 2.2, 2.2, LIGHT_PINK, None)

    elif spec.key == "troll_grin":
        # Crisp white comic face silhouette with huge mischief grin
        _fill_ellipse(bp, cx, cy, 16.5, 13.5, WHITE_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx + 3, cy + 6, 13.0, 9.0, WHITE_IDX, BLACK_IDX)
        # Squinted mischievous brows + eyes
        _draw_line(bp, cx - 11, cy - 5, cx - 3, cy - 3, BLACK_IDX, thickness=2)
        _draw_line(bp, cx + 11, cy - 5, cx + 3, cy - 3, BLACK_IDX, thickness=2)
        _fill_rect(bp, cx - 10, cy - 2, cx - 4, cy + 1, BLACK_IDX)
        _fill_rect(bp, cx + 4, cy - 2, cx + 10, cy + 1, BLACK_IDX)
        bp[cy - 1, cx - 7] = WHITE_IDX
        bp[cy - 1, cx + 7] = WHITE_IDX
        # Huge wide toothy grin
        _fill_ellipse(bp, cx, cy + 6, 12.0, 4.5, WHITE_IDX, BLACK_IDX)
        _draw_line(bp, cx - 11, cy + 6, cx + 11, cy + 6, BLACK_IDX)
        for tx in range(cx - 9, cx + 10, 3):
            _draw_line(bp, tx, cy + 3, tx, cy + 9, BLACK_IDX)

    elif spec.key == "capybara_chill":
        # Steaming light-blue hot spring tub + super chill brown capybara with yuzu orange on head
        _fill_ellipse(bp, cx, cy + 10, 18.0, 8.5, LIGHT_BLUE, BLUE_IDX)
        # Capybara rounded snout & head
        _fill_ellipse(bp, cx, cy + 4, 13.5, 10.5, BROWN, BLACK_IDX)
        _fill_ellipse(bp, cx + 5, cy + 5, 9.0, 7.5, DARK_BROWN, None)
        _fill_ellipse(bp, cx - 2, cy + 2, 10.5, 8.5, BROWN, None)
        # Tiny ears
        _fill_ellipse(bp, cx - 8, cy - 5, 2.5, 2.2, DARK_BROWN, BLACK_IDX)
        # Blissed-out closed eyes
        _draw_line(bp, cx - 5, cy, cx - 1, cy, BLACK_IDX, thickness=2)
        _draw_line(bp, cx + 4, cy, cx + 8, cy, BLACK_IDX, thickness=2)
        # Bright orange yuzu fruit balanced on top of head with green leaf!
        _fill_ellipse(bp, cx, cy - 9, 4.2, 3.6, ORANGE_IDX, BLACK_IDX)
        bp[cy - 10, cx - 1] = YELLOW_IDX
        _draw_line(bp, cx, cy - 13, cx + 3, cy - 15, GREEN_IDX, thickness=2)
        # Steam curls
        _draw_line(bp, 6, cy - 6, 6, cy - 12, WHITE_IDX)
        _draw_line(bp, cw - 7, cy - 5, cw - 7, cy - 11, WHITE_IDX)

    elif spec.key == "wizard_frog":
        # Starry pointed indigo wizard hat + plump green frog + glowing golden staff
        for ry in range(16):
            hw = max(1, int(ry * 0.72))
            _draw_line(bp, cx - hw, 2 + ry, cx + hw, 2 + ry, INDIGO)
            bp[2 + ry, max(0, cx - hw)] = BLACK_IDX
            bp[2 + ry, min(cw - 1, cx + hw)] = BLACK_IDX
        # Hat brim + gold stars
        _fill_ellipse(bp, cx, 18, 15.0, 3.2, DARK_PURPLE, BLACK_IDX)
        bp[7, cx] = YELLOW_IDX
        bp[12, cx - 3] = YELLOW_IDX
        bp[13, cx + 4] = PALE_YELLOW
        # Frog face + glowing eyes
        _fill_ellipse(bp, cx, cy + 8, 12.5, 8.5, GREEN_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 10, 9.5, 5.5, LIGHT_GREEN, None)
        _fill_ellipse(bp, cx - 5, cy + 5, 2.5, 2.5, YELLOW_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx + 5, cy + 5, 2.5, 2.5, YELLOW_IDX, BLACK_IDX)
        bp[cy + 5, cx - 5] = BLACK_IDX
        bp[cy + 5, cx + 5] = BLACK_IDX
        _draw_line(bp, cx - 4, cy + 11, cx + 4, cy + 11, RED_IDX)
        # Magic staff with glowing orb
        _draw_line(bp, cw - 8, 12, cw - 8, ch - 3, BROWN, thickness=2)
        _fill_ellipse(bp, cw - 8, 10, 3.5, 3.5, YELLOW_IDX, ORANGE_IDX)
        bp[9, cw - 9] = WHITE_IDX

    elif spec.key == "rubber_duck":
        # Glossy cartoon yellow bath duck floating on cyan ripples
        _fill_ellipse(bp, cx, ch - 6, 17.0, 4.5, LIGHT_BLUE, BLUE_IDX)
        _fill_ellipse(bp, cx - 2, cy + 8, 14.0, 9.0, YELLOW_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx + 4, cy - 4, 9.5, 8.5, YELLOW_IDX, BLACK_IDX)
        # Gloss highlight + cute wing
        _fill_ellipse(bp, cx + 2, cy - 8, 3.5, 2.2, PALE_YELLOW, None)
        _fill_ellipse(bp, cx - 4, cy + 8, 7.0, 4.0, PALE_YELLOW, ORANGE_IDX)
        # Bright orange bill
        _fill_ellipse(bp, cx + 14, cy - 2, 5.5, 3.0, ORANGE_IDX, BLACK_IDX)
        _draw_line(bp, cx + 11, cy - 2, cx + 18, cy - 2, DARK_RED_IDX)
        # Big shiny cartoon eye
        _fill_ellipse(bp, cx + 7, cy - 6, 2.4, 2.4, BLACK_IDX, None)
        bp[cy - 7, cx + 6] = WHITE_IDX

    elif spec.key == "cartoon_robot":
        # Retro boxy Saturday-morning cartoon robot
        _draw_line(bp, cx, 3, cx, 10, DARK_GRAY, thickness=2)
        _fill_ellipse(bp, cx, 4, 2.5, 2.5, RED_IDX, BLACK_IDX)
        # Head
        _fill_rect(bp, cx - 12, 9, cx + 13, 27, LIGHT_TEAL, BLACK_IDX)
        _fill_rect(bp, cx - 10, 11, cx + 11, 13, WHITE_IDX)
        # Dark visor with bright glowing yellow eyes
        _fill_rect(bp, cx - 9, 14, cx + 10, 21, DARK_BLUE_IDX, BLACK_IDX)
        _fill_rect(bp, cx - 7, 16, cx - 2, 19, YELLOW_IDX)
        _fill_rect(bp, cx + 3, 16, cx + 8, 19, YELLOW_IDX)
        # Speaker grin
        _fill_rect(bp, cx - 6, 23, cx + 7, 25, ORANGE_IDX, BLACK_IDX)
        # Torso with glowing heart core
        _fill_rect(bp, cx - 10, 28, cx + 11, ch - 2, TEAL, BLACK_IDX)
        _fill_ellipse(bp, cx, 34, 3.5, 3.2, PINK_IDX, BLACK_IDX)
        # Ear bolts
        _fill_rect(bp, cx - 15, 15, cx - 12, 21, ORANGE_IDX, BLACK_IDX)
        _fill_rect(bp, cx + 13, 15, cx + 16, 21, ORANGE_IDX, BLACK_IDX)

    elif spec.key == "taco_cat":
        # Cute white/orange cat peeking out of a loaded crunchy yellow pixel taco!
        # Cat head peeking from top
        _draw_line(bp, cx - 7, 14, cx - 4, 6, BLACK_IDX, thickness=2)
        _draw_line(bp, cx + 7, 14, cx + 4, 6, BLACK_IDX, thickness=2)
        _fill_ellipse(bp, cx, 15, 8.5, 7.0, WHITE_IDX, BLACK_IDX)
        bp[14, cx - 3] = BLACK_IDX
        bp[14, cx + 3] = BLACK_IDX
        bp[16, cx] = PINK_IDX
        # Lettuce, tomatoes, cheese peeking out the taco top
        _fill_ellipse(bp, cx - 9, 21, 6.0, 3.5, LIGHT_GREEN, DARK_GREEN)
        _fill_ellipse(bp, cx + 9, 21, 6.0, 3.5, LIGHT_GREEN, DARK_GREEN)
        _fill_ellipse(bp, cx - 5, 22, 2.5, 2.5, RED_IDX, None)
        _fill_ellipse(bp, cx + 6, 22, 2.5, 2.5, RED_IDX, None)
        # Big crunchy half-moon golden taco shell
        yy, xx = np.ogrid[:ch, :cw]
        shell = (((xx - cx) / 17.5) ** 2 + ((yy - 34) / 14.5) ** 2 <= 1.0) & (yy <= 34)
        shell_out = (((xx - cx) / 18.8) ** 2 + ((yy - 34) / 15.8) ** 2 <= 1.0) & (yy <= 35)
        bp[shell_out] = BLACK_IDX
        bp[shell] = YELLOW_IDX
        # Toasty taco spots
        for sx, sy in ((cx - 10, 29), (cx - 3, 27), (cx + 5, 30), (cx + 11, 28), (cx, 32)):
            bp[sy, sx] = ORANGE_IDX

    elif spec.key == "gigachad_bust":
        # Classic monochrome side-profile silhouette: swept hair, strong nose, chiseled jaw & beard
        # Broad shoulders & neck
        _fill_rect(bp, 8, 34, 34, ch - 1, DARK_GRAY, BLACK_IDX)
        _fill_rect(bp, 14, 29, 25, 36, GRAY, BLACK_IDX)
        # Side-profile face facing right
        _fill_ellipse(bp, 21, 19, 9.5, 11.5, LIGHT_GRAY, BLACK_IDX)
        _fill_ellipse(bp, 23, 18, 7.5, 9.5, WHITE_IDX, None)
        # Swept wavy dark pompadour hair on top and back
        _fill_ellipse(bp, 18, 9, 10.5, 5.5, DARK_GRAY, BLACK_IDX)
        _fill_ellipse(bp, 12, 16, 4.5, 7.5, DARK_GRAY, BLACK_IDX)
        _draw_line(bp, 13, 8, 25, 7, GRAY, thickness=2)
        # Strong profile nose & chiseled jaw jutting right
        _fill_rect(bp, 29, 16, 33, 21, WHITE_IDX, BLACK_IDX)
        _fill_rect(bp, 27, 21, 32, 29, LIGHT_GRAY, BLACK_IDX)
        # Chiseled dark beard & mustache along lower jaw
        _fill_rect(bp, 16, 24, 31, 30, DARK_GRAY)
        _draw_line(bp, 24, 24, 31, 24, BLACK_IDX)
        _draw_line(bp, 24, 26, 29, 26, WHITE_IDX)  # confident grin
        # Deep brow & focused eye
        _draw_line(bp, 22, 15, 29, 15, BLACK_IDX, thickness=2)
        bp[17, 26] = BLACK_IDX
        bp[17, 25] = DARK_GRAY
        # Chin gleam sparkle
        _draw_line(bp, 33, 28, 39, 28, YELLOW_IDX)
        _draw_line(bp, 36, 25, 36, 31, YELLOW_IDX)

    else:  # "scream_possum"
        # Chaotic wide-eyed screaming cartoon opossum
        _fill_ellipse(bp, cx - 10, 10, 4.5, 5.0, DARK_GRAY, BLACK_IDX)
        _fill_ellipse(bp, cx + 10, 10, 4.5, 5.0, DARK_GRAY, BLACK_IDX)
        _fill_ellipse(bp, cx - 10, 10, 2.5, 2.8, LIGHT_PINK, None)
        _fill_ellipse(bp, cx + 10, 10, 2.5, 2.8, LIGHT_PINK, None)
        _fill_ellipse(bp, cx, cy + 2, 14.5, 13.5, WHITE_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy - 4, 12.0, 6.0, LIGHT_GRAY, None)
        # Huge wide black eyes
        _fill_ellipse(bp, cx - 6, cy - 2, 3.0, 3.0, BLACK_IDX, None)
        _fill_ellipse(bp, cx + 6, cy - 2, 3.0, 3.0, BLACK_IDX, None)
        bp[cy - 3, cx - 7] = WHITE_IDX
        bp[cy - 3, cx + 5] = WHITE_IDX
        # Pink snout + giant screaming mouth with tiny teeth
        _fill_ellipse(bp, cx, cy + 3, 2.5, 1.8, PINK_IDX, None)
        _fill_ellipse(bp, cx, cy + 10, 7.5, 5.2, RED_IDX, BLACK_IDX)
        _fill_ellipse(bp, cx, cy + 12, 5.2, 2.5, LIGHT_PINK, None)
        for tx in (cx - 4, cx - 1, cx + 2, cx + 5):
            bp[cy + 6, tx] = WHITE_IDX
        _draw_mini_text(bp, "AAAA!", 12, 2, color=YELLOW_IDX, outline=BLACK_IDX)

    bp = _add_cutout_outline(bp, BLACK_IDX)
    return _resize_nn(bp, target_h, target_w), spec
