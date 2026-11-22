"""Crisp procedural library of 32 recognizable national/community flags & cross-map Transverse Ribbons."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
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
LIGHT_PINK = NAME_TO_IDX["light_pink"]
DARK_GREEN = NAME_TO_IDX["dark_green"]
LIGHT_GREEN = NAME_TO_IDX["light_green"]
BROWN = NAME_TO_IDX["brown"]
DARK_BROWN = NAME_TO_IDX["dark_brown"]
INDIGO = NAME_TO_IDX["indigo"]
TEAL = NAME_TO_IDX["teal"]


@dataclass(frozen=True)
class FlagMeta:
    key: str
    display_name: str
    subreddit: str
    slogan: str
    primary: int
    secondary: int
    accent: int


# 32 common national and community flags
NATIONAL_FLAGS: list[FlagMeta] = [
    FlagMeta("germany", "r/placeDE", "de", "Einigkeit und Pixel!", BLACK_IDX, DARK_RED_IDX, YELLOW_IDX),
    FlagMeta("france", "r/placeFR", "france", "Pour la France!", DARK_BLUE_IDX, WHITE_IDX, DARK_RED_IDX),
    FlagMeta("ukraine", "r/placeUkraine", "ukraine", "Stand with Ukraine!", BLUE_IDX, YELLOW_IDX, WHITE_IDX),
    FlagMeta("italy", "r/placeIT", "italy", "Avanti Italia!", DARK_GREEN, WHITE_IDX, DARK_RED_IDX),
    FlagMeta("japan", "r/placeJP", "japan", "Rising Sun pixels!", WHITE_IDX, DARK_RED_IDX, RED_IDX),
    FlagMeta("usa", "r/AmericanFlagInPlace", "usa", "Stars and Stripes forever!", DARK_BLUE_IDX, DARK_RED_IDX, WHITE_IDX),
    FlagMeta("uk", "r/ukplace", "unitedkingdom", "Hold the Union Jack!", DARK_BLUE_IDX, DARK_RED_IDX, WHITE_IDX),
    FlagMeta("canada", "r/placeCanada", "canada", "Protect the maple leaf!", DARK_RED_IDX, WHITE_IDX, RED_IDX),
    FlagMeta("turkey", "r/placeTR", "Turkey", "Crescent and star strong!", DARK_RED_IDX, WHITE_IDX, RED_IDX),
    FlagMeta("brazil", "r/placeBR", "brasil", "Ordem e Pixel!", DARK_GREEN, YELLOW_IDX, DARK_BLUE_IDX),
    FlagMeta("mexico", "r/placeMX", "mexico", "Viva Mexico!", DARK_GREEN, WHITE_IDX, DARK_RED_IDX),
    FlagMeta("argentina", "r/PlaceArg", "argentina", "Vamos Argentina!", LIGHT_BLUE, WHITE_IDX, YELLOW_IDX),
    FlagMeta("sweden", "r/NordicPlace SE", "sweden", "Hold the Nordic North!", BLUE_IDX, YELLOW_IDX, DARK_BLUE_IDX),
    FlagMeta("norway", "r/NordicPlace NO", "norge", "Northern fjords unite!", DARK_RED_IDX, DARK_BLUE_IDX, WHITE_IDX),
    FlagMeta("finland", "r/NordicPlace FI", "Suomi", "Suomi holds the wall!", WHITE_IDX, DARK_BLUE_IDX, BLUE_IDX),
    FlagMeta("denmark", "r/NordicPlace DK", "Denmark", "Dannebrog stands!", DARK_RED_IDX, WHITE_IDX, RED_IDX),
    FlagMeta("switzerland", "r/placeCH", "Switzerland", "Neutral pixels forever!", DARK_RED_IDX, WHITE_IDX, RED_IDX),
    FlagMeta("netherlands", "r/PlaceNL", "thenetherlands", "Gekoloniseerd!", DARK_RED_IDX, WHITE_IDX, DARK_BLUE_IDX),
    FlagMeta("poland", "r/placePL", "Polska", "Polska strong!", WHITE_IDX, DARK_RED_IDX, RED_IDX),
    FlagMeta("ireland", "r/placeIE", "ireland", "Emerald pixels!", DARK_GREEN, WHITE_IDX, ORANGE_IDX),
    FlagMeta("spain", "r/placeES", "es", "Adelante Espana!", DARK_RED_IDX, YELLOW_IDX, RED_IDX),
    FlagMeta("portugal", "r/placePT", "portugal", "Portugal caralho!", DARK_GREEN, DARK_RED_IDX, YELLOW_IDX),
    FlagMeta("greece", "r/placeGR", "greece", "Aegean stripes hold!", BLUE_IDX, WHITE_IDX, DARK_BLUE_IDX),
    FlagMeta("south_korea", "r/placeKR", "korea", "Taegeuk united!", WHITE_IDX, DARK_RED_IDX, DARK_BLUE_IDX),
    FlagMeta("india", "r/IndiaPlace", "india", "Tricolor across the map!", ORANGE_IDX, WHITE_IDX, DARK_GREEN),
    FlagMeta("vietnam", "r/placeVN", "VietNam", "Gold star shines bright!", DARK_RED_IDX, YELLOW_IDX, RED_IDX),
    FlagMeta("guatemala", "r/placeGT", "guatemala", "Quetzal watches the border!", LIGHT_BLUE, WHITE_IDX, DARK_GREEN),
    FlagMeta("colombia", "r/placeCO", "Colombia", "Gold blue crimson!", YELLOW_IDX, DARK_BLUE_IDX, DARK_RED_IDX),
    FlagMeta("chile", "r/placeCL", "chile", "Lone star of the south!", DARK_BLUE_IDX, DARK_RED_IDX, WHITE_IDX),
    FlagMeta("belgium", "r/placeBE", "belgium", "Waffles and stripes!", BLACK_IDX, YELLOW_IDX, DARK_RED_IDX),
    FlagMeta("pride", "r/RainbowPlace", "lgbt", "All colors together!", RED_IDX, ORANGE_IDX, YELLOW_IDX),
    FlagMeta("trans", "r/transplace", "trans", "Pastel stripes hold strong!", LIGHT_BLUE, LIGHT_PINK, WHITE_IDX),
]

FLAG_BY_KEY: dict[str, FlagMeta] = {f.key: f for f in NATIONAL_FLAGS}


def _draw_circle(bp: np.ndarray, cx: float, cy: float, r: float, col: int) -> None:
    h, w = bp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bp[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = col


def _draw_star5(bp: np.ndarray, cx: float, cy: float, r_out: float, col: int) -> None:
    h, w = bp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    ang = np.arctan2(dy, dx) + math.pi / 2.0
    sector = (ang % (2.0 * math.pi / 5.0)) / (2.0 * math.pi / 5.0)
    tri = 1.0 - np.abs(sector * 2.0 - 1.0)
    rad = r_out * 0.42 + (r_out * 0.58) * tri
    bp[np.sqrt(dx * dx + dy * dy) <= rad] = col


def _h_stripes(bp: np.ndarray, colors: list[int], weights: list[float] | None = None) -> None:
    h, _ = bp.shape
    if weights is None:
        weights = [1.0] * len(colors)
    tot = sum(weights)
    acc = 0.0
    for i, (c, wt) in enumerate(zip(colors, weights)):
        y0 = int(round(acc / tot * h))
        acc += wt
        y1 = h if i == len(colors) - 1 else int(round(acc / tot * h))
        bp[y0:max(y0 + 1, y1), :] = c


def _v_stripes(bp: np.ndarray, colors: list[int], weights: list[float] | None = None) -> None:
    _, w = bp.shape
    if weights is None:
        weights = [1.0] * len(colors)
    tot = sum(weights)
    acc = 0.0
    for i, (c, wt) in enumerate(zip(colors, weights)):
        x0 = int(round(acc / tot * w))
        acc += wt
        x1 = w if i == len(colors) - 1 else int(round(acc / tot * w))
        bp[:, x0:max(x0 + 1, x1)] = c


def generate_national_flag(flag_key: str, w: int, h: int, add_border: bool = True) -> np.ndarray:
    """Render any of the 32 iconic national/community flags cleanly at Micro, Medium, or Macro scale."""
    bp = np.full((h, w), WHITE_IDX, dtype=np.int16)
    k = flag_key.lower().strip()
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    mn = min(w, h)

    if k == "germany":
        _h_stripes(bp, [BLACK_IDX, DARK_RED_IDX, YELLOW_IDX])
    elif k == "france":
        _v_stripes(bp, [DARK_BLUE_IDX, WHITE_IDX, DARK_RED_IDX])
    elif k == "ukraine":
        _h_stripes(bp, [BLUE_IDX, YELLOW_IDX])
    elif k == "italy":
        _v_stripes(bp, [DARK_GREEN, WHITE_IDX, DARK_RED_IDX])
    elif k == "japan":
        bp[:, :] = WHITE_IDX
        _draw_circle(bp, cx, cy, max(2.2, mn * 0.28), DARK_RED_IDX)
    elif k == "usa":
        n_stripes = 13 if h >= 26 else (9 if h >= 18 else 7)
        for y in range(h):
            s = int(y * n_stripes / max(1, h))
            bp[y, :] = DARK_RED_IDX if (s % 2 == 0) else WHITE_IDX
        cw = max(5, int(w * 0.42))
        ch = max(4, int(h * 0.54))
        bp[:ch, :cw] = DARK_BLUE_IDX
        step = 2 if cw < 24 else 3
        for sy in range(1, ch - 1, step):
            for sx in range(1 + ((sy // step) % 2), cw - 1, step):
                bp[sy, sx] = WHITE_IDX
    elif k == "uk":
        bp[:, :] = DARK_BLUE_IDX
        yy, xx = np.mgrid[0:h, 0:w]
        # Diagonal saltires scaled by aspect ratio
        d1 = np.abs((xx / max(1, w - 1)) - (yy / max(1, h - 1)))
        d2 = np.abs((xx / max(1, w - 1)) + (yy / max(1, h - 1)) - 1.0)
        w_thick = max(0.08, 2.2 / max(1, mn))
        r_thick = max(0.035, 1.0 / max(1, mn))
        bp[(d1 < w_thick) | (d2 < w_thick)] = WHITE_IDX
        bp[(d1 < r_thick) | (d2 < r_thick)] = DARK_RED_IDX
        # White & red central St George cross
        cw = max(2, int(h * 0.24))
        ch = max(2, int(w * 0.18))
        bp[max(0, h // 2 - cw // 2) : min(h, h // 2 + (cw + 1) // 2), :] = WHITE_IDX
        bp[:, max(0, w // 2 - ch // 2) : min(w, w // 2 + (ch + 1) // 2)] = WHITE_IDX
        rw = max(1, cw // 2)
        rh = max(1, ch // 2)
        bp[max(0, h // 2 - rw // 2) : min(h, h // 2 + (rw + 1) // 2), :] = DARK_RED_IDX
        bp[:, max(0, w // 2 - rh // 2) : min(w, w // 2 + (rh + 1) // 2)] = DARK_RED_IDX
    elif k == "canada":
        _v_stripes(bp, [DARK_RED_IDX, WHITE_IDX, DARK_RED_IDX], [1.0, 2.0, 1.0])
        # Crisp pixel maple leaf silhouette
        r = max(2.2, mn * 0.26)
        _draw_star5(bp, cx, cy - r * 0.08, r, DARK_RED_IDX)
        stem_h = max(2, int(r * 0.65))
        bp[int(cy) : min(h - 1, int(cy) + stem_h), int(cx)] = DARK_RED_IDX
    elif k == "turkey":
        bp[:, :] = DARK_RED_IDX
        _draw_circle(bp, w * 0.38, cy, max(2.5, mn * 0.27), WHITE_IDX)
        _draw_circle(bp, w * 0.44, cy, max(1.9, mn * 0.21), DARK_RED_IDX)
        _draw_star5(bp, w * 0.63, cy, max(1.8, mn * 0.14), WHITE_IDX)
    elif k == "brazil":
        bp[:, :] = DARK_GREEN
        yy, xx = np.mgrid[0:h, 0:w]
        diamond = (np.abs(xx - cx) / max(1.0, w * 0.40)) + (np.abs(yy - cy) / max(1.0, h * 0.40)) <= 1.0
        bp[diamond] = YELLOW_IDX
        r = max(2.0, mn * 0.21)
        _draw_circle(bp, cx, cy, r, DARK_BLUE_IDX)
        if r >= 3.5:
            band = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & (np.abs((yy - cy) - (xx - cx) * 0.18) <= 0.7)
            bp[band] = WHITE_IDX
    elif k == "mexico":
        _v_stripes(bp, [DARK_GREEN, WHITE_IDX, DARK_RED_IDX])
        _draw_circle(bp, cx, cy, max(1.6, mn * 0.14), BROWN)
        if mn >= 22:
            bp[int(cy) : int(cy) + 2, int(cx) - 1 : int(cx) + 2] = DARK_GREEN
    elif k == "argentina":
        _h_stripes(bp, [LIGHT_BLUE, WHITE_IDX, LIGHT_BLUE])
        _draw_circle(bp, cx, cy, max(1.5, mn * 0.13), YELLOW_IDX)
        if mn >= 26:
            bp[int(cy), int(cx)] = ORANGE_IDX
    elif k == "sweden":
        bp[:, :] = BLUE_IDX
        t = max(2, int(mn * 0.20))
        x_bar = int(w * 0.34)
        y_bar = h // 2
        bp[max(0, y_bar - t // 2) : min(h, y_bar + (t + 1) // 2), :] = YELLOW_IDX
        bp[:, max(0, x_bar - t // 2) : min(w, x_bar + (t + 1) // 2)] = YELLOW_IDX
    elif k == "norway":
        bp[:, :] = DARK_RED_IDX
        tw = max(3, int(mn * 0.26))
        tb = max(1, tw // 2)
        x_bar = int(w * 0.34)
        y_bar = h // 2
        bp[max(0, y_bar - tw // 2) : min(h, y_bar + (tw + 1) // 2), :] = WHITE_IDX
        bp[:, max(0, x_bar - tw // 2) : min(w, x_bar + (tw + 1) // 2)] = WHITE_IDX
        bp[max(0, y_bar - tb // 2) : min(h, y_bar + (tb + 1) // 2), :] = DARK_BLUE_IDX
        bp[:, max(0, x_bar - tb // 2) : min(w, x_bar + (tb + 1) // 2)] = DARK_BLUE_IDX
    elif k == "finland":
        bp[:, :] = WHITE_IDX
        t = max(2, int(mn * 0.22))
        x_bar = int(w * 0.34)
        y_bar = h // 2
        bp[max(0, y_bar - t // 2) : min(h, y_bar + (t + 1) // 2), :] = DARK_BLUE_IDX
        bp[:, max(0, x_bar - t // 2) : min(w, x_bar + (t + 1) // 2)] = DARK_BLUE_IDX
    elif k == "denmark":
        bp[:, :] = DARK_RED_IDX
        t = max(2, int(mn * 0.18))
        x_bar = int(w * 0.34)
        y_bar = h // 2
        bp[max(0, y_bar - t // 2) : min(h, y_bar + (t + 1) // 2), :] = WHITE_IDX
        bp[:, max(0, x_bar - t // 2) : min(w, x_bar + (t + 1) // 2)] = WHITE_IDX
    elif k == "switzerland":
        bp[:, :] = DARK_RED_IDX
        arm = max(2, int(mn * 0.20))
        span = max(4, int(mn * 0.58))
        bp[max(0, h // 2 - arm // 2) : min(h, h // 2 + (arm + 1) // 2), max(1, w // 2 - span // 2) : min(w - 1, w // 2 + (span + 1) // 2)] = WHITE_IDX
        bp[max(1, h // 2 - span // 2) : min(h - 1, h // 2 + (span + 1) // 2), max(0, w // 2 - arm // 2) : min(w, w // 2 + (arm + 1) // 2)] = WHITE_IDX
    elif k == "netherlands":
        _h_stripes(bp, [DARK_RED_IDX, WHITE_IDX, DARK_BLUE_IDX])
    elif k == "poland":
        _h_stripes(bp, [WHITE_IDX, DARK_RED_IDX])
    elif k == "ireland":
        _v_stripes(bp, [DARK_GREEN, WHITE_IDX, ORANGE_IDX])
    elif k == "spain":
        _h_stripes(bp, [DARK_RED_IDX, YELLOW_IDX, DARK_RED_IDX], [1.0, 2.0, 1.0])
        # Tiny heraldic coat of arms
        ex = int(w * 0.30)
        er = max(1, int(mn * 0.13))
        bp[max(0, int(cy) - er) : min(h, int(cy) + er + 1), max(0, ex - er) : min(w, ex + er + 1)] = DARK_RED_IDX
        bp[int(cy), ex] = YELLOW_IDX
    elif k == "portugal":
        _v_stripes(bp, [DARK_GREEN, DARK_RED_IDX], [2.0, 3.0])
        sx = int(w * 0.40)
        _draw_circle(bp, sx, cy, max(1.8, mn * 0.18), YELLOW_IDX)
        _draw_circle(bp, sx, cy, max(1.0, mn * 0.10), DARK_RED_IDX)
    elif k == "greece":
        for y in range(h):
            s = int(y * 9 / max(1, h))
            bp[y, :] = BLUE_IDX if (s % 2 == 0) else WHITE_IDX
        cw = max(5, int(w * 0.36))
        ch = max(5, int(h * 0.56))
        bp[:ch, :cw] = BLUE_IDX
        t = max(1, ch // 5)
        bp[ch // 2 - t // 2 : ch // 2 + (t + 1) // 2, :cw] = WHITE_IDX
        bp[:ch, cw // 2 - t // 2 : cw // 2 + (t + 1) // 2] = WHITE_IDX
    elif k == "south_korea":
        bp[:, :] = WHITE_IDX
        r = max(2.2, mn * 0.24)
        yy, xx = np.mgrid[0:h, 0:w]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        bp[inside & (yy <= cy)] = DARK_RED_IDX
        bp[inside & (yy > cy)] = DARK_BLUE_IDX
        # 4 corner black trigrams
        if mn >= 18:
            off_x, off_y = max(3, int(w * 0.25)), max(3, int(h * 0.25))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    px = int(cx + sx * off_x)
                    py = int(cy + sy * off_y)
                    bp[max(1, py - 1) : min(h - 1, py + 2), max(1, px - 1) : min(w - 1, px + 2)] = BLACK_IDX
    elif k == "india":
        _h_stripes(bp, [ORANGE_IDX, WHITE_IDX, DARK_GREEN])
        _draw_circle(bp, cx, cy, max(1.5, mn * 0.12), DARK_BLUE_IDX)
        if mn >= 28:
            _draw_circle(bp, cx, cy, max(0.8, mn * 0.06), WHITE_IDX)
    elif k == "vietnam":
        bp[:, :] = DARK_RED_IDX
        _draw_star5(bp, cx, cy, max(2.5, mn * 0.28), YELLOW_IDX)
    elif k == "guatemala":
        _v_stripes(bp, [LIGHT_BLUE, WHITE_IDX, LIGHT_BLUE])
        _draw_circle(bp, cx, cy, max(1.5, mn * 0.13), DARK_GREEN)
        bp[int(cy), int(cx)] = YELLOW_IDX
    elif k == "colombia":
        _h_stripes(bp, [YELLOW_IDX, DARK_BLUE_IDX, DARK_RED_IDX], [2.0, 1.0, 1.0])
    elif k == "chile":
        _h_stripes(bp, [WHITE_IDX, DARK_RED_IDX])
        cw = max(4, w // 3)
        ch = max(4, h // 2)
        bp[:ch, :cw] = DARK_BLUE_IDX
        _draw_star5(bp, cw / 2.0, ch / 2.0, max(1.4, min(cw, ch) * 0.30), WHITE_IDX)
    elif k == "belgium":
        _v_stripes(bp, [BLACK_IDX, YELLOW_IDX, DARK_RED_IDX])
    elif k == "pride":
        _h_stripes(bp, [RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX])
    elif k == "trans":
        _h_stripes(bp, [LIGHT_BLUE, LIGHT_PINK, WHITE_IDX, LIGHT_PINK, LIGHT_BLUE])
    else:
        _h_stripes(bp, [DARK_BLUE_IDX, WHITE_IDX, DARK_RED_IDX])

    if add_border and w >= 12 and h >= 10:
        bp[0, :] = BLACK_IDX
        bp[-1, :] = BLACK_IDX
        bp[:, 0] = BLACK_IDX
        bp[:, -1] = BLACK_IDX
    return bp


# Named Transverse Ribbon patterns (straight stripes & patterned highways crossing the entire map)
RIBBON_PATTERNS: list[tuple[str, str, list[int], str]] = [
    ("germany", "German Highway", [BLACK_IDX, DARK_RED_IDX, YELLOW_IDX], "stripes"),
    ("rainbow", "Rainbow Highway", [RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX], "stripes"),
    ("france", "French Ribbon", [DARK_BLUE_IDX, WHITE_IDX, DARK_RED_IDX], "stripes"),
    ("trans", "Trans Bridge", [LIGHT_BLUE, LIGHT_PINK, WHITE_IDX, LIGHT_PINK, LIGHT_BLUE], "stripes"),
    ("ukraine", "Ukraine Stripe", [BLUE_IDX, YELLOW_IDX], "stripes"),
    ("netherlands", "Dutch Canal Line", [DARK_RED_IDX, WHITE_IDX, DARK_BLUE_IDX], "stripes"),
    ("italy", "Italian Autostrada", [DARK_GREEN, WHITE_IDX, DARK_RED_IDX], "stripes"),
    ("checker_green", "Green Lattice Track", [DARK_GREEN, LIGHT_GREEN, BLACK_IDX], "checker"),
    ("chevron_neon", "Neon Chevron Beam", [PINK_IDX, LIGHT_BLUE, PURPLE_IDX, BLACK_IDX], "chevron"),
    ("caution", "Caution Tape Line", [YELLOW_IDX, BLACK_IDX], "diagonal_hatch"),
    ("railroad", "Pixel Railroad", [DARK_BROWN, BROWN, WHITE_IDX, BLACK_IDX], "railroad"),
    ("laser_cyan", "Cyan Plasma Beam", [DARK_BLUE_IDX, BLUE_IDX, LIGHT_BLUE, WHITE_IDX, LIGHT_BLUE, BLUE_IDX, DARK_BLUE_IDX], "stripes"),
]


def generate_transverse_ribbon(
    pattern_key: str,
    axis: str,
    length: int,
    thickness: int,
) -> np.ndarray:
    """Generate a straight horizontal ('h'), vertical ('v'), or 45-deg diagonal ('d1'/'d2') Transverse Ribbon.

    Diagonal ribbons return a [length, length] cutout (-1 outside the stripe) so they slice across diagonally!
    """
    spec = next((p for p in RIBBON_PATTERNS if p[0] == pattern_key), RIBBON_PATTERNS[0])
    _, _, cols, style = spec
    n_c = len(cols)

    def _fill_pixel(u: np.ndarray, v: np.ndarray, thick: int) -> np.ndarray:
        """u = coordinate along travel axis; v = coordinate across stripe thickness (0..thick-1)."""
        if style == "stripes":
            band = np.clip((v * n_c) // max(1, thick), 0, n_c - 1)
            out = np.take(np.array(cols, dtype=np.int16), band)
            # Crisp 1px dark edge lines when thick >= 7
            if thick >= 7 and cols[0] != BLACK_IDX and cols[-1] != BLACK_IDX:
                out = np.where((v == 0) | (v == thick - 1), BLACK_IDX, out)
            return out
        if style == "checker":
            chk = ((u // 2) + (v // 2)) % 2
            out = np.where(chk == 0, cols[0], cols[1]).astype(np.int16)
            out = np.where((v == 0) | (v == thick - 1), BLACK_IDX, out)
            return out
        if style == "chevron":
            mid = thick // 2
            phase = (u + np.abs(v - mid)) // 2
            idx = phase % 3
            out = np.take(np.array(cols[:3], dtype=np.int16), idx)
            out = np.where((v == 0) | (v == thick - 1), BLACK_IDX, out)
            return out
        if style == "diagonal_hatch":
            hatch = ((u + v) // 3) % 2
            out = np.where(hatch == 0, cols[0], cols[1]).astype(np.int16)
            out = np.where((v == 0) | (v == thick - 1), BLACK_IDX, out)
            return out
        # railroad
        out = np.full_like(u, BROWN, dtype=np.int16)
        ties = (u % 4) == 0
        out = np.where(ties, DARK_BROWN, out)
        rail1 = max(1, thick // 4)
        rail2 = max(rail1 + 1, thick - 1 - thick // 4)
        out = np.where((v == rail1) | (v == rail2), WHITE_IDX, out)
        out = np.where((v == 0) | (v == thick - 1), BLACK_IDX, out)
        return out

    length = max(16, int(length))
    thickness = max(5, min(15, int(thickness)))

    if axis == "h":
        yy, xx = np.mgrid[0:thickness, 0:length]
        return _fill_pixel(xx, yy, thickness)
    if axis == "v":
        yy, xx = np.mgrid[0:length, 0:thickness]
        return _fill_pixel(yy, xx, thickness)

    # 45-degree diagonal cutout ('d1' top-left -> bottom-right, 'd2' bottom-left -> top-right)
    size = min(length, 512)
    bp = np.full((size, size), -1, dtype=np.int16)
    yy, xx = np.mgrid[0:size, 0:size]
    if axis == "d1":
        diff = xx - yy
        u = (xx + yy) // 2
    else:
        diff = (xx + yy) - (size - 1)
        u = (xx - yy + size) // 2
    half = thickness // 2
    mask = (diff >= -half) & (diff < -half + thickness)
    v = np.clip(diff + half, 0, thickness - 1)
    filled = _fill_pixel(u, v, thickness)
    bp[mask] = filled[mask]
    return bp
