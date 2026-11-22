"""Crisp procedural pixel-art generators (flags, shields, wide word banners, 5x7 outlined font, borders)."""

from __future__ import annotations

import hashlib
import math
import random

import numpy as np

from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    DARK_BLUE_IDX,
    GREEN_IDX,
    NUM_COLORS,
    ORANGE_IDX,
    PINK_IDX,
    PURPLE_IDX,
    RED_IDX,
    WHITE_IDX,
    YELLOW_IDX,
)

# Crisp 5x7 bold bitmap pixel font for unmistakable r/place banners & mural nameplates
_FONT_5X7: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10011", "10101", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["10010", "10010", "10010", "11111", "00010", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    " ": ["000", "000", "000", "000", "000", "000", "000"],
}

# Compact 3x5 fallback font when mural is narrow
_FONT_3X5: dict[str, list[str]] = {
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["110", "101", "101", "101", "101"],
    "O": ["010", "101", "101", "101", "010"],
    "P": ["110", "101", "110", "100", "100"],
    "Q": ["010", "101", "101", "111", "001"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["110", "001", "010", "100", "111"],
    "3": ["110", "001", "010", "001", "110"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "110", "001", "110"],
    "6": ["011", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "110"],
    "/": ["001", "001", "010", "100", "100"],
    "!": ["010", "010", "010", "000", "010"],
    "-": ["000", "000", "111", "000", "000"],
    " ": ["00", "00", "00", "00", "00"],
}


def _measure_text(text: str, font: dict[str, list[str]], spacing: int, scale: int = 1) -> int:
    if not text:
        return 0
    return sum((len(font[ch][0]) + spacing) * scale for ch in text) - spacing * scale


def stamp_outlined_text(
    bp: np.ndarray,
    text: str,
    fg: int = WHITE_IDX,
    outline: int = BLACK_IDX,
    ribbon_bg: int | None = None,
    y: int | None = None,
    scale: int = 1,
    prefer_large: bool = True,
) -> None:
    """Stamp crisp outlined pixel-art typography (5x7 or 3x5 fallback) onto an int16 blueprint."""
    h, w = bp.shape
    raw = text.strip().upper()
    if not raw:
        return

    # Try 5x7 at requested scale, then 5x7 at 1x, then compact 3x5
    chosen_font = _FONT_5X7
    chosen_scale = scale
    spacing = 1
    clean = "".join(ch for ch in raw if ch in _FONT_5X7)
    if not clean:
        return

    if not prefer_large or _measure_text(clean, _FONT_5X7, spacing, chosen_scale) > w - 6:
        chosen_scale = 1
    if _measure_text(clean, _FONT_5X7, spacing, 1) > w - 6:
        clean_s = "".join(ch for ch in raw if ch in _FONT_3X5)
        if _measure_text(clean_s, _FONT_3X5, 1, 1) <= w - 4:
            chosen_font = _FONT_3X5
            clean = clean_s
        else:
            # Trim cleanly at word boundary (no awkward mid-word clipping like 'EMERALD DREA')
            chosen_font = _FONT_3X5
            max_chars = max(3, (w - 6) // 4)
            trimmed = clean_s[:max_chars]
            if len(clean_s) > max_chars and " " in trimmed:
                cand_trim = trimmed.rsplit(" ", 1)[0].strip()
                if len(cand_trim) >= 3:
                    trimmed = cand_trim
            clean = trimmed

    if not clean:
        return

    gh = len(next(iter(chosen_font.values()))) * chosen_scale
    tw = _measure_text(clean, chosen_font, spacing, chosen_scale)
    x0 = max(2, (w - tw) // 2)
    y0 = (h - gh - 3) if y is None else max(1, min(h - gh - 1, y))

    if ribbon_bg is not None:
        ry0 = max(0, y0 - 2)
        ry1 = min(h, y0 + gh + 2)
        rx0 = max(0, x0 - 4)
        rx1 = min(w, x0 + tw + 4)
        bp[ry0:ry1, rx0:rx1] = ribbon_bg
        # Thin border lines above and below ribbon
        if ry0 < h:
            bp[ry0, rx0:rx1] = outline
        if ry1 - 1 < h:
            bp[ry1 - 1, rx0:rx1] = outline

    # Build foreground pixel mask first, then draw 1px crisp outline underneath
    fg_coords: list[tuple[int, int]] = []
    cur_x = x0
    for ch in clean:
        glyph = chosen_font[ch]
        gw = len(glyph[0])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    for sy in range(chosen_scale):
                        for sx in range(chosen_scale):
                            py = y0 + gy * chosen_scale + sy
                            px = cur_x + gx * chosen_scale + sx
                            if 0 <= py < h and 0 <= px < w:
                                fg_coords.append((py, px))
        cur_x += (gw + spacing) * chosen_scale

    # 8-connected outline so letters stay readable over busy pixel art
    for py, px in fg_coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    bp[ny, nx] = outline
    for py, px in fg_coords:
        bp[py, px] = fg


def stamp_nameplate_ribbon(
    bp: np.ndarray,
    text: str,
    primary: int,
    accent: int = WHITE_IDX,
    position: str = "bottom",
) -> np.ndarray:
    """Stamp a crisp r/place nameplate tightly under a mural (floating pill, never a full-width box)."""
    out = bp.copy()
    h, w = out.shape
    if h < 28 or w < 32 or not text.strip():
        return out
    fg = WHITE_IDX if primary != WHITE_IDX else BLACK_IDX
    # If transparent cutout, anchor tightly right below the lowest silhouette pixel
    if bool((out < 0).any()):
        rows_with_art = np.where((out >= 0).any(axis=1))[0]
        if len(rows_with_art) > 0:
            bottom_y = int(rows_with_art[-1])
            y_pos = min(h - 9, max(3, bottom_y - 6)) if position == "bottom" else max(2, int(rows_with_art[0]))
        else:
            y_pos = h - 10
        ribbon = None  # Outlined floating pixel letters without a background box
    else:
        y_pos = 3 if position == "top" else (h - 10)
        ribbon = primary % NUM_COLORS
    stamp_outlined_text(
        out,
        text,
        fg=fg,
        outline=BLACK_IDX,
        ribbon_bg=ribbon,
        y=y_pos,
        scale=1,
        prefer_large=True,
    )
    return out


def add_silhouette_fortify(bp: np.ndarray, primary: int, secondary: int, accent: int) -> np.ndarray:
    """Fortify a blueprint organically: die-cut silhouette aura for cutouts, ornate frame only for solid rectangles."""
    # When >= 12% pixels are transparent, outline the curved silhouette directly
    if float((bp < 0).mean()) >= 0.12:
        out = bp.copy()
        h, w = out.shape
        valid = out >= 0

        def _dilate4(m: np.ndarray) -> np.ndarray:
            d = m.copy()
            d[1:, :] |= m[:-1, :]
            d[:-1, :] |= m[1:, :]
            d[:, 1:] |= m[:, :-1]
            d[:, :-1] |= m[:, 1:]
            return d

        ring1 = _dilate4(valid) & (~valid)
        ring2 = _dilate4(valid | ring1) & (~(valid | ring1))
        ring3 = _dilate4(valid | ring1 | ring2) & (~(valid | ring1 | ring2))
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx + yy) % 2) == 0
        out[ring1] = np.where(checker[ring1], primary % NUM_COLORS, secondary % NUM_COLORS)
        out[ring2] = np.where(checker[ring2], secondary % NUM_COLORS, accent % NUM_COLORS)
        out[ring3] = BLACK_IDX
        return out
    return add_ornate_border(bp, primary, secondary, accent)


def add_ornate_border(bp: np.ndarray, primary: int, secondary: int, accent: int) -> np.ndarray:
    """Wrap a blueprint with a crisp pixel-art frame and corner gems (used by FORTIFY)."""
    out = bp.copy()
    h, w = out.shape
    if h < 12 or w < 12:
        return out
    out[0, :] = BLACK_IDX
    out[-1, :] = BLACK_IDX
    out[:, 0] = BLACK_IDX
    out[:, -1] = BLACK_IDX
    for x in range(1, w - 1):
        c = primary if (x % 2 == 0) else secondary
        out[1, x] = c
        out[-2, x] = c
    for y in range(1, h - 1):
        c = primary if (y % 2 == 0) else secondary
        out[y, 1] = c
        out[y, -2] = c
    out[2, 2:-2] = BLACK_IDX
    out[-3, 2:-2] = BLACK_IDX
    out[2:-2, 2] = BLACK_IDX
    out[2:-2, -3] = BLACK_IDX
    for cy in (1, h - 3):
        for cx in (1, w - 3):
            out[cy : cy + 2, cx : cx + 2] = accent
    return out


def _draw_star(bp: np.ndarray, cx: float, cy: float, r_outer: float, r_inner: float, col: int, outline: int = BLACK_IDX) -> None:
    h, w = bp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    ang = np.arctan2(dy, dx) + math.pi / 2.0
    sector = (ang % (2.0 * math.pi / 5.0)) / (2.0 * math.pi / 5.0)
    tri = 1.0 - np.abs(sector * 2.0 - 1.0)
    rad = r_inner + (r_outer - r_inner) * tri
    dist = np.sqrt(dx * dx + dy * dy)
    bp[(dist <= rad + 1.1) & (dist > rad - 0.3)] = outline
    bp[dist <= rad - 0.3] = col


def _draw_heart(bp: np.ndarray, cx: float, cy: float, scale: float, fill: int, outline: int = BLACK_IDX) -> None:
    h, w = bp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - cx) / max(1.0, scale)
    ny = -(yy - cy) / max(1.0, scale)
    val = (nx * nx + ny * ny - 1.0) ** 3 - nx * nx * (ny ** 3)
    bp[(val <= 0.18) & (val > 0.0)] = outline
    bp[val <= 0.0] = fill


def generate_word_banner(
    text: str,
    w: int,
    h: int,
    primary: int,
    secondary: int,
    accent: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate a bold horizontal r/place word/subreddit banner (e.g. 112x32) with crisp 5x7 letters."""
    rng = random.Random(seed)
    primary = int(primary) % NUM_COLORS
    secondary = int(secondary) % NUM_COLORS
    accent = int(accent) % NUM_COLORS
    if primary == WHITE_IDX:
        primary = BLUE_IDX
    if secondary == primary:
        secondary = PINK_IDX if primary != PINK_IDX else YELLOW_IDX

    bp = np.full((h, w), primary, dtype=np.int16)
    style = rng.choice(["striped", "rainbow_trim", "checker_band", "solid_split"])

    if style == "striped":
        band = max(4, h // 4)
        for y in range(h):
            if (y // band) % 2 == 1:
                bp[y, :] = secondary
    elif style == "rainbow_trim":
        rb = [RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX, PINK_IDX]
        for x in range(w):
            col = rb[(x // 5) % len(rb)]
            bp[2:5, x] = col
            bp[h - 5 : h - 2, x] = col
    elif style == "checker_band":
        yy, xx = np.mgrid[0:h, 0:w]
        bp[((xx // 4 + yy // 4) % 2) == 1] = secondary
    else:
        bp[h // 2 :, :] = secondary

    bp = add_ornate_border(bp, primary, secondary, accent)

    # Pick big 2x scale if short text fits, otherwise crisp 1x 5x7
    clean = text.strip().upper() or "R/PLACE"
    use_scale = 2 if (h >= 28 and _measure_text(clean, _FONT_5X7, 1, 2) <= w - 12) else 1
    gh = 7 * use_scale
    y_center = max(4, (h - gh) // 2)
    stamp_outlined_text(
        bp,
        clean,
        fg=WHITE_IDX,
        outline=BLACK_IDX,
        ribbon_bg=None,
        y=y_center,
        scale=use_scale,
        prefer_large=True,
    )
    return bp


def generate_procedural(
    prompt: str,
    w: int,
    h: int,
    primary: int,
    secondary: int,
    accent: int,
    label: str = "",
    banner_text: str = "",
    stage: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Generate a rich, crisp procedural pixel-art blueprint [h, w] int16."""
    p = prompt.lower()
    digest = int(hashlib.md5(f"{prompt}:{seed}:{stage}".encode()).hexdigest()[:8], 16)
    rng = random.Random(digest)

    primary = int(primary) % NUM_COLORS
    secondary = int(secondary) % NUM_COLORS
    accent = int(accent) % NUM_COLORS
    if primary == WHITE_IDX:
        primary = BLUE_IDX
    if secondary == primary:
        secondary = YELLOW_IDX if primary != YELLOW_IDX else RED_IDX
    if accent in (primary, secondary):
        accent = WHITE_IDX

    # When banner_text is empty, leave pure artwork so AI Typography Studio handles all lettering cleanly
    tag_text = banner_text.strip().upper()[:14] if banner_text else ""

    # Explicit text/sign/banner prompt (not a winding road) -> dedicated horizontal word banner
    if (
        any(k in p for k in ("word banner", "text banner", "sign", "slogan", "subreddit"))
        or (w >= h * 2.2 and not any(k in p for k in ("road", "rainbow", "river", "wave")))
    ):
        return generate_word_banner(tag_text, w, h, primary, secondary, accent, seed=digest)

    bp = np.full((h, w), -1, dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5

    if any(k in p for k in ("rainbow", "prism", "spectrum", "pride", "road")):
        motif = "winding_road"
    elif any(k in p for k in ("flag", "tricolor", "stripe")):
        motif = "flag"
    elif any(k in p for k in ("nordic", "cross", "kingdom")):
        motif = "nordic"
    elif any(k in p for k in ("heart", "love", "peace", "alliance")):
        motif = "heart_cutout"
    elif any(k in p for k in ("castle", "tower", "fort", "citadel", "city")):
        motif = "castle_silhouette"
    elif any(k in p for k in ("mountain", "sunset", "horizon", "synthwave", "moon", "planet")):
        motif = "medallion"
    elif any(k in p for k in ("spiral", "portal", "vortex", "galaxy", "nebula")):
        motif = "spiral"
    elif any(k in p for k in ("lattice", "checker", "mosaic", "chess", "grid", "geometric")):
        motif = "organic_mosaic"
    elif any(k in p for k in ("corner",)):
        motif = "corner_flood"
    elif any(k in p for k in ("shield", "crest", "emblem", "guild", "knight", "order")):
        motif = "shield"
    else:
        motif = rng.choice(["winding_road", "heart_cutout", "medallion", "spiral", "shield", "organic_mosaic", "flag", "nordic", "starburst_cutout"])

    if motif == "corner_flood":
        # Curved corner wedge flooding outward from the canvas corner
        if "top_left" in p:
            corner_id = 0
        elif "top_right" in p:
            corner_id = 1
        elif "bottom_left" in p:
            corner_id = 2
        elif "bottom_right" in p:
            corner_id = 3
        else:
            corner_id = digest % 4
        ax = 0.0 if (corner_id & 1) == 0 else float(w - 1)
        ay = 0.0 if (corner_id & 2) == 0 else float(h - 1)
        dx = (xx - ax) / max(1.0, w)
        dy = (yy - ay) / max(1.0, h)
        r = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(np.abs(dy), np.abs(dx) + 1e-5)
        coast = 0.88 + 0.11 * np.sin(5.0 * ang + seed) + 0.05 * np.cos(9.0 * ang)
        inside = r <= coast
        edge = (r > coast) & (r <= coast + 0.045)
        bp[inside] = primary
        bp[inside & (((xx + yy) % 7) == 0)] = secondary
        bp[edge] = BLACK_IDX
        if tag_text:
            ty = int(h * 0.25) if ay == 0 else int(h * 0.65)
            stamp_outlined_text(bp, tag_text[:12], fg=WHITE_IDX, outline=BLACK_IDX, y=ty, scale=1)
        return bp

    elif motif == "winding_road":
        # S-curved rainbow ribbon over transparent space
        bands = [RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX, PINK_IDX]
        diag = (xx / max(1.0, w)) * math.pi * 1.8 + (digest % 7)
        center_y = cy + (h * 0.26) * np.sin(diag)
        band_half = max(7.0, h * 0.18)
        dy_c = yy - center_y
        in_road = np.abs(dy_c) <= band_half
        out_edge = (np.abs(dy_c) > band_half) & (np.abs(dy_c) <= band_half + 1.6)
        stripe_idx = np.clip(((dy_c + band_half) / (2.0 * band_half) * len(bands)).astype(np.int32), 0, len(bands) - 1)
        for i, col in enumerate(bands):
            bp[in_road & (stripe_idx == i)] = col
        bp[out_edge] = BLACK_IDX
        if tag_text and w >= 48:
            stamp_outlined_text(bp, tag_text[:10], fg=WHITE_IDX, outline=BLACK_IDX, y=int(cy - 3), scale=1)
        return bp

    elif motif == "heart_cutout":
        # Heart cutout with transparent exterior
        _draw_heart(bp, cx, cy + h * 0.03, min(w, h) * 0.36, primary, BLACK_IDX)
        _draw_heart(bp, cx, cy + h * 0.03, min(w, h) * 0.21, secondary, BLACK_IDX)
        if stage >= 2:
            _draw_heart(bp, cx, cy + h * 0.03, min(w, h) * 0.10, accent, BLACK_IDX)
        if tag_text and w >= 44:
            stamp_outlined_text(bp, tag_text[:9], fg=WHITE_IDX, outline=BLACK_IDX, y=int(cy - 3), scale=1)
        return bp

    elif motif == "medallion":
        # Circular sunset / ringed badge with transparent exterior
        rad = min(w, h) * 0.46
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inside = dist <= rad
        ring = (dist > rad) & (dist <= rad + 2.0)
        bp[inside & (yy < cy - rad * 0.25)] = PURPLE_IDX
        bp[inside & (yy >= cy - rad * 0.25) & (yy < cy + rad * 0.12)] = PINK_IDX
        bp[inside & (yy >= cy + rad * 0.12) & (yy < cy + rad * 0.42)] = YELLOW_IDX
        bp[inside & (yy >= cy + rad * 0.42)] = DARK_BLUE_IDX
        sun_d = np.sqrt((xx - cx) ** 2 + (yy - (cy - rad * 0.05)) ** 2)
        bp[inside & (sun_d <= rad * 0.38) & (yy < cy + rad * 0.42)] = primary
        ridge = cy + rad * 0.55 - np.maximum(0.0, (rad * 0.85 - np.abs(xx - cx)) * 0.75)
        bp[inside & (yy >= ridge) & (yy < cy + rad * 0.55)] = secondary
        bp[ring] = BLACK_IDX
        if tag_text and w >= 48:
            stamp_outlined_text(bp, tag_text[:9], fg=WHITE_IDX, outline=BLACK_IDX, y=int(cy + rad * 0.52), scale=1)
        return bp

    elif motif == "organic_mosaic":
        # Organic irregular cluster of checkerboard diamond tiles (like Green Lattice)
        tile = max(3, min(w, h) // 10)
        nx = (xx - cx) / max(1.0, w * 0.46)
        ny = (yy - cy) / max(1.0, h * 0.46)
        ang = np.arctan2(ny, nx)
        r = np.sqrt(nx * nx + ny * ny)
        blob = r <= (0.86 + 0.13 * np.sin(4.0 * ang + seed) + 0.07 * np.cos(7.0 * ang))
        cx_t = xx // tile
        cy_t = yy // tile
        bp[blob] = np.where((((cx_t + cy_t) % 2) == 0)[blob], primary, secondary)
        lx = np.abs((xx % tile) - tile // 2)
        ly = np.abs((yy % tile) - tile // 2)
        bp[blob & ((lx + ly) <= max(1, tile // 4))] = accent
        if tag_text and w >= 48:
            stamp_outlined_text(bp, tag_text[:9], fg=WHITE_IDX, outline=BLACK_IDX, y=int(cy - 3), scale=1)
        return bp

    elif motif == "castle_silhouette":
        # Castle keep cutout on transparent background
        gy = int(h * 0.86)
        wx0, wx1 = int(w * 0.16), int(w * 0.84)
        wy0 = int(h * 0.42)
        bp[wy0:gy, wx0:wx1] = primary
        tw = max(6, w // 5)
        bp[int(h * 0.20) : gy, wx0 : wx0 + tw] = secondary
        bp[int(h * 0.20) : gy, wx1 - tw : wx1] = secondary
        # Pointed spire roofs
        for tx in (wx0 + tw // 2, wx1 - tw // 2):
            for ry in range(int(h * 0.06), int(h * 0.20)):
                span = max(1, int((ry - h * 0.06) * 0.55))
                bp[ry, max(0, tx - span) : min(w, tx + span + 1)] = accent
        for x in range(wx0, wx1, 4):
            bp[max(0, wy0 - 3) : wy0, x : min(wx1, x + 2)] = primary
        gw = max(6, w // 5)
        bp[int(h * 0.58) : gy, int(cx - gw // 2) : int(cx + gw // 2)] = BLACK_IDX
        bp[int(h * 0.30) : int(h * 0.38), wx0 + tw // 2 - 1 : wx0 + tw // 2 + 2] = YELLOW_IDX
        bp[int(h * 0.30) : int(h * 0.38), wx1 - tw // 2 - 1 : wx1 - tw // 2 + 2] = YELLOW_IDX
        # Crisp 1px black silhouette outline around castle
        valid = bp >= 0
        dil = np.zeros_like(valid)
        dil[1:, :] |= valid[:-1, :]
        dil[:-1, :] |= valid[1:, :]
        dil[:, 1:] |= valid[:, :-1]
        dil[:, :-1] |= valid[:, 1:]
        bp[dil & (~valid)] = BLACK_IDX
        if tag_text and w >= 48:
            stamp_outlined_text(bp, tag_text[:9], fg=WHITE_IDX, outline=BLACK_IDX, y=gy - 2, scale=1)
        return bp

    elif motif == "starburst_cutout":
        # 8-point pixel star cutout on transparent exterior
        _draw_star(bp, cx, cy, min(w, h) * 0.46, min(w, h) * 0.22, primary, BLACK_IDX)
        _draw_star(bp, cx, cy, min(w, h) * 0.26, min(w, h) * 0.12, secondary, BLACK_IDX)
        if tag_text and w >= 48:
            stamp_outlined_text(bp, tag_text[:8], fg=WHITE_IDX, outline=BLACK_IDX, y=int(cy - 3), scale=1)
        return bp

    elif motif == "flag":
        if rng.random() < 0.55:
            t1, t2 = h // 3, (2 * h) // 3
            bp[:t1, :] = primary
            bp[t1:t2, :] = secondary
            bp[t2:, :] = accent
        else:
            t1, t2 = w // 3, (2 * w) // 3
            bp[:, :t1] = primary
            bp[:, t1:t2] = secondary
            bp[:, t2:] = accent
        _draw_star(bp, cx, cy - 3, min(w, h) * 0.21, min(w, h) * 0.09, WHITE_IDX if secondary != WHITE_IDX else BLACK_IDX)

    elif motif == "nordic":
        bp[:, :] = primary
        cw = max(4, min(w, h) // 6)
        cx_cross = int(w * 0.36)
        cy_cross = h // 2
        bp[cy_cross - cw // 2 - 2 : cy_cross + cw // 2 + 2, :] = accent
        bp[:, cx_cross - cw // 2 - 2 : cx_cross + cw // 2 + 2] = accent
        bp[cy_cross - cw // 2 : cy_cross + cw // 2, :] = secondary
        bp[:, cx_cross - cw // 2 : cx_cross + cw // 2] = secondary

    elif motif == "spiral":
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)
        arms = 3 + (stage % 3)
        wave = np.sin(arms * ang + dist * 0.35)
        inside = dist <= min(w, h) * 0.46
        bp[inside] = DARK_BLUE_IDX
        bp[inside & (wave > 0.15)] = primary
        bp[inside & (wave > 0.62)] = secondary
        bp[dist < min(w, h) * 0.14] = accent
        bp[(dist > min(w, h) * 0.46) & (dist <= min(w, h) * 0.49)] = BLACK_IDX
        return bp

    else:  # shield
        nx = (xx - cx) / max(1.0, w * 0.42)
        ny = (yy - cy) / max(1.0, h * 0.44)
        half_w = np.where(ny < 0.0, 1.0, np.maximum(0.0, 1.0 - (np.maximum(0.0, ny) ** 1.65)))
        inside = (ny >= -0.95) & (ny <= 0.98) & (np.abs(nx) <= half_w)
        border = (ny >= -1.03) & (ny <= 1.06) & (np.abs(nx) <= half_w + 0.10) & (~inside)
        q = ((xx >= cx) ^ (yy >= cy))
        bp[inside] = np.where(q[inside], primary, secondary)
        bp[border] = BLACK_IDX
        _draw_star(bp, cx, cy - h * 0.04, min(w, h) * 0.18, min(w, h) * 0.08, accent)
        return bp

    # Only true flags/nordic flags get thin crisp flag edges + nameplate
    bp = add_ornate_border(bp, primary, secondary, accent)
    if tag_text and w >= 44 and h >= 44:
        bp = stamp_nameplate_ribbon(bp, tag_text[:10], primary=primary, accent=accent, position="bottom")

    return bp
