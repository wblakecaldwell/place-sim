"""Multi-engine local AI typography studio for authentic r/place pixel-art lettering.

Unites:
  - Engine 1: VLM-directed multi-font crisp 1-bit TrueType & bitmap pixel renderer
    (10 distinct font families, 6 color fills/gradients, 5 3D/glow/sticker border effects,
     6 geometric layouts including wavy, arched, italic, stacked, and vertical).
  - Engine 3: qwen2.5:7b-instruct sandboxed Python PIL custom flourish/wordmark code artist
    (draws custom crowns, stars, swords, ribbons, sparkles, and brackets around letters).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pixel_battle.art.diffusion import _add_1px_silhouette_outline, _flood_strip_background_perceptual
from pixel_battle.art.procedural import _FONT_3X5, _FONT_5X7
from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    COLOR_NAMES,
    DARK_BLUE_IDX,
    GREEN_IDX,
    NUM_COLORS,
    ORANGE_IDX,
    PALETTE_RGB,
    PINK_IDX,
    PURPLE_IDX,
    RED_IDX,
    WHITE_IDX,
    YELLOW_IDX,
    resolve_color,
    rgb_to_palette_indices,
)

log = logging.getLogger(__name__)

# Distinct real macOS system fonts mapped by artistic personality (with clean fallbacks)
_FONT_CANDIDATES: dict[str, list[str]] = {
    "gothic": [
        "/System/Library/Fonts/Supplemental/Luminari.ttf",
        "/System/Library/Fonts/Supplemental/Herculanum.ttf",
        "/System/Library/Fonts/Supplemental/Copperplate.ttc",
    ],
    "block": [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    ],
    "arcade": [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
    ],
    "bubble": [
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
    ],
    "graffiti": [
        "/System/Library/Fonts/MarkerFelt.ttc",
        "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    ],
    "condensed": [
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    ],
    "serif": [
        "/System/Library/Fonts/Supplemental/Copperplate.ttc",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/BigCaslon.ttf",
    ],
    "script": [
        "/System/Library/Fonts/Supplemental/Brush Script.ttf",
        "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    ],
    "pixel_mono": [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ],
}

FONT_STYLES = (
    "gothic",
    "block",
    "arcade",
    "bubble",
    "graffiti",
    "condensed",
    "serif",
    "script",
    "pixel_mono",
    "pixel_retro",
)

_RESOLVED_FONT_PATHS: dict[str, str | None] = {}


def _get_font_path(style: str) -> str | None:
    if style in _RESOLVED_FONT_PATHS:
        return _RESOLVED_FONT_PATHS[style]
    for path in _FONT_CANDIDATES.get(style, []):
        if Path(path).exists():
            _RESOLVED_FONT_PATHS[style] = path
            return path
    _RESOLVED_FONT_PATHS[style] = None
    return None


@dataclass
class TypographyStyleSpec:
    include_text: bool = True
    text: str = ""
    casing: Literal["upper", "title", "subreddit"] = "upper"
    font_style: str = "arcade"
    size_tier: Literal["tiny", "medium", "large"] = "medium"
    color_effect: Literal[
        "solid",
        "vertical_gradient",
        "per_letter_palette",
        "rainbow_letters",
        "glossy_highlight",
        "striped_fill",
    ] = "vertical_gradient"
    border_effect: Literal[
        "outline_1px",
        "extrude_3d",
        "double_sticker",
        "neon_glow",
        "none",
    ] = "extrude_3d"
    layout: Literal[
        "straight",
        "wavy",
        "arched",
        "italic_slant",
        "stacked",
        "vertical_side",
    ] = "straight"
    placement: Literal[
        "bottom_center",
        "top_center",
        "center",
        "bottom_right_tag",
        "top_left_tag",
        "left_vertical",
    ] = "bottom_center"
    fill_color: int = WHITE_IDX
    fill_color_2: int = YELLOW_IDX
    fill_color_3: int = ORANGE_IDX
    outline_color: int = BLACK_IDX
    shadow_color: int = DARK_BLUE_IDX
    use_ribbon: bool = False
    ribbon_color: int = BLACK_IDX
    engine: Literal["styled", "flux", "code"] = "styled"


def _format_text(raw: str, casing: str) -> str:
    clean = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in " /!-+<>.")[:16].strip()
    if not clean:
        return ""
    if casing == "subreddit":
        body = clean.lower().lstrip("r/").replace(" ", "")[:9]
        return f"r/{body}" if body else ""
    if casing == "title":
        return clean.title()
    return clean.upper()


def _render_bitmap_word_mask(text: str, font_dict: dict[str, list[str]], scale: int = 1) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render a crisp 1-bit pixel bitmap word mask [gh, gw] and per-character x spans."""
    clean = "".join(ch for ch in text.upper() if ch in font_dict)
    if not clean:
        return np.zeros((1, 1), dtype=bool), []
    gh_base = len(next(iter(font_dict.values())))
    spacing = 1
    spans: list[tuple[int, int]] = []
    cur_x = 0
    for ch in clean:
        cw = len(font_dict[ch][0]) * scale
        spans.append((cur_x, cur_x + cw))
        cur_x += cw + spacing * scale
    total_w = max(1, cur_x - spacing * scale)
    mask = np.zeros((gh_base * scale, total_w), dtype=bool)
    for ch, (sx, _) in zip(clean, spans):
        glyph = font_dict[ch]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    mask[gy * scale : (gy + 1) * scale, sx + gx * scale : sx + (gx + 1) * scale] = True
    return mask, spans


def _render_ttf_word_mask(
    text: str,
    font_style: str,
    target_px_h: int,
    max_w: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render crisp 1-bit aliased TrueType glyphs + per-letter horizontal spans."""
    font_path = _get_font_path(font_style)
    if font_path is None or font_style == "pixel_retro":
        if target_px_h <= 8:
            return _render_bitmap_word_mask(text, _FONT_3X5, scale=1)
        sc = 2 if target_px_h >= 14 and len(text) * 12 <= max_w else 1
        return _render_bitmap_word_mask(text, _FONT_5X7, scale=sc)

    # Fit font size cleanly inside max_w
    px_size = max(9, target_px_h)
    font = None
    while px_size >= 9:
        try:
            cand = ImageFont.truetype(font_path, size=px_size)
        except Exception:
            break
        tmp = Image.new("1", (max(max_w * 2, 256), px_size * 3), 0)
        td = ImageDraw.Draw(tmp)
        td.fontmode = "1"
        bbox = td.textbbox((0, 0), text, font=cand)
        tw = bbox[2] - bbox[0]
        if tw <= max_w or px_size == 9:
            font = cand
            break
        px_size -= 2

    if font is None:
        return _render_bitmap_word_mask(text, _FONT_5X7, scale=1)

    # Render character by character onto a 1-bit mask so we have exact spans for wavy/arched/per-letter colors
    pad_h = px_size * 3
    pad_w = max(max_w * 2, 256)
    img = Image.new("1", (pad_w, pad_h), 0)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "1"

    spans: list[tuple[int, int]] = []
    cur_x = 4
    base_y = px_size // 2
    for ch in text:
        cb = draw.textbbox((cur_x, base_y), ch, font=font)
        draw.text((cur_x, base_y), ch, font=font, fill=1)
        adv = max(3, int(round(draw.textlength(ch, font=font))))
        spans.append((cur_x, cur_x + adv))
        cur_x += adv

    arr = np.asarray(img, dtype=bool)
    ys, xs = np.where(arr)
    if len(ys) == 0:
        return _render_bitmap_word_mask(text, _FONT_5X7, scale=1)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = arr[y0:y1, x0:x1]
    adj_spans = [(max(0, a - x0), max(1, min(x1 - x0, b - x0))) for a, b in spans]

    # If still slightly wider than max_w, crisp nearest-neighbor shrink horizontally
    if cropped.shape[1] > max_w and max_w >= 12:
        new_h = max(6, int(round(cropped.shape[0] * (max_w / cropped.shape[1]))))
        pil_m = Image.fromarray((cropped.astype(np.uint8) * 255), mode="L").resize(
            (max_w, new_h), Image.Resampling.NEAREST
        )
        scale_x = max_w / cropped.shape[1]
        cropped = np.asarray(pil_m) > 127
        adj_spans = [(int(a * scale_x), max(int(a * scale_x) + 1, int(b * scale_x))) for a, b in adj_spans]

    return cropped, adj_spans


def _apply_geometry(
    mask: np.ndarray,
    spans: list[tuple[int, int]],
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a 1-bit word mask according to layout and return (warped_mask, letter_id_map)."""
    h, w = mask.shape
    letter_ids = np.full((h, w), -1, dtype=np.int16)
    for idx, (a, b) in enumerate(spans):
        letter_ids[:, a:b] = idx
    letter_ids[~mask] = -1

    if layout in ("wavy", "arched") and len(spans) >= 2:
        amp = max(2, min(5, h // 3))
        out_h = h + amp * 2 + 2
        out_m = np.zeros((out_h, w), dtype=bool)
        out_id = np.full((out_h, w), -1, dtype=np.int16)
        n_ch = max(1, len(spans) - 1)
        for idx, (a, b) in enumerate(spans):
            t = idx / n_ch
            if layout == "wavy":
                dy = int(round(amp * math.sin(idx * 1.15))) + amp
            else:  # gentle upward rainbow arch
                dy = int(round(amp * 2.0 * ((t - 0.5) * 2.0) ** 2))
            sub = mask[:, a:b]
            out_m[dy : dy + h, a:b] |= sub
            cur = out_id[dy : dy + h, a:b]
            cur[sub] = idx
        return out_m, out_id

    if layout == "italic_slant":
        max_shift = max(2, h // 3)
        out_w = w + max_shift
        out_m = np.zeros((h, out_w), dtype=bool)
        out_id = np.full((h, out_w), -1, dtype=np.int16)
        for y in range(h):
            sh = int(round(((h - 1 - y) / max(1, h - 1)) * max_shift))
            out_m[y, sh : sh + w] = mask[y]
            out_id[y, sh : sh + w] = letter_ids[y]
        return out_m, out_id

    return mask, letter_ids


def render_styled_wordmark_patch(
    spec: TypographyStyleSpec,
    max_w: int,
    max_h: int,
) -> np.ndarray | None:
    """Render a standalone styled transparent-outside [ph, pw] int16 wordmark patch (Engine 1)."""
    text = _format_text(spec.text, spec.casing)
    if not text:
        return None

    # Target pixel height based on size_tier and available canvas box
    if spec.size_tier == "tiny":
        target_h = max(7, min(11, max_h // 4))
    elif spec.size_tier == "large":
        target_h = max(14, min(26, int(max_h * 0.58)))
    else:
        target_h = max(10, min(16, int(max_h * 0.34)))

    # Handle stacked two-line layout or vertical sidebar layout
    if spec.layout == "stacked" and " " in text and max_h >= 24:
        parts = text.split(" ", 1)
        m1, s1 = _render_ttf_word_mask(parts[0], spec.font_style, max(9, target_h), max_w - 6)
        m2, s2 = _render_ttf_word_mask(parts[1], spec.font_style, max(8, int(target_h * 0.78)), max_w - 6)
        cw = max(m1.shape[1], m2.shape[1])
        gap = 2
        ch = m1.shape[0] + gap + m2.shape[0]
        mask = np.zeros((ch, cw), dtype=bool)
        lid = np.full((ch, cw), -1, dtype=np.int16)
        ox1 = (cw - m1.shape[1]) // 2
        ox2 = (cw - m2.shape[1]) // 2
        mask[: m1.shape[0], ox1 : ox1 + m1.shape[1]] = m1
        for i, (a, b) in enumerate(s1):
            sub = lid[: m1.shape[0], ox1 + a : ox1 + b]
            sub[m1[:, a:b]] = i
        y2 = m1.shape[0] + gap
        mask[y2 : y2 + m2.shape[0], ox2 : ox2 + m2.shape[1]] = m2
        off_i = len(s1)
        for i, (a, b) in enumerate(s2):
            sub = lid[y2 : y2 + m2.shape[0], ox2 + a : ox2 + b]
            sub[m2[:, a:b]] = off_i + i
    elif spec.layout == "vertical_side" and len(text) <= 9:
        letters = [c for c in text if c != " "][:7]
        glyphs = [_render_ttf_word_mask(c, spec.font_style, max(8, min(12, max_h // max(len(letters), 1) - 2)), 18)[0] for c in letters]
        cw = max(g.shape[1] for g in glyphs)
        ch = sum(g.shape[0] + 2 for g in glyphs)
        mask = np.zeros((ch, cw), dtype=bool)
        lid = np.full((ch, cw), -1, dtype=np.int16)
        cy = 0
        for i, g in enumerate(glyphs):
            ox = (cw - g.shape[1]) // 2
            mask[cy : cy + g.shape[0], ox : ox + g.shape[1]] = g
            sub = lid[cy : cy + g.shape[0], ox : ox + g.shape[1]]
            sub[g] = i
            cy += g.shape[0] + 2
    else:
        raw_m, spans = _render_ttf_word_mask(text, spec.font_style, target_h, max(16, max_w - 8))
        mask, lid = _apply_geometry(raw_m, spans, spec.layout)

    gh, gw = mask.shape
    pad = 4
    ph, pw = gh + pad * 2, gw + pad * 2
    out = np.full((ph, pw), -1, dtype=np.int16)

    # Shift mask into padded frame
    m_pad = np.zeros((ph, pw), dtype=bool)
    l_pad = np.full((ph, pw), -1, dtype=np.int16)
    m_pad[pad : pad + gh, pad : pad + gw] = mask
    l_pad[pad : pad + gh, pad : pad + gw] = lid

    c1 = int(spec.fill_color) % NUM_COLORS
    c2 = int(spec.fill_color_2) % NUM_COLORS
    c3 = int(spec.fill_color_3) % NUM_COLORS
    oc = int(spec.outline_color) % NUM_COLORS
    sc = int(spec.shadow_color) % NUM_COLORS
    if sc == c1:
        sc = BLACK_IDX

    # Optional translucent/solid ribbon backing (only when VLM explicitly asks)
    if spec.use_ribbon:
        out[:, :] = int(spec.ribbon_color) % NUM_COLORS
        out[0, :] = oc
        out[-1, :] = oc
        out[:, 0] = oc
        out[:, -1] = oc

    def _dilate8(m: np.ndarray) -> np.ndarray:
        d = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ys0 = max(0, dy)
                ys1 = ph + min(0, dy)
                xs0 = max(0, dx)
                xs1 = pw + min(0, dx)
                d[ys0:ys1, xs0:xs1] |= m[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
        return d

    # Border / 3D extrusion effects drawn underneath letter fill
    if spec.border_effect == "extrude_3d":
        for step in (3, 2):
            sh = np.zeros_like(m_pad)
            sh[step:, step:] = m_pad[:-step, :-step]
            out[sh] = sc
        ring1 = _dilate8(m_pad) & (~m_pad)
        out[ring1] = oc
    elif spec.border_effect == "double_sticker":
        ring1 = _dilate8(m_pad) & (~m_pad)
        ring2 = _dilate8(m_pad | ring1) & (~(m_pad | ring1))
        out[ring2] = c2 if c2 != oc else WHITE_IDX
        out[ring1] = oc
    elif spec.border_effect == "neon_glow":
        ring1 = _dilate8(m_pad) & (~m_pad)
        ring2 = _dilate8(m_pad | ring1) & (~(m_pad | ring1))
        out[ring2] = sc
        out[ring1] = c2
    elif spec.border_effect == "outline_1px":
        ring1 = _dilate8(m_pad) & (~m_pad)
        out[ring1] = oc

    # Letter foreground fill effects
    yy, xx = np.mgrid[0:ph, 0:pw]
    rel_y = np.clip((yy - pad) / max(1.0, gh - 1), 0.0, 1.0)
    rainbow_pal = [RED_IDX, ORANGE_IDX, YELLOW_IDX, GREEN_IDX, BLUE_IDX, PURPLE_IDX, PINK_IDX]

    fill = np.full((ph, pw), c1, dtype=np.int16)
    if spec.color_effect == "vertical_gradient":
        fill = np.where(rel_y < 0.36, c1, np.where(rel_y < 0.72, c2, c3)).astype(np.int16)
    elif spec.color_effect == "per_letter_palette":
        pal3 = np.array([c1, c2, c3], dtype=np.int16)
        fill = pal3[np.maximum(0, l_pad) % 3]
    elif spec.color_effect == "rainbow_letters":
        rb = np.array(rainbow_pal, dtype=np.int16)
        fill = rb[np.maximum(0, l_pad) % len(rainbow_pal)]
    elif spec.color_effect == "glossy_highlight":
        fill = np.where(rel_y < 0.30, WHITE_IDX if c1 != WHITE_IDX else c2, np.where(rel_y > 0.78, c2, c1)).astype(np.int16)
    elif spec.color_effect == "striped_fill":
        fill = np.where(((yy // 2) % 2) == 0, c1, c2).astype(np.int16)

    out[m_pad] = fill[m_pad]
    return out


def composite_styled_text(bp: np.ndarray, spec: TypographyStyleSpec) -> np.ndarray:
    """Composite a styled AI-designed wordmark onto a mural blueprint at its chosen placement."""
    if not spec.include_text or not spec.text.strip():
        return bp
    out = bp.copy()
    h, w = out.shape
    patch = render_styled_wordmark_patch(spec, max_w=w - 4, max_h=h - 4)
    if patch is None:
        return out
    ph, pw = patch.shape
    if ph > h or pw > w:
        # Shrink patch cleanly if needed
        scale = min((w - 2) / max(1, pw), (h - 2) / max(1, ph))
        nw, nh = max(8, int(pw * scale)), max(6, int(ph * scale))
        ys = np.linspace(0, ph - 1, nh).astype(np.int32)
        xs = np.linspace(0, pw - 1, nw).astype(np.int32)
        patch = patch[ys[:, None], xs[None, :]]
        ph, pw = patch.shape

    # Compute intentional placement anchored cleanly to silhouette or mural area
    if spec.placement == "top_center":
        px, py = (w - pw) // 2, 2
    elif spec.placement == "center":
        px, py = (w - pw) // 2, (h - ph) // 2
    elif spec.placement == "bottom_right_tag":
        px, py = max(1, w - pw - 2), max(1, h - ph - 2)
    elif spec.placement == "top_left_tag":
        px, py = 2, 2
    elif spec.placement == "left_vertical":
        px, py = 2, (h - ph) // 2
    else:  # bottom_center (hugging lowest art row for cutouts)
        px = (w - pw) // 2
        if bool((out < 0).any()):
            art_rows = np.where((out >= 0).any(axis=1))[0]
            if len(art_rows) > 0:
                py = min(h - ph - 1, max(1, int(art_rows[-1]) - ph + 3))
            else:
                py = max(1, h - ph - 2)
        else:
            py = max(1, h - ph - 2)

    px = max(0, min(w - pw, px))
    py = max(0, min(h - ph, py))
    mask = patch >= 0
    sub = out[py : py + ph, px : px + pw]
    sub[mask] = patch[mask]
    return out


# ==============================================================================
# ENGINE 3: qwen2.5:7b-instruct Sandboxed Custom Pixel-Art Typography Code Artist
# ==============================================================================

_CODE_ARTIST_SYSTEM = """You are a pixel-art logo designer for r/place.
Write a short, complete, syntactically valid Python function `def draw_flourish(draw, W, H, PAL):` (8 to 20 lines, NO imports, NO comments)
that draws creative pixel-art decorations BEHIND or AROUND a centered wordmark on a (W, H) PIL ImageDraw canvas.
Available variables ONLY:
- `draw`: PIL ImageDraw.Draw on an RGB canvas (already pure white (255,255,255); white strips to transparent afterward!)
- `W`, `H`: integer canvas width and height
- `PAL`: dict mapping color names ('red','orange','yellow','green','blue','light_blue','dark_blue','indigo','purple','pink','magenta','white','black','teal','lavender') to (R,G,B) tuples
- `math`, `random`
Draw crisp shapes with `draw.polygon`, `draw.ellipse`, `draw.rectangle`, `draw.line` — e.g., wavy pennant ribbon banner, tilted crown on top-left, twinkling 4-point pixel stars, crossed swords, dripping neon drops, floating hearts, or curved scroll brackets.
NEVER call `draw.text()` (the stylized lettering is composited on top automatically!).
Return JSON {"code": "def draw_flourish(draw, W, H, PAL):\\n    ..."} ONLY."""


def generate_code_wordmark(
    spec: TypographyStyleSpec,
    w: int,
    h: int,
    theme: str,
    ollama_url: str = "http://localhost:11434",
    text_model: str = "qwen2.5:7b-instruct",
    timeout: float = 30.0,
) -> np.ndarray | None:
    """Engine 3: Ask qwen2.5:7b-instruct to write PIL flourish code around styled lettering."""
    import json as _json

    pal_dict = {name: tuple(int(v) for v in PALETTE_RGB[i]) for i, name in enumerate(COLOR_NAMES)}
    c1_name = COLOR_NAMES[spec.fill_color % NUM_COLORS]
    c2_name = COLOR_NAMES[spec.fill_color_2 % NUM_COLORS]
    prompt = (
        f"Canvas: W={w}, H={h}. Faction theme: '{theme}'. Wordmark: '{spec.text}'. "
        f"Colors to use from PAL: '{c1_name}', '{c2_name}', 'black', 'white', 'yellow'. "
        "Write a concise 10-18 line `def draw_flourish(draw, W, H, PAL):` drawing a bespoke non-rectangular decorative pixel-art backing & flourishes around the centered wordmark."
    )
    code = ""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": text_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": _CODE_ARTIST_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "format": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                    "options": {"temperature": 0.75, "num_predict": 900},
                },
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            code = str(_json.loads(raw).get("code", ""))
    except Exception as exc:
        log.warning("Code wordmark LLM call failed (%s)", exc)
        return None

    # Strip optional markdown code fences
    if "```" in code:
        parts = code.split("```")
        for p in parts:
            if "def draw_flourish" in p:
                code = p.lstrip("python").lstrip("py").strip()
                break

    # Strict AST / keyword safety filter before sandboxed exec
    banned = ("import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "globals", "locals", "getattr", "setattr")
    if "def draw_flourish" not in code or any(b in code for b in banned):
        log.warning("Code wordmark rejected by safety filter.")
        return None

    safe_builtins = {
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "abs": abs,
        "int": int,
        "float": float,
        "round": round,
        "tuple": tuple,
        "list": list,
        "enumerate": enumerate,
        "zip": zip,
    }
    ns: dict = {"__builtins__": safe_builtins, "math": math, "random": random.Random(hash(theme) & 0xFFFFFFFF)}
    try:
        exec(compile(code, "<wordmark_artist>", "exec"), ns, ns)
        fn = ns.get("draw_flourish")
        if not callable(fn):
            return None
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.fontmode = "1"
        fn(draw, w, h, pal_dict)
    except Exception as exc:
        log.warning("Sandboxed wordmark flourish script errored (%s)", exc)
        return None

    rgb = np.asarray(img, dtype=np.uint8)
    bp = rgb_to_palette_indices(rgb).astype(np.int16)
    bp = _flood_strip_background_perceptual(bp, lab_tol=18.0)
    if bool((bp < 0).any()):
        bp = _add_1px_silhouette_outline(bp, BLACK_IDX)

    # Composite Engine 1 styled typography on top of generated flourishes
    hero_spec = TypographyStyleSpec(
        include_text=True,
        text=spec.text,
        casing=spec.casing,
        font_style=spec.font_style,
        size_tier="large",
        color_effect=spec.color_effect,
        border_effect=spec.border_effect if spec.border_effect != "none" else "extrude_3d",
        layout=spec.layout if spec.layout != "vertical_side" else "wavy",
        placement="center",
        fill_color=spec.fill_color,
        fill_color_2=spec.fill_color_2,
        fill_color_3=spec.fill_color_3,
        outline_color=spec.outline_color,
        shadow_color=spec.shadow_color,
        use_ribbon=False,
    )
    return composite_styled_text(bp, hero_spec)
