"""Local SDXL-Turbo + nerijs/pixel-art-xl pixel-art generator and organic cutout img2img morpher."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageFilter

from pixel_battle.palette import BLACK_IDX, PALETTE_LAB, rgb_to_palette_indices

log = logging.getLogger(__name__)


def _crisp_downsample_to_palette(
    img: Image.Image,
    target_w: int,
    target_h: int,
    strip_bg: bool = True,
    add_outline: bool = True,
) -> np.ndarray:
    """Downsample a 512x512 diffusion image to crisp [target_h, target_w] palette grid.

    For cutout sprites (strip_bg=True), flood-strips border-connected background pixels
    using perceptual CIE-LAB tolerance so sprites float organically with NO rectangular box,
    then optionally adds a crisp 1px silhouette outline.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    src_h, src_w, _ = arr.shape

    ys = np.linspace(0, src_h, target_h + 1).astype(np.int32)
    xs = np.linspace(0, src_w, target_w + 1).astype(np.int32)

    out_rgb = np.empty((target_h, target_w, 3), dtype=np.uint8)
    for j in range(target_h):
        y0, y1 = ys[j], max(ys[j] + 1, ys[j + 1])
        cy0 = y0 + (y1 - y0) // 4
        cy1 = max(cy0 + 1, y1 - (y1 - y0) // 4)
        for i in range(target_w):
            x0, x1 = xs[i], max(xs[i] + 1, xs[i + 1])
            cx0 = x0 + (x1 - x0) // 4
            cx1 = max(cx0 + 1, x1 - (x1 - x0) // 4)
            cell = arr[cy0:cy1, cx0:cx1].reshape(-1, 3)
            out_rgb[j, i] = np.median(cell, axis=0).astype(np.uint8)

    bp = rgb_to_palette_indices(out_rgb).astype(np.int16)

    if strip_bg:
        bp = _flood_strip_background_perceptual(bp)
        if add_outline and bool((bp < 0).any()):
            bp = _add_1px_silhouette_outline(bp, BLACK_IDX)

    return bp


def _flood_strip_background_perceptual(bp: np.ndarray, lab_tol: float = 24.0) -> np.ndarray:
    """Perceptual CIE-LAB border flood-fill to turn sprites into organic transparent-outside cutouts."""
    h, w = bp.shape
    border = np.concatenate([bp[0, :], bp[-1, :], bp[:, 0], bp[:, -1]])
    vals, counts = np.unique(border, return_counts=True)
    order = np.argsort(counts)[::-1]
    dom_col = int(vals[order[0]])
    dom_lab = PALETTE_LAB[dom_col]

    # Collect all palette indices visually close to the dominant border color (handles shaded/dithered bgs)
    dists = np.linalg.norm(PALETTE_LAB - dom_lab[None, :], axis=1)
    bg_set = set(int(i) for i in np.where(dists <= lab_tol)[0])
    # Also include 2nd most common border color if top-2 cover >= 55% of border and they are somewhat related
    border_bg_frac = float(np.isin(border, list(bg_set)).mean())
    if border_bg_frac < 0.42 and len(order) >= 2:
        sec_col = int(vals[order[1]])
        sec_dists = np.linalg.norm(PALETTE_LAB - PALETTE_LAB[sec_col][None, :], axis=1)
        cand_set = bg_set | set(int(i) for i in np.where(sec_dists <= 14.0)[0])
        if float(np.isin(border, list(cand_set)).mean()) >= 0.50:
            bg_set = cand_set
            border_bg_frac = float(np.isin(border, list(bg_set)).mean())

    if border_bg_frac < 0.35:
        # Even if a scene fills the box, gently feather the 4 rectangular corners into an organic shape
        return _organic_feather_edges(bp)

    is_bg_color = np.isin(bp, list(bg_set))
    out = bp.copy()
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        if is_bg_color[0, x]:
            q.append((0, x))
            visited[0, x] = True
        if is_bg_color[h - 1, x]:
            q.append((h - 1, x))
            visited[h - 1, x] = True
    for y in range(h):
        if is_bg_color[y, 0] and not visited[y, 0]:
            q.append((y, 0))
            visited[y, 0] = True
        if is_bg_color[y, w - 1] and not visited[y, w - 1]:
            q.append((y, w - 1))
            visited[y, w - 1] = True

    stripped = 0
    while q:
        y, x = q.popleft()
        out[y, x] = -1
        stripped += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg_color[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    # Safety: if stripping ate > 82% of pixels, fall back to organic feathered edge instead of blank
    if stripped > 0.82 * h * w:
        return _organic_feather_edges(bp)
    # Remove tiny 1-2px floating dust specks left in transparent margin
    out = _remove_isolated_specks(out)
    return out


def _organic_feather_edges(bp: np.ndarray) -> np.ndarray:
    """Turn a full-bleed square crop into a soft organic blob silhouette."""
    h, w = bp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - (w - 1) * 0.5) / max(1.0, w * 0.48)
    ny = (yy - (h - 1) * 0.5) / max(1.0, h * 0.48)
    ang = np.arctan2(ny, nx)
    r = np.sqrt(nx * nx + ny * ny)
    # Gentle multi-lobe organic boundary
    boundary = 0.92 + 0.08 * np.sin(3.0 * ang + 0.7) + 0.05 * np.cos(5.0 * ang - 0.4)
    out = bp.copy()
    out[r > boundary] = -1
    return out


def _remove_isolated_specks(bp: np.ndarray) -> np.ndarray:
    valid = bp >= 0
    h, w = bp.shape
    nbr = np.zeros((h, w), dtype=np.int8)
    nbr[1:, :] += valid[:-1, :]
    nbr[:-1, :] += valid[1:, :]
    nbr[:, 1:] += valid[:, :-1]
    nbr[:, :-1] += valid[:, 1:]
    out = bp.copy()
    out[valid & (nbr == 0)] = -1
    return out


def _add_1px_silhouette_outline(bp: np.ndarray, outline_col: int = BLACK_IDX) -> np.ndarray:
    """Add a crisp 1px dark silhouette outline around an organic transparent cutout."""
    valid = bp >= 0
    h, w = bp.shape
    dil = np.zeros((h, w), dtype=bool)
    dil[1:, :] |= valid[:-1, :]
    dil[:-1, :] |= valid[1:, :]
    dil[:, 1:] |= valid[:, :-1]
    dil[:, :-1] |= valid[:, 1:]
    out = bp.copy()
    out[dil & (~valid)] = outline_col
    return out


class LocalPixelDiffusion:
    """Lazy-loaded SDXL-Turbo + nerijs/pixel-art-xl pipeline sharing weights for txt2img & img2img."""

    def __init__(
        self,
        base_model: str = "stabilityai/sdxl-turbo",
        lora_repo: str = "nerijs/pixel-art-xl",
        lora_weight: str = "pixel-art-xl.safetensors",
        device: torch.device | None = None,
        steps: int = 6,
    ) -> None:
        self.base_model = base_model
        self.lora_repo = lora_repo
        self.lora_weight = lora_weight
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.steps = steps
        self._txt2img = None
        self._img2img = None
        self._failed = False

    def ensure_loaded(self) -> bool:
        if self._txt2img is not None:
            return True
        if self._failed:
            return False

        try:
            from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

            log.info("Loading local pixel-art diffusion (%s + %s) on %s...", self.base_model, self.lora_repo, self.device)
            dtype = torch.float16 if self.device.type == "mps" else torch.float32
            variant = "fp16" if dtype == torch.float16 else None

            pipe = AutoPipelineForText2Image.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                variant=variant,
                use_safetensors=True,
            )
            if hasattr(pipe, "vae") and pipe.vae is not None and hasattr(pipe.vae, "config"):
                pipe.vae.config.force_upcast = True

            pipe.to(self.device)
            pipe.set_progress_bar_config(disable=True)

            pipe.load_lora_weights(self.lora_repo, weight_name=self.lora_weight)
            pipe.fuse_lora(lora_scale=1.0)

            self._txt2img = pipe
            self._img2img = AutoPipelineForImage2Image.from_pipe(pipe)
            self._img2img.set_progress_bar_config(disable=True)
            log.info("Local pixel-art diffusion ready on %s.", self.device)
            return True
        except Exception as exc:
            log.warning("Failed to load local diffusion model (%s). Falling back to procedural art.", exc)
            self._failed = True
            return False

    def _build_prompt(self, prompt: str, color_theme: str = "", cutout: bool = True) -> str:
        clean = prompt.strip().rstrip(".")
        theme_part = f", vibrant {color_theme}" if color_theme else ""
        bg_part = ", isolated sprite on pure solid white background, no frame, no box, no background" if cutout else ""
        return (
            f"pixel art, {clean}{theme_part}{bg_part}, crisp 16-bit retro game sprite, "
            f"saturated flat colors, clean black outlines, iconic silhouette, centered"
        )

    def generate_txt2img(
        self,
        prompt: str,
        target_w: int,
        target_h: int,
        strip_bg: bool = True,
        color_theme: str = "",
        seed: int = 42,
    ) -> np.ndarray | None:
        if not self.ensure_loaded():
            return None
        gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        full_prompt = self._build_prompt(prompt, color_theme=color_theme, cutout=strip_bg)
        with torch.inference_mode():
            res = self._txt2img(
                prompt=full_prompt,
                num_inference_steps=max(4, self.steps),
                guidance_scale=0.0,
                width=512,
                height=512,
                generator=gen,
            )
        img: Image.Image = res.images[0]
        return _crisp_downsample_to_palette(img, target_w, target_h, strip_bg=strip_bg, add_outline=strip_bg)

    def morph_img2img(
        self,
        prompt: str,
        init_rgb: np.ndarray,
        target_w: int,
        target_h: int,
        strength: float = 0.82,
        strip_bg: bool = True,
        color_theme: str = "",
        seed: int = 42,
    ) -> np.ndarray | None:
        """Morph an existing live canvas crop into an evolved floating pixel-art cutout."""
        if not self.ensure_loaded():
            return None
        init_img = Image.fromarray(init_rgb.astype(np.uint8), mode="RGB").resize(
            (512, 512), Image.Resampling.NEAREST
        ).filter(ImageFilter.GaussianBlur(radius=6))
        strength = float(max(0.45, min(0.95, strength)))
        total_steps = max(self.steps + 2, int(np.ceil(5.0 / strength)))
        gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        full_prompt = self._build_prompt(prompt, color_theme=color_theme, cutout=strip_bg)
        with torch.inference_mode():
            res = self._img2img(
                prompt=full_prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=total_steps,
                guidance_scale=0.0,
                generator=gen,
            )
        img: Image.Image = res.images[0]
        return _crisp_downsample_to_palette(img, target_w, target_h, strip_bg=strip_bg, add_outline=strip_bg)
