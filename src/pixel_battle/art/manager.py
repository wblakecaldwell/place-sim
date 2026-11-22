"""Hybrid ArtManager uniting SDXL-Turbo cutouts, 32 flags, micro sprites, fine-art museum frames, ribbons, fractals, clean-space, and 3-engine AI Typography Studio."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
import torch
from PIL import Image

from pixel_battle.art.comics import (
    generate_cartoon_meme,
    generate_comic_strip,
)
from pixel_battle.art.diffusion import LocalPixelDiffusion
from pixel_battle.art.flags import (
    FLAG_BY_KEY,
    generate_national_flag,
    generate_transverse_ribbon,
)
from pixel_battle.art.flux_wordmark import FluxWordmarkGenerator
from pixel_battle.art.procedural import (
    add_silhouette_fortify,
    generate_procedural,
)
from pixel_battle.art.sprites import (
    FINE_ART_CATALOG,
    generate_black_hole,
    generate_clean_space,
    generate_fractal,
    generate_micro_sprite,
    generate_procedural_painting,
    generate_void_monster,
    wrap_museum_frame,
)
from pixel_battle.art.typography import (
    TypographyStyleSpec,
    composite_styled_text,
    generate_code_wordmark,
    render_styled_wordmark_patch,
)
from pixel_battle.config import SimConfig
from pixel_battle.palette import NUM_COLORS, PALETTE_RGB, WHITE_IDX

log = logging.getLogger(__name__)

OCRVerifier = Callable[[Image.Image, str], tuple[bool, str]]


class ArtManager:
    """Thread-safe coordinator for sprites, flags, paintings, ribbons, fractals, clean-space, and 3-engine AI typography."""

    def __init__(self, cfg: SimConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._lock = threading.Lock()
        self.diffusion: LocalPixelDiffusion | None = None
        self.flux: FluxWordmarkGenerator | None = None
        if cfg.use_diffusion:
            self.diffusion = LocalPixelDiffusion(
                base_model=cfg.diffusion_base_model,
                lora_repo=cfg.diffusion_lora_repo,
                lora_weight=cfg.diffusion_lora_weight,
                device=device,
                steps=cfg.diffusion_steps,
            )
        if cfg.use_flux_typography:
            self.flux = FluxWordmarkGenerator(
                model_repo=cfg.flux_model_repo,
                steps=cfg.flux_steps,
            )

    def warmup(self) -> bool:
        ok = False
        if self.diffusion is not None:
            with self._lock:
                ok = self.diffusion.ensure_loaded()
        return ok

    def create_wordmark_banner(
        self,
        spec: TypographyStyleSpec,
        w: int,
        h: int,
        theme: str,
        color_theme: str,
        flux_style_desc: str = "",
        ocr_verifier: OCRVerifier | None = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, str]:
        """Generate a hero wordmark using Engine 2 (FLUX), Engine 3 (Qwen Code), or Engine 1 (Styled)."""
        # Small micro tags (< 46px wide) look sharpest with Engine 1 directly (no slow FLUX/code needed)
        if w < 46 or h < 20:
            canvas = np.full((h, w), -1, dtype=np.int16)
            out = composite_styled_text(canvas, spec)
            if self._is_valid_blueprint(out):
                return out, "styled"

        # Engine 2: FLUX.1-schnell 4-bit (MLX) illustrated logo + Qwen2.5-VL OCR legibility gate
        if spec.engine == "flux" and self.flux is not None and w >= 56 and h >= 24:
            with self._lock:
                bp = self.flux.generate_wordmark(
                    text=spec.text,
                    theme=theme,
                    style_desc=flux_style_desc or f"{spec.font_style} retro pixel logo",
                    color_theme=color_theme,
                    target_w=w,
                    target_h=h,
                    seed=seed,
                )
            if bp is not None and self._is_valid_blueprint(bp):
                if ocr_verifier is not None:
                    rgb = PALETTE_RGB[np.where(bp < 0, WHITE_IDX, bp)]
                    check_img = Image.fromarray(rgb, mode="RGB").resize(
                        (w * 4, h * 4), Image.Resampling.NEAREST
                    )
                    ok, note = ocr_verifier(check_img, spec.text)
                    log.info("FLUX wordmark OCR check for '%s': %s", spec.text, note)
                    if ok:
                        return bp, "flux"
                else:
                    return bp, "flux"

        # Engine 3: qwen2.5:7b-instruct sandboxed custom PIL flourish script + styled multi-font letters
        if spec.engine in ("code", "flux") and w >= 52 and h >= 22:
            bp_code = generate_code_wordmark(
                spec=spec,
                w=w,
                h=h,
                theme=theme,
                ollama_url=self.cfg.ollama_url,
                text_model=self.cfg.text_model,
            )
            if bp_code is not None and self._is_valid_blueprint(bp_code):
                return bp_code, "code"

        # Engine 1: Standalone floating styled multi-font wordmark cutout (100% crisp, zero rectangular box)
        canvas = np.full((h, w), -1, dtype=np.int16)
        hero = TypographyStyleSpec(
            include_text=True,
            text=spec.text,
            casing=spec.casing,
            font_style=spec.font_style,
            size_tier="large" if h >= 28 else ("medium" if h >= 18 else "tiny"),
            color_effect=spec.color_effect,
            border_effect=spec.border_effect if spec.border_effect != "none" else "extrude_3d",
            layout=spec.layout if spec.layout != "vertical_side" else "wavy",
            placement="center",
            fill_color=spec.fill_color,
            fill_color_2=spec.fill_color_2,
            fill_color_3=spec.fill_color_3,
            outline_color=spec.outline_color,
            shadow_color=spec.shadow_color,
            use_ribbon=spec.use_ribbon,
            ribbon_color=spec.ribbon_color,
            engine="styled",
        )
        out = composite_styled_text(canvas, hero)
        if self._is_valid_blueprint(out):
            return out, "styled"
        patch = render_styled_wordmark_patch(hero, max_w=w, max_h=h)
        if patch is not None:
            ph, pw = min(h, patch.shape[0]), min(w, patch.shape[1])
            canvas[:ph, :pw] = patch[:ph, :pw]
        return canvas, "styled"

    def create_blueprint(
        self,
        art_mode: str,
        prompt: str,
        w: int,
        h: int,
        primary: int,
        secondary: int,
        accent: int,
        label: str = "",
        theme: str = "",
        color_theme: str = "",
        typography: TypographyStyleSpec | None = None,
        flux_style_desc: str = "",
        ocr_verifier: OCRVerifier | None = None,
        live_crop_rgb: np.ndarray | None = None,
        seed: int = 42,
        extra_key: str = "",
        ribbon_axis: str = "h",
    ) -> tuple[np.ndarray, str]:
        """Create an initial [h, w] int16 blueprint and return (blueprint, text_engine_used)."""
        # Dedicated art modes
        if art_mode == "flag":
            fkey = extra_key or prompt.split()[0].lower()
            if fkey not in FLAG_BY_KEY:
                for cand_k in FLAG_BY_KEY:
                    if cand_k in prompt.lower() or cand_k in theme.lower():
                        fkey = cand_k
                        break
            bp = generate_national_flag(fkey, w, h, add_border=(min(w, h) >= 14))
            return bp, "none"

        if art_mode == "micro_sprite":
            bp, _ = generate_micro_sprite(sprite_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return bp, "none"

        if art_mode == "comic_strip":
            bp, _ = generate_comic_strip(comic_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return bp, "none"

        if art_mode == "cartoon_meme":
            bp, _ = generate_cartoon_meme(meme_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return bp, "none"

        if art_mode == "clean_space":
            style = "corner_ticks" if (seed % 2 == 0) else "pure_white"
            return generate_clean_space(w, h, style=style, seed=seed), "none"

        if art_mode == "ribbon":
            thick = min(w, h) if ribbon_axis in ("h", "v") else max(5, min(11, extra_key.count(":") and 7 or 7))
            length = max(w, h)
            if ribbon_axis in ("d1", "d2"):
                thick = max(5, min(11, int(extra_key.split("|")[1]) if "|" in extra_key else 7))
                pkey = extra_key.split("|")[0]
            else:
                pkey = extra_key or "rainbow"
            return generate_transverse_ribbon(pkey, ribbon_axis, length=length, thickness=thick), "none"

        if art_mode == "fractal":
            fkind = extra_key or ("sierpinski", "mandelbrot", "carpet", "diamond")[seed % 4]
            return generate_fractal(fkind, w, h, seed=seed), "none"

        if art_mode == "black_hole":
            return generate_black_hole(w, h, seed=seed), "none"

        if art_mode == "void_monster":
            return generate_void_monster(w, h, seed=seed), "none"

        if art_mode == "painting":
            spec = next((s for s in FINE_ART_CATALOG if s.key == extra_key), FINE_ART_CATALOG[seed % len(FINE_ART_CATALOG)])
            bp_art: np.ndarray | None = None
            if self.diffusion is not None and min(w, h) >= 26:
                with self._lock:
                    bp_art = self.diffusion.generate_txt2img(
                        prompt=f"pixel art painting of {spec.prompt}, full canvas oil masterpiece",
                        target_w=w,
                        target_h=h,
                        strip_bg=False,
                        color_theme=spec.color_theme,
                        seed=seed,
                    )
                if bp_art is not None and self._is_valid_blueprint(bp_art, art_mode="painting"):
                    bp_art = wrap_museum_frame(bp_art)
                else:
                    bp_art = None
            if bp_art is None:
                bp_art = generate_procedural_painting(spec, w, h, seed=seed)
            return bp_art, "none"

        # --- Original Wordmark Banner / Diffusion Cutout / Organic Procedural ---
        if art_mode == "banner" and typography is not None:
            return self.create_wordmark_banner(
                spec=typography,
                w=w,
                h=h,
                theme=theme or label,
                color_theme=color_theme,
                flux_style_desc=flux_style_desc,
                ocr_verifier=ocr_verifier,
                seed=seed,
            )

        bp: np.ndarray | None = None
        if art_mode == "diffusion" and self.diffusion is not None:
            with self._lock:
                bp = self.diffusion.generate_txt2img(
                    prompt=prompt,
                    target_w=w,
                    target_h=h,
                    strip_bg=True,
                    color_theme=color_theme,
                    seed=seed,
                )
            if bp is not None and not self._is_valid_blueprint(bp, art_mode=art_mode):
                bp = None

        if bp is None:
            bp = generate_procedural(
                prompt=prompt,
                w=w,
                h=h,
                primary=primary,
                secondary=secondary,
                accent=accent,
                label=label,
                banner_text="",
                stage=1,
                seed=seed,
            )

        # Apply faction typography when the canvas region is large enough to stay legible
        if typography is not None and typography.include_text and max(w, h) >= 34:
            bp = composite_styled_text(bp, typography)
            return bp, "styled"
        return bp, "none"

    def morph_blueprint(
        self,
        art_mode: str,
        new_prompt: str,
        live_crop_rgb: np.ndarray,
        target_w: int,
        target_h: int,
        primary: int,
        secondary: int,
        accent: int,
        label: str = "",
        theme: str = "",
        color_theme: str = "",
        typography: TypographyStyleSpec | None = None,
        ocr_verifier: OCRVerifier | None = None,
        stage: int = 2,
        strength: float = 0.82,
        seed: int = 42,
        extra_key: str = "",
        ribbon_axis: str = "h",
    ) -> np.ndarray:
        """Evolve/morph a faction's artwork while preserving art_mode & unique AI typography."""
        if art_mode in (
            "flag",
            "micro_sprite",
            "comic_strip",
            "cartoon_meme",
            "clean_space",
            "ribbon",
            "fractal",
            "black_hole",
            "void_monster",
            "painting",
        ):
            bp, _ = self.create_blueprint(
                art_mode=art_mode,
                prompt=new_prompt,
                w=target_w,
                h=target_h,
                primary=primary,
                secondary=secondary,
                accent=accent,
                label=label,
                theme=theme,
                color_theme=color_theme,
                typography=typography,
                ocr_verifier=ocr_verifier,
                seed=seed,
                extra_key=extra_key,
                ribbon_axis=ribbon_axis,
            )
            return bp

        if art_mode == "banner" and typography is not None:
            bp, _ = self.create_wordmark_banner(
                spec=typography,
                w=target_w,
                h=target_h,
                theme=theme or label,
                color_theme=color_theme,
                ocr_verifier=ocr_verifier,
                seed=seed,
            )
            return bp

        bp: np.ndarray | None = None
        if art_mode == "diffusion" and self.diffusion is not None:
            with self._lock:
                bp = self.diffusion.morph_img2img(
                    prompt=new_prompt,
                    init_rgb=live_crop_rgb,
                    target_w=target_w,
                    target_h=target_h,
                    strength=strength,
                    strip_bg=True,
                    color_theme=color_theme,
                    seed=seed,
                )
            if bp is not None and not self._is_valid_blueprint(bp, art_mode=art_mode):
                bp = None

        if bp is None:
            bp = generate_procedural(
                prompt=new_prompt,
                w=target_w,
                h=target_h,
                primary=primary,
                secondary=secondary,
                accent=accent,
                label=label,
                banner_text="",
                stage=stage,
                seed=seed,
            )

        if typography is not None and typography.include_text and max(target_w, target_h) >= 34:
            bp = composite_styled_text(bp, typography)
        return bp

    def fortify_blueprint(
        self,
        bp: np.ndarray,
        primary: int,
        secondary: int,
        accent: int,
        typography: TypographyStyleSpec | None = None,
        already_stamped: bool = False,
        art_mode: str = "procedural",
    ) -> np.ndarray:
        """Add ONE clean organic die-cut fortify aura (never double-stamping text or framing empty space/paintings)."""
        if art_mode in ("clean_space", "painting", "ribbon", "micro_sprite", "comic_strip"):
            return bp.copy()
        framed = add_silhouette_fortify(bp, primary % NUM_COLORS, secondary % NUM_COLORS, accent % NUM_COLORS)
        if (not already_stamped) and typography is not None and typography.include_text and max(bp.shape) >= 34:
            framed = composite_styled_text(framed, typography)
        return framed

    @staticmethod
    def _is_valid_blueprint(bp: np.ndarray, art_mode: str = "diffusion") -> bool:
        """Sanity-check against blank/monochrome outputs while accepting clean_space and diagonal ribbons."""
        valid = bp[bp >= 0]
        if art_mode == "clean_space":
            return valid.size >= bp.size * 0.20
        if art_mode == "ribbon":
            return valid.size >= max(24, int(bp.size * 0.015))
        if valid.size < bp.size * 0.08:
            return False
        non_white = valid[valid != WHITE_IDX]
        if non_white.size < bp.size * 0.04:
            return False
        return len(np.unique(valid)) >= 2
