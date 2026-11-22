"""AI-first ArtManager uniting MLX FLUX.1-schnell Pixel-Art LoRA, LLM-written pixel code, multilingual typography, flags, ribbons, Void variants, and offline fallbacks."""

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
from pixel_battle.art.flux_art import FluxPixelArtGenerator, run_sandboxed_pixel_code
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
from pixel_battle.palette import NUM_COLORS, PALETTE_RGB, WHITE_IDX, remap_indices_to_allowed

log = logging.getLogger(__name__)

OCRVerifier = Callable[[Image.Image, str], tuple[bool, str]]


class ArtManager:
    """Thread-safe coordinator for AI-generated pixel art, LLM-coded geometric chaos, flags, ribbons, and multilingual typography."""

    def __init__(self, cfg: SimConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._lock = threading.Lock()
        self.diffusion: LocalPixelDiffusion | None = None
        self.flux_art: FluxPixelArtGenerator | None = None

        if cfg.use_flux_art or cfg.use_flux_typography:
            self.flux_art = FluxPixelArtGenerator(
                model_repo=cfg.flux_model_repo,
                lora_repo=cfg.flux_pixel_lora_repo,
                lora_file=cfg.flux_pixel_lora_file,
                lora_scale=cfg.flux_pixel_lora_scale,
                steps=cfg.flux_steps,
            )
        if cfg.use_diffusion:
            self.diffusion = LocalPixelDiffusion(
                base_model=cfg.diffusion_base_model,
                lora_repo=cfg.diffusion_lora_repo,
                lora_weight=cfg.diffusion_lora_weight,
                device=device,
                steps=cfg.diffusion_steps,
            )

    def warmup(self) -> bool:
        ok = False
        if self.flux_art is not None:
            with self._lock:
                ok = self.flux_art.ensure_loaded()
        elif self.diffusion is not None:
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
        allowed_colors: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[np.ndarray, str]:
        """Generate a multilingual or stylized hero wordmark using FLUX Pixel-Art LoRA or Engine 1 Styled Typography."""
        # Non-ASCII / CJK or micro tags render crisply via Engine 1 Unicode pixel typography directly
        has_non_ascii = any(ord(ch) > 127 for ch in spec.text)
        if w < 46 or h < 20 or has_non_ascii:
            canvas = np.full((h, w), -1, dtype=np.int16)
            out = composite_styled_text(canvas, spec)
            out = remap_indices_to_allowed(out, allowed_colors)
            if self._is_valid_blueprint(out):
                return out, "styled"

        # Engine 2: FLUX.1-schnell 4-bit + Pixel-Art LoRA illustrated wordmark + VLM OCR gate
        if spec.engine == "flux" and self.flux_art is not None and w >= 56 and h >= 24:
            wm_prompt = (
                f'retro pixel art game logo typography clearly spelling "{spec.text}", '
                f"{flux_style_desc or spec.font_style + ' bold pixel logo'}, inspired by {theme}, "
                f"chunky legible pixel lettering with thick black outline"
            )
            with self._lock:
                bp = self.flux_art.generate_pixel_art(
                    prompt=wm_prompt,
                    target_w=w,
                    target_h=h,
                    strip_bg=True,
                    add_outline=True,
                    allowed_colors=allowed_colors,
                    color_theme=color_theme,
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

        # Engine 1: Standalone floating styled multi-font wordmark cutout (crisp, fluid contour, zero rectangle)
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
        out = remap_indices_to_allowed(out, allowed_colors)
        if self._is_valid_blueprint(out):
            return out, "styled"
        patch = render_styled_wordmark_patch(hero, max_w=w, max_h=h)
        if patch is not None:
            ph, pw = min(h, patch.shape[0]), min(w, patch.shape[1])
            canvas[:ph, :pw] = patch[:ph, :pw]
        return remap_indices_to_allowed(canvas, allowed_colors), "styled"

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
        allowed_colors: tuple[int, ...] | list[int] | None = None,
        fluid_cutout: bool = True,
        ai_code: str = "",
    ) -> tuple[np.ndarray, str]:
        """Create an initial [h, w] int16 blueprint strictly quantized to allowed_colors."""
        # 1. Dedicated world primitives (Flags, Clean Space, Ribbons, Void variants)
        if art_mode == "flag":
            fkey = extra_key or prompt.split()[0].lower()
            if fkey not in FLAG_BY_KEY:
                for cand_k in FLAG_BY_KEY:
                    if cand_k in prompt.lower() or cand_k in theme.lower():
                        fkey = cand_k
                        break
            bp = generate_national_flag(fkey, w, h, add_border=(min(w, h) >= 14))
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "clean_space":
            style = "corner_ticks" if (seed % 2 == 0) else "pure_white"
            bp = generate_clean_space(w, h, style=style, seed=seed)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "ribbon":
            thick = min(w, h) if ribbon_axis in ("h", "v") else 7
            length = max(w, h)
            if ribbon_axis in ("d1", "d2"):
                thick = max(5, min(11, int(extra_key.split("|")[1]) if "|" in extra_key else 7))
                pkey = extra_key.split("|")[0]
            else:
                pkey = extra_key or "rainbow"
            bp = generate_transverse_ribbon(pkey, ribbon_axis, length=length, thickness=thick)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "fractal":
            fkind = extra_key or ("sierpinski", "mandelbrot", "carpet", "diamond")[seed % 4]
            bp = generate_fractal(fkind, w, h, seed=seed)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "black_hole":
            return remap_indices_to_allowed(generate_black_hole(w, h, seed=seed), allowed_colors), "none"

        if art_mode == "void_monster":
            return remap_indices_to_allowed(generate_void_monster(w, h, seed=seed), allowed_colors), "none"

        # 2. AI-First Visionless Geometric Chaos via LLM-written sandboxed code
        if art_mode == "ai_code":
            bp_code: np.ndarray | None = None
            if ai_code:
                bp_code = run_sandboxed_pixel_code(ai_code, w, h, allowed_colors=allowed_colors)
            if bp_code is None and self.flux_art is not None:
                with self._lock:
                    bp_code = self.flux_art.generate_pixel_art(
                        prompt=f"abstract geometric pixel art pattern, {prompt}, no characters, pure geometry",
                        target_w=w,
                        target_h=h,
                        strip_bg=fluid_cutout,
                        add_outline=False,
                        allowed_colors=allowed_colors,
                        color_theme=color_theme,
                        seed=seed,
                    )
            if bp_code is None or not self._is_valid_blueprint(bp_code, art_mode="fractal"):
                bp_code = generate_fractal(("sierpinski", "mandelbrot", "carpet", "diamond")[seed % 4], w, h, seed=seed)
            return remap_indices_to_allowed(bp_code, allowed_colors), "none"

        # 3. AI-First FLUX Pixel-Art LoRA (fluid cutouts, comics, streamer raids, paintings, pop culture)
        if art_mode in ("flux_art", "flux_comic", "streamer_raid", "painting"):
            is_rect_art = art_mode in ("flux_comic", "painting")
            strip = (not is_rect_art) and fluid_cutout
            bp_ai: np.ndarray | None = None
            if self.flux_art is not None:
                p_use = prompt
                if art_mode == "painting":
                    p_use = f"classic oil painting masterpiece in 16-bit pixel art: {prompt}"
                with self._lock:
                    bp_ai = self.flux_art.generate_pixel_art(
                        prompt=p_use,
                        target_w=w,
                        target_h=h,
                        strip_bg=strip,
                        add_outline=strip,
                        allowed_colors=allowed_colors,
                        color_theme=color_theme,
                        seed=seed,
                    )
                if bp_ai is not None and not self._is_valid_blueprint(bp_ai, art_mode=art_mode):
                    bp_ai = None
            if bp_ai is not None:
                if art_mode == "painting":
                    bp_ai = wrap_museum_frame(bp_ai)
                if typography is not None and typography.include_text and max(w, h) >= 42 and art_mode != "flux_comic":
                    bp_ai = composite_styled_text(bp_ai, typography)
                return remap_indices_to_allowed(bp_ai, allowed_colors), "flux"

            # Offline/error fallback
            if art_mode == "painting":
                spec = next((s for s in FINE_ART_CATALOG if s.key == extra_key), FINE_ART_CATALOG[seed % len(FINE_ART_CATALOG)])
                return remap_indices_to_allowed(generate_procedural_painting(spec, w, h, seed=seed), allowed_colors), "none"
            if art_mode == "flux_comic":
                bp_fb, _ = generate_comic_strip(extra_key or None, w, h, seed=seed)
                return remap_indices_to_allowed(bp_fb, allowed_colors), "none"
            bp_fb = generate_procedural(
                prompt=prompt, w=w, h=h, primary=primary, secondary=secondary, accent=accent,
                label=label, banner_text="", stage=1, seed=seed,
            )
            return remap_indices_to_allowed(bp_fb, allowed_colors), "none"

        # 4. Multilingual & Stylized Wordmark Banners
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
                allowed_colors=allowed_colors,
            )

        # 5. Legacy / offline template fallbacks (micro_sprite, comic_strip, cartoon_meme, diffusion, procedural)
        if art_mode == "micro_sprite":
            bp, _ = generate_micro_sprite(sprite_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "comic_strip":
            bp, _ = generate_comic_strip(comic_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        if art_mode == "cartoon_meme":
            bp, _ = generate_cartoon_meme(meme_key=extra_key or None, target_w=w, target_h=h, seed=seed)
            return remap_indices_to_allowed(bp, allowed_colors), "none"

        bp: np.ndarray | None = None
        if art_mode == "diffusion":
            if self.flux_art is not None:
                with self._lock:
                    bp = self.flux_art.generate_pixel_art(
                        prompt=prompt, target_w=w, target_h=h, strip_bg=True, add_outline=True,
                        allowed_colors=allowed_colors, color_theme=color_theme, seed=seed,
                    )
            elif self.diffusion is not None:
                with self._lock:
                    bp = self.diffusion.generate_txt2img(
                        prompt=prompt, target_w=w, target_h=h, strip_bg=True,
                        color_theme=color_theme, seed=seed,
                    )
            if bp is not None and not self._is_valid_blueprint(bp, art_mode=art_mode):
                bp = None

        if bp is None:
            bp = generate_procedural(
                prompt=prompt, w=w, h=h, primary=primary, secondary=secondary, accent=accent,
                label=label, banner_text="", stage=1, seed=seed,
            )

        if typography is not None and typography.include_text and max(w, h) >= 34:
            bp = composite_styled_text(bp, typography)
            return remap_indices_to_allowed(bp, allowed_colors), "styled"
        return remap_indices_to_allowed(bp, allowed_colors), "none"

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
        allowed_colors: tuple[int, ...] | list[int] | None = None,
    ) -> np.ndarray:
        """Evolve/morph a faction's artwork while honoring the currently unlocked palette."""
        if art_mode in (
            "flag", "micro_sprite", "comic_strip", "cartoon_meme", "clean_space",
            "ribbon", "fractal", "black_hole", "void_monster", "painting",
            "flux_art", "flux_comic", "streamer_raid", "ai_code",
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
                allowed_colors=allowed_colors,
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
                allowed_colors=allowed_colors,
            )
            return bp

        bp: np.ndarray | None = None
        if art_mode == "diffusion":
            if self.flux_art is not None:
                with self._lock:
                    bp = self.flux_art.generate_pixel_art(
                        prompt=new_prompt, target_w=target_w, target_h=target_h,
                        strip_bg=True, add_outline=True, allowed_colors=allowed_colors,
                        color_theme=color_theme, seed=seed,
                    )
            elif self.diffusion is not None:
                with self._lock:
                    bp = self.diffusion.morph_img2img(
                        prompt=new_prompt, init_rgb=live_crop_rgb, target_w=target_w, target_h=target_h,
                        strength=strength, strip_bg=True, color_theme=color_theme, seed=seed,
                    )
            if bp is not None and not self._is_valid_blueprint(bp, art_mode=art_mode):
                bp = None

        if bp is None:
            bp = generate_procedural(
                prompt=new_prompt, w=target_w, h=target_h, primary=primary, secondary=secondary,
                accent=accent, label=label, banner_text="", stage=stage, seed=seed,
            )

        if typography is not None and typography.include_text and max(target_w, target_h) >= 34:
            bp = composite_styled_text(bp, typography)
        return remap_indices_to_allowed(bp, allowed_colors)

    def fortify_blueprint(
        self,
        bp: np.ndarray,
        primary: int,
        secondary: int,
        accent: int,
        typography: TypographyStyleSpec | None = None,
        already_stamped: bool = False,
        art_mode: str = "procedural",
        allowed_colors: tuple[int, ...] | list[int] | None = None,
    ) -> np.ndarray:
        """Add one clean organic die-cut fortify aura (never double-stamping text or framing comics/paintings)."""
        if art_mode in ("clean_space", "painting", "ribbon", "micro_sprite", "comic_strip", "flux_comic"):
            return remap_indices_to_allowed(bp.copy(), allowed_colors)
        framed = add_silhouette_fortify(bp, primary % NUM_COLORS, secondary % NUM_COLORS, accent % NUM_COLORS)
        if (not already_stamped) and typography is not None and typography.include_text and max(bp.shape) >= 34:
            framed = composite_styled_text(framed, typography)
        return remap_indices_to_allowed(framed, allowed_colors)

    @staticmethod
    def _is_valid_blueprint(bp: np.ndarray, art_mode: str = "flux_art") -> bool:
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
