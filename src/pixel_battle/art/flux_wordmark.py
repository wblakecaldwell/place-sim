"""Engine 2: Local 4-bit FLUX.1-schnell (Apple Silicon MLX via mflux) illustrated wordmark generator."""

from __future__ import annotations

import logging
import threading

import numpy as np

from pixel_battle.art.diffusion import _crisp_downsample_to_palette

log = logging.getLogger(__name__)


class FluxWordmarkGenerator:
    """Lazy-loaded 4-bit FLUX.1-schnell on MLX for illustrated pixel-art typography logos."""

    def __init__(
        self,
        model_repo: str = "dhairyashil/FLUX.1-schnell-mflux-4bit",
        steps: int = 4,
    ) -> None:
        self.model_repo = model_repo
        self.steps = steps
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                from mflux.models.common.config.model_config import ModelConfig
                from mflux.models.flux.variants.txt2img.flux import Flux1

                log.info("Loading local 4-bit FLUX.1-schnell via MLX (%s)...", self.model_repo)
                self._model = Flux1(
                    model_config=ModelConfig.from_name(
                        model_name=self.model_repo, base_model="schnell"
                    ),
                )
                log.info("FLUX.1-schnell 4-bit (MLX) ready for illustrated typography!")
                return True
            except Exception as exc:
                log.warning("Failed to load FLUX.1-schnell MLX (%s); falling back to styled/code typography.", exc)
                self._failed = True
                return False

    def generate_wordmark(
        self,
        text: str,
        theme: str,
        style_desc: str,
        color_theme: str,
        target_w: int,
        target_h: int,
        seed: int = 42,
    ) -> np.ndarray | None:
        """Generate an illustrated pixel-art wordmark [target_h, target_w] int16 cutout."""
        if not self.ensure_loaded():
            return None

        # Match aspect ratio cleanly to multiples of 64 for FLUX
        gen_w = 512
        gen_h = 256 if target_w >= target_h * 1.4 else 384

        prompt = (
            f'16-bit retro pixel art game logo typography clearly spelling "{text}", '
            f"{style_desc}, inspired by {theme}, bold vivid {color_theme}, "
            f"chunky crisp pixel lettering with clean black outlines, flat saturated colors, "
            f"isolated floating cutout on pure solid white background, no frame, no rectangular box"
        )
        try:
            with self._lock:
                res = self._model.generate_image(
                    seed=int(seed) & 0x7FFFFFFF,
                    prompt=prompt,
                    num_inference_steps=max(2, self.steps),
                    width=gen_w,
                    height=gen_h,
                )
            pil_img = res.image
            return _crisp_downsample_to_palette(
                pil_img,
                target_w=target_w,
                target_h=target_h,
                strip_bg=True,
                add_outline=True,
            )
        except Exception as exc:
            log.warning("FLUX wordmark generation failed (%s)", exc)
            return None
