"""Unified local MLX 4-bit FLUX.1-schnell + Pixel-Art LoRA generator and sandboxed LLM pixel-code runner."""

from __future__ import annotations

import ast
import concurrent.futures
import logging
import math
import re
import threading
from pathlib import Path

import numpy as np

from pixel_battle.art.diffusion import _crisp_downsample_to_palette
from pixel_battle.palette import remap_indices_to_allowed

log = logging.getLogger(__name__)


class FluxPixelArtGenerator:
    """Lazy-loaded 4-bit FLUX.1-schnell + Pixel-Art LoRA on Apple Silicon MLX.

    All MLX calls execute on a dedicated single worker thread so MLX thread-local
    streams stay valid regardless of which caller thread invokes warmup or generation.
    """

    def __init__(
        self,
        model_repo: str = "dhairyashil/FLUX.1-schnell-mflux-4bit",
        lora_repo: str = "hliu143/dr-diffusion-flux-1-s-pixel-art-lora",
        lora_file: str = "pixelart_schnell_v1.safetensors",
        lora_scale: float = 0.95,
        steps: int = 4,
    ) -> None:
        self.model_repo = model_repo
        self.lora_repo = lora_repo
        self.lora_file = lora_file
        self.lora_scale = lora_scale
        self.steps = steps
        self._model = None
        self._failed = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-flux"
        )
        self._lock = threading.Lock()

    def _load_on_mlx_thread(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            import mlx.core as mx
            from huggingface_hub import hf_hub_download
            from mflux.models.common.config.model_config import ModelConfig
            from mflux.models.flux.variants.txt2img.flux import Flux1

            # Cap MLX Metal buffer cache so LM Studio / PyTorch MPS retain unified memory headroom
            try:
                mx.set_cache_limit(512 * 1024 * 1024)
            except Exception:
                pass

            lora_paths: list[str] | None = None
            lora_scales: list[float] | None = None
            if self.lora_repo and self.lora_file:
                try:
                    lp = hf_hub_download(self.lora_repo, self.lora_file)
                    if Path(lp).exists():
                        lora_paths = [lp]
                        lora_scales = [float(self.lora_scale)]
                except Exception as lexc:
                    log.warning("Could not load FLUX Pixel-Art LoRA (%s); using base FLUX.", lexc)

            log.info(
                "Loading local 4-bit FLUX.1-schnell (MLX) %s...",
                f"+ {self.lora_file}" if lora_paths else "(base)",
            )
            self._model = Flux1(
                model_config=ModelConfig.from_name(
                    model_name=self.model_repo, base_model="schnell"
                ),
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            # Pre-compile fixed 512x512 MLX Metal graph once during warmup so live sprites take ~6-10s
            try:
                _ = self._model.generate_image(
                    seed=1,
                    prompt="pixel art star icon on white background",
                    num_inference_steps=1,
                    width=512,
                    height=512,
                )
            except Exception as warm_exc:
                log.debug("MLX FLUX pre-compile pass skipped: %s", warm_exc)
            try:
                mx.clear_cache()
            except Exception:
                pass
            log.info("FLUX.1-schnell 4-bit Pixel-Art engine ready on MLX.")
            return True
        except Exception as exc:
            log.warning("Failed to load FLUX.1-schnell MLX (%s).", exc)
            self._failed = True
            return False

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            return bool(self._executor.submit(self._load_on_mlx_thread).result())

    def _clear_mlx_cache_on_thread(self) -> None:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _generate_pil_on_mlx_thread(
        self, full_prompt: str, gen_w: int, gen_h: int, seed: int
    ):
        if not self._load_on_mlx_thread():
            return None
        try:
            res = self._model.generate_image(
                seed=int(seed) & 0x7FFFFFFF,
                prompt=full_prompt,
                num_inference_steps=max(2, self.steps),
                width=gen_w,
                height=gen_h,
            )
            pil_img = res.image
            self._clear_mlx_cache_on_thread()
            return pil_img
        except Exception as exc:
            log.warning("FLUX pixel art generation failed (%s)", exc)
            self._clear_mlx_cache_on_thread()
            return None

    def generate_pixel_art(
        self,
        prompt: str,
        target_w: int,
        target_h: int,
        strip_bg: bool = True,
        add_outline: bool = True,
        allowed_colors: tuple[int, ...] | list[int] | None = None,
        color_theme: str = "",
        seed: int = 42,
    ) -> np.ndarray | None:
        """Generate an [target_h, target_w] int16 pixel-art blueprint using FLUX + Pixel-Art LoRA."""
        if self._failed:
            return None

        # Always generate at a single compiled 512x512 MLX shape to avoid 45s Metal shader recompiles!
        # Aspect ratio is preserved cleanly via centered aspect crop before quantization.
        gen_w, gen_h = 512, 512

        color_clause = f"vivid {color_theme} palette, " if color_theme else "flat vivid saturated colors, "
        bg_clause = (
            ", isolated floating cutout sprite centered on pure solid white background, no frame, no box"
            if strip_bg
            else ", crisp bold pixel composition filling the frame"
        )
        full_prompt = (
            f"pixel art, 16-bit retro sprite style, {color_clause}sharp crisp pixel edges, "
            f"{prompt}{bg_clause}"
        )

        with self._lock:
            pil_img = self._executor.submit(
                self._generate_pil_on_mlx_thread, full_prompt, gen_w, gen_h, seed
            ).result()

        if pil_img is None:
            return None

        # Aspect-ratio crop without squashing; for wide non-cutout frames (e.g. comic strips),
        # avoid slicing across a mid-image horizontal panel gutter by cropping from the top panel row.
        ratio = target_w / max(1.0, float(target_h))
        if ratio > 1.15:
            crop_h = max(64, int(round(pil_img.width / ratio)))
            if (not strip_bg) and ratio >= 1.55:
                top = 0
            else:
                top = max(0, (pil_img.height - crop_h) // 2)
            pil_img = pil_img.crop((0, top, pil_img.width, min(pil_img.height, top + crop_h)))
        elif ratio < 0.87:
            crop_w = max(64, int(round(pil_img.height * ratio)))
            left = max(0, (pil_img.width - crop_w) // 2)
            pil_img = pil_img.crop((left, 0, min(pil_img.width, left + crop_w), pil_img.height))
        return _crisp_downsample_to_palette(
            pil_img,
            target_w=target_w,
            target_h=target_h,
            strip_bg=strip_bg,
            add_outline=add_outline and strip_bg,
            allowed_colors=allowed_colors,
        )


# ---------------------------------------------------------------------------
# Sandboxed LLM-Written Pixel Code Runner (for Visionless Geometric Chaos)
# ---------------------------------------------------------------------------

_SAFE_IMPORT_RE = re.compile(
    r"^\s*(?:"
    r"import\s+(?:numpy(?:\s+as\s+np)?|math)(?:\s*,\s*(?:numpy(?:\s+as\s+np)?|math))*"
    r"|from\s+(?:math|numpy)\s+import\s+[A-Za-z0-9_,\s\*]+"
    r")\s*(?:#.*)?$"
)
_FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "dir", "breakpoint", "input",
}
_FORBIDDEN_ATTRS = {"__class__", "__bases__", "__subclasses__", "__globals__", "__builtins__", "__dict__"}


def _validate_pixel_code_ast(source: str) -> tuple[bool, str]:
    """Reject any imports, dunder attributes, or unsafe builtins via Python AST inspection."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax:{exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Raise)):
            return False, f"node:{type(node).__name__}"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS):
            return False, f"attr:{node.attr}"
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _FORBIDDEN_CALLS):
            return False, f"name:{node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            return False, f"call:{node.func.id}"
    return True, "ok"


def run_sandboxed_pixel_code(
    code: str,
    w: int,
    h: int,
    allowed_colors: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray | None:
    """Safely execute LLM-written `def draw(grid, w, h):` with only numpy and math."""
    clean = re.sub(r"<think>.*?</think>", "", code, flags=re.S).strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", clean, flags=re.S)
    if m:
        clean = m.group(1).strip()
    elif clean.startswith("```"):
        # Handle truncated code block with no closing ```
        clean = re.sub(r"^```(?:python)?\s*", "", clean).strip()

    # Strip trailing `if __name__ == "__main__":` examples if the model appended one
    main_idx = clean.find('if __name__')
    if main_idx >= 0:
        clean = clean[:main_idx].rstrip()

    # Strip harmless numpy/math import lines (even indented inside draw())
    filtered_lines = []
    for ln in clean.splitlines():
        if _SAFE_IMPORT_RE.match(ln):
            continue
        filtered_lines.append(ln)
    clean = "\n".join(filtered_lines).strip()
    if not clean or len(clean) > 7000:
        return None
    ok, reason = _validate_pixel_code_ast(clean)
    if not ok:
        log.warning("Rejected unsafe LLM pixel-code script (%s).", reason)
        return None

    from pixel_battle.palette import COLOR_NAMES

    color_consts: dict[str, int] = {}
    for idx, cname in enumerate(COLOR_NAMES):
        key = cname.upper().replace(" ", "_").replace("-", "_")
        color_consts[key] = idx
    color_consts.setdefault("GOLD", 6)
    color_consts.setdefault("CYAN", 14)
    color_consts.setdefault("MAGENTA", 21)
    color_consts.setdefault("VIOLET", 19)
    color_consts.setdefault("GRAY", 28)
    color_consts.setdefault("GREY", 28)

    grid = np.full((h, w), -1, dtype=np.int16)
    safe_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "abs": abs,
            "int": int,
            "float": float,
            "round": round,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "set": set,
            "dict": dict,
            "enumerate": enumerate,
            "zip": zip,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "pow": pow,
            "divmod": divmod,
            "any": any,
            "all": all,
        },
        "np": np,
        "numpy": np,
        "math": math,
        "pi": math.pi,
        "tau": math.tau,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "hypot": math.hypot,
        "atan2": math.atan2,
        "floor": math.floor,
        "ceil": math.ceil,
        "exp": math.exp,
        "log": math.log,
        **color_consts,
    }
    safe_locals: dict = {}
    try:
        exec(compile(clean, "<ai_pixel_code>", "exec"), safe_globals, safe_locals)
        fn = safe_locals.get("draw")
        if callable(fn):
            fn(grid, w, h)
        grid = np.where(grid < 0, -1, np.clip(grid, 0, 31)).astype(np.int16)
        valid_cnt = int((grid >= 0).sum())
        if valid_cnt < max(20, int(w * h * 0.05)):
            return None
        if len(np.unique(grid[grid >= 0])) < 2:
            return None
        return remap_indices_to_allowed(grid, allowed_colors)
    except Exception as exc:
        log.warning("Sandboxed LLM pixel-code failed (%s)", exc)
        return None
