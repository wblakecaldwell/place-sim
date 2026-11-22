"""Main simulation engine uniting GPU MPS pixel loop, 200-faction power-law ecosystem, progressive ribbons, 4-variant Void, and async VLM/diffusion strategist."""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pixel_battle.art.comics import (
    CARTOON_MEME_CATALOG,
    COMIC_CATALOG,
)
from pixel_battle.art.flags import (
    FLAG_BY_KEY,
    NATIONAL_FLAGS,
    RIBBON_PATTERNS,
    generate_transverse_ribbon,
)
from pixel_battle.art.manager import ArtManager
from pixel_battle.art.sprites import (
    FINE_ART_CATALOG,
    SPRITE_KEYS,
    generate_black_hole,
    generate_fractal,
    generate_void_monster,
)
from pixel_battle.art.typography import FONT_STYLES, TypographyStyleSpec
from pixel_battle.brain import COLOR_THEMES, FactionDecision, LocalVLMBrain
from pixel_battle.canvas import CanvasState
from pixel_battle.config import SimConfig
from pixel_battle.detector import (
    find_open_exodus_spot,
    find_pareidolia_candidates,
    find_ribbon_track,
    find_void_spawn_spot,
    update_faction_metrics,
)
from pixel_battle.events import EventLogger
from pixel_battle.factions import (
    DiplomacyBoard,
    Faction,
    compute_edge_weights,
    compute_shared_border_rect,
    describe_neighbors,
)
from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    BURGUNDY_IDX,
    COLOR_NAMES,
    DARK_RED_IDX,
    GREEN_IDX,
    ORANGE_IDX,
    PINK_IDX,
    PURPLE_IDX,
    RED_IDX,
    WHITE_IDX,
    YELLOW_IDX,
    resolve_color,
)
from pixel_battle.policies.base import PixelPolicy
from pixel_battle.policies.neural import NeuralRedditPolicy
from pixel_battle.policies.stochastic import StochasticPolicy
from pixel_battle.subreddits import SubredditSampler

log = logging.getLogger(__name__)

_DIR_MAP: dict[str, tuple[int, int]] = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
    "NONE": (0, 0),
}

# Empirical placeAtlas 2017/2022/2023 size-tier shares (REQ-SIM-010)
_TIER_TARGETS: dict[str, float] = {
    "micro": 0.60,
    "medium": 0.24,
    "macro": 0.12,
    "ribbon": 0.04,
}

# Per-tier max expansion dimensions so Micro & Medium factions stay properly scaled
_TIER_MAX_DIM: dict[str, tuple[int, int]] = {
    "micro": (28, 28),
    "medium": (58, 58),
    "macro": (112, 112),
    "ribbon": (512, 512),
}

_LOGO_WORDMARKS: list[tuple[str, str]] = [
    ("STONKS", "Pixel portfolio only goes up."),
    ("BONK!", "No chaos near the comic district."),
    ("r/MEMES", "Meme headquarters on the grid."),
    ("r/COMICS", "Daily pixel comic strips."),
    ("PIXEL", "One pixel at a time."),
    ("ARCADE", "Insert coin to paint."),
    ("LO-FI", "Chill beats and clean pixels."),
    ("DOGE", "Much color, very art, wow."),
    ("MEOW", "Feline pixel sanctuary."),
    ("RETRO", "16-bit nostalgia forever."),
    ("HOLD FAST", "Hold the line against the Void."),
    ("NO VOID", "Keep the canvas bright."),
    ("VIBES", "Good vibes across the border."),
    ("GG EZ", "Well played, neighbors."),
    ("CHILL", "Relaxed pixel builders."),
    ("r/CATS", "Cats rule the canvas."),
    ("r/GAMING", "Press start to defend."),
    ("r/ANIME", "Chibi alliance united."),
    ("TOON", "Saturday morning pixel squad."),
    ("BRUH", "Who painted over my corner?"),
]


_CLEAN_SPACE_NAMES: list[tuple[str, str]] = [
    ("Whiteout Guild", "Keep this square pure white!"),
    ("Blank Space", "Defend negative space!"),
    ("The Erasers", "Clean canvas forever!"),
    ("Pure Canvas", "Zero clutter, pure white!"),
    ("Zen Margin", "Peace in empty pixels!"),
    ("r/CleanCanvas", "Wipe the chaos clean!"),
    ("Null Zone", "Nothingness is art!"),
    ("Snow Square", "Hold the white tile!"),
]

_FRACTAL_NAMES: list[tuple[str, str, str]] = [
    ("Sierpinski Cult", "sierpinski", "Infinite triangles forever!"),
    ("Mandelbrot Set", "mandelbrot", "Zoom into infinity!"),
    ("Cantor Carpet", "carpet", "Recurse every pixel!"),
    ("Sacred Diamond", "diamond", "Symmetry rules the board!"),
]

_VOID_VARIANTS = ("tendrils", "black_hole", "fractal", "monster")


def _snap_to_walls(x0: int, y0: int, w: int, h: int, W: int, H: int, snap: int = 16) -> tuple[int, int]:
    """Snap a territory bounding box flush against canvas walls/corners when close to an edge."""
    x0 = max(0, min(W - w, x0))
    y0 = max(0, min(H - h, y0))
    if x0 <= snap:
        x0 = 0
    elif W - (x0 + w) <= snap:
        x0 = W - w
    if y0 <= snap:
        y0 = 0
    elif H - (y0 + h) <= snap:
        y0 = H - h
    return x0, y0


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


class SimulationEngine:
    """Thread-safe r/place engine with real-time GPU ticking, 200-faction power-law ecosystem, and async VLM/diffusion strategist."""

    def __init__(
        self,
        cfg: SimConfig,
        policy: PixelPolicy | None = None,
        resume: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.rng = random.Random(cfg.seed)

        self.canvas = CanvasState(cfg.width, cfg.height, self.device)
        if policy is not None:
            self.policy: PixelPolicy = policy
            self.policy_name = type(policy).__name__
        else:
            use_neural = (cfg.policy_mode == "neural") or (
                cfg.policy_mode == "auto" and cfg.neural_model_path.exists()
            )
            if use_neural:
                if not cfg.neural_model_path.exists():
                    raise FileNotFoundError(
                        f"Neural policy requested but model not found at {cfg.neural_model_path}. "
                        "Run `uv run pixel-battle train-reddit` first."
                    )
                self.policy = NeuralRedditPolicy(
                    model_path=cfg.neural_model_path,
                    num_users=cfg.num_users,
                    width=cfg.width,
                    height=cfg.height,
                    base_cooldown=cfg.base_cooldown,
                    device=self.device,
                    seed=cfg.seed,
                )
                self.policy_name = "neural (Reddit r/place 2022, batched)"
            else:
                self.policy = StochasticPolicy(
                    num_users=cfg.num_users,
                    width=cfg.width,
                    height=cfg.height,
                    base_cooldown=cfg.base_cooldown,
                    device=self.device,
                    seed=cfg.seed,
                )
                self.policy_name = "stochastic"

        self.art = ArtManager(cfg, self.device)
        self.brain = LocalVLMBrain(ollama_url=cfg.ollama_url, vlm_model=cfg.vlm_model)
        self.subreddits = SubredditSampler(seed=cfg.seed)
        self.diplomacy = DiplomacyBoard()
        self.logger = EventLogger(cfg.run_dir, cfg.width, cfg.height, resume=resume)

        self.factions: list[Faction] = []
        self.next_faction_id: int = 0
        self.shared_borders: list[tuple[int, int, tuple[int, int, int, int]]] = []
        self._void_cycle_idx: int = 0

        # Simulation runtime state
        self.tick: int = 0
        self.epoch: int = 0
        self.start_wall_time: float = time.time()
        self.paused: bool = False
        self.stop_requested: bool = False
        self.sim_error: Exception | None = None
        self.target_tps: float = cfg.target_tps
        self.actual_tps: float = 0.0
        self.pixels_per_sec: float = 0.0
        self.brain_status: str = "Primordial chaos — citizens wandering..."
        self.ollama_ready: bool = self.brain.check_health()

        self._state_lock = threading.RLock()
        self._job_queue: queue.Queue[str] = queue.Queue(maxsize=12)
        self._brain_busy = threading.Event()
        self._sim_thread: threading.Thread | None = None
        self._brain_thread: threading.Thread | None = None

        if resume:
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        if self.cfg.max_hours > 0:
            return (time.time() - self.start_wall_time) >= self.cfg.max_hours * 3600.0
        return self.tick >= self.cfg.max_ticks

    def start(self) -> None:
        self.stop_requested = False
        self.start_wall_time = time.time()
        horizon_str = (
            f"{self.cfg.max_hours:.2f}h wall-clock"
            if self.cfg.max_hours > 0
            else f"{self.cfg.max_ticks:,} ticks"
        )
        self.logger.log_chronicle(
            self.tick,
            "system",
            f"Simulation started on {self.cfg.width}x{self.cfg.height} canvas "
            f"({self.cfg.num_users:,} online slots, max_factions={self.cfg.max_active_factions}, "
            f"device={self.device}, policy={self.policy_name}, horizon={horizon_str}, "
            f"ollama={'online' if self.ollama_ready else 'fallback'}, "
            f"subreddits={len(self.subreddits.entries):,}).",
        )
        self._brain_thread = threading.Thread(target=self._strategist_loop, name="strategist", daemon=True)
        self._brain_thread.start()
        self._sim_thread = threading.Thread(target=self._tick_loop, name="sim-loop", daemon=True)
        self._sim_thread.start()

    def stop(self) -> None:
        self.stop_requested = True
        try:
            self._job_queue.put_nowait("STOP")
        except queue.Full:
            pass
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=3.0)
        if self._brain_thread is not None:
            self._brain_thread.join(timeout=3.0)
        self.save_checkpoint()
        self.logger.close()
        self.brain.close()

    def queue_job(self, job: str) -> None:
        try:
            self._job_queue.put_nowait(job)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Fast GPU Tick Loop
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        t_window = time.perf_counter()
        ticks_window = 0
        pix_window = 0

        while not self.stop_requested and not self.is_finished():
            if self.paused:
                time.sleep(0.05)
                t_window = time.perf_counter()
                ticks_window = 0
                pix_window = 0
                continue

            if (self.target_tps == 0 and (self._brain_busy.is_set() or not self._job_queue.empty())) or (
                self._brain_busy.is_set() and not self._job_queue.empty()
            ):
                time.sleep(0.05)
                continue

            step_start = time.perf_counter()
            try:
                n_placed = self.step_once()
            except Exception as exc:
                self.sim_error = exc
                self.brain_status = f"Simulation error at tick {self.tick}: {exc}"
                log.exception("Tick loop failed at tick %d", self.tick)
                self.stop_requested = True
                break
            ticks_window += 1
            pix_window += n_placed

            now = time.perf_counter()
            if now - t_window >= 1.0:
                dt = now - t_window
                self.actual_tps = ticks_window / dt
                self.pixels_per_sec = pix_window / dt
                t_window = now
                ticks_window = 0
                pix_window = 0

            if self.target_tps > 0:
                min_dt = 1.0 / self.target_tps
                elapsed = time.perf_counter() - step_start
                if elapsed < min_dt:
                    time.sleep(min_dt - elapsed)

        if self.is_finished():
            self.brain_status = f"Finished at tick {self.tick} ({self.logger.total_events:,} pixels)!"
            self.save_checkpoint()
            try:
                self.canvas.to_pil(scale=1).save(self.cfg.run_dir / "final.png")
                self.canvas.to_pil(scale=4).save(self.cfg.run_dir / "final_2k.png")
            except Exception:
                pass
            pop = self.policy.get_population_stats() if hasattr(self.policy, "get_population_stats") else {}
            self.logger.log_chronicle(
                self.tick,
                "system",
                f"Simulation completed at tick {self.tick:,} with {self.logger.total_events:,} placements "
                f"({pop.get('total_unique_users', 0):,} unique users, {pop.get('one_shot_pct', 20.0):.1f}% one-shot).",
            )

    def step_once(self) -> int:
        """Advance the pixel simulation by exactly 1 tick on GPU."""
        with self._state_lock:
            self.tick += 1
            tick = self.tick
            self.canvas.decay_activity(0.94)

            batch = self.policy.step(tick, self.canvas, self.factions, self.shared_borders)
            n_placed = int(batch.xs.numel())
            if n_placed > 0:
                self.canvas.apply_placements(tick, batch.xs, batch.ys, batch.colors, batch.factions)
                self.logger.log_pixels(
                    tick,
                    batch.xs.detach().cpu().numpy(),
                    batch.ys.detach().cpu().numpy(),
                    batch.colors.detach().cpu().numpy(),
                    batch.factions.detach().cpu().numpy(),
                    batch.user_ids.detach().cpu().numpy(),
                )

            if tick % 15 == 0:
                update_faction_metrics(self.canvas, self.factions, self.policy.user_factions)
                self.policy.social_step(self.factions, self.canvas)
                self._refresh_shared_borders()

            # Advance progressive Transverse Ribbons cleanly every 65 ticks
            if tick >= self.cfg.chaos_ticks + 65 and tick % 65 == 0:
                self._advance_ribbons()

            if (
                tick >= self.cfg.chaos_ticks
                and (tick - self.cfg.chaos_ticks) % self.cfg.discovery_interval == 0
            ):
                self.queue_job("DISCOVER")

            if (
                tick >= self.cfg.chaos_ticks + 40
                and (tick - self.cfg.chaos_ticks - 40) % self.cfg.epoch_interval == 0
            ):
                self.queue_job("EPOCH")

            if tick % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        return n_placed

    def _refresh_shared_borders(self) -> None:
        borders: list[tuple[int, int, tuple[int, int, int, int]]] = []
        seen: set[tuple[int, int]] = set()
        alive = {f.id: f for f in self.factions if f.alive and not f.is_void}
        for fa_id, fa in alive.items():
            for fb_id in fa.allies:
                if fb_id not in alive:
                    continue
                pair = (min(fa_id, fb_id), max(fa_id, fb_id))
                if pair in seen:
                    continue
                seen.add(pair)
                rect = compute_shared_border_rect(fa, alive[fb_id])
                if rect is not None:
                    borders.append((pair[0], pair[1], rect))
        self.shared_borders = borders

    # ------------------------------------------------------------------
    # Progressive Transverse Ribbon Advancement (REQ-DIP-009, REQ-ART-012)
    # ------------------------------------------------------------------

    def _advance_ribbons(self) -> None:
        """Advance each active Transverse Ribbon step-by-step across the Canvas, respecting ally tunnels."""
        W, H = self.cfg.width, self.cfg.height
        fac_by_id = {f.id: f for f in self.factions if f.alive}

        for fac in self.factions:
            if not (fac.alive and fac.art_mode == "ribbon"):
                continue
            if fac.integrity < 0.35:
                continue

            step = 52
            axis = fac.ribbon_axis or "h"
            thick = max(5, min(15, fac.ribbon_thickness))
            nx0, ny0, nw, nh = fac.x0, fac.y0, fac.w, fac.h

            if axis == "h":
                if nw >= W:
                    continue
                if nx0 == 0:
                    nw = min(W, nw + step)
                elif nx0 + nw >= W:
                    add = min(step, nx0)
                    nx0 -= add
                    nw += add
                else:
                    left_add = min(step // 2, nx0)
                    nx0 -= left_add
                    nw = min(W - nx0, nw + step)
            elif axis == "v":
                if nh >= H:
                    continue
                if ny0 == 0:
                    nh = min(H, nh + step)
                elif ny0 + nh >= H:
                    add = min(step, ny0)
                    ny0 -= add
                    nh += add
                else:
                    top_add = min(step // 2, ny0)
                    ny0 -= top_add
                    nh = min(H - ny0, nh + step)
            else:
                cur_s = max(nw, nh)
                if cur_s >= min(W, H):
                    continue
                new_s = min(min(W, H), cur_s + step)
                if axis == "d1":
                    nx0 = min(nx0, W - new_s)
                    ny0 = min(ny0, H - new_s)
                else:
                    nx0 = min(nx0, W - new_s)
                    ny0 = max(0, (ny0 + nh) - new_s)
                nw = nh = new_s

            bp_np = generate_transverse_ribbon(
                pattern_key=fac.extra_key or "rainbow",
                axis=axis,
                length=max(nw, nh),
                thickness=thick,
            )
            # Clip cleanly to canvas
            bp_np = bp_np[: min(bp_np.shape[0], H - ny0), : min(bp_np.shape[1], W - nx0)]

            # Respect allied Factions by leaving transparent overpass tunnels over allies
            for ally_id in fac.allies:
                ally = fac_by_id.get(ally_id)
                if ally is None or ally.is_void:
                    continue
                ix0 = max(nx0, ally.x0)
                iy0 = max(ny0, ally.y0)
                ix1 = min(nx0 + bp_np.shape[1], ally.x1)
                iy1 = min(ny0 + bp_np.shape[0], ally.y1)
                if ix1 > ix0 and iy1 > iy0:
                    bp_np[iy0 - ny0 : iy1 - ny0, ix0 - nx0 : ix1 - nx0] = -1

            fac.set_blueprint(bp_np, x0=nx0, y0=ny0, device=self.device)
            if (axis == "h" and fac.w >= W) or (axis == "v" and fac.h >= H):
                fac.remember("Spanned all 512px across the entire Canvas!")

    # ------------------------------------------------------------------
    # Async VLM + Multi-Scale Strategist Worker
    # ------------------------------------------------------------------

    def _strategist_loop(self) -> None:
        while not self.stop_requested:
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job == "STOP":
                break
            self._brain_busy.set()
            try:
                if job == "DISCOVER":
                    self.run_pareidolia_discovery()
                elif job == "EPOCH":
                    self.run_geopolitical_epoch()
                elif job == "VOID":
                    self.spawn_void_eruption(force=True)
            except Exception as exc:
                log.exception("Strategist job %s failed: %s", job, exc)
                self.brain_status = f"Strategist recovered from error: {exc}"
            finally:
                self._brain_busy.clear()

    def _pick_next_tier(self, alive_art: list[Faction]) -> str:
        """Choose next size tier to track empirical placeAtlas shares (60% micro, 24% med, 12% macro, 4% ribbon)."""
        n = len(alive_art)
        # Seed initial iconic structure fast: corner macros + early ribbon + medium/micro
        if n < 2:
            return "macro"
        if n == 2:
            return "ribbon"
        if n < 5:
            return "medium"
        counts = Counter(f.size_tier for f in alive_art)
        best_tier = "micro"
        best_deficit = -1e9
        denom = float(n + 1)
        for tier, target_share in _TIER_TARGETS.items():
            cur_share = counts.get(tier, 0) / denom
            deficit = target_share - cur_share
            if deficit > best_deficit:
                best_deficit = deficit
                best_tier = tier
        return best_tier

    def _pick_category_for_tier(self, tier: str) -> str:
        """Sample artwork category emphasizing comic strips, cartoons, memes, and text logos with restrained flags."""
        r = self.rng.random()
        if tier == "ribbon":
            return "ribbon"
        if tier == "macro":
            if r < 0.28:
                return "comic_strip"
            if r < 0.52:
                return "banner"
            if r < 0.76:
                return "cartoon_meme"
            if r < 0.86:
                return "flag"
            if r < 0.94:
                return "diffusion"
            return "painting" if self.rng.random() < 0.5 else ("procedural" if self.rng.random() < 0.6 else "fractal")
        if tier == "medium":
            if r < 0.26:
                return "cartoon_meme"
            if r < 0.50:
                return "comic_strip"
            if r < 0.72:
                return "banner"
            if r < 0.82:
                return "flag"
            if r < 0.90:
                return "diffusion"
            if r < 0.95:
                return "painting"
            return "fractal" if self.rng.random() < 0.5 else "clean_space"
        # micro tier (~60% of active factions)
        if r < 0.52:
            return "micro_sprite"
        if r < 0.72:
            return "banner"
        if r < 0.84:
            return "cartoon_meme"
        if r < 0.91:
            return "flag"
        if r < 0.96:
            return "clean_space"
        return "painting" if self.rng.random() < 0.5 else "fractal"

    def _target_initial_members(self, tier: str, valid_pixels: int) -> int:
        """Scale initial citizen recruitment proportionally to Blueprint area (REQ-SIM-011)."""
        if tier == "micro":
            return max(42, min(125, int(38 + valid_pixels * 0.16)))
        if tier == "medium":
            return max(145, min(330, int(115 + valid_pixels * 0.11)))
        if tier == "ribbon":
            return self.rng.randint(360, 560)
        return max(380, min(780, int(280 + valid_pixels * 0.08)))

    def _fast_typography_spec(
        self,
        text: str,
        primary: int,
        secondary: int,
        accent: int,
        f_idx: int,
        is_micro: bool = False,
        allow_flux: bool = False,
    ) -> TypographyStyleSpec:
        """Varied multi-font typography spec for micro/medium/macro text logos."""
        font = FONT_STYLES[f_idx % len(FONT_STYLES)]
        ce = ("vertical_gradient", "per_letter_palette", "glossy_highlight", "rainbow_letters", "striped_fill", "solid")[f_idx % 6]
        be = ("extrude_3d", "double_sticker", "neon_glow", "drop_shadow", "outline_1px")[(f_idx // 2) % 5]
        lay = ("wavy", "arched", "straight", "italic_slant")[(f_idx // 3) % 4]
        use_rib = (not is_micro) and (f_idx % 3 == 0)
        if is_micro:
            engine_choice = "styled"
        elif allow_flux and f_idx % 3 == 0:
            engine_choice = "flux"
        elif f_idx % 3 == 1:
            engine_choice = "code"
        else:
            engine_choice = "styled"
        return TypographyStyleSpec(
            include_text=True,
            text=text[:12],
            casing="subreddit" if text.lower().startswith("r/") else ("upper" if f_idx % 2 == 0 else "title"),
            font_style=font,
            size_tier="tiny" if is_micro else "large",
            color_effect=ce,  # type: ignore[arg-type]
            border_effect="outline_1px" if is_micro else be,  # type: ignore[arg-type]
            layout="straight" if is_micro else lay,  # type: ignore[arg-type]
            placement="center",
            fill_color=primary if primary != BLACK_IDX else YELLOW_IDX,
            fill_color_2=secondary if secondary != primary else accent,
            fill_color_3=accent,
            outline_color=BLACK_IDX,
            shadow_color=secondary,
            use_ribbon=use_rib,
            ribbon_color=secondary if secondary != primary else BLACK_IDX,
            engine=engine_choice,  # type: ignore[arg-type]
        )

    def run_pareidolia_discovery(self, max_new: int = 6) -> list[Faction]:
        """Multi-scale discovery wave spawning up to 6 factions across Macro/Medium/Micro/Ribbon tiers."""
        with self._state_lock:
            alive_art = [f for f in self.factions if f.alive and not f.is_void]
            slots = self.cfg.max_active_factions - len(alive_art)
            if slots <= 0:
                return []
            # Fill early canvas briskly (up to 6 per wave), then gently replace fallen factions
            wave_size = min(max_new, slots)
            existing_names = {f.name.lower() for f in self.factions}
            existing_names_list = [f.name for f in self.factions]
            existing_themes_list = [f.theme for f in self.factions]
            existing_flag_keys = {f.extra_key for f in alive_art if f.category == "flag"}
            existing_paint_keys = {f.extra_key for f in alive_art if f.category == "painting"}
            existing_comic_keys = {f.extra_key for f in alive_art if f.art_mode == "comic_strip"}
            existing_meme_keys = {f.extra_key for f in alive_art if f.art_mode == "cartoon_meme"}
            existing_rib_keys = {f.extra_key for f in alive_art if f.art_mode == "ribbon"}
            tick = self.tick
            total_founded = len(self.factions)

        created: list[Faction] = []
        vlm_calls_in_wave = 0
        sim_alive = list(alive_art)

        for i in range(wave_size):
            if self.stop_requested:
                break

            tier = self._pick_next_tier(sim_alive)
            cat_choice = self._pick_category_for_tier(tier)
            f_idx = total_founded + i
            theme_desc, def_c1_name, def_c2_name, def_c3_name = COLOR_THEMES[f_idx % len(COLOR_THEMES)]
            def_c1 = resolve_color(def_c1_name, BLUE_IDX)
            def_c2 = resolve_color(def_c2_name, YELLOW_IDX)
            def_c3 = resolve_color(def_c3_name, WHITE_IDX)

            # Cap total active flags (<= 18 overall, <= 3 macro) so flags stay balanced
            alive_flags = sum(1 for f in sim_alive if f.category == "flag")
            alive_macro_flags = sum(1 for f in sim_alive if f.category == "flag" and f.size_tier == "macro")
            if cat_choice == "flag" and (alive_flags >= 18 or (tier == "macro" and alive_macro_flags >= 3)):
                if tier == "micro":
                    cat_choice = "micro_sprite" if self.rng.random() < 0.65 else "banner"
                else:
                    cat_choice = self.rng.choice(["comic_strip", "cartoon_meme", "banner"])

            # Cap VLM+diffusion spawns to <= 2 per wave without falling back to flags
            fast_banner_override = False
            if cat_choice == "diffusion" and tier != "micro" and vlm_calls_in_wave >= 2:
                cat_choice = "comic_strip" if self.rng.random() < 0.5 else "cartoon_meme"
            elif cat_choice == "banner" and tier != "micro" and vlm_calls_in_wave >= 1:
                fast_banner_override = True

            # Pre-select comic / cartoon-meme spec so bounding box matches layout aspect ratio
            pre_comic = None
            pre_meme = None
            if cat_choice == "comic_strip":
                avail_c = [c for c in COMIC_CATALOG if c.key not in existing_comic_keys] or COMIC_CATALOG
                pre_comic = self.rng.choice(avail_c)
            elif cat_choice == "cartoon_meme":
                avail_m = [m for m in CARTOON_MEME_CATALOG if m.key not in existing_meme_keys] or CARTOON_MEME_CATALOG
                pre_meme = self.rng.choice(avail_m)

            # --- 1. Determine candidate bounding box & dimensions ---
            is_cameo = False
            is_corner = False
            is_edge = False
            cand_crop_pil = None
            cand_crop_rgb = None
            dom_names: list[str] = []
            ribbon_axis = ""
            ribbon_thick = 8
            extra_key = ""

            if tier == "ribbon":
                axis_r = self.rng.random()
                ribbon_axis = "h" if axis_r < 0.55 else ("v" if axis_r < 0.85 else self.rng.choice(["d1", "d2"]))
                ribbon_thick = self.rng.choice([6, 7, 8, 9, 10, 12])
                with self._state_lock:
                    x0, y0, bp_w, bp_h = find_ribbon_track(self.canvas, self.factions, ribbon_axis, ribbon_thick, self.rng)
                avail_ribs = [p for p in RIBBON_PATTERNS if p[0] not in existing_rib_keys] or RIBBON_PATTERNS
                pkey, pname, pcols, _ = self.rng.choice(avail_ribs)
                existing_rib_keys.add(pkey)
                extra_key = f"{pkey}|{ribbon_thick}" if ribbon_axis in ("d1", "d2") else pkey
                fac_name = pname if pname.lower() not in existing_names else f"{pname} {f_idx + 1}"
                fac_theme = f"cross-map {pkey} ribbon"
                fac_slogan = f"Extend the {pname} across all 512px!"
                art_mode = "ribbon"
                fac_category = "ribbon"
                art_prompt = f"straight {ribbon_axis} {pkey} patterned ribbon"
                c1, c2, c3 = pcols[0], pcols[min(1, len(pcols) - 1)], pcols[-1]
                sub_name = f"place_{pkey}"
                typo_spec = None
                flux_style_desc = ""

            else:
                # Choose target Blueprint dimensions strictly within tier bounds (REQ-SIM-010)
                if tier == "micro":
                    if cat_choice == "flag":
                        bp_w, bp_h = self.rng.choice([(20, 13), (22, 14), (24, 15)])
                    elif cat_choice == "painting":
                        bp_w, bp_h = self.rng.choice([(24, 20), (26, 22), (28, 22)])
                    elif cat_choice == "banner":
                        bp_w, bp_h = self.rng.choice([(28, 12), (32, 14), (30, 14)])
                    elif cat_choice == "cartoon_meme":
                        if pre_meme is not None and pre_meme.key == "nyan_cat":
                            bp_w, bp_h = 30, 20
                        else:
                            bp_w, bp_h = self.rng.choice([(24, 24), (26, 26), (28, 28)])
                    elif cat_choice == "clean_space":
                        bp_w, bp_h = self.rng.choice([(18, 18), (22, 18), (24, 20)])
                    else:
                        s = self.rng.choice([16, 18, 20, 22, 24, 26])
                        bp_w, bp_h = s, s
                    scan_win, scan_stride = max(bp_w, bp_h), 10
                    allow_cameo = (cat_choice in ("micro_sprite", "banner")) and (len(sim_alive) >= 28) and (self.rng.random() < 0.25)
                elif tier == "medium":
                    if cat_choice == "flag":
                        bp_w, bp_h = self.rng.choice([(38, 24), (42, 28), (46, 30)])
                    elif cat_choice == "comic_strip":
                        if pre_comic is not None and pre_comic.layout == "3h":
                            bp_w, bp_h = self.rng.choice([(64, 28), (72, 30)])
                        elif pre_comic is not None and pre_comic.layout == "2v":
                            bp_w, bp_h = self.rng.choice([(48, 38), (54, 42)])
                        else:
                            bp_w, bp_h = self.rng.choice([(50, 42), (56, 46)])
                    elif cat_choice == "cartoon_meme":
                        if pre_meme is not None and pre_meme.key == "nyan_cat":
                            bp_w, bp_h = self.rng.choice([(52, 34), (58, 36)])
                        else:
                            bp_w, bp_h = self.rng.choice([(38, 38), (44, 44), (48, 48)])
                    elif cat_choice == "painting":
                        bp_w, bp_h = self.rng.choice([(36, 28), (42, 34), (48, 38)])
                    elif cat_choice == "banner":
                        bp_w, bp_h = self.rng.choice([(60, 24), (70, 26), (78, 30)])
                    elif cat_choice == "clean_space":
                        bp_w, bp_h = self.rng.choice([(32, 26), (36, 28)])
                    else:
                        s = self.rng.choice([36, 42, 48, 52])
                        bp_w, bp_h = s, s
                    scan_win, scan_stride = min(54, max(bp_w, bp_h)), 18
                    allow_cameo = False
                else:  # macro
                    if cat_choice == "flag":
                        bp_w, bp_h = self.rng.choice([(64, 40), (72, 46), (78, 48)])
                    elif cat_choice == "comic_strip":
                        if pre_comic is not None and pre_comic.layout == "3h":
                            bp_w, bp_h = self.rng.choice([(84, 34), (94, 36), (102, 40)])
                        elif pre_comic is not None and pre_comic.layout == "2v":
                            bp_w, bp_h = self.rng.choice([(60, 48), (68, 54)])
                        else:
                            bp_w, bp_h = self.rng.choice([(62, 52), (70, 58)])
                    elif cat_choice == "cartoon_meme":
                        if pre_meme is not None and pre_meme.key == "nyan_cat":
                            bp_w, bp_h = self.rng.choice([(68, 42), (76, 46)])
                        else:
                            bp_w, bp_h = self.rng.choice([(52, 52), (58, 58), (64, 64)])
                    elif cat_choice == "painting":
                        bp_w, bp_h = self.rng.choice([(54, 42), (62, 48), (68, 54)])
                    elif cat_choice == "banner":
                        bp_w, bp_h = self.rng.choice([(96, 34), (110, 38), (124, 42)])
                    else:
                        s = self.rng.choice([60, 68, 74])
                        bp_w, bp_h = s, s
                    scan_win, scan_stride = min(68, max(bp_w, bp_h)), 24
                    allow_cameo = False

                with self._state_lock:
                    cands = find_pareidolia_candidates(
                        self.canvas,
                        self.factions,
                        window=scan_win,
                        stride=scan_stride,
                        min_density=0.06 if tier != "micro" else 0.01,
                        top_k=1,
                        allow_cameo=allow_cameo,
                        rng=self.rng,
                    )
                    if not cands:
                        # Fallback to least-occupied spot for dense packing
                        fx0, fy0 = find_open_exodus_spot(self.canvas, self.factions, bp_w, bp_h)
                        x0, y0 = fx0, fy0
                        dom_colors = [def_c1, def_c2]
                        cand_crop_pil = self.canvas.crop_pil(x0, y0, x0 + bp_w, y0 + bp_h, scale=6)
                        cand_crop_rgb = self.canvas.crop_rgb_numpy(x0, y0, x0 + bp_w, y0 + bp_h)
                    else:
                        c = cands[0]
                        is_cameo = c.is_cameo
                        is_corner = c.is_corner
                        is_edge = c.is_edge
                        dom_colors = c.dominant_colors
                        raw_x0 = c.x0 - (bp_w - c.w) // 2
                        raw_y0 = c.y0 - (bp_h - c.h) // 2
                        if is_cameo:
                            x0 = max(0, min(self.cfg.width - bp_w, raw_x0))
                            y0 = max(0, min(self.cfg.height - bp_h, raw_y0))
                        else:
                            x0, y0 = _snap_to_walls(
                                raw_x0, raw_y0, bp_w, bp_h, self.cfg.width, self.cfg.height,
                                snap=14 if tier != "micro" else 6,
                            )
                        cand_crop_pil = self.canvas.crop_pil(c.x0, c.y0, c.x0 + c.w, c.y0 + c.h, scale=6)
                        cand_crop_rgb = self.canvas.crop_rgb_numpy(c.x0, c.y0, c.x0 + c.w, c.y0 + c.h)
                    dom_names = [COLOR_NAMES[idx] for idx in dom_colors if 0 <= idx < len(COLOR_NAMES)]

                # --- 2. Build metadata & art spec for the selected category ---
                sub_entry = self.subreddits.sample(size_tier=tier, avoid_names=existing_names)
                sub_name = sub_entry.name
                typo_spec = None
                flux_style_desc = ""

                if cat_choice == "flag":
                    avail_flags = [fl for fl in NATIONAL_FLAGS if fl.key not in existing_flag_keys] or NATIONAL_FLAGS
                    fl = self.rng.choice(avail_flags)
                    existing_flag_keys.add(fl.key)
                    extra_key = fl.key
                    fac_name = fl.display_name if fl.display_name.lower() not in existing_names else f"{fl.display_name} {f_idx + 1}"
                    fac_theme = f"{fl.key} national flag"
                    fac_slogan = fl.slogan
                    art_mode = "flag"
                    fac_category = "flag"
                    art_prompt = f"{fl.key} national flag"
                    c1, c2, c3 = fl.primary, fl.secondary, fl.accent
                    sub_name = fl.subreddit

                elif cat_choice == "micro_sprite":
                    skey = self.rng.choice(SPRITE_KEYS)
                    extra_key = skey
                    from pixel_battle.art.sprites import SPRITE_BY_KEY

                    tmpl = SPRITE_BY_KEY[skey]
                    label_sub = f"r/{sub_name[:12]}"
                    fac_name = label_sub if label_sub.lower() not in existing_names else f"{tmpl.title} {f_idx + 1}"
                    fac_theme = f"{tmpl.title} ({tmpl.category})"
                    fac_slogan = f"{label_sub} holds these pixels!"
                    art_mode = "micro_sprite"
                    fac_category = tmpl.category  # "game" | "meme" | "anime"
                    art_prompt = f"micro pixel sprite {tmpl.title}"
                    pal0 = tmpl.palettes[0]
                    c1 = pal0.get("P", def_c1)
                    c2 = pal0.get("S", def_c2)
                    c3 = pal0.get("A", def_c3)

                elif cat_choice == "clean_space":
                    cname, cslogan = _CLEAN_SPACE_NAMES[f_idx % len(_CLEAN_SPACE_NAMES)]
                    fac_name = cname if cname.lower() not in existing_names else f"{cname} {f_idx + 1}"
                    fac_theme = "pure white empty space"
                    fac_slogan = cslogan
                    art_mode = "clean_space"
                    fac_category = "clean_space"
                    art_prompt = "pure white empty space enclave"
                    c1, c2, c3 = WHITE_IDX, WHITE_IDX, BLACK_IDX
                    sub_name = "CleanCanvas"

                elif cat_choice == "painting":
                    avail_p = [pa for pa in FINE_ART_CATALOG if pa.key not in existing_paint_keys] or FINE_ART_CATALOG
                    pa = self.rng.choice(avail_p)
                    existing_paint_keys.add(pa.key)
                    extra_key = pa.key
                    fac_name = f"r/Art {pa.name}" if f"r/Art {pa.name}".lower() not in existing_names else f"{pa.name} {f_idx + 1}"
                    fac_theme = f"{pa.artist} {pa.name}"
                    fac_slogan = f"Preserve {pa.name} in pixel gold!"
                    art_mode = "painting"
                    fac_category = "painting"
                    art_prompt = pa.prompt
                    c1, c2, c3 = pa.primary, pa.secondary, pa.accent
                    theme_desc = pa.color_theme
                    sub_name = "Art"

                elif cat_choice == "fractal":
                    fname, fkind, fslogan = _FRACTAL_NAMES[f_idx % len(_FRACTAL_NAMES)]
                    extra_key = fkind
                    fac_name = fname if fname.lower() not in existing_names else f"{fname} {f_idx + 1}"
                    fac_theme = f"{fkind} mathematical fractal"
                    fac_slogan = fslogan
                    art_mode = "fractal"
                    fac_category = "fractal"
                    art_prompt = f"{fkind} pixel fractal"
                    c1, c2, c3 = BLACK_IDX, PURPLE_IDX, def_c2
                    sub_name = "fractals"

                elif cat_choice == "comic_strip" and pre_comic is not None:
                    existing_comic_keys.add(pre_comic.key)
                    extra_key = pre_comic.key
                    fac_name = pre_comic.title if pre_comic.title.lower() not in existing_names else f"{pre_comic.title} {f_idx + 1}"
                    fac_theme = f"pixel comic strip: {pre_comic.title}"
                    fac_slogan = pre_comic.slogan
                    art_mode = "comic_strip"
                    fac_category = "comic"
                    art_prompt = f"multi-panel comic strip {pre_comic.title}"
                    c1, c2, c3 = pre_comic.primary, pre_comic.secondary, pre_comic.accent
                    sub_name = "comics"

                elif cat_choice == "cartoon_meme" and pre_meme is not None:
                    existing_meme_keys.add(pre_meme.key)
                    extra_key = pre_meme.key
                    fac_name = pre_meme.title if pre_meme.title.lower() not in existing_names else f"{pre_meme.title} {f_idx + 1}"
                    fac_theme = f"pixel {pre_meme.category}: {pre_meme.title}"
                    fac_slogan = pre_meme.slogan
                    art_mode = "cartoon_meme"
                    fac_category = pre_meme.category
                    art_prompt = f"pixel art {pre_meme.category} {pre_meme.title}"
                    c1, c2, c3 = pre_meme.primary, pre_meme.secondary, pre_meme.accent
                    sub_name = "memes" if pre_meme.category == "meme" else "cartoons"

                elif cat_choice == "banner" and (tier == "micro" or fast_banner_override):
                    if tier == "micro":
                        if f_idx % 2 == 0:
                            short_tag, slogan_txt = _LOGO_WORDMARKS[f_idx % len(_LOGO_WORDMARKS)]
                        else:
                            short_tag = f"r/{sub_name[:6]}" if len(sub_name) <= 6 else sub_name[:6].upper()
                            slogan_txt = f"Represent {short_tag}!"
                    else:
                        if f_idx % 3 != 0:
                            short_tag, slogan_txt = _LOGO_WORDMARKS[f_idx % len(_LOGO_WORDMARKS)]
                        else:
                            short_tag = f"r/{sub_name[:8]}" if len(sub_name) <= 8 else sub_name[:8].upper()
                            slogan_txt = f"Represent {short_tag}!"
                    fac_name = short_tag if short_tag.lower() not in existing_names else f"{short_tag} {f_idx + 1}"
                    fac_theme = f"stylized text logo {short_tag}"
                    fac_slogan = slogan_txt
                    art_mode = "banner"
                    fac_category = "logo"
                    art_prompt = f"stylized pixel text logo {short_tag}"
                    c1, c2, c3 = def_c1, def_c2, def_c3
                    typo_spec = self._fast_typography_spec(
                        short_tag,
                        c1,
                        c2,
                        c3,
                        f_idx,
                        is_micro=(tier == "micro"),
                        allow_flux=(tier == "macro"),
                    )

                else:
                    # Full Qwen2.5-VL Pareidolia + Typography Studio for Macro/Medium Diffusion, Procedural, & Hero Wordmarks
                    vlm_calls_in_wave += 1
                    desired_mode = cat_choice if cat_choice in ("diffusion", "procedural", "banner") else "diffusion"
                    if desired_mode == "diffusion" and not self.cfg.use_diffusion:
                        desired_mode = "procedural"
                    if is_corner and desired_mode == "banner":
                        desired_mode = "procedural"

                    self.brain_status = f"Pareidolia ({tier}): Qwen2.5-VL inspecting ({x0},{y0}) for r/{sub_name}..."
                    proposal, theme_desc = self.brain.pareidolia_discover(
                        cluster_img=cand_crop_pil,
                        dominant_color_names=dom_names,
                        existing_names=existing_names_list,
                        existing_themes=existing_themes_list,
                        faction_index=f_idx,
                        desired_mode=desired_mode,
                        is_corner=is_corner,
                        is_edge=is_edge,
                        subreddit_hint=sub_name,
                    )
                    art_mode = proposal.art_mode
                    if art_mode == "diffusion" and not self.cfg.use_diffusion:
                        art_mode = "procedural"
                    c1 = resolve_color(proposal.primary_color, def_c1)
                    c2 = resolve_color(proposal.secondary_color, def_c2)
                    c3 = resolve_color(proposal.accent_color, def_c3)
                    fac_name = proposal.name.strip()[:28]
                    if fac_name.lower() in existing_names:
                        fac_name = f"{fac_name[:22]} {f_idx + 1}"
                    fac_theme = proposal.theme.strip()[:40]
                    fac_slogan = proposal.slogan.strip()[:90]
                    art_prompt = proposal.art_prompt.strip()[:120]
                    fac_category = "logo" if art_mode == "banner" else ("game" if f_idx % 3 == 0 else ("anime" if f_idx % 3 == 1 else "meme"))
                    btxt = (proposal.banner_text or f"r/{sub_name[:8]}").strip()[:14]

                    if is_corner and art_mode == "procedural":
                        ctag = (
                            "top_left" if (x0 == 0 and y0 == 0)
                            else "top_right" if (x0 > 0 and y0 == 0)
                            else "bottom_left" if (x0 == 0 and y0 > 0)
                            else "bottom_right"
                        )
                        art_prompt = f"corner {ctag} {art_prompt}"

                    self.brain_status = f"Typography Studio designing '{fac_name}'..."
                    typo_spec, flux_style_desc = self.brain.design_typography(
                        name=fac_name,
                        theme=fac_theme,
                        art_mode=art_mode,
                        art_prompt=art_prompt,
                        banner_text=btxt,
                        color_theme=theme_desc,
                        primary_idx=c1,
                        secondary_idx=c2,
                        accent_idx=c3,
                        faction_index=f_idx,
                    )

            # --- 3. Generate Blueprint ---
            self.brain_status = f"Creating {tier}/{art_mode} blueprint for '{fac_name}'..."
            bp_np, text_engine_used = self.art.create_blueprint(
                art_mode=art_mode,
                prompt=art_prompt,
                w=bp_w,
                h=bp_h,
                primary=c1,
                secondary=c2,
                accent=c3,
                label=fac_name,
                theme=fac_theme,
                color_theme=theme_desc,
                typography=typo_spec,
                flux_style_desc=flux_style_desc,
                ocr_verifier=self.brain.verify_wordmark_ocr,
                live_crop_rgb=cand_crop_rgb,
                seed=self.cfg.seed + self.next_faction_id * 97 + tick + i * 31,
                extra_key=extra_key,
                ribbon_axis=ribbon_axis or "h",
            )
            bp_h_actual, bp_w_actual = int(bp_np.shape[0]), int(bp_np.shape[1])
            x0 = max(0, min(self.cfg.width - bp_w_actual, x0))
            y0 = max(0, min(self.cfg.height - bp_h_actual, y0))
            valid_px = int((bp_np >= 0).sum())

            # --- 4. Commit Faction & Proportional Recruitment ---
            with self._state_lock:
                fid = self.next_faction_id
                self.next_faction_id += 1
                bp_t = torch.from_numpy(bp_np.astype(np.int16)).to(self.device)

                fac = Faction(
                    id=fid,
                    name=fac_name,
                    theme=fac_theme,
                    slogan=fac_slogan,
                    art_mode=art_mode,
                    art_prompt=art_prompt,
                    primary_color=c1,
                    secondary_color=c2,
                    accent_color=c3,
                    x0=x0,
                    y0=y0,
                    w=bp_w_actual,
                    h=bp_h_actual,
                    blueprint=bp_t,
                    edge_weight=compute_edge_weights(bp_t),
                    created_tick=self.tick,
                    banner_text=(typo_spec.text if (typo_spec and typo_spec.include_text) else ""),
                    color_theme=theme_desc,
                    typography_spec=asdict(typo_spec) if typo_spec is not None else None,
                    text_stamped=(text_engine_used != "none"),
                    size_tier=tier,
                    category=fac_category,
                    subreddit=sub_name,
                    extra_key=extra_key.split("|")[0],
                    ribbon_axis=ribbon_axis,
                    ribbon_thickness=ribbon_thick,
                    is_cameo=is_cameo,
                    last_morph_tick=self.tick,
                )
                style_tag = (
                    f"{text_engine_used}:{typo_spec.font_style}/{typo_spec.color_effect}"
                    if (typo_spec is not None and typo_spec.include_text)
                    else art_mode
                )
                fac.remember(f"Born ({tier}/{fac_category}) at ({x0},{y0}) [{bp_w_actual}x{bp_h_actual}].")
                self.factions.append(fac)
                sim_alive.append(fac)

                target_rec = self._target_initial_members(tier, valid_px)
                recruited = self.policy.recruit_initial_members(fac, target_count=target_rec)
                fac.member_count = recruited
                self.diplomacy.post_public(self.tick, fac, fac.slogan, kind="propaganda")
                cameo_tag = " [CAMEO]" if is_cameo else ""
                self.logger.log_chronicle(
                    self.tick,
                    "founded",
                    f"New {tier} {fac_category} '{fac.name}' ({bp_w_actual}x{bp_h_actual}, {style_tag}){cameo_tag} at ({x0},{y0}) — {recruited} citizens!",
                    faction_id=fac.id,
                    faction_name=fac.name,
                    size_tier=tier,
                    category=fac_category,
                    art_mode=art_mode,
                    bbox=[fac.x0, fac.y0, fac.x1, fac.y1],
                )
                existing_names.add(fac.name.lower())
                existing_names_list.append(fac.name)
                existing_themes_list.append(fac.theme)
                created.append(fac)

        self.brain_status = f"Active ({len([f for f in self.factions if f.alive])} factions competing)"
        return created

    def _run_autonomous_micro_and_flag_diplomacy(self, alive_art: list[Faction]) -> None:
        """Fast local diplomacy so 200 Micro/Flag factions form Indie-Alliance clusters & flag rivalries without bottlenecking Ollama."""
        fac_by_id = {f.id: f for f in alive_art}
        # 1. Micro-sprite neighborhood alliances
        micros = [f for f in alive_art if f.size_tier == "micro" and f.art_mode != "clean_space" and f.integrity >= 0.45]
        self.rng.shuffle(micros)
        for mf in micros[:14]:
            if len(mf.allies) >= 3:
                continue
            nbrs = describe_neighbors(mf, self.factions, max_gap=18)
            for n in nbrs[:2]:
                other = fac_by_id.get(n["id"])
                if other is None or other.is_void or other.id in mf.allies or other.art_mode == "clean_space":
                    continue
                if other.size_tier == "micro" and self.rng.random() < 0.45:
                    mf.allies.add(other.id)
                    other.allies.add(mf.id)
                    mf.enemies.discard(other.id)
                    other.enemies.discard(mf.id)
                    mf.remember(f"Joined micro alliance with {other.name}!")
                    break

        # 2. Adjacent national flag border rivalries and peace treaties
        flags = [f for f in alive_art if f.category == "flag" and f.integrity >= 0.55]
        self.rng.shuffle(flags)
        for fl in flags[:6]:
            nbrs = describe_neighbors(fl, self.factions, max_gap=28)
            flag_nbrs = [
                fac_by_id[n["id"]]
                for n in nbrs
                if n["id"] in fac_by_id and fac_by_id[n["id"]].category == "flag"
            ]
            if not flag_nbrs:
                continue
            rival = flag_nbrs[0]
            if rival.id in fl.allies or rival.id in fl.enemies:
                continue
            if self.rng.random() < 0.52:
                # Flag border rivalry
                fl.raid_target_id = rival.id
                fl.enemies.add(rival.id)
                rival.enemies.add(fl.id)
                fl.role_weights = {"builder": 0.35, "defender": 0.30, "expander": 0.35}
                self.logger.log_chronicle(
                    self.tick,
                    "war",
                    f"FLAG RIVALRY: '{fl.name}' contests borders against '{rival.name}'!",
                    attacker=fl.id,
                    defender=rival.id,
                )
            else:
                # Flag border peace treaty
                fl.allies.add(rival.id)
                rival.allies.add(fl.id)
                self.logger.log_chronicle(
                    self.tick,
                    "alliance",
                    f"FLAG TREATY: '{fl.name}' & '{rival.name}' sealed border hearts!",
                    faction_a=fl.id,
                    faction_b=rival.id,
                )

    def run_geopolitical_epoch(self) -> None:
        """Run one geopolitical round: VLM leads 6 frontline/macro/medium factions + fast autonomy for 200-faction ecosystem."""
        with self._state_lock:
            self.epoch += 1
            ep = self.epoch
            tick = self.tick
            update_faction_metrics(self.canvas, self.factions, self.policy.user_factions)
            self._manage_void_lifecycle()
            alive_art = [f for f in self.factions if f.alive and not f.is_void]
            if not alive_art:
                self.brain_status = "Waiting for factions to emerge..."
                return

            # Natural era turnover + tier-appropriate collapse check across all active factions
            for f in alive_art:
                age = tick - f.created_tick
                if age > 2800:
                    f.hype = max(0.32, f.hype * 0.96)
                min_mem = 14 if f.size_tier == "micro" else (45 if f.size_tier == "medium" else 75)
                if f.integrity < 0.14 and f.member_count < min_mem:
                    f.low_integrity_epochs += 1
                elif age > 5000 and f.integrity < 0.35 and f.member_count < min_mem * 2:
                    f.low_integrity_epochs += 1
                else:
                    f.low_integrity_epochs = 0

                if f.low_integrity_epochs >= 4:
                    f.alive = False
                    self.policy.release_faction_members(f.id)
                    self.logger.log_chronicle(
                        self.tick,
                        "collapse",
                        f"'{f.name}' ({f.size_tier}) dissolved back into free agents.",
                        faction_id=f.id,
                    )

            alive_art = [f for f in self.factions if f.alive and not f.is_void]
            if not alive_art:
                return

            # Local diplomacy for micro neighborhoods and national flag borders
            self._run_autonomous_micro_and_flag_diplomacy(alive_art)

            # Select up to 6 active factions for full Qwen2.5-VL epoch decisions (keeps epoch <= 25s)
            def _urgency(f: Faction) -> float:
                u = self.rng.random() * 0.45
                if f.size_tier in ("macro", "ribbon"):
                    u += 0.9
                elif f.size_tier == "medium":
                    u += 0.5
                if f.category == "flag":
                    u += 0.35
                if f.pending_ally_offers:
                    u += 2.0
                if f.integrity >= 0.78:
                    u += 1.1
                if f.recent_churn >= 0.8:
                    u += 1.0
                if f.integrity < 0.35:
                    u += 0.8
                return u

            chosen = sorted(alive_art, key=_urgency, reverse=True)[:6]

        for fac in chosen:
            if self.stop_requested or not fac.alive:
                continue

            with self._state_lock:
                view_img = self.canvas.render_faction_view_pil(fac, self.factions, margin=36)
                neighbors = describe_neighbors(fac, self.factions, max_gap=44)
                inbox = self.diplomacy.pop_inbox(fac.id)
                public_feed = self.diplomacy.recent_public(6)
                cur_tick = self.tick

            can_morph = (
                (cur_tick - fac.last_morph_tick >= 280)
                and (fac.evolution_stage < 5)
                and (fac.art_mode not in ("flag", "micro_sprite", "clean_space", "ribbon", "painting"))
            )
            self.brain_status = f"Epoch {ep}: Qwen2.5-VL leading '{fac.name}' ({fac.size_tier})..."
            decision = self.brain.faction_epoch_turn(
                faction=fac,
                view_img=view_img,
                neighbors=neighbors,
                inbox=inbox,
                public_feed=public_feed,
                tick=cur_tick,
                can_morph=can_morph,
            )
            self._apply_faction_decision(fac, decision, neighbors, can_morph=can_morph)

        # Trigger Void variant cycle (tendrils, black hole, fractal, monster)
        with self._state_lock:
            has_void = any(f.alive and f.is_void for f in self.factions)
            if (
                self.cfg.enable_void
                and not has_void
                and self.tick >= self.cfg.void_min_tick
                and len(alive_art) >= 8
                and self.rng.random() < self.cfg.void_chance_per_epoch
            ):
                self.spawn_void_eruption(force=False)

            self._refresh_shared_borders()

        self.brain_status = f"Epoch {ep} complete — {len([f for f in self.factions if f.alive])} active factions"

    def _apply_faction_decision(
        self, fac: Faction, dec: FactionDecision, neighbors: list[dict], can_morph: bool = True
    ) -> None:
        s = max(1e-4, dec.builder_ratio + dec.defender_ratio + dec.expander_ratio)
        fac.role_weights = {
            "builder": dec.builder_ratio / s,
            "defender": dec.defender_ratio / s,
            "expander": dec.expander_ratio / s,
        }
        fac.last_action = dec.action
        fac.raid_target_id = None
        fac.expand_dir = _DIR_MAP.get(dec.direction, (0, 0))

        fac_by_id = {f.id: f for f in self.factions if f.alive}
        tgt = fac_by_id.get(dec.target_faction_id) if dec.target_faction_id is not None else None
        if tgt is not None and tgt.id == fac.id:
            tgt = None

        action = dec.action

        if action == "ACCEPT_ALLIANCE":
            aid = tgt.id if (tgt is not None and tgt.id in fac.pending_ally_offers) else (
                next(iter(fac.pending_ally_offers)) if fac.pending_ally_offers else None
            )
            partner = fac_by_id.get(aid) if aid is not None else None
            if partner is not None and not partner.is_void:
                with self._state_lock:
                    fac.allies.add(partner.id)
                    partner.allies.add(fac.id)
                    fac.enemies.discard(partner.id)
                    partner.enemies.discard(fac.id)
                    fac.pending_ally_offers.discard(partner.id)
                    partner.pending_ally_offers.discard(fac.id)
                    fac.hype = min(2.0, fac.hype + 0.25)
                    partner.hype = min(2.0, partner.hype + 0.25)
                    fac.remember(f"Signed border-heart alliance with {partner.name}.")
                    partner.remember(f"{fac.name} accepted our alliance!")
                    msg = dec.diplomacy_message or f"Alliance sealed between {fac.name} and {partner.name}!"
                    self.diplomacy.post_public(self.tick, fac, msg, kind="accept")
                    self.logger.log_chronicle(
                        self.tick,
                        "alliance",
                        f"ALLIANCE SIGNED: '{fac.name}' & '{partner.name}' agreed to weave border hearts! ({dec.reasoning})",
                        faction_a=fac.id,
                        faction_b=partner.id,
                    )
                return
            action = "BUILD"

        if action == "PROPOSE_ALLIANCE":
            if tgt is None and neighbors:
                for n in neighbors:
                    cand = fac_by_id.get(n["id"])
                    if cand is not None and not cand.is_void and cand.id not in fac.allies:
                        tgt = cand
                        break
            if tgt is not None and not tgt.is_void and tgt.id not in fac.allies:
                with self._state_lock:
                    tgt.pending_ally_offers.add(fac.id)
                    msg = dec.diplomacy_message or f"Truce and border hearts, {tgt.name}?"
                    self.diplomacy.send_direct(self.tick, fac, tgt.id, msg, kind="proposal")
                    fac.remember(f"Proposed alliance to {tgt.name}.")
                    self.logger.log_chronicle(
                        self.tick,
                        "diplomacy",
                        f"'{fac.name}' proposed an alliance to '{tgt.name}': \"{msg}\"",
                        faction_a=fac.id,
                        faction_b=tgt.id,
                    )
                return
            action = "BUILD"

        if action == "RAID":
            if tgt is None and neighbors:
                for n in neighbors:
                    cand = fac_by_id.get(n["id"])
                    if cand is not None and cand.id not in fac.allies:
                        tgt = cand
                        break
            if tgt is not None:
                with self._state_lock:
                    fac.raid_target_id = tgt.id
                    fac.enemies.add(tgt.id)
                    tgt.enemies.add(fac.id)
                    was_ally = tgt.id in fac.allies
                    fac.allies.discard(tgt.id)
                    tgt.allies.discard(fac.id)
                    fac.role_weights = {"builder": 0.30, "defender": 0.25, "expander": 0.45}
                    msg = dec.diplomacy_message or f"{fac.name} marches on {tgt.name}!"
                    self.diplomacy.send_direct(self.tick, fac, tgt.id, msg, kind="threat")
                    self.diplomacy.post_public(self.tick, fac, msg, kind="threat")
                    fac.remember(f"Launched raid against {tgt.name}.")
                    kind_tag = "betrayal" if was_ally else "war"
                    prefix = "BETRAYAL! " if was_ally else "WAR: "
                    self.logger.log_chronicle(
                        self.tick,
                        kind_tag,
                        f"{prefix}'{fac.name}' launched a pixel raid on '{tgt.name}': \"{msg}\" ({dec.reasoning})",
                        attacker=fac.id,
                        defender=tgt.id,
                    )
                return
            action = "BUILD"

        # Enforce per-tier expansion size limits
        tier_max_w, tier_max_h = _TIER_MAX_DIM.get(fac.size_tier, (58, 58))
        if fac.art_mode == "banner":
            tier_max_w = 136 if fac.size_tier == "macro" else (84 if fac.size_tier == "medium" else 34)
            tier_max_h = 44 if fac.size_tier == "macro" else (30 if fac.size_tier == "medium" else 16)
        elif fac.art_mode == "flag":
            tier_max_w = 80 if fac.size_tier == "macro" else (50 if fac.size_tier == "medium" else 26)
            tier_max_h = 52 if fac.size_tier == "macro" else (34 if fac.size_tier == "medium" else 17)
        elif fac.art_mode == "comic_strip":
            tier_max_w = 104 if fac.size_tier == "macro" else 74
            tier_max_h = 58 if fac.size_tier == "macro" else 48

        if action == "MORPH_ART" and not can_morph:
            action = "FORTIFY" if not fac.has_border_frame else "BUILD"
        elif action == "EXPAND":
            if fac.art_mode == "ribbon":
                with self._state_lock:
                    self._advance_ribbons()
                return
            if fac.size_tier == "micro" or (fac.w >= tier_max_w and fac.h >= tier_max_h):
                action = "FORTIFY" if not fac.has_border_frame else "BUILD"

        if action == "FORTIFY":
            with self._state_lock:
                if not fac.has_border_frame:
                    bp_np = fac.blueprint.detach().cpu().numpy()
                    t_spec = TypographyStyleSpec(**fac.typography_spec) if fac.typography_spec else None
                    fort_np = self.art.fortify_blueprint(
                        bp_np,
                        fac.primary_color,
                        fac.secondary_color,
                        fac.accent_color,
                        typography=t_spec,
                        already_stamped=fac.text_stamped,
                        art_mode=fac.art_mode,
                    )
                    fac.set_blueprint(fort_np, device=self.device)
                    fac.has_border_frame = True
                    if t_spec is not None and t_spec.include_text and max(fac.w, fac.h) >= 34:
                        fac.text_stamped = True
                fac.role_weights = {"builder": 0.35, "defender": 0.55, "expander": 0.10}
                fac.remember("Fortified territory and raised defenses.")
                if dec.diplomacy_message:
                    self.diplomacy.post_public(self.tick, fac, dec.diplomacy_message, kind="propaganda")
                self.logger.log_chronicle(
                    self.tick,
                    "fortify",
                    f"'{fac.name}' fortified its borders. ({dec.reasoning})",
                    faction_id=fac.id,
                )
            return

        if action == "EXODUS":
            with self._state_lock:
                nx0, ny0 = find_open_exodus_spot(self.canvas, self.factions, fac.w, fac.h, fac.id)
                old_pos = (fac.x0, fac.y0)
                fac.x0, fac.y0 = nx0, ny0
                fac.low_integrity_epochs = 0
                fac.morale = 0.72
                fac.hype = 1.15
                fac.remember(f"Exodus from {old_pos} to ({nx0},{ny0})!")
                msg = dec.diplomacy_message or f"{fac.name} relocates to ({nx0},{ny0})!"
                self.diplomacy.post_public(self.tick, fac, msg, kind="propaganda")
                self.logger.log_chronicle(
                    self.tick,
                    "exodus",
                    f"EXODUS: '{fac.name}' migrated from {old_pos} to ({nx0},{ny0})! ({dec.reasoning})",
                    faction_id=fac.id,
                    new_bbox=[fac.x0, fac.y0, fac.x1, fac.y1],
                )
            return

        if action in ("MORPH_ART", "EXPAND"):
            raw_new = dec.new_art_prompt.strip()
            if raw_new and len(raw_new) >= 6:
                new_prompt = raw_new
            else:
                base = fac.art_prompt.replace("evolved ", "").split(",")[0].strip()[:65]
                flair = self.rng.choice(
                    [
                        "with pixel sparkles",
                        "upgraded detailed 16-bit emblem",
                        "with arcade shading",
                        "with festive pixel ribbons",
                    ]
                )
                new_prompt = f"{base} {flair}"
            with self._state_lock:
                nx0, ny0, nw, nh = fac.x0, fac.y0, fac.w, fac.h
                if action == "EXPAND":
                    grow = 10 if fac.size_tier == "medium" else 14
                    dx, dy = fac.expand_dir
                    if dx == 0 and dy == 0:
                        dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    if dx > 0 and nw + grow <= tier_max_w:
                        nw = min(tier_max_w, nw + grow)
                    elif dx < 0 and nw + grow <= tier_max_w:
                        shift = min(grow, nx0)
                        nx0 -= shift
                        nw += shift
                    elif dy > 0 and nh + grow <= tier_max_h:
                        nh = min(tier_max_h, nh + grow)
                    elif dy < 0 and nh + grow <= tier_max_h:
                        shift = min(grow, ny0)
                        ny0 -= shift
                        nh += shift
                    else:
                        nw = min(tier_max_w, nw + 8)
                        nh = min(tier_max_h, nh + (4 if fac.art_mode == "banner" else 8))
                    nx0, ny0 = _snap_to_walls(nx0, ny0, nw, nh, self.cfg.width, self.cfg.height, snap=12)

                live_crop_rgb = self.canvas.crop_rgb_numpy(nx0, ny0, nx0 + nw, ny0 + nh)
                next_stage = min(5, fac.evolution_stage + 1)
                seed_val = self.cfg.seed + fac.id * 1009 + self.tick

            self.brain_status = f"Morphing '{fac.name}' Stage {next_stage} ({fac.art_mode})..."
            t_spec = TypographyStyleSpec(**fac.typography_spec) if fac.typography_spec else None
            morphed_np = self.art.morph_blueprint(
                art_mode=fac.art_mode,
                new_prompt=new_prompt,
                live_crop_rgb=live_crop_rgb,
                target_w=nw,
                target_h=nh,
                primary=fac.primary_color,
                secondary=fac.secondary_color,
                accent=fac.accent_color,
                label=fac.name,
                theme=fac.theme,
                color_theme=fac.color_theme,
                typography=t_spec,
                ocr_verifier=self.brain.verify_wordmark_ocr,
                stage=next_stage,
                strength=0.82 if action == "MORPH_ART" else 0.86,
                seed=seed_val,
                extra_key=fac.extra_key,
                ribbon_axis=fac.ribbon_axis or "h",
            )

            with self._state_lock:
                fac.set_blueprint(morphed_np, x0=nx0, y0=ny0, device=self.device)
                fac.art_prompt = new_prompt[:120]
                fac.evolution_stage = next_stage
                fac.has_border_frame = False
                fac.text_stamped = bool(t_spec is not None and t_spec.include_text and max(nw, nh) >= 34)
                fac.last_morph_tick = self.tick
                fac.hype = min(2.2, fac.hype + 0.30)
                fac.remember(f"Stage {next_stage} {action}: '{new_prompt}'.")
                msg = dec.diplomacy_message or f"{fac.name} evolves: {new_prompt}!"
                self.diplomacy.post_public(self.tick, fac, msg, kind="propaganda")
                self.logger.log_chronicle(
                    self.tick,
                    "morph" if action == "MORPH_ART" else "expand",
                    f"{action}: '{fac.name}' ({fac.size_tier}) -> Stage {next_stage} ({nw}x{nh}) ({dec.reasoning})",
                    faction_id=fac.id,
                    stage=next_stage,
                    bbox=[fac.x0, fac.y0, fac.x1, fac.y1],
                )
            return

        with self._state_lock:
            if dec.diplomacy_message:
                self.diplomacy.post_public(self.tick, fac, dec.diplomacy_message, kind="propaganda")
            fac.remember(f"Focused on building ({dec.reasoning[:50]}).")

    # ------------------------------------------------------------------
    # 4-Variant Void Lifecycle (REQ-DIP-010, REQ-ART-013)
    # ------------------------------------------------------------------

    def spawn_void_eruption(self, force: bool = False, variant: str | None = None) -> Faction | None:
        with self._state_lock:
            if any(f.alive and f.is_void for f in self.factions) and not force:
                return None
            v_kind = variant or _VOID_VARIANTS[self._void_cycle_idx % len(_VOID_VARIANTS)]
            self._void_cycle_idx += 1

            size = 86 if v_kind in ("black_hole", "monster") else (76 if v_kind == "fractal" else 92)
            vx0, vy0 = find_void_spawn_spot(self.canvas, self.factions, size=size)
            fid = self.next_faction_id
            self.next_faction_id += 1
            seed_v = self.cfg.seed + fid * 313 + self.tick

            if v_kind == "black_hole":
                bp_np = generate_black_hole(size, size, seed=seed_v)
                v_name, v_slogan = "THE BLACK HOLE", "ALL PIXELS FALL INTO THE SINGULARITY"
            elif v_kind == "fractal":
                fk = self.rng.choice(["sierpinski", "mandelbrot", "carpet", "diamond"])
                bp_np = generate_fractal(fk, size, size, seed=seed_v)
                v_name, v_slogan = f"VOID FRACTAL ({fk.upper()})", "RECURSIVE ABYSS CONSUMES THE BOARD"
            elif v_kind == "monster":
                bp_np = generate_void_monster(size, size, seed=seed_v)
                v_name, v_slogan = "VOID MOTHER", "THE ABYSS OPENS ITS EYES"
            else:
                bp_np = np.full((size, size), BLACK_IDX, dtype=np.int16)
                v_name, v_slogan = "THE VOID", "CONSUME ALL PIXELS"

            bp_t = torch.from_numpy(bp_np.astype(np.int16)).to(self.device)
            void_fac = Faction(
                id=fid,
                name=v_name,
                theme=f"void {v_kind}",
                slogan=v_slogan,
                art_mode="void",
                art_prompt=f"void {v_kind} anomaly",
                primary_color=BLACK_IDX,
                secondary_color=BURGUNDY_IDX,
                accent_color=DARK_RED_IDX,
                x0=vx0,
                y0=vy0,
                w=size,
                h=size,
                blueprint=bp_t,
                edge_weight=compute_edge_weights(bp_t),
                created_tick=self.tick,
                size_tier="macro",
                category="void",
                void_variant=v_kind,
                is_void=True,
                hype=1.65,
            )
            self.factions.append(void_fac)
            recruited = self.policy.recruit_initial_members(void_fac, target_count=1500)
            void_fac.member_count = recruited
            self.diplomacy.post_public(self.tick, void_fac, v_slogan, kind="threat")
            self.logger.log_chronicle(
                self.tick,
                "void",
                f"VOID ERUPTION [{v_kind.upper()}]: '{v_name}' opened at ({vx0},{vy0}) with {recruited} shadow pixels!",
                faction_id=void_fac.id,
                void_variant=v_kind,
                bbox=[void_fac.x0, void_fac.y0, void_fac.x1, void_fac.y1],
            )
            return void_fac

    def _manage_void_lifecycle(self) -> None:
        for f in self.factions:
            if not (f.alive and f.is_void):
                continue
            age = self.tick - f.created_tick
            if f.void_variant == "tendrils" and age < self.cfg.void_max_duration_ticks // 2 and f.w < 128:
                f.x0 = max(0, f.x0 - 6)
                f.y0 = max(0, f.y0 - 6)
                f.w = min(self.cfg.width - f.x0, f.w + 12)
                f.h = min(self.cfg.height - f.y0, f.h + 12)
            elif age >= self.cfg.void_max_duration_ticks:
                f.alive = False
                self.policy.release_faction_members(f.id)
                self.logger.log_chronicle(
                    self.tick,
                    "void",
                    f"'{f.name}' ({f.void_variant}) receded into the abyss, leaving scars for factions to reclaim!",
                    faction_id=f.id,
                )

    # ------------------------------------------------------------------
    # Checkpointing & UI Snapshot
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        with self._state_lock:
            ckpt_dir = self.cfg.run_dir / "checkpoint"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ckpt_dir / "canvas.npz",
                colors=self.canvas.colors_numpy(),
                prev_colors=self.canvas.prev_colors_numpy(),
                owner=self.canvas.owner_numpy(),
                last_edit_tick=self.canvas.last_edit_tick.detach().cpu().numpy(),
                user_factions=self.policy.user_factions.detach().cpu().numpy(),
            )
            fac_meta = []
            bp_dict: dict[str, np.ndarray] = {}
            for f in self.factions:
                bp_dict[f"bp_{f.id}"] = f.blueprint.detach().cpu().numpy()
                fac_meta.append(
                    {
                        "id": f.id,
                        "name": f.name,
                        "theme": f.theme,
                        "slogan": f.slogan,
                        "art_mode": f.art_mode,
                        "art_prompt": f.art_prompt,
                        "primary_color": f.primary_color,
                        "secondary_color": f.secondary_color,
                        "accent_color": f.accent_color,
                        "x0": f.x0,
                        "y0": f.y0,
                        "w": f.w,
                        "h": f.h,
                        "created_tick": f.created_tick,
                        "banner_text": f.banner_text,
                        "color_theme": f.color_theme,
                        "typography_spec": f.typography_spec,
                        "text_stamped": f.text_stamped,
                        "size_tier": f.size_tier,
                        "category": f.category,
                        "subreddit": f.subreddit,
                        "extra_key": f.extra_key,
                        "ribbon_axis": f.ribbon_axis,
                        "ribbon_thickness": f.ribbon_thickness,
                        "void_variant": f.void_variant,
                        "is_cameo": f.is_cameo,
                        "is_void": f.is_void,
                        "alive": f.alive,
                        "role_weights": f.role_weights,
                        "expand_dir": list(f.expand_dir),
                        "raid_target_id": f.raid_target_id,
                        "last_action": f.last_action,
                        "last_morph_tick": f.last_morph_tick,
                        "evolution_stage": f.evolution_stage,
                        "has_border_frame": f.has_border_frame,
                        "allies": sorted(f.allies),
                        "enemies": sorted(f.enemies),
                        "morale": f.morale,
                        "hype": f.hype,
                        "memory": f.memory,
                    }
                )
            if bp_dict:
                np.savez_compressed(ckpt_dir / "blueprints.npz", **bp_dict)
            pop_stats = self.policy.get_population_stats() if hasattr(self.policy, "get_population_stats") else {}
            meta = {
                "tick": self.tick,
                "epoch": self.epoch,
                "next_faction_id": self.next_faction_id,
                "total_events": self.logger.total_events,
                "void_cycle_idx": self._void_cycle_idx,
                "population": pop_stats,
                "factions": fac_meta,
            }
            (ckpt_dir / "state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.logger.flush()

    def _load_checkpoint(self) -> None:
        ckpt_dir = self.cfg.run_dir / "checkpoint"
        state_file = ckpt_dir / "state.json"
        canvas_file = ckpt_dir / "canvas.npz"
        if not state_file.exists() or not canvas_file.exists():
            return
        meta = json.loads(state_file.read_text(encoding="utf-8"))
        c_npz = np.load(canvas_file)
        self.canvas.colors = torch.from_numpy(c_npz["colors"]).to(self.device)
        if "prev_colors" in c_npz:
            self.canvas.prev_colors = torch.from_numpy(c_npz["prev_colors"]).to(self.device)
        else:
            self.canvas.prev_colors = self.canvas.colors.clone()
        self.canvas.owner = torch.from_numpy(c_npz["owner"]).to(self.device)
        self.canvas.last_edit_tick = torch.from_numpy(c_npz["last_edit_tick"]).to(self.device)
        uf = torch.from_numpy(c_npz["user_factions"]).to(self.device)
        if uf.numel() == self.policy.user_factions.numel():
            self.policy.user_factions = uf

        self.tick = int(meta["tick"])
        self.epoch = int(meta["epoch"])
        self.next_faction_id = int(meta["next_faction_id"])
        self._void_cycle_idx = int(meta.get("void_cycle_idx", 0))
        pop = meta.get("population", {})
        if pop and hasattr(self.policy, "next_user_id"):
            self.policy.total_unique_users = int(pop.get("total_unique_users", self.policy.total_unique_users))
            self.policy.next_user_id = self.policy.total_unique_users + 1
            self.policy.total_completed_sessions = int(pop.get("completed_sessions", 0))
            self.policy.total_one_shot_sessions = int(pop.get("one_shot_sessions", 0))

        bp_file = ckpt_dir / "blueprints.npz"
        bp_npz = np.load(bp_file) if bp_file.exists() else {}

        self.factions = []
        for fd in meta.get("factions", []):
            key = f"bp_{fd['id']}"
            bp_np = bp_npz[key] if key in bp_npz else np.full((fd["h"], fd["w"]), -1, dtype=np.int16)
            bp_t = torch.from_numpy(bp_np.astype(np.int16)).to(self.device)
            fac = Faction(
                id=fd["id"],
                name=fd["name"],
                theme=fd["theme"],
                slogan=fd["slogan"],
                art_mode=fd["art_mode"],
                art_prompt=fd["art_prompt"],
                primary_color=fd["primary_color"],
                secondary_color=fd["secondary_color"],
                accent_color=fd["accent_color"],
                x0=fd["x0"],
                y0=fd["y0"],
                w=fd["w"],
                h=fd["h"],
                blueprint=bp_t,
                edge_weight=compute_edge_weights(bp_t),
                created_tick=fd["created_tick"],
                banner_text=fd.get("banner_text", ""),
                color_theme=fd.get("color_theme", ""),
                typography_spec=fd.get("typography_spec"),
                text_stamped=fd.get("text_stamped", False),
                size_tier=fd.get("size_tier", "medium"),
                category=fd.get("category", "art"),
                subreddit=fd.get("subreddit", ""),
                extra_key=fd.get("extra_key", ""),
                ribbon_axis=fd.get("ribbon_axis", ""),
                ribbon_thickness=fd.get("ribbon_thickness", 8),
                void_variant=fd.get("void_variant", "tendrils"),
                is_cameo=fd.get("is_cameo", False),
                is_void=fd["is_void"],
                alive=fd["alive"],
                role_weights=fd["role_weights"],
                expand_dir=tuple(fd["expand_dir"]),
                raid_target_id=fd["raid_target_id"],
                last_action=fd["last_action"],
                last_morph_tick=fd["last_morph_tick"],
                evolution_stage=fd["evolution_stage"],
                has_border_frame=fd["has_border_frame"],
                allies=set(fd["allies"]),
                enemies=set(fd["enemies"]),
                morale=fd["morale"],
                hype=fd["hype"],
                memory=fd.get("memory", []),
            )
            self.factions.append(fac)
        self._refresh_shared_borders()

    def get_ui_state(self) -> dict[str, Any]:
        with self._state_lock:
            alive = [f for f in self.factions if f.alive]
            alive.sort(key=lambda f: f.member_count, reverse=True)
            free_count = int((self.policy.user_factions == -1).sum().item())
            alliances: list[tuple[int, int]] = []
            wars: list[tuple[int, int]] = []
            seen_a: set[tuple[int, int]] = set()
            seen_w: set[tuple[int, int]] = set()
            alive_ids = {f.id for f in alive}
            for f in alive:
                for a in f.allies:
                    if a in alive_ids:
                        p = (min(f.id, a), max(f.id, a))
                        if p not in seen_a:
                            seen_a.add(p)
                            alliances.append(p)
                for e in f.enemies:
                    if e in alive_ids:
                        p = (min(f.id, e), max(f.id, e))
                        if p not in seen_w:
                            seen_w.add(p)
                            wars.append(p)
            pop = self.policy.get_population_stats() if hasattr(self.policy, "get_population_stats") else {}
            tier_counts = dict(Counter(f.size_tier for f in alive if not f.is_void))
            category_counts = dict(Counter(f.category for f in alive))
            return {
                "tick": self.tick,
                "max_ticks": self.cfg.max_ticks,
                "epoch": self.epoch,
                "width": self.cfg.width,
                "height": self.cfg.height,
                "total_placements": self.logger.total_events,
                "actual_tps": round(self.actual_tps, 1),
                "pixels_per_sec": int(self.pixels_per_sec),
                "target_tps": self.target_tps,
                "paused": self.paused,
                "free_agents": free_count,
                "num_users": self.cfg.num_users,
                "population": pop,
                "tier_counts": tier_counts,
                "category_counts": category_counts,
                "device": str(self.device),
                "policy_name": self.policy_name,
                "ollama_ready": self.ollama_ready,
                "use_diffusion": self.cfg.use_diffusion,
                "brain_status": self.brain_status,
                "factions": [f.to_summary_dict() for f in alive],
                "alliances": alliances,
                "wars": wars,
                "chronicle": self.logger.recent_chronicle[-60:][::-1],
            }
