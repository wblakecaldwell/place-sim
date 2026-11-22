"""Main simulation engine uniting GPU MPS pixel loop, 4-act r/place expansion, AI-first factions, timezones, flash-raids, and async VLM/FLUX strategist."""

from __future__ import annotations

import json
import logging
import queue
import random
import threading
import time
from collections import Counter, deque
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
from pixel_battle.brain import AIFactionIdea, COLOR_THEMES, FactionDecision, LocalVLMBrain
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
    FLAG_TIMEZONES,
    Faction,
    compute_edge_weights,
    compute_shared_border_rect,
    describe_neighbors,
    timezone_awake_factor,
)
from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    BURGUNDY_IDX,
    COLOR_NAMES,
    DARK_RED_IDX,
    PURPLE_IDX,
    WHITE_IDX,
    YELLOW_IDX,
    remap_indices_to_allowed,
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

# Empirical placeAtlas size-tier shares (REQ-SIM-010)
_TIER_TARGETS: dict[str, float] = {
    "micro": 0.58,
    "medium": 0.27,
    "macro": 0.12,
    "ribbon": 0.03,
}

_LOGO_WORDMARKS: list[tuple[str, str, str, str]] = [
    ("像素联盟", "一起守护像素画布！", "ASIA_PACIFIC", "zh"),
    ("¡VIVA!", "Colores unidos en el lienzo.", "LATAM", "es"),
    ("FRANÇAIS", "Artistes du soir unis.", "EUROPE", "fr"),
    ("PIXELN", "Gemeinsam Pixel für Pixel.", "EUROPE", "de"),
    ("日本画", "ドット絵の仲間たち！", "ASIA_PACIFIC", "ja"),
    ("픽셀단", "함께 만드는 픽셀 아트!", "ASIA_PACIFIC", "ko"),
    ("ARCADE", "Insert coin to paint.", "AMERICAS", "en"),
    ("LO-FI", "Chill beats and clean pixels.", "GLOBAL", "en"),
    ("RETRO", "16-bit nostalgia forever.", "AMERICAS", "en"),
    ("NO VOID", "Keep the canvas bright.", "GLOBAL", "en"),
    ("BRASIL", "Unidos pelo canvas!", "LATAM", "pt"),
    ("MERHABA", "Renkli piksel dostları!", "EUROPE", "tr"),
]

_CLEAN_SPACE_NAMES: list[tuple[str, str]] = [
    ("Whiteout Guild", "Keep this space pure white!"),
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
_REGIONAL_TIMEZONES = ("AMERICAS", "EUROPE", "ASIA_PACIFIC", "LATAM")


def _snap_to_walls(x0: int, y0: int, w: int, h: int, W: int, H: int, snap: int = 16) -> tuple[int, int]:
    """Snap a territory bounding box flush against active canvas walls/corners when close to an edge."""
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
    """Thread-safe r/place engine with real-time GPU ticking, 4-act canvas expansion, AI-first factions, timezones, and flash-raids."""

    def __init__(
        self,
        cfg: SimConfig,
        policy: PixelPolicy | None = None,
        resume: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.rng = random.Random(cfg.seed)

        # Act 1 starts at W/2 x H/2 with 16 initial colors when expansion is enabled
        can_expand = cfg.enable_expansion and cfg.width >= 256 and cfg.height >= 256
        init_w = (cfg.width // 2) if can_expand else cfg.width
        init_h = (cfg.height // 2) if can_expand else cfg.height
        init_tier = 1 if can_expand else 3

        self.canvas = CanvasState(
            cfg.width,
            cfg.height,
            self.device,
            active_w=init_w,
            active_h=init_h,
            palette_tier=init_tier,
        )
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

        # Align initial citizen positions and timezone clock with Act 1 active viewport
        if hasattr(self.policy, "active_w"):
            self.policy.active_w = init_w
            self.policy.active_h = init_h
            self.policy.enable_timezones = cfg.enable_timezones
            self.policy.pos_x = self.policy.pos_x % init_w
            self.policy.pos_y = self.policy.pos_y % init_h

        self.art = ArtManager(cfg, self.device)
        llm_endpoint = cfg.llm_url or cfg.ollama_url
        llm_model = cfg.llm_model or cfg.vlm_model
        self.brain = LocalVLMBrain(
            ollama_url=llm_endpoint,
            vlm_model=llm_model,
            keychain_service=cfg.keychain_service,
        )
        self.subreddits = SubredditSampler(seed=cfg.seed)
        self.diplomacy = DiplomacyBoard()
        self.logger = EventLogger(cfg.run_dir, cfg.width, cfg.height, resume=resume)

        self.factions: list[Faction] = []
        self.next_faction_id: int = 0
        self.shared_borders: list[tuple[int, int, tuple[int, int, int, int]]] = []
        self._void_cycle_idx: int = 0
        self._idea_queue: deque[AIFactionIdea] = deque()
        self._used_flag_keys_ever: set[str] = set()

        # 4-Act progression & 72-hour virtual clock state
        self.act: int = 1 if can_expand else 3
        self.virtual_hour: float = 0.0
        self._land_rush_region: tuple[int, int, int, int] | None = None
        self._land_rush_until_tick: int = 0

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
        self.brain_status: str = "Act 1 primordial chaos — citizens arriving..."
        self.ollama_ready: bool = self.brain.check_health()

        self._state_lock = threading.RLock()
        self._job_queue: queue.Queue[str] = queue.Queue(maxsize=16)
        self._queued_jobs: set[str] = set()
        self._brain_busy = threading.Event()
        self._sim_thread: threading.Thread | None = None
        self._brain_thread: threading.Thread | None = None

        if resume:
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Lifecycle & 4-Act Milestone Controller
    # ------------------------------------------------------------------

    def get_progress(self) -> float:
        if self.cfg.max_hours > 0:
            return min(1.0, max(0.0, (time.time() - self.start_wall_time) / (self.cfg.max_hours * 3600.0)))
        return min(1.0, max(0.0, self.tick / max(1, self.cfg.max_ticks)))

    def is_finished(self) -> bool:
        if self.cfg.max_hours > 0:
            return (time.time() - self.start_wall_time) >= self.cfg.max_hours * 3600.0
        return self.tick >= self.cfg.max_ticks

    def _prime_timezone_now(self) -> str:
        """Return the regional timezone currently closest to its prime-time peak."""
        h = self.virtual_hour % 24.0
        return max(_REGIONAL_TIMEZONES, key=lambda tz: timezone_awake_factor(tz, h))

    def _scatter_free_agents_into(self, x0: int, y0: int, x1: int, y1: int, frac: float = 0.35) -> None:
        """Send a wave of free agents rushing into newly unlocked Eastern/Southern territory."""
        free_idx = torch.nonzero(self.policy.user_factions == -1, as_tuple=False).squeeze(-1)
        if free_idx.numel() == 0:
            return
        n_rush = max(1, int(free_idx.numel() * frac))
        perm = torch.randperm(free_idx.numel(), device=self.device)[:n_rush]
        chosen = free_idx[perm]
        self.policy.pos_x[chosen] = torch.randint(x0, max(x0 + 1, x1), (n_rush,), dtype=torch.int32, device=self.device)
        self.policy.pos_y[chosen] = torch.randint(y0, max(y0 + 1, y1), (n_rush,), dtype=torch.int32, device=self.device)

    def _save_peak_snapshots(self) -> None:
        """Preserve final colorful canvas snapshot right before Act 4 Whiteout begins."""
        try:
            self.canvas.to_pil(scale=1, active_only=False).save(self.cfg.run_dir / "final_peak.png")
            s_2k = max(1, 2048 // max(1, self.cfg.width))
            self.canvas.to_pil(scale=s_2k, active_only=False).save(self.cfg.run_dir / "final_peak_2k.png")
        except Exception as exc:
            log.warning("Could not save final_peak snapshot: %s", exc)

    def _check_act_milestones(self) -> None:
        """Drive 4-Act r/place 2022 viewport expansions, palette unlocks, and Whiteout finale."""
        progress = self.get_progress()
        self.virtual_hour = progress * self.cfg.virtual_hours_total
        if hasattr(self.policy, "virtual_hour_utc"):
            self.policy.virtual_hour_utc = self.virtual_hour % 24.0
            self.policy.enable_timezones = self.cfg.enable_timezones

        # Act 1 -> Act 2: Expand East (W x H/2), unlock 24 colors
        if self.cfg.enable_expansion and self.act == 1 and progress >= self.cfg.act2_progress:
            new_w, new_h = self.cfg.width, max(64, self.cfg.height // 2)
            self.act = 2
            self.canvas.expand_viewport(new_w, new_h, 2)
            self._land_rush_region = (self.cfg.width // 2, 0, new_w, new_h)
            self._land_rush_until_tick = self.tick + 700
            self._scatter_free_agents_into(self.cfg.width // 2, 0, new_w, new_h, frac=0.40)
            self.logger.log_chronicle(
                self.tick,
                "expand_canvas",
                f"ACT 2 EXPANSION: Canvas doubled East to {new_w}x{new_h} and unlocked 24 colors! Eastern land rush begins!",
                act=2,
                active_w=new_w,
                active_h=new_h,
                palette_size=24,
            )
            self.queue_job("DISCOVER")

        # Act 2 -> Act 3: Expand South (W x H), unlock all 32 colors
        if self.cfg.enable_expansion and self.act == 2 and progress >= self.cfg.act3_progress:
            new_w, new_h = self.cfg.width, self.cfg.height
            self.act = 3
            self.canvas.expand_viewport(new_w, new_h, 3)
            self._land_rush_region = (0, self.cfg.height // 2, new_w, new_h)
            self._land_rush_until_tick = self.tick + 800
            self._scatter_free_agents_into(0, self.cfg.height // 2, new_w, new_h, frac=0.40)
            self.logger.log_chronicle(
                self.tick,
                "expand_canvas",
                f"ACT 3 EXPANSION: Canvas expanded South to full {new_w}x{new_h} with all 32 colors unlocked! Southern land rush begins!",
                act=3,
                active_w=new_w,
                active_h=new_h,
                palette_size=32,
            )
            self.queue_job("DISCOVER")

        # Act 3 -> Act 4: Save peak colorful canvas, then lock palette to White-only finale
        if self.cfg.enable_whiteout and self.act < 4 and progress >= self.cfg.whiteout_progress:
            self._save_peak_snapshots()
            self.act = 4
            self.canvas.expand_viewport(self.cfg.width, self.cfg.height, 4)
            self._land_rush_region = None
            self.brain_status = "ACT 4 WHITEOUT FINALE — only white pixels remain..."
            self.logger.log_chronicle(
                self.tick,
                "whiteout",
                "ACT 4 WHITEOUT FINALE: All colors locked except pure white (#FFFFFF). The Great Whiteout sweeps the canvas!",
                act=4,
                active_w=self.cfg.width,
                active_h=self.cfg.height,
                palette_size=1,
            )

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
            f"Simulation started ({self.cfg.preset}): {self.canvas.active_w}x{self.canvas.active_h} "
            f"-> {self.cfg.width}x{self.cfg.height} canvas, palette={len(self.canvas.allowed_colors)} colors, "
            f"{self.cfg.num_users:,} online slots, max_factions={self.cfg.max_active_factions}, "
            f"device={self.device}, policy={self.policy_name}, horizon={horizon_str}, "
            f"llm={self.brain.vlm_model} ({'online' if self.ollama_ready else 'fallback'}), "
            f"flux_art={'on' if self.cfg.use_flux_art else 'off'}.",
            act=self.act,
            active_w=self.canvas.active_w,
            active_h=self.canvas.active_h,
            palette_size=len(self.canvas.allowed_colors),
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
            self._brain_thread.join(timeout=5.0)
        self.save_checkpoint()
        self.logger.close()
        self.brain.close()

    def queue_job(self, job: str) -> None:
        with self._state_lock:
            if job != "STOP" and job in self._queued_jobs:
                return
            try:
                self._job_queue.put_nowait(job)
                if job != "STOP":
                    self._queued_jobs.add(job)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Non-Blocking Fast GPU Tick Loop (REQ-SYS-006)
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

            if self.canvas.is_whiteout:
                # Pace Act 4 Whiteout smoothly (~10-15k px/s) so the finale sweeps cleanly without flooding events.bin
                elapsed = time.perf_counter() - step_start
                min_wo_dt = max(0.55, (1.0 / self.target_tps) if self.target_tps > 0 else 0.55)
                if elapsed < min_wo_dt:
                    time.sleep(min_wo_dt - elapsed)
            elif self._brain_busy.is_set() and self.device.type == "mps":
                # Give LM Studio MLX and FLUX clean Metal queue slices without stopping GPU pixel placements
                time.sleep(0.22)
            elif self.target_tps > 0:
                min_dt = 1.0 / self.target_tps
                elapsed = time.perf_counter() - step_start
                if elapsed < min_dt:
                    time.sleep(min_dt - elapsed)

        if self.is_finished():
            self.brain_status = f"Finished at tick {self.tick} ({self.logger.total_events:,} pixels)!"
            self.save_checkpoint()
            try:
                self.canvas.to_pil(scale=1, active_only=False).save(self.cfg.run_dir / "final.png")
                s_2k = max(1, 2048 // max(1, self.cfg.width))
                self.canvas.to_pil(scale=s_2k, active_only=False).save(self.cfg.run_dir / "final_2k.png")
                if not (self.cfg.run_dir / "final_peak.png").exists():
                    self._save_peak_snapshots()
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
            self._check_act_milestones()
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

            if not self.canvas.is_whiteout:
                if tick % 20 == 0:
                    update_faction_metrics(self.canvas, self.factions, self.policy.user_factions)
                    self.policy.social_step(self.factions, self.canvas)
                    self._refresh_shared_borders()
                    self._manage_flash_raids()

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

                if (
                    self.cfg.enable_flash_raids
                    and tick >= self.cfg.chaos_ticks + 220
                    and (tick - self.cfg.chaos_ticks) % self.cfg.flash_raid_interval_ticks == 0
                ):
                    self.queue_job("FLASH_RAID")

            if tick % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()
            if tick % 100 == 0 and self.device.type == "mps":
                torch.mps.empty_cache()

        return n_placed

    def _manage_flash_raids(self) -> None:
        """Disperse temporary streamer flash-raid mobs once their blitz timer expires."""
        for f in self.factions:
            if f.alive and f.is_streamer_raid and f.raid_end_tick > 0 and self.tick >= f.raid_end_tick:
                f.alive = False
                self.policy.release_faction_members(f.id)
                self.logger.log_chronicle(
                    self.tick,
                    "raid",
                    f"RAID DISPERSED: Streamer mob '{f.name}' wrapped up its blitz ({f.timezone_region}), leaving its mark on the canvas!",
                    faction_id=f.id,
                )

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
        """Advance each active Transverse Ribbon step-by-step across the active viewport, respecting ally tunnels."""
        W, H = self.canvas.active_w, self.canvas.active_h
        allowed = self.canvas.allowed_colors
        fac_by_id = {f.id: f for f in self.factions if f.alive}
        step = max(52, W // 12)

        for fac in self.factions:
            if not (fac.alive and fac.art_mode == "ribbon"):
                continue
            if fac.integrity < 0.30:
                continue

            axis = fac.ribbon_axis or "h"
            thick = max(6, min(18, fac.ribbon_thickness))
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
            bp_np = bp_np[: min(bp_np.shape[0], H - ny0), : min(bp_np.shape[1], W - nx0)]
            bp_np = remap_indices_to_allowed(bp_np, allowed)

            # Leave transparent overpass tunnels over allied factions
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
                fac.remember(f"Spanned across the entire {W}x{H} viewport!")

    # ------------------------------------------------------------------
    # Async VLM + FLUX Strategist Worker
    # ------------------------------------------------------------------

    def _strategist_loop(self) -> None:
        while not self.stop_requested:
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._state_lock:
                self._queued_jobs.discard(job)
            if job == "STOP":
                break
            if self.canvas.is_whiteout and job != "STOP":
                continue
            self._brain_busy.set()
            try:
                if job == "DISCOVER":
                    self.run_pareidolia_discovery()
                elif job == "EPOCH":
                    self.run_geopolitical_epoch()
                elif job == "FLASH_RAID":
                    self.spawn_streamer_flash_raid()
                elif job == "VOID":
                    self.spawn_void_eruption(force=True)
            except Exception as exc:
                log.exception("Strategist job %s failed: %s", job, exc)
                self.brain_status = f"Strategist recovered from error: {exc}"
            finally:
                self._brain_busy.clear()

    def _pick_next_tier(self, alive_art: list[Faction]) -> str:
        """Choose next size tier to track empirical placeAtlas power-law shares."""
        n = len(alive_art)
        if n < 2:
            return "macro"
        if n == 2:
            return "ribbon"
        if n < 6:
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

    def _size_scale(self) -> float:
        """Scale art dimensions cleanly between 512px, 1024px, and 2000px canvases."""
        if self.cfg.width >= 1800:
            return 1.38
        if self.cfg.width >= 960:
            return 1.15
        return 1.0

    def _dims_for_tier(self, tier: str, art_mode: str, aspect: str = "square") -> tuple[int, int]:
        s = self._size_scale()
        if tier == "micro":
            if art_mode == "flag":
                w, h = self.rng.choice([(22, 14), (26, 16), (28, 18)])
            elif art_mode == "banner":
                w, h = self.rng.choice([(32, 14), (36, 16), (38, 16)])
            elif art_mode == "clean_space":
                w, h = self.rng.choice([(20, 18), (24, 20)])
            elif aspect == "wide":
                w, h = self.rng.choice([(32, 22), (36, 24)])
            elif aspect == "tall":
                w, h = self.rng.choice([(22, 30), (24, 34)])
            else:
                d = self.rng.choice([20, 24, 26, 28, 30])
                w, h = d, d
        elif tier == "medium":
            if art_mode == "flag":
                w, h = self.rng.choice([(44, 28), (50, 32), (56, 36)])
            elif art_mode == "flux_comic":
                w, h = self.rng.choice([(82, 36), (92, 40)])
            elif art_mode == "banner":
                w, h = self.rng.choice([(66, 24), (76, 28), (84, 30)])
            elif art_mode == "clean_space":
                w, h = self.rng.choice([(34, 28), (38, 30)])
            elif aspect == "wide":
                w, h = self.rng.choice([(62, 42), (68, 44)])
            elif aspect == "tall":
                w, h = self.rng.choice([(40, 56), (44, 62)])
            else:
                d = self.rng.choice([42, 46, 52, 56])
                w, h = d, d
        else:  # macro
            if art_mode == "flag":
                w, h = self.rng.choice([(72, 46), (82, 52), (90, 56)])
            elif art_mode == "flux_comic":
                w, h = self.rng.choice([(104, 44), (116, 48), (124, 52)])
            elif art_mode == "banner":
                w, h = self.rng.choice([(98, 34), (114, 38), (128, 44)])
            elif aspect == "wide":
                w, h = self.rng.choice([(96, 62), (108, 68)])
            elif aspect == "tall":
                w, h = self.rng.choice([(64, 92), (72, 102)])
            else:
                d = self.rng.choice([68, 76, 84, 92])
                w, h = d, d

        max_w = max(24, self.canvas.active_w // 4)
        max_h = max(24, self.canvas.active_h // 4)
        return min(max_w, int(round(w * s))), min(max_h, int(round(h * s)))

    def _target_initial_members(self, tier: str, valid_pixels: int) -> int:
        """Scale initial citizen recruitment proportionally to Blueprint area & total slots."""
        if self.cfg.max_active_factions >= 500:
            if tier == "micro":
                return max(36, min(95, int(30 + valid_pixels * 0.08)))
            if tier == "medium":
                return max(105, min(240, int(90 + valid_pixels * 0.05)))
            if tier == "ribbon":
                return self.rng.randint(320, 520)
            return max(260, min(580, int(200 + valid_pixels * 0.04)))
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
        """Varied multi-font multilingual typography spec for floating pixel wordmarks."""
        has_non_ascii = any(ord(ch) > 127 for ch in text)
        font = FONT_STYLES[f_idx % len(FONT_STYLES)]
        ce = ("vertical_gradient", "per_letter_palette", "glossy_highlight", "rainbow_letters", "striped_fill", "solid")[f_idx % 6]
        be = ("extrude_3d", "double_sticker", "neon_glow", "drop_shadow", "outline_1px")[(f_idx // 2) % 5]
        lay = ("straight", "wavy", "arched", "italic_slant")[(f_idx // 3) % 4]
        use_rib = (not is_micro) and (not has_non_ascii) and (f_idx % 4 == 0)
        if is_micro or has_non_ascii:
            engine_choice = "styled"
        elif allow_flux and f_idx % 3 == 0:
            engine_choice = "flux"
        else:
            engine_choice = "styled"
        return TypographyStyleSpec(
            include_text=True,
            text=text[:14],
            casing="subreddit" if text.lower().startswith("r/") else ("upper" if (not has_non_ascii and f_idx % 2 == 0) else "title"),
            font_style=font,
            size_tier="tiny" if is_micro else "large",
            color_effect=ce,  # type: ignore[arg-type]
            border_effect="outline_1px" if is_micro else be,  # type: ignore[arg-type]
            layout="straight" if (is_micro or has_non_ascii) else lay,  # type: ignore[arg-type]
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

    def _refill_ai_idea_queue(
        self, existing_names: list[str], allowed_colors: tuple[int, ...]
    ) -> None:
        """Ask local Qwen3-VL-30B to invent a batch of 10 unique multilingual/pop-culture/comic/geometric factions."""
        if not self.ollama_ready:
            return
        prime_tz = self._prime_timezone_now()
        sub_hints = [self.subreddits.sample(size_tier="medium").name for _ in range(6)]
        self.brain_status = f"Act {self.act} (Hr {self.virtual_hour:.1f}) — Qwen3-VL brainstorming fresh communities ({prime_tz})..."
        batch = self.brain.brainstorm_faction_batch(
            count=8,
            act=self.act,
            virtual_hour=self.virtual_hour,
            prime_tz=prime_tz,
            unlocked_colors=allowed_colors,
            existing_names=existing_names,
            subreddit_hints=sub_hints,
        )
        if batch:
            for idea in batch:
                self._idea_queue.append(idea)
            log.info("Brainstormed %d original AI faction ideas (queue=%d)", len(batch), len(self._idea_queue))

    def _pick_spawn_slot(
        self,
        tier: str,
        bp_w: int,
        bp_h: int,
        allow_cameo: bool,
        def_c1: int,
        def_c2: int,
    ) -> tuple[int, int, bool, bool, bool, Any, np.ndarray, list[str]]:
        """Locate an open or emergent pareidolia bounding box inside the active viewport."""
        prefer_region = (
            self._land_rush_region
            if (self._land_rush_region is not None and self.tick <= self._land_rush_until_tick)
            else None
        )
        scan_win = min(max(bp_w, bp_h), 72)
        scan_stride = max(12, min(48, self.canvas.active_w // 28))
        with self._state_lock:
            W_act, H_act = self.canvas.active_w, self.canvas.active_h
            cands = find_pareidolia_candidates(
                self.canvas,
                self.factions,
                window=scan_win,
                stride=scan_stride,
                min_density=0.04 if tier != "micro" else 0.01,
                top_k=1,
                allow_cameo=allow_cameo,
                rng=self.rng,
                prefer_region=prefer_region,
            )
            if not cands:
                x0, y0 = find_open_exodus_spot(
                    self.canvas, self.factions, bp_w, bp_h, prefer_region=prefer_region
                )
                is_cameo = is_corner = is_edge = False
                dom_colors = [def_c1, def_c2]
                crop_pil = self.canvas.crop_pil(x0, y0, x0 + bp_w, y0 + bp_h, scale=4)
                crop_rgb = self.canvas.crop_rgb_numpy(x0, y0, x0 + bp_w, y0 + bp_h)
            else:
                c = cands[0]
                is_cameo = c.is_cameo
                is_corner = c.is_corner
                is_edge = c.is_edge
                dom_colors = c.dominant_colors
                raw_x0 = c.x0 - (bp_w - c.w) // 2
                raw_y0 = c.y0 - (bp_h - c.h) // 2
                if is_cameo:
                    x0 = max(0, min(W_act - bp_w, raw_x0))
                    y0 = max(0, min(H_act - bp_h, raw_y0))
                else:
                    x0, y0 = _snap_to_walls(
                        raw_x0, raw_y0, bp_w, bp_h, W_act, H_act, snap=14 if tier != "micro" else 6
                    )
                crop_pil = self.canvas.crop_pil(c.x0, c.y0, c.x0 + c.w, c.y0 + c.h, scale=4)
                crop_rgb = self.canvas.crop_rgb_numpy(c.x0, c.y0, c.x0 + c.w, c.y0 + c.h)
            dom_names = [COLOR_NAMES[idx] for idx in dom_colors if 0 <= idx < len(COLOR_NAMES)]
        return x0, y0, is_cameo, is_corner, is_edge, crop_pil, crop_rgb, dom_names

    def run_pareidolia_discovery(self, max_new: int = 4) -> list[Faction]:
        """AI-first multi-scale discovery wave (>= 80% live AI-invented factions + restrained flags/ribbons/cameos)."""
        if self.canvas.is_whiteout:
            return []

        with self._state_lock:
            alive_art = [f for f in self.factions if f.alive and not f.is_void]
            area_frac = (self.canvas.active_w * self.canvas.active_h) / max(1, self.cfg.width * self.cfg.height)
            if area_frac < 0.99:
                eff_max_factions = max(24, int(round(self.cfg.max_active_factions * min(1.0, area_frac + 0.08))))
            else:
                eff_max_factions = self.cfg.max_active_factions
            slots = eff_max_factions - len(alive_art)
            if slots <= 0:
                return []
            wave_size = min(max_new, slots)
            existing_names = {f.name.lower() for f in self.factions}
            existing_names_list = [f.name for f in self.factions]
            existing_themes_list = [f.theme for f in self.factions]
            active_flag_keys = {f.extra_key for f in alive_art if f.category == "flag"}
            active_rib_keys = {f.extra_key for f in alive_art if f.art_mode == "ribbon"}
            active_comic_keys = {f.extra_key for f in alive_art if f.art_mode == "comic_strip"}
            active_meme_keys = {f.extra_key for f in alive_art if f.art_mode == "cartoon_meme"}
            active_sprite_keys = {f.extra_key for f in alive_art if f.art_mode == "micro_sprite"}
            tick = self.tick
            total_founded = len(self.factions)
            allowed_colors = self.canvas.allowed_colors

        # Keep AI idea queue primed from Qwen3-VL-30B (non-blocking, outside lock)
        if self.ollama_ready and len(self._idea_queue) < wave_size + 2:
            self._refill_ai_idea_queue(existing_names_list, allowed_colors)

        created: list[Faction] = []
        flux_calls_in_wave = 0
        max_flux_per_wave = 2 if self.cfg.use_flux_art else 0
        sim_alive = list(alive_art)

        for i in range(wave_size):
            if self.stop_requested or self.canvas.is_whiteout:
                break

            tier = self._pick_next_tier(sim_alive)
            f_idx = total_founded + i
            theme_desc, def_c1_name, def_c2_name, def_c3_name = COLOR_THEMES[f_idx % len(COLOR_THEMES)]
            def_c1 = resolve_color(def_c1_name, BLUE_IDX)
            def_c2 = resolve_color(def_c2_name, YELLOW_IDX)
            def_c3 = resolve_color(def_c3_name, WHITE_IDX)

            # Decide world primitive vs AI-invented faction (~84%+ live AI-invented!)
            alive_flags = sum(1 for f in sim_alive if f.category == "flag")
            alive_ribbons = sum(1 for f in sim_alive if f.art_mode == "ribbon")
            max_flags = min(len(NATIONAL_FLAGS), max(12, int(self.cfg.max_active_factions * 0.09)))
            max_ribbons = 8 if self.cfg.width >= 1800 else 4

            r = self.rng.random()
            if tier == "ribbon" and alive_ribbons < max_ribbons:
                spawn_kind = "ribbon"
            elif alive_flags < max_flags and r < 0.095:
                spawn_kind = "flag"
            elif tier == "micro" and r < 0.115:
                spawn_kind = "amogus_cameo"
            elif tier in ("micro", "medium") and r < 0.132:
                spawn_kind = "clean_space"
            else:
                spawn_kind = "ai_invented"
                if tier == "ribbon":
                    tier = "medium"

            # Defaults
            is_cameo = False
            extra_key = ""
            ribbon_axis = ""
            ribbon_thick = 8
            typo_spec: TypographyStyleSpec | None = None
            flux_style_desc = ""
            fluid_cutout = True
            ai_code_str = ""
            ai_origin = False
            tz_region = "GLOBAL"
            lang_code = "en"
            cand_crop_rgb: np.ndarray | None = None
            sub_entry = self.subreddits.sample(size_tier=tier, avoid_names=existing_names)
            sub_name = sub_entry.name

            if spawn_kind == "ribbon":
                axis_r = self.rng.random()
                ribbon_axis = "h" if axis_r < 0.55 else ("v" if axis_r < 0.85 else self.rng.choice(["d1", "d2"]))
                ribbon_thick = self.rng.choice([7, 8, 10, 12]) if self.cfg.width >= 1500 else self.rng.choice([6, 7, 8, 10])
                with self._state_lock:
                    x0, y0, bp_w, bp_h = find_ribbon_track(self.canvas, self.factions, ribbon_axis, ribbon_thick, self.rng)
                avail_ribs = [p for p in RIBBON_PATTERNS if p[0] not in active_rib_keys] or RIBBON_PATTERNS
                pkey, pname, pcols, _ = self.rng.choice(avail_ribs)
                active_rib_keys.add(pkey)
                extra_key = f"{pkey}|{ribbon_thick}" if ribbon_axis in ("d1", "d2") else pkey
                fac_name = pname if pname.lower() not in existing_names else f"{pname} {f_idx + 1}"
                fac_theme = f"cross-map {pkey} ribbon"
                fac_slogan = f"Extend the {pname} across the canvas!"
                art_mode = "ribbon"
                fac_category = "ribbon"
                art_prompt = f"straight {ribbon_axis} {pkey} patterned ribbon"
                c1, c2, c3 = pcols[0], pcols[min(1, len(pcols) - 1)], pcols[-1]
                sub_name = f"place_{pkey}"

            elif spawn_kind == "flag":
                # Strict <= 1 active copy per country, preferring countries never spawned yet
                unused_ever = [fl for fl in NATIONAL_FLAGS if fl.key not in active_flag_keys and fl.key not in self._used_flag_keys_ever]
                avail_flags = unused_ever or [fl for fl in NATIONAL_FLAGS if fl.key not in active_flag_keys]
                if not avail_flags:
                    spawn_kind = "ai_invented"
                else:
                    fl = self.rng.choice(avail_flags)
                    active_flag_keys.add(fl.key)
                    self._used_flag_keys_ever.add(fl.key)
                    extra_key = fl.key
                    tz_region = FLAG_TIMEZONES.get(fl.key, "GLOBAL")
                    bp_w, bp_h = self._dims_for_tier(tier, "flag", aspect="wide")
                    x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, fl.primary, fl.secondary)
                    fac_name = fl.display_name if fl.display_name.lower() not in existing_names else f"{fl.display_name} {f_idx + 1}"
                    fac_theme = f"{fl.key} national flag"
                    fac_slogan = fl.slogan
                    art_mode = "flag"
                    fac_category = "flag"
                    art_prompt = f"{fl.key} national flag"
                    c1, c2, c3 = fl.primary, fl.secondary, fl.accent
                    sub_name = fl.subreddit
                    fluid_cutout = False

            if spawn_kind == "amogus_cameo":
                # Subtle hidden 10x12 Among Us crewmate cameo tucked onto the canvas
                bp_w, bp_h = self.rng.choice([(10, 12), (12, 14)])
                x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(
                    "micro", bp_w, bp_h, allow_cameo=(len(sim_alive) >= 12), def_c1=def_c1, def_c2=def_c2
                )
                extra_key = "amogus"
                fac_name = f"Sus Crew #{f_idx + 1}"
                fac_theme = "hidden mini crewmate cameo"
                fac_slogan = "shhh... blend into the background!"
                art_mode = "micro_sprite"
                fac_category = "meme"
                art_prompt = "tiny hidden pixel crewmate"
                c1, c2, c3 = def_c1, def_c2, WHITE_IDX
                sub_name = "AmongUs"
                tier = "micro"

            elif spawn_kind == "clean_space":
                bp_w, bp_h = self._dims_for_tier(tier, "clean_space")
                x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, WHITE_IDX, BLACK_IDX)
                cname, cslogan = _CLEAN_SPACE_NAMES[f_idx % len(_CLEAN_SPACE_NAMES)]
                fac_name = cname if cname.lower() not in existing_names else f"{cname} {f_idx + 1}"
                fac_theme = "pure white empty space"
                fac_slogan = cslogan
                art_mode = "clean_space"
                fac_category = "clean_space"
                art_prompt = "pure white empty space enclave"
                c1, c2, c3 = WHITE_IDX, WHITE_IDX, BLACK_IDX
                sub_name = "CleanCanvas"
                fluid_cutout = False

            elif spawn_kind == "ai_invented":
                idea: AIFactionIdea | None = None
                while self._idea_queue:
                    cand_idea = self._idea_queue.popleft()
                    if cand_idea.name.lower() not in existing_names:
                        idea = cand_idea
                        break

                if idea is not None:
                    ai_origin = True
                    tz_region = idea.timezone_region
                    lang_code = idea.language or "en"
                    art_mode = idea.art_mode
                    # Throttle FLUX calls per wave so fast ai_code / banner ideas keep discovery snappy
                    if art_mode in ("flux_art", "flux_comic") and (not self.cfg.use_flux_art or flux_calls_in_wave >= max_flux_per_wave):
                        if idea.banner_text and len(idea.banner_text.strip()) >= 2:
                            art_mode = "banner"
                        else:
                            art_mode = "ai_code"
                    if art_mode == "flux_comic" and tier == "micro":
                        tier = "medium"

                    bp_w, bp_h = self._dims_for_tier(tier, art_mode, aspect=idea.aspect)
                    x0, y0, is_cameo, is_corner, is_edge, cand_crop_pil, cand_crop_rgb, dom_names = self._pick_spawn_slot(
                        tier, bp_w, bp_h, allow_cameo=(tier == "micro" and len(sim_alive) >= 20 and self.rng.random() < 0.2),
                        def_c1=def_c1, def_c2=def_c2,
                    )

                    # Every ~6th macro/medium AI spawn, run live Vision Pareidolia on the actual canvas crop!
                    if (
                        self.ollama_ready
                        and art_mode == "flux_art"
                        and tier != "micro"
                        and (f_idx % 6 == 0)
                        and flux_calls_in_wave < max_flux_per_wave
                    ):
                        self.brain_status = f"Act {self.act} Vision Pareidolia at ({x0},{y0})..."
                        prop, theme_desc = self.brain.pareidolia_discover(
                            cluster_img=cand_crop_pil,
                            dominant_color_names=dom_names,
                            existing_names=existing_names_list,
                            existing_themes=existing_themes_list,
                            faction_index=f_idx,
                            desired_mode="diffusion",
                            is_corner=is_corner,
                            is_edge=is_edge,
                            subreddit_hint=sub_name,
                        )
                        if prop.name.lower() not in existing_names:
                            fac_name = prop.name.strip()[:28]
                            fac_theme = prop.theme.strip()[:42]
                            fac_slogan = prop.slogan.strip()[:90]
                            art_prompt = prop.art_prompt.strip()[:160]
                            c1 = resolve_color(prop.primary_color, def_c1)
                            c2 = resolve_color(prop.secondary_color, def_c2)
                            c3 = resolve_color(prop.accent_color, def_c3)
                            fac_category = idea.category or "art"
                            fluid_cutout = True
                        else:
                            fac_name = idea.name[:28]
                            fac_theme = idea.category[:40]
                            fac_slogan = (idea.slogan or f"{fac_name} unites!")[:90]
                            art_prompt = idea.art_prompt[:180]
                            c1 = resolve_color(idea.primary_color, def_c1)
                            c2 = resolve_color(idea.secondary_color, def_c2)
                            c3 = resolve_color(idea.accent_color, def_c3)
                            fac_category = idea.category or "art"
                            fluid_cutout = idea.fluid_cutout
                    else:
                        fac_name = idea.name[:28]
                        fac_theme = (idea.category or "emergent community")[:40]
                        fac_slogan = (idea.slogan or f"{fac_name} holds the line!")[:90]
                        art_prompt = idea.art_prompt[:180]
                        c1 = resolve_color(idea.primary_color, def_c1)
                        c2 = resolve_color(idea.secondary_color, def_c2)
                        c3 = resolve_color(idea.accent_color, def_c3)
                        fac_category = idea.category or "art"
                        fluid_cutout = False if art_mode == "flux_comic" else True

                    if art_mode in ("flux_art", "flux_comic"):
                        flux_calls_in_wave += 1
                        if idea.banner_text and art_mode == "flux_art" and max(bp_w, bp_h) >= 48 and self.rng.random() < 0.35:
                            typo_spec = self._fast_typography_spec(
                                idea.banner_text, c1, c2, c3, f_idx, is_micro=True, allow_flux=False
                            )
                            typo_spec.placement = "bottom_center"
                            typo_spec.size_tier = "tiny"
                    elif art_mode == "banner":
                        btxt = (idea.banner_text or fac_name[:10]).strip()[:12]
                        typo_spec = self._fast_typography_spec(
                            btxt, c1, c2, c3, f_idx, is_micro=(tier == "micro"), allow_flux=False
                        )
                        fac_category = "logo"
                    elif art_mode == "ai_code":
                        fac_category = "geometric"
                        if self.ollama_ready:
                            self.brain_status = f"Qwen3-VL coding geometric pattern for '{fac_name}'..."
                            ai_code_str = self.brain.write_geometric_code(
                                name=fac_name,
                                art_prompt=art_prompt,
                                w=bp_w,
                                h=bp_h,
                                allowed_colors=allowed_colors,
                            )
                else:
                    # Offline fallback when LM Studio/Ollama is unreachable (still <= 1 active copy per template!)
                    fb_pick = self.rng.random()
                    if fb_pick < 0.32:
                        wm_txt, wm_slo, tz_region, lang_code = _LOGO_WORDMARKS[f_idx % len(_LOGO_WORDMARKS)]
                        bp_w, bp_h = self._dims_for_tier(tier, "banner", aspect="wide")
                        x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, def_c1, def_c2)
                        fac_name = wm_txt if wm_txt.lower() not in existing_names else f"{wm_txt} {f_idx + 1}"
                        fac_theme = f"wordmark {wm_txt}"
                        fac_slogan = wm_slo
                        art_mode = "banner"
                        fac_category = "logo"
                        art_prompt = f"stylized wordmark {wm_txt}"
                        c1, c2, c3 = def_c1, def_c2, def_c3
                        typo_spec = self._fast_typography_spec(wm_txt, c1, c2, c3, f_idx, is_micro=(tier == "micro"), allow_flux=False)
                    elif fb_pick < 0.50:
                        avail_c = [c for c in COMIC_CATALOG if c.key not in active_comic_keys]
                        if avail_c and tier != "micro":
                            cm = self.rng.choice(avail_c)
                            active_comic_keys.add(cm.key)
                            extra_key = cm.key
                            bp_w, bp_h = self._dims_for_tier(tier, "flux_comic", aspect="wide")
                            x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, cm.primary, cm.secondary)
                            fac_name = cm.title if cm.title.lower() not in existing_names else f"{cm.title} {f_idx + 1}"
                            fac_theme = f"comic: {cm.title}"
                            fac_slogan = cm.slogan
                            art_mode = "comic_strip"
                            fac_category = "comic"
                            art_prompt = f"comic strip {cm.title}"
                            c1, c2, c3 = cm.primary, cm.secondary, cm.accent
                            fluid_cutout = False
                        else:
                            fname, fkind, fslogan = _FRACTAL_NAMES[f_idx % len(_FRACTAL_NAMES)]
                            extra_key = fkind
                            bp_w, bp_h = self._dims_for_tier(tier, "fractal")
                            x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, BLACK_IDX, PURPLE_IDX)
                            fac_name = fname if fname.lower() not in existing_names else f"{fname} {f_idx + 1}"
                            fac_theme = f"{fkind} fractal"
                            fac_slogan = fslogan
                            art_mode = "fractal"
                            fac_category = "geometric"
                            art_prompt = f"{fkind} geometric fractal"
                            c1, c2, c3 = BLACK_IDX, PURPLE_IDX, def_c2
                    elif fb_pick < 0.72:
                        avail_m = [m for m in CARTOON_MEME_CATALOG if m.key not in active_meme_keys]
                        if avail_m:
                            mm = self.rng.choice(avail_m)
                            active_meme_keys.add(mm.key)
                            extra_key = mm.key
                            bp_w, bp_h = self._dims_for_tier(tier, "flux_art")
                            x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, mm.primary, mm.secondary)
                            fac_name = mm.title if mm.title.lower() not in existing_names else f"{mm.title} {f_idx + 1}"
                            fac_theme = mm.title
                            fac_slogan = mm.slogan
                            art_mode = "cartoon_meme"
                            fac_category = mm.category
                            art_prompt = mm.title
                            c1, c2, c3 = mm.primary, mm.secondary, mm.accent
                        else:
                            avail_s = [k for k in SPRITE_KEYS if k not in active_sprite_keys] or SPRITE_KEYS
                            skey = self.rng.choice(avail_s)
                            active_sprite_keys.add(skey)
                            extra_key = skey
                            bp_w, bp_h = self._dims_for_tier("micro", "micro_sprite")
                            x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot("micro", bp_w, bp_h, False, def_c1, def_c2)
                            fac_name = f"r/{sub_name[:12]}" if f"r/{sub_name[:12]}".lower() not in existing_names else f"Pixel {f_idx + 1}"
                            fac_theme = f"pixel sprite {skey}"
                            fac_slogan = f"{fac_name} holds the grid!"
                            art_mode = "micro_sprite"
                            fac_category = "game"
                            art_prompt = f"pixel sprite {skey}"
                            c1, c2, c3 = def_c1, def_c2, def_c3
                    else:
                        bp_w, bp_h = self._dims_for_tier(tier, "procedural")
                        x0, y0, is_cameo, _, _, _, cand_crop_rgb, _ = self._pick_spawn_slot(tier, bp_w, bp_h, False, def_c1, def_c2)
                        fac_name = f"r/{sub_name[:12]}" if f"r/{sub_name[:12]}".lower() not in existing_names else f"Guild {f_idx + 1}"
                        fac_theme = f"emergent pixel emblem ({theme_desc})"
                        fac_slogan = f"Defend {fac_name}!"
                        art_mode = "procedural"
                        fac_category = "art"
                        art_prompt = f"pixel emblem for {fac_name}"
                        c1, c2, c3 = def_c1, def_c2, def_c3

            # --- 3. Generate Blueprint strictly within unlocked palette ---
            with self._state_lock:
                cur_allowed = self.canvas.allowed_colors
                W_act, H_act = self.canvas.active_w, self.canvas.active_h

            self.brain_status = f"Act {self.act} forging {tier}/{art_mode} '{fac_name}' [{tz_region}]..."
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
                ocr_verifier=None,  # Styled Unicode typography is guaranteed crisp; FLUX art has pure visuals
                live_crop_rgb=cand_crop_rgb,
                seed=self.cfg.seed + self.next_faction_id * 97 + tick + i * 31,
                extra_key=extra_key,
                ribbon_axis=ribbon_axis or "h",
                allowed_colors=cur_allowed,
                fluid_cutout=fluid_cutout,
                ai_code=ai_code_str,
            )
            bp_h_actual, bp_w_actual = int(bp_np.shape[0]), int(bp_np.shape[1])
            x0 = max(0, min(W_act - bp_w_actual, x0))
            y0 = max(0, min(H_act - bp_h_actual, y0))
            valid_px = int((bp_np >= 0).sum())

            # --- 4. Commit Faction & Recruit Citizens ---
            with self._state_lock:
                if self.canvas.is_whiteout:
                    break
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
                    timezone_region=tz_region,
                    language=lang_code,
                    ai_origin=ai_origin,
                    burnout_rate=round(self.rng.uniform(0.75, 1.45), 2),
                )
                fac.remember(f"Born ({tier}/{fac_category}/{tz_region}) at ({x0},{y0}) [{bp_w_actual}x{bp_h_actual}].")
                self.factions.append(fac)
                sim_alive.append(fac)

                target_rec = self._target_initial_members(tier, valid_px)
                recruited = self.policy.recruit_initial_members(fac, target_count=target_rec)
                fac.member_count = recruited
                self.diplomacy.post_public(self.tick, fac, fac.slogan, kind="propaganda")
                ai_tag = " [AI]" if ai_origin else ""
                self.logger.log_chronicle(
                    self.tick,
                    "founded",
                    f"New {tier} {fac_category}{ai_tag} '{fac.name}' ({bp_w_actual}x{bp_h_actual}, {art_mode}, {tz_region}/{lang_code}) "
                    f"at ({x0},{y0}) — {recruited} citizens!",
                    faction_id=fac.id,
                    faction_name=fac.name,
                    size_tier=tier,
                    category=fac_category,
                    art_mode=art_mode,
                    ai_origin=ai_origin,
                    timezone_region=tz_region,
                    language=lang_code,
                    bbox=[fac.x0, fac.y0, fac.x1, fac.y1],
                )
                existing_names.add(fac.name.lower())
                existing_names_list.append(fac.name)
                existing_themes_list.append(fac.theme)
                created.append(fac)

        self.brain_status = (
            f"Act {self.act} (Hr {self.virtual_hour:.1f}/72) — "
            f"{len([f for f in self.factions if f.alive])} active factions"
        )
        return created

    # ------------------------------------------------------------------
    # Regional Streamer Flash-Raids (REQ-DIP-008)
    # ------------------------------------------------------------------

    def spawn_streamer_flash_raid(self) -> Faction | None:
        """Spawn a prime-time regional streamer blitz right onto established (often sleeping) territory."""
        if self.canvas.is_whiteout:
            return None

        with self._state_lock:
            alive_art = [f for f in self.factions if f.alive and not f.is_void and not f.is_streamer_raid]
            if len(alive_art) < 6:
                return None
            # Only 1 active flash-raid at a time so each blitz is dramatic and clean
            if any(f.alive and f.is_streamer_raid for f in self.factions):
                return None
            prime_tz = self._prime_timezone_now()
            W_act, H_act = self.canvas.active_w, self.canvas.active_h
            allowed = self.canvas.allowed_colors

            # Prefer raiding a faction in a sleeping timezone (authentic overnight r/place ambushes!)
            def _raid_appeal(f: Faction) -> float:
                awake = timezone_awake_factor(f.timezone_region, self.virtual_hour % 24.0)
                return (1.6 - awake) + (0.6 if f.size_tier in ("macro", "medium") else 0.0) + self.rng.random() * 0.5

            victims = sorted(alive_art, key=_raid_appeal, reverse=True)
            anchor = victims[0]
            nearby_names = [f.name for f in victims[:4]]
            tick = self.tick
            s = self._size_scale()
            bp_w = min(W_act // 4, int(round(self.rng.choice([76, 86, 94]) * s)))
            bp_h = min(H_act // 4, int(round(self.rng.choice([76, 86, 94]) * s)))
            rx0 = max(0, min(W_act - bp_w, int(anchor.cx) - bp_w // 2))
            ry0 = max(0, min(H_act - bp_h, int(anchor.cy) - bp_h // 2))

        self.brain_status = f"FLASH RAID ({prime_tz}) mobilizing against '{anchor.name}'..."
        idea = (
            self.brain.invent_streamer_raid(prime_tz, self.virtual_hour, nearby_names, allowed)
            if self.ollama_ready
            else None
        )
        if idea is not None:
            r_name = idea.name[:28]
            r_slogan = (idea.slogan or f"{r_name} RAID! TAKE THE CANVAS!")[:90]
            r_prompt = idea.art_prompt[:180]
            r_lang = idea.language or "en"
            c1 = resolve_color(idea.primary_color, PURPLE_IDX)
            c2 = resolve_color(idea.secondary_color, YELLOW_IDX)
            c3 = resolve_color(idea.accent_color, WHITE_IDX)
        else:
            r_name = f"{prime_tz.title()} Blitz #{self.next_faction_id + 1}"
            r_slogan = f"LIVE RAID! {r_name} storms the grid!"
            r_prompt = "bold fiery neon crown pixel mascot skull emblem wearing headphones, crisp 16-bit arcade sprite"
            r_lang = "en"
            c1, c2, c3 = PURPLE_IDX, YELLOW_IDX, WHITE_IDX

        bp_np, _ = self.art.create_blueprint(
            art_mode="streamer_raid",
            prompt=r_prompt,
            w=bp_w,
            h=bp_h,
            primary=c1,
            secondary=c2,
            accent=c3,
            label=r_name,
            theme=f"{prime_tz} streamer raid",
            seed=self.cfg.seed + tick * 19 + 777,
            allowed_colors=allowed,
            fluid_cutout=True,
        )
        bp_h_a, bp_w_a = int(bp_np.shape[0]), int(bp_np.shape[1])

        with self._state_lock:
            if self.canvas.is_whiteout:
                return None
            fid = self.next_faction_id
            self.next_faction_id += 1
            bp_t = torch.from_numpy(bp_np.astype(np.int16)).to(self.device)
            duration = self.rng.randint(260, 440)
            raid_fac = Faction(
                id=fid,
                name=r_name,
                theme=f"streamer flash-raid ({prime_tz})",
                slogan=r_slogan,
                art_mode="streamer_raid",
                art_prompt=r_prompt,
                primary_color=c1,
                secondary_color=c2,
                accent_color=c3,
                x0=max(0, min(self.canvas.active_w - bp_w_a, rx0)),
                y0=max(0, min(self.canvas.active_h - bp_h_a, ry0)),
                w=bp_w_a,
                h=bp_h_a,
                blueprint=bp_t,
                edge_weight=compute_edge_weights(bp_t),
                created_tick=self.tick,
                size_tier="macro",
                category="streamer_raid",
                timezone_region=prime_tz,
                language=r_lang,
                ai_origin=True,
                is_streamer_raid=True,
                raid_end_tick=self.tick + duration,
                hype=2.4,
                role_weights={"builder": 0.78, "defender": 0.18, "expander": 0.04},
            )
            self.factions.append(raid_fac)
            surge_target = min(7500, max(1200, self.cfg.num_users // 20))
            if hasattr(self.policy, "mobilize_flash_raid"):
                recruited = self.policy.mobilize_flash_raid(raid_fac, surge_target)
            else:
                recruited = self.policy.recruit_initial_members(raid_fac, surge_target)
            raid_fac.member_count = recruited
            self.diplomacy.post_public(self.tick, raid_fac, r_slogan, kind="threat")
            self.logger.log_chronicle(
                self.tick,
                "raid",
                f"STREAMER FLASH-RAID [{prime_tz}]: '{r_name}' mobilized {recruited:,} citizens onto ({raid_fac.x0},{raid_fac.y0}) over '{anchor.name}' for {duration} ticks!",
                faction_id=raid_fac.id,
                timezone_region=prime_tz,
                bbox=[raid_fac.x0, raid_fac.y0, raid_fac.x1, raid_fac.y1],
            )
            return raid_fac

    # ------------------------------------------------------------------
    # Geopolitical Epochs, Expiring Alliances & Burnout (REQ-SIM-017)
    # ------------------------------------------------------------------

    def _form_alliance(self, fa: Faction, fb: Faction) -> bool:
        """Seal a time-limited alliance respecting per-faction partner caps (<= 3, Macro <= 2)."""
        cap_a = 2 if fa.size_tier == "macro" else 3
        cap_b = 2 if fb.size_tier == "macro" else 3
        if len(fa.allies) >= cap_a or len(fb.allies) >= cap_b:
            return False
        expiry = self.tick + self.rng.randint(1500, 2600)
        fa.allies.add(fb.id)
        fb.allies.add(fa.id)
        fa.ally_expiry[fb.id] = expiry
        fb.ally_expiry[fa.id] = expiry
        fa.enemies.discard(fb.id)
        fb.enemies.discard(fa.id)
        fa.pending_ally_offers.discard(fb.id)
        fb.pending_ally_offers.discard(fa.id)
        return True

    def _expire_stale_alliances(self, alive_art: list[Faction]) -> None:
        fac_by_id = {f.id: f for f in alive_art}
        for f in alive_art:
            expired = [aid for aid, exp_t in list(f.ally_expiry.items()) if self.tick >= exp_t or aid not in fac_by_id]
            for aid in expired:
                f.allies.discard(aid)
                f.ally_expiry.pop(aid, None)
                other = fac_by_id.get(aid)
                if other is not None:
                    other.allies.discard(f.id)
                    other.ally_expiry.pop(f.id, None)

    def _run_autonomous_micro_and_flag_diplomacy(self, alive_art: list[Faction]) -> None:
        """Fast local diplomacy for micro neighborhoods and national flag border rivalries."""
        fac_by_id = {f.id: f for f in alive_art}
        micros = [f for f in alive_art if f.size_tier == "micro" and f.art_mode != "clean_space" and f.integrity >= 0.45]
        self.rng.shuffle(micros)
        for mf in micros[:16]:
            if len(mf.allies) >= 3:
                continue
            nbrs = describe_neighbors(mf, self.factions, max_gap=20)
            for n in nbrs[:2]:
                other = fac_by_id.get(n["id"])
                if other is None or other.is_void or other.id in mf.allies or other.art_mode == "clean_space":
                    continue
                if other.size_tier == "micro" and self.rng.random() < 0.38:
                    if self._form_alliance(mf, other):
                        mf.remember(f"Joined micro alliance with {other.name}!")
                    break

        flags = [f for f in alive_art if f.category == "flag" and f.integrity >= 0.55]
        self.rng.shuffle(flags)
        for fl in flags[:6]:
            nbrs = describe_neighbors(fl, self.factions, max_gap=32)
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
            if self.rng.random() < 0.58:
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
                if self._form_alliance(fl, rival):
                    self.logger.log_chronicle(
                        self.tick,
                        "alliance",
                        f"FLAG TREATY: '{fl.name}' & '{rival.name}' sealed border hearts!",
                        faction_a=fl.id,
                        faction_b=rival.id,
                    )

    def run_geopolitical_epoch(self) -> None:
        """Run one geopolitical round: alliance expiry, burnout turnover, autonomous diplomacy, and VLM leadership."""
        if self.canvas.is_whiteout:
            return

        with self._state_lock:
            self.epoch += 1
            ep = self.epoch
            tick = self.tick
            update_faction_metrics(self.canvas, self.factions, self.policy.user_factions)
            self._manage_void_lifecycle()
            self._manage_flash_raids()
            alive_art = [f for f in self.factions if f.alive and not f.is_void and not f.is_streamer_raid]
            if not alive_art:
                self.brain_status = "Waiting for factions to emerge..."
                return

            self._expire_stale_alliances(alive_art)

            # Community burnout & dynamic turnover across all 72 virtual hours (REQ-SIM-017)
            for f in alive_art:
                age = tick - f.created_tick
                if self.cfg.enable_burnout and age > 1600:
                    decay = max(0.86, 0.955 - 0.025 * f.burnout_rate)
                    if f.integrity < 0.45:
                        decay -= 0.03
                    f.hype = max(0.08, f.hype * decay)
                min_mem = 12 if f.size_tier == "micro" else (36 if f.size_tier == "medium" else 65)
                if f.hype < 0.18 or (f.integrity < 0.22 and f.member_count < min_mem):
                    f.low_integrity_epochs += 1
                elif age > 3600 and f.integrity < 0.42 and f.member_count < min_mem * 2:
                    f.low_integrity_epochs += 1
                else:
                    f.low_integrity_epochs = 0

                if f.low_integrity_epochs >= 3:
                    f.alive = False
                    self.policy.release_faction_members(f.id)
                    self.logger.log_chronicle(
                        self.tick,
                        "collapse",
                        f"'{f.name}' ({f.size_tier}, {f.timezone_region}) burned out and dissolved, opening territory for new art.",
                        faction_id=f.id,
                    )

            alive_art = [f for f in self.factions if f.alive and not f.is_void and not f.is_streamer_raid]
            if not alive_art:
                return

            self._run_autonomous_micro_and_flag_diplomacy(alive_art)

            # Select up to 4 high-urgency frontline factions for Qwen3-VL vision leadership
            def _urgency(f: Faction) -> float:
                u = self.rng.random() * 0.5
                if f.size_tier in ("macro", "ribbon"):
                    u += 0.9
                elif f.size_tier == "medium":
                    u += 0.5
                if f.pending_ally_offers:
                    u += 2.0
                if f.recent_churn >= 0.8:
                    u += 1.1
                if f.integrity < 0.40:
                    u += 0.9
                return u

            chosen = sorted(alive_art, key=_urgency, reverse=True)[:4]

        for fac in chosen:
            if self.stop_requested or not fac.alive or self.canvas.is_whiteout:
                continue

            with self._state_lock:
                view_img = self.canvas.render_faction_view_pil(fac, self.factions, margin=36)
                neighbors = describe_neighbors(fac, self.factions, max_gap=48)
                inbox = self.diplomacy.pop_inbox(fac.id)
                public_feed = self.diplomacy.recent_public(6)
                cur_tick = self.tick

            can_morph = (
                (cur_tick - fac.last_morph_tick >= 420)
                and (fac.evolution_stage < 4)
                and (fac.art_mode not in ("flag", "micro_sprite", "clean_space", "ribbon", "painting", "flux_comic", "comic_strip"))
            )
            self.brain_status = f"Act {self.act} Epoch {ep}: Qwen3-VL leading '{fac.name}' ({fac.timezone_region})..."
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

        with self._state_lock:
            has_void = any(f.alive and f.is_void for f in self.factions)
            if (
                self.cfg.enable_void
                and not self.canvas.is_whiteout
                and not has_void
                and self.tick >= self.cfg.void_min_tick
                and len(alive_art) >= 8
                and self.rng.random() < self.cfg.void_chance_per_epoch
            ):
                self.spawn_void_eruption(force=False)

            self._refresh_shared_borders()

        self.brain_status = (
            f"Act {self.act} Epoch {ep} complete — {len([f for f in self.factions if f.alive])} active factions"
        )

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
                    if self._form_alliance(fac, partner):
                        fac.hype = min(2.0, fac.hype + 0.22)
                        partner.hype = min(2.0, partner.hype + 0.22)
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
            cap = 2 if fac.size_tier == "macro" else 3
            if tgt is not None and not tgt.is_void and tgt.id not in fac.allies and len(fac.allies) < cap:
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
                    fac.ally_expiry.pop(tgt.id, None)
                    tgt.ally_expiry.pop(fac.id, None)
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

        sc = self._size_scale()
        tier_max_w = int(round((34 if fac.size_tier == "micro" else (68 if fac.size_tier == "medium" else 132)) * sc))
        tier_max_h = int(round((34 if fac.size_tier == "micro" else (68 if fac.size_tier == "medium" else 132)) * sc))
        if fac.art_mode == "banner":
            tier_max_w = int(round((140 if fac.size_tier == "macro" else (88 if fac.size_tier == "medium" else 42)) * sc))
            tier_max_h = int(round((48 if fac.size_tier == "macro" else (32 if fac.size_tier == "medium" else 18)) * sc))

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
                        allowed_colors=self.canvas.allowed_colors,
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
                    f"'{fac.name}' fortified its silhouette. ({dec.reasoning})",
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
                flair = self.rng.choice(["with pixel sparkles", "upgraded detailed 16-bit emblem", "with arcade shading", "with festive pixel ribbons"])
                new_prompt = f"{base} {flair}"
            with self._state_lock:
                W_act, H_act = self.canvas.active_w, self.canvas.active_h
                allowed = self.canvas.allowed_colors
                nx0, ny0, nw, nh = fac.x0, fac.y0, fac.w, fac.h
                if action == "EXPAND":
                    grow = 10 if fac.size_tier == "medium" else 16
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
                    nx0, ny0 = _snap_to_walls(nx0, ny0, nw, nh, W_act, H_act, snap=12)

                live_crop_rgb = self.canvas.crop_rgb_numpy(nx0, ny0, nx0 + nw, ny0 + nh)
                next_stage = min(4, fac.evolution_stage + 1)
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
                stage=next_stage,
                strength=0.82 if action == "MORPH_ART" else 0.86,
                seed=seed_val,
                extra_key=fac.extra_key,
                ribbon_axis=fac.ribbon_axis or "h",
                allowed_colors=allowed,
            )

            with self._state_lock:
                if self.canvas.is_whiteout:
                    return
                fac.set_blueprint(morphed_np, x0=nx0, y0=ny0, device=self.device)
                fac.art_prompt = new_prompt[:140]
                fac.evolution_stage = next_stage
                fac.has_border_frame = False
                fac.text_stamped = bool(t_spec is not None and t_spec.include_text and max(nw, nh) >= 34)
                fac.last_morph_tick = self.tick
                fac.hype = min(2.2, fac.hype + 0.25)
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
            if self.canvas.is_whiteout:
                return None
            if any(f.alive and f.is_void for f in self.factions) and not force:
                return None
            v_kind = variant or _VOID_VARIANTS[self._void_cycle_idx % len(_VOID_VARIANTS)]
            self._void_cycle_idx += 1

            s = self._size_scale()
            base_sz = 92 if v_kind in ("black_hole", "monster") else (84 if v_kind == "fractal" else 102)
            size = min(self.canvas.active_w // 4, int(round(base_sz * s)))
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

            bp_np = remap_indices_to_allowed(bp_np, self.canvas.allowed_colors)
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
                hype=1.85,
            )
            self.factions.append(void_fac)
            void_target = min(5500, max(1500, self.cfg.num_users // 24))
            recruited = self.policy.recruit_initial_members(void_fac, target_count=void_target)
            void_fac.member_count = recruited
            self.diplomacy.post_public(self.tick, void_fac, v_slogan, kind="threat")
            self.logger.log_chronicle(
                self.tick,
                "void",
                f"VOID ERUPTION [{v_kind.upper()}]: '{v_name}' opened at ({vx0},{vy0}) with {recruited:,} shadow pixels!",
                faction_id=void_fac.id,
                void_variant=v_kind,
                bbox=[void_fac.x0, void_fac.y0, void_fac.x1, void_fac.y1],
            )
            return void_fac

    def _manage_void_lifecycle(self) -> None:
        max_dim = int(round(140 * self._size_scale()))
        for f in self.factions:
            if not (f.alive and f.is_void):
                continue
            age = self.tick - f.created_tick
            if f.void_variant == "tendrils" and age < self.cfg.void_max_duration_ticks // 2 and f.w < max_dim:
                f.x0 = max(0, f.x0 - 8)
                f.y0 = max(0, f.y0 - 8)
                f.w = min(self.canvas.active_w - f.x0, f.w + 16)
                f.h = min(self.canvas.active_h - f.y0, f.h + 16)
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
                        "timezone_region": f.timezone_region,
                        "language": f.language,
                        "ai_origin": f.ai_origin,
                        "is_streamer_raid": f.is_streamer_raid,
                        "raid_end_tick": f.raid_end_tick,
                        "burnout_rate": f.burnout_rate,
                        "ally_expiry": {str(k): v for k, v in f.ally_expiry.items()},
                    }
                )
            if bp_dict:
                np.savez_compressed(ckpt_dir / "blueprints.npz", **bp_dict)
            pop_stats = self.policy.get_population_stats() if hasattr(self.policy, "get_population_stats") else {}
            meta = {
                "tick": self.tick,
                "epoch": self.epoch,
                "act": self.act,
                "virtual_hour": self.virtual_hour,
                "active_w": self.canvas.active_w,
                "active_h": self.canvas.active_h,
                "palette_tier": self.canvas.palette_tier,
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
        self.act = int(meta.get("act", self.act))
        self.virtual_hour = float(meta.get("virtual_hour", 0.0))
        self.canvas.expand_viewport(
            int(meta.get("active_w", self.cfg.width)),
            int(meta.get("active_h", self.cfg.height)),
            int(meta.get("palette_tier", 3)),
        )
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
                timezone_region=fd.get("timezone_region", "GLOBAL"),
                language=fd.get("language", "en"),
                ai_origin=fd.get("ai_origin", False),
                is_streamer_raid=fd.get("is_streamer_raid", False),
                raid_end_tick=fd.get("raid_end_tick", 0),
                burnout_rate=fd.get("burnout_rate", 1.0),
                ally_expiry={int(k): int(v) for k, v in fd.get("ally_expiry", {}).items()},
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
            non_void = [f for f in alive if not f.is_void]
            ai_pct = round(100.0 * sum(1 for f in non_void if f.ai_origin) / max(1, len(non_void)), 1)
            tier_counts = dict(Counter(f.size_tier for f in non_void))
            category_counts = dict(Counter(f.category for f in alive))
            return {
                "tick": self.tick,
                "max_ticks": self.cfg.max_ticks,
                "epoch": self.epoch,
                "act": self.act,
                "virtual_hour": round(self.virtual_hour, 1),
                "width": self.cfg.width,
                "height": self.cfg.height,
                "active_w": self.canvas.active_w,
                "active_h": self.canvas.active_h,
                "palette_tier": self.canvas.palette_tier,
                "palette_size": len(self.canvas.allowed_colors),
                "total_placements": self.logger.total_events,
                "actual_tps": round(self.actual_tps, 1),
                "pixels_per_sec": int(self.pixels_per_sec),
                "target_tps": self.target_tps,
                "paused": self.paused,
                "free_agents": free_count,
                "num_users": self.cfg.num_users,
                "population": pop,
                "ai_faction_pct": ai_pct,
                "tier_counts": tier_counts,
                "category_counts": category_counts,
                "device": str(self.device),
                "policy_name": self.policy_name,
                "ollama_ready": self.ollama_ready,
                "use_flux_art": self.cfg.use_flux_art,
                "use_diffusion": self.cfg.use_diffusion,
                "brain_status": self.brain_status,
                "factions": [f.to_summary_dict() for f in alive[:200]],
                "total_active_factions": len(alive),
                "alliances": alliances[:250],
                "wars": wars[:250],
                "chronicle": self.logger.recent_chronicle[-60:][::-1],
            }
