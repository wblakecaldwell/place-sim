"""Fast high-level community macro-planner & rapid storyboard preview renderer.

Simulates an entire multi-hour, 4-Act r/place community lifecycle (canvas expansions,
regional land rushes, multi-scale art packing, timezone sleep/wake waves, burnout,
session turnover vs recruitment, streamer flash-raids, Void eruptions, and Whiteout)
in ~1 second without running FLUX, Qwen3-VL, or per-pixel CNN passes.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from pixel_battle.factions import timezone_awake_factor
from pixel_battle.palette import PALETTE_RGB, WHITE_IDX

_REGIONS = ("AMERICAS", "EUROPE", "ASIA_PACIFIC", "LATAM", "GLOBAL")

_MODE_COLORS: dict[str, tuple[int, int, int]] = {
    "flux_art": (56, 189, 248),       # cyan-blue die-cut AI characters
    "flux_comic": (168, 85, 247),     # purple multi-panel comics
    "ai_code": (52, 211, 153),        # emerald geometric chaos
    "banner": (251, 191, 36),         # amber multilingual logos
    "flag": (239, 68, 68),            # red/white/blue flags
    "ribbon": (244, 114, 182),        # pink transverse ribbons
    "streamer_raid": (249, 115, 22),  # orange flash-raids
    "void": (30, 27, 46),             # abyssal void
}

_SAMPLE_PREFIXES = (
    "Pixel", "Kawaii", "Neon", "Retro", "Boba", "Chibi", "Mecha", "Ghibli",
    "Taco", "Ramen", "Voxel", "Arcade", "Synth", "Pika", "Corgi", "Neko",
    "Cyber", "LoFi", "Sushi", "Turbo", "Waffle", "Mochi", "Otaku", "Baguette",
)
_SAMPLE_SUFFIXES = (
    "Guild", "Squad", "Crew", "Club", "Empire", "Union", "Collective", "Corner",
    "Sanctuary", "Alliance", "Bots", "Cats", "Pals", "Core", "Nodes", "Verse",
)
_FLAG_NAMES = (
    "r/PlaceNL", "r/placeFR", "r/placeDE", "r/placeUK", "r/AmericanFlagInPlace",
    "r/placeJP", "r/placeKR", "r/IndiaPlace", "r/placeBR", "r/placeMX",
    "r/placeES", "r/placeIT", "r/NordicPlace SE", "r/placeUkraine", "r/placeCA",
    "r/placePL", "r/placeTR", "r/placeGR", "r/transplace", "r/RainbowPlace",
)


@dataclass
class MacroPlanConfig:
    """High-level community planning knobs for rapid multi-hour dry-runs."""

    width: int = 2000
    height: int = 2000
    hours: float = 8.0
    virtual_hours: float = 72.0
    online_slots: int = 120_000
    max_active_factions: int = 1200

    # 4-Act milestones
    enable_expansion: bool = True
    enable_whiteout: bool = True
    act2_progress: float = 0.31
    act3_progress: float = 0.62
    whiteout_progress: float = 0.95

    # Tunable Rev 2.1 community planning controls
    factions_per_sim_min: float = 3.2          # Discovery throughput (baseline was ~1.4/min)
    size_multiplier: float = 1.65              # Art size scale factor
    burnout_min_coverage: float = 0.72         # Crowd-gated burnout (0.0 = always burn out)
    burnout_age_vhours: float = 24.0           # Virtual hours before burnout decay begins
    land_rush_target_cov: float = 0.65         # Keep preferring newly opened wing until cov reaches this
    target_organized_share: float = 0.82       # Target share of online slots inside real factions
    free_agent_clustering: float = 0.85        # 0.0 = rainbow static, 1.0 = cohesive regional colonies
    raid_interval_vhours: float = 3.5
    void_interval_vhours: float = 10.0
    seed: int = 2022


@dataclass
class PlannedFaction:
    id: int
    name: str
    art_mode: str
    size_tier: str
    timezone_region: str
    x0: int
    y0: int
    w: int
    h: int
    born_vh: float
    died_vh: float | None = None
    color_idx: int = 14
    secondary_idx: int = 6
    ribbon_axis: str = ""
    hype: float = 1.0
    is_raid: bool = False
    is_void: bool = False
    raid_end_vh: float = 0.0

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h


@dataclass
class PlanFrameMetrics:
    vhour: float
    wall_min: float
    act: int
    active_w: int
    active_h: int
    palette_size: int
    alive_factions: int
    cumulative_founded: int
    cumulative_burned_out: int
    cov_overall: float
    cov_act1: float
    cov_act2: float
    cov_act3: float
    organized_pct: float
    free_agent_pct: float
    est_pixels_m: float
    est_users_m: float


@dataclass
class MacroPlanResult:
    config: MacroPlanConfig
    factions: list[PlannedFaction]
    timeline: list[PlanFrameMetrics]
    events: list[dict[str, Any]]

    def summary_dict(self) -> dict[str, Any]:
        # Peak frame is the last frame before Act 4 Whiteout
        pre_wo = [m for m in self.timeline if m.act < 4] or self.timeline
        peak = pre_wo[-1]
        return {
            "width": self.config.width,
            "height": self.config.height,
            "hours": self.config.hours,
            "peak_vhour": round(peak.vhour, 1),
            "peak_alive_factions": peak.alive_factions,
            "total_founded": peak.cumulative_founded,
            "total_burned_out": peak.cumulative_burned_out,
            "peak_cov_overall_pct": round(peak.cov_overall * 100, 1),
            "peak_cov_act1_pct": round(peak.cov_act1 * 100, 1),
            "peak_cov_act2_pct": round(peak.cov_act2 * 100, 1),
            "peak_cov_act3_pct": round(peak.cov_act3 * 100, 1),
            "peak_organized_pct": round(peak.organized_pct * 100, 1),
            "peak_free_agent_pct": round(peak.free_agent_pct * 100, 1),
            "est_total_pixels_m": round(self.timeline[-1].est_pixels_m, 1),
            "est_total_users_m": round(self.timeline[-1].est_users_m, 2),
        }


def preset_baseline_2000() -> MacroPlanConfig:
    """Reproduces last night's 8h 2000x2000 run behavior for direct comparison."""
    return MacroPlanConfig(
        width=2000,
        height=2000,
        factions_per_sim_min=1.35,
        size_multiplier=1.0,
        burnout_min_coverage=0.0,
        burnout_age_vhours=9.0,
        land_rush_target_cov=0.08,
        target_organized_share=0.14,
        free_agent_clustering=0.12,
    )


def preset_tuned_1024() -> MacroPlanConfig:
    """Rev 2.1 dense wall-to-wall 1024x1024 plan (512 -> 1024x512 -> 1024x1024)."""
    return MacroPlanConfig(
        width=1024,
        height=1024,
        online_slots=80_000,
        max_active_factions=650,
        factions_per_sim_min=2.4,
        size_multiplier=1.30,
        burnout_min_coverage=0.78,
        burnout_age_vhours=26.0,
        land_rush_target_cov=0.72,
        target_organized_share=0.84,
        free_agent_clustering=0.88,
    )


def preset_tuned_2000() -> MacroPlanConfig:
    """Rev 2.1 wall-to-wall 1:1 Reddit 2022 2000x2000 plan."""
    return MacroPlanConfig(
        width=2000,
        height=2000,
        online_slots=120_000,
        max_active_factions=1400,
        factions_per_sim_min=4.2,
        size_multiplier=1.75,
        burnout_min_coverage=0.75,
        burnout_age_vhours=26.0,
        land_rush_target_cov=0.70,
        target_organized_share=0.82,
        free_agent_clustering=0.88,
    )


def preset_ytshort() -> MacroPlanConfig:
    """Rev 2.1 9:16 YouTube Short plan (1080x1920 native -> 4K 2160x3840 @ 60fps)."""
    return MacroPlanConfig(
        width=1080,
        height=1920,
        hours=6.0,
        online_slots=90_000,
        max_active_factions=1200,
        factions_per_sim_min=3.7,
        size_multiplier=0.95,
        burnout_min_coverage=0.60,
        burnout_age_vhours=24.0,
        land_rush_target_cov=0.70,
        target_organized_share=0.80,
        free_agent_clustering=0.24,
    )


def run_macro_plan(cfg: MacroPlanConfig, steps: int = 144) -> MacroPlanResult:
    """Execute a complete 72-virtual-hour macro community simulation on a coarse occupancy grid."""
    rng = random.Random(cfg.seed)
    W, H = cfg.width, cfg.height
    # Use a 10px-per-cell coarse grid (200x200 for 2000x2000) so full 8h packing runs in < 0.25s!
    cell = max(4, W // 200)
    gw, gh = W // cell, H // cell
    occ = np.zeros((gh, gw), dtype=np.int16)  # count of alive factions covering each coarse cell

    factions: list[PlannedFaction] = []
    alive: list[PlannedFaction] = []
    timeline: list[PlanFrameMetrics] = []
    events: list[dict[str, Any]] = []

    half_w, half_h = W // 2, H // 2
    act = 1 if cfg.enable_expansion else 3
    active_w = half_w if cfg.enable_expansion else W
    active_h = half_h if cfg.enable_expansion else H
    pal_size = 16 if cfg.enable_expansion else 32

    used_flags: set[str] = set()
    spawn_carry = 0.0
    next_raid_vh = cfg.raid_interval_vhours
    next_void_vh = cfg.void_interval_vhours
    burned_total = 0
    org_share = 0.05

    dt_vh = cfg.virtual_hours / max(1, steps)
    dt_min = (cfg.hours * 60.0) / max(1, steps)

    def _mark_occ(fac: PlannedFaction, delta: int) -> None:
        gx0 = max(0, fac.x0 // cell)
        gy0 = max(0, fac.y0 // cell)
        gx1 = min(gw, max(gx0 + 1, (fac.x1 + cell - 1) // cell))
        gy1 = min(gh, max(gy0 + 1, (fac.y1 + cell - 1) // cell))
        occ[gy0:gy1, gx0:gx1] = np.maximum(0, occ[gy0:gy1, gx0:gx1] + delta)

    def _region_cov(x0: int, y0: int, x1: int, y1: int) -> float:
        gx0, gy0 = x0 // cell, y0 // cell
        gx1, gy1 = max(gx0 + 1, x1 // cell), max(gy0 + 1, y1 // cell)
        sub = occ[gy0:gy1, gx0:gx1]
        return float((sub > 0).mean()) if sub.size else 0.0

    def _pick_spot(bw: int, bh: int, prefer: tuple[int, int, int, int] | None) -> tuple[int, int]:
        rx0, ry0, rx1, ry1 = prefer if prefer is not None else (0, 0, active_w, active_h)
        max_x = max(rx0, rx1 - bw)
        max_y = max(ry0, ry1 - bh)
        best = (rx0, ry0)
        best_score = 1e9
        cw = max(1, bw // cell)
        ch = max(1, bh // cell)
        for _ in range(18):
            cx = rng.randint(rx0, max_x) if max_x > rx0 else rx0
            cy = rng.randint(ry0, max_y) if max_y > ry0 else ry0
            gx, gy = cx // cell, cy // cell
            overlap = float((occ[gy : min(gh, gy + ch), gx : min(gw, gx + cw)] > 0).mean())
            if overlap < best_score:
                best_score = overlap
                best = (cx, cy)
                if overlap == 0.0:
                    break
        return best

    for step_i in range(steps + 1):
        progress = step_i / float(max(1, steps))
        vh = progress * cfg.virtual_hours
        wall_min = progress * cfg.hours * 60.0

        # Act transitions
        if cfg.enable_expansion and act == 1 and progress >= cfg.act2_progress:
            act, active_w, active_h, pal_size = 2, W, half_h, 24
            events.append({"vhour": round(vh, 1), "kind": "expand_canvas", "text": f"Act 2: Expanded East to {W}x{half_h} (24 colors)"})
        if cfg.enable_expansion and act == 2 and progress >= cfg.act3_progress:
            act, active_w, active_h, pal_size = 3, W, H, 32
            events.append({"vhour": round(vh, 1), "kind": "expand_canvas", "text": f"Act 3: Expanded South to {W}x{H} (32 colors)"})
        if cfg.enable_whiteout and act < 4 and progress >= cfg.whiteout_progress:
            act, active_w, active_h, pal_size = 4, W, H, 1
            events.append({"vhour": round(vh, 1), "kind": "whiteout", "text": "Act 4: Whiteout Finale begins"})

        # Compute current coverage per wing
        cov_all = _region_cov(0, 0, active_w, active_h)
        cov_a1 = _region_cov(0, 0, half_w, half_h)
        cov_a2 = _region_cov(half_w, 0, W, half_h) if act >= 2 else 0.0
        cov_a3 = _region_cov(0, half_h, W, H) if act >= 3 else 0.0

        if act < 4 and step_i > 0:
            # 1. Disperse expired raids & voids
            for fac in list(alive):
                if (fac.is_raid or fac.is_void) and vh >= fac.raid_end_vh:
                    fac.died_vh = vh
                    _mark_occ(fac, -1)
                    alive.remove(fac)

            # 2. Crowd-gated community burnout
            allow_burnout = cov_all >= cfg.burnout_min_coverage
            if allow_burnout:
                for fac in list(alive):
                    if fac.is_raid or fac.is_void:
                        continue
                    age = vh - fac.born_vh
                    if age > cfg.burnout_age_vhours:
                        fac.hype *= 0.92
                        if fac.hype < 0.24:
                            fac.died_vh = vh
                            _mark_occ(fac, -1)
                            alive.remove(fac)
                            burned_total += 1

            # 3. Persistent regional land-rush targeting when a newly unlocked wing is still underfilled
            prefer_region: tuple[int, int, int, int] | None = None
            if act == 2 and cov_a2 < cfg.land_rush_target_cov:
                prefer_region = (half_w, 0, W, half_h)
            elif act == 3 and cov_a3 < cfg.land_rush_target_cov:
                prefer_region = (0, half_h, W, H)

            # 4. Spawn new factions at planned discovery rate
            area_frac = (active_w * active_h) / float(W * H)
            eff_cap = max(24, int(round(cfg.max_active_factions * min(1.0, area_frac + 0.10))))
            spawn_carry += cfg.factions_per_sim_min * dt_min
            n_spawn = min(int(spawn_carry), max(0, eff_cap - len(alive)))
            spawn_carry -= n_spawn

            base_scale = (1.38 if W >= 1800 else (1.15 if W >= 960 else 1.0)) * cfg.size_multiplier
            for _ in range(n_spawn):
                r = rng.random()
                if r < 0.03:
                    tier, mode = "ribbon", "ribbon"
                elif r < 0.12 and len(used_flags) < len(_FLAG_NAMES):
                    tier = rng.choice(["medium", "macro", "medium", "micro"])
                    mode = "flag"
                elif r < 0.24:
                    tier = rng.choice(["macro", "medium"])
                    mode = "flux_comic"
                elif r < 0.42:
                    tier = rng.choice(["micro", "medium", "micro"])
                    mode = "ai_code"
                elif r < 0.54:
                    tier = rng.choice(["micro", "medium", "macro"])
                    mode = "banner"
                else:
                    tier = rng.choices(["micro", "medium", "macro"], weights=[0.50, 0.34, 0.16])[0]
                    mode = "flux_art"

                axis = ""
                if mode == "ribbon":
                    axis = "h" if rng.random() < 0.65 else "v"
                    thick = int(round(rng.choice([8, 10, 12]) * cfg.size_multiplier))
                    if axis == "h":
                        bw, bh = active_w, thick
                        x0, y0 = 0, rng.randint(0, max(0, active_h - bh))
                    else:
                        bw, bh = thick, active_h
                        x0, y0 = rng.randint(0, max(0, active_w - bw)), 0
                else:
                    if tier == "micro":
                        bw = int(round(rng.choice([24, 28, 32]) * base_scale))
                        bh = int(round(rng.choice([22, 26, 30]) * base_scale))
                    elif tier == "medium":
                        bw = int(round((rng.choice([82, 94]) if mode in ("flux_comic", "banner") else rng.choice([48, 56, 62])) * base_scale))
                        bh = int(round((rng.choice([36, 40]) if mode == "flux_comic" else (rng.choice([26, 32]) if mode == "banner" else rng.choice([46, 54]))) * base_scale))
                    else:
                        bw = int(round((rng.choice([112, 128]) if mode in ("flux_comic", "banner") else rng.choice([80, 94, 108])) * base_scale))
                        bh = int(round((rng.choice([48, 56]) if mode == "flux_comic" else (rng.choice([38, 44]) if mode == "banner" else rng.choice([76, 90]))) * base_scale))
                    bw = max(16, min(active_w // 3, bw))
                    bh = max(16, min(active_h // 3, bh))
                    x0, y0 = _pick_spot(bw, bh, prefer_region)

                if mode == "flag":
                    avail = [n for n in _FLAG_NAMES if n not in used_flags]
                    name = rng.choice(avail)
                    used_flags.add(name)
                else:
                    name = f"{rng.choice(_SAMPLE_PREFIXES)}{rng.choice(_SAMPLE_SUFFIXES)} #{len(factions) + 1}"

                tz = rng.choice(_REGIONS)
                fac = PlannedFaction(
                    id=len(factions),
                    name=name,
                    art_mode=mode,
                    size_tier=tier,
                    timezone_region=tz,
                    x0=x0,
                    y0=y0,
                    w=bw,
                    h=bh,
                    born_vh=vh,
                    color_idx=rng.randint(2, min(31, pal_size - 1)),
                    secondary_idx=rng.randint(2, min(31, pal_size - 1)),
                    ribbon_axis=axis,
                )
                factions.append(fac)
                alive.append(fac)
                _mark_occ(fac, +1)

            # 5. Prime-time Streamer Flash-Raid
            if vh >= next_raid_vh and len(alive) >= 8:
                next_raid_vh = vh + cfg.raid_interval_vhours
                prime_tz = max(_REGIONS[:4], key=lambda tz: timezone_awake_factor(tz, vh % 24.0))
                anchor = rng.choice(alive)
                rw = min(active_w // 4, int(round(92 * base_scale)))
                rh = min(active_h // 4, int(round(92 * base_scale)))
                rx0 = max(0, min(active_w - rw, anchor.x0 + anchor.w // 2 - rw // 2))
                ry0 = max(0, min(active_h - rh, anchor.y0 + anchor.h // 2 - rh // 2))
                raid = PlannedFaction(
                    id=len(factions),
                    name=f"{prime_tz.title()} Blitz #{len(factions) + 1}",
                    art_mode="streamer_raid",
                    size_tier="macro",
                    timezone_region=prime_tz,
                    x0=rx0,
                    y0=ry0,
                    w=rw,
                    h=rh,
                    born_vh=vh,
                    color_idx=5,
                    secondary_idx=6,
                    is_raid=True,
                    raid_end_vh=vh + 1.4,
                )
                factions.append(raid)
                alive.append(raid)
                _mark_occ(raid, +1)
                events.append({"vhour": round(vh, 1), "kind": "raid", "text": f"Flash-Raid [{prime_tz}] '{raid.name}' hits '{anchor.name}'"})

            # 6. Void eruption
            if vh >= next_void_vh and len(alive) >= 12:
                next_void_vh = vh + cfg.void_interval_vhours
                vw = min(active_w // 4, int(round(108 * base_scale)))
                vx0 = rng.randint(0, max(0, active_w - vw))
                vy0 = rng.randint(0, max(0, active_h - vw))
                void_f = PlannedFaction(
                    id=len(factions),
                    name=f"THE VOID #{len(factions) + 1}",
                    art_mode="void",
                    size_tier="macro",
                    timezone_region="GLOBAL",
                    x0=vx0,
                    y0=vy0,
                    w=vw,
                    h=vw,
                    born_vh=vh,
                    color_idx=1,
                    secondary_idx=2,
                    is_void=True,
                    raid_end_vh=vh + 2.2,
                )
                factions.append(void_f)
                alive.append(void_f)
                _mark_occ(void_f, +1)
                events.append({"vhour": round(vh, 1), "kind": "void", "text": f"Void Eruption opened at ({vx0},{vy0})"})

        # Smooth citizen organization share ODE (reflects recruitment vs session recycling)
        if act == 4:
            org_share = max(0.0, org_share * 0.65)
        else:
            desired_org = min(cfg.target_organized_share, 0.08 + cov_all * 1.15) if alive else 0.0
            org_share += (desired_org - org_share) * 0.22

        # Estimate cumulative pixels & unique users (at ~6,500 px/s sustained, ~15.5 px/user)
        est_px_m = (wall_min * 60.0 * 6500.0) / 1e6
        est_users_m = (est_px_m / 15.5) + (cfg.online_slots / 1e6)

        timeline.append(
            PlanFrameMetrics(
                vhour=round(vh, 2),
                wall_min=round(wall_min, 1),
                act=act,
                active_w=active_w,
                active_h=active_h,
                palette_size=pal_size,
                alive_factions=len(alive),
                cumulative_founded=len(factions),
                cumulative_burned_out=burned_total,
                cov_overall=round(_region_cov(0, 0, active_w, active_h), 4),
                cov_act1=round(_region_cov(0, 0, half_w, half_h), 4),
                cov_act2=round(_region_cov(half_w, 0, W, half_h) if act >= 2 else 0.0, 4),
                cov_act3=round(_region_cov(0, half_h, W, H) if act >= 3 else 0.0, 4),
                organized_pct=round(org_share, 4),
                free_agent_pct=round(1.0 - org_share, 4),
                est_pixels_m=round(est_px_m, 2),
                est_users_m=round(est_users_m, 2),
            )
        )

    return MacroPlanResult(config=cfg, factions=factions, timeline=timeline, events=events)


def export_plan_artifacts(result: MacroPlanResult, out_dir: Path, render_mp4: bool = True) -> dict[str, Path]:
    """Save plan_trace.json and optionally render a rapid 10-second 720p MP4 storyboard."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_path = out_dir / "plan_trace.json"
    payload = {
        "summary": result.summary_dict(),
        "config": asdict(result.config),
        "timeline": [asdict(m) for m in result.timeline],
        "events": result.events,
        "factions": [asdict(f) for f in result.factions],
    }
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out: dict[str, Path] = {"trace": trace_path}

    if render_mp4:
        mp4_path = out_dir / "plan_storyboard.mp4"
        _render_plan_mp4(result, mp4_path)
        out["mp4"] = mp4_path
    return out


def _render_plan_mp4(result: MacroPlanResult, out_path: Path, size: int = 720, fps: int = 24) -> Path:
    """Render a rapid 6-second visual storyboard MP4 showing 4-Act macro community growth."""
    cfg = result.config
    W, H = cfg.width, cfg.height
    hud_h = 48
    if H > W:
        # Vertical 9:16 storyboard preview
        canvas_w = int(round(size * (W / float(H))))
        canvas_w += canvas_w % 2
        canvas_h = size + (size % 2)
    else:
        canvas_w = size + (size % 2)
        canvas_h = int(round(size * (H / float(W))))
        canvas_h += canvas_h % 2
    out_w, out_h = canvas_w, canvas_h + hud_h

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-", "-an", "-vcodec", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for m in result.timeline:
            img = Image.new("RGB", (out_w, out_h), (12, 16, 24))
            draw = ImageDraw.Draw(img)

            # Dynamic camera viewport scaling
            scale = min(canvas_w / float(m.active_w), canvas_h / float(m.active_h))
            fw = int(round(m.active_w * scale))
            fh = int(round(m.active_h * scale))
            ox = (canvas_w - fw) // 2
            oy = (canvas_h - fh) // 2

            # Background base (whiteout fade in Act 4, subtle clustered tone otherwise)
            bg = (245, 247, 250) if m.act == 4 else (24, 30, 44)
            draw.rectangle([ox, oy, ox + fw, oy + fh], fill=bg, outline=(71, 85, 105), width=2)

            if m.act < 4:
                for fac in result.factions:
                    if fac.born_vh > m.vhour:
                        continue
                    if fac.died_vh is not None and fac.died_vh <= m.vhour:
                        continue
                    x0 = ox + int(fac.x0 * scale)
                    y0 = oy + int(fac.y0 * scale)
                    x1 = ox + max(x0 - ox + 2, int(fac.x1 * scale))
                    y1 = oy + max(y0 - oy + 2, int(fac.y1 * scale))
                    col = _MODE_COLORS.get(fac.art_mode, (148, 163, 184))
                    if fac.art_mode in ("flux_art", "ai_code", "streamer_raid", "void"):
                        draw.ellipse([x0, y0, x1, y1], fill=col, outline=(10, 12, 18))
                    else:
                        draw.rectangle([x0, y0, x1, y1], fill=col, outline=(10, 12, 18))

            # HUD footer
            draw.rectangle([0, canvas_h, out_w, out_h], fill=(15, 20, 30))
            draw.text(
                (10, canvas_h + 6),
                f"ACT {m.act} ({m.active_w}x{m.active_h}, {m.palette_size}c) | Hr {m.vhour:4.1f}/72 | {m.alive_factions} fac | Cov {m.cov_overall*100:4.1f}%",
                fill=(56, 189, 248),
            )
            draw.text(
                (10, canvas_h + 26),
                f"Org {m.organized_pct*100:4.1f}% | Free {m.free_agent_pct*100:4.1f}% | S:{m.cov_act3*100:4.1f}% | {m.est_pixels_m:.0f}M px",
                fill=(226, 232, 240),
            )
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())

        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg plan render failed ({ret}): {err[-600:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
    return out_path
