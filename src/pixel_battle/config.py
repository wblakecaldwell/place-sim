"""Simulation configuration dataclass and Reddit r/place scale presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    # Preset ("standard" | "large" | "reddit2022")
    preset: str = "standard"

    # Canvas & population (online GPU slots cycling through unique 32-bit user_ids; 20% one-shot)
    width: int = 512
    height: int = 512
    num_users: int = 36_000
    base_cooldown: int = 18

    # 4-Act r/place 2022 Canvas Expansion & Palette Unlocks (16 -> 24 -> 32 -> Whiteout)
    enable_expansion: bool = True
    enable_whiteout: bool = True
    act2_progress: float = 0.31     # Expand East (W x H/2), unlock 24 colors
    act3_progress: float = 0.62     # Expand South (W x H), unlock 32 colors
    whiteout_progress: float = 0.95 # Lock palette to white-only finale (last 5%)

    # Virtual 72-hour clock, timezones, burnout, and streamer flash-raids
    virtual_hours_total: float = 72.0
    enable_timezones: bool = True
    enable_flash_raids: bool = True
    flash_raid_interval_ticks: int = 750
    enable_burnout: bool = True

    # Rev 2.1 Community Planning Controls (defaults preserve legacy behavior unless set / preset)
    spawn_rate_per_min: float = 0.0        # If > 0, pace discovery waves to track factions/wall-clock min
    size_multiplier: float = 1.0           # Artwork dimension scale multiplier
    burnout_min_coverage: float = 0.0      # Crowd-gated burnout (0.0 = always retire old art; 0.60 = only >= 60%)
    burnout_age_vhours: float = 24.0       # Virtual hours before crowd-gated burnout decay begins
    land_rush_target_cov: float = 0.0      # If > 0, keep preferring newly opened wing until coverage reaches target
    target_organized_share: float = 0.80   # Target share of online citizen slots inside active factions (0..0.95)
    free_agent_clustering: float = 0.65    # 0.0 = rainbow confetti; 0.90+ = calm corner blobs

    # Rev 2.2 Deterministic Step-Locked & Clean-Canvas Controls
    sync_mode: bool = False                # True = step-locked deterministic ticks (finish all work before tick+1)
    day0_factions: int = 0                 # Prepped Day-0 Launch Wave spawned across Ticks 1..45
    target_total_factions: int = 0         # Exact total factions to spawn across the run (in sync_mode)
    factions_per_wave: int = 4             # Factions spawned per discovery wave (in sync_mode)
    targeted_chaos_only: bool = False      # True = pure white empty space; chaotic players attack active art only
    no_generic_fallbacks: bool = False     # True = zero procedural shield spam & <= 1 copy ever of each template
    art_bank_dir: Path | None = None       # Optional pre-forged Art Bank directory (decoupled Qwen3-VL -> FLUX)

    # Simulation horizon & pacing
    max_ticks: int = 8_000          # Placement horizon (or overridden by max_hours)
    max_hours: float = 0.0          # If > 0, run for this many wall-clock hours cleanly
    chaos_ticks: int = 140          # Pure free-agent chaos before first discovery
    discovery_interval: int = 45    # Ticks between discovery / AI-forge pulls
    epoch_interval: int = 130       # Ticks between faction VLM strategic/diplomacy turns
    max_active_factions: int = 200
    checkpoint_interval: int = 300
    target_tps: float = 18.0        # Soft ticks/sec cap when watching live (0 = uncapped)

    # Void dynamics
    enable_void: bool = True
    void_min_tick: int = 900
    void_chance_per_epoch: float = 0.14
    void_max_duration_ticks: int = 480

    # Local AI models (OpenAI-compatible LM Studio / Ollama + MLX FLUX Pixel-Art LoRA)
    llm_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen3-vl-30b-a3b-instruct-mlx"
    keychain_service: str = "place-sim"
    # Legacy aliases preserved for backward compatibility
    ollama_url: str = "http://localhost:11434"
    vlm_model: str = "qwen3-vl-30b-a3b-instruct-mlx"
    text_model: str = "qwen3-vl-30b-a3b-instruct-mlx"

    # Generative pixel-art engine (MLX 4-bit FLUX.1-schnell + Pixel-Art LoRA)
    use_flux_art: bool = True
    flux_model_repo: str = "dhairyashil/FLUX.1-schnell-mflux-4bit"
    flux_pixel_lora_repo: str = "hliu143/dr-diffusion-flux-1-s-pixel-art-lora"
    flux_pixel_lora_file: str = "pixelart_schnell_v1.safetensors"
    flux_pixel_lora_scale: float = 0.95
    flux_steps: int = 4
    use_flux_typography: bool = True
    use_diffusion: bool = False     # Optional legacy SDXL-Turbo fallback (off by default in Rev 2.0)
    diffusion_base_model: str = "stabilityai/sdxl-turbo"
    diffusion_lora_repo: str = "nerijs/pixel-art-xl"
    diffusion_lora_weight: str = "pixel-art-xl.safetensors"
    diffusion_steps: int = 6

    # Pixel painting policy ("auto" -> neural if trained checkpoint exists, else stochastic)
    policy_mode: str = "auto"
    neural_model_path: Path = field(default_factory=lambda: Path("models/reddit_citizen.pt"))

    # Device & output paths
    device: str = "auto"
    run_dir: Path = field(default_factory=lambda: Path("runs/latest"))
    seed: int = 42

    # Web server
    host: str = "127.0.0.1"
    port: int = 8765


def apply_preset(cfg: SimConfig, preset: str) -> SimConfig:
    """Apply a named scale preset ('standard', 'large', 'reddit2022', 'ytshort') to SimConfig."""
    p = preset.strip().lower()
    cfg.preset = p
    if p in ("ytshort", "short", "youtube_short"):
        # Fixed 9:16 360x640 canvas (230k px — exact 9:16 twin of 512x512; 6x -> 4K 2160x3840 & 3x -> 1080p 1080x1920)
        # Corners/edges + bold landmarks first, Niche-Carvers seal every white pocket, focused rival wars, no whiteout
        cfg.preset = "ytshort"
        cfg.width = 360
        cfg.height = 640
        cfg.num_users = 72_000
        cfg.base_cooldown = 16
        cfg.max_ticks = 10_000
        cfg.max_hours = 0.0
        cfg.max_active_factions = 300
        cfg.chaos_ticks = 1
        cfg.discovery_interval = 20
        cfg.epoch_interval = 160
        cfg.checkpoint_interval = 500
        cfg.void_min_tick = 700
        cfg.flash_raid_interval_ticks = 720
        cfg.enable_expansion = False
        cfg.enable_whiteout = False
        cfg.enable_timezones = True
        cfg.enable_flash_raids = True
        cfg.enable_burnout = True
        # Deterministic Rev 2.3 controls: Hour-1 full pack + AI visual niche-hunters + white-claiming chaos
        cfg.sync_mode = True
        cfg.day0_factions = 150
        cfg.target_total_factions = 440
        cfg.factions_per_wave = 8
        cfg.targeted_chaos_only = False
        cfg.no_generic_fallbacks = True
        cfg.spawn_rate_per_min = 0.0
        cfg.size_multiplier = 1.0
        cfg.burnout_min_coverage = 0.90
        cfg.burnout_age_vhours = 44.0
        cfg.land_rush_target_cov = 0.0
        cfg.target_organized_share = 0.86
        cfg.free_agent_clustering = 0.36
    elif p == "reddit2022":
        # Exact 1:1 Reddit r/place 2022 canvas (1000x1000 -> 2000x1000 -> 2000x2000 -> Whiteout)
        # 120k online slots cycling ~15.5 px/user reaches ~10M unique users & 2M 1-shot over 6-8h.
        cfg.width = 2000
        cfg.height = 2000
        cfg.num_users = 120_000
        cfg.base_cooldown = 18
        cfg.max_active_factions = 1_200
        cfg.chaos_ticks = 120
        cfg.discovery_interval = 30
        cfg.epoch_interval = 140
        cfg.checkpoint_interval = 400
        cfg.void_min_tick = 1_200
        cfg.flash_raid_interval_ticks = 850
        cfg.enable_expansion = True
        cfg.enable_whiteout = True
        cfg.enable_timezones = True
        cfg.enable_flash_raids = True
        cfg.enable_burnout = True
    elif p == "large":
        # Half-scale (512x512 -> 1024x512 -> 1024x1024 -> Whiteout)
        cfg.width = 1024
        cfg.height = 1024
        cfg.num_users = 80_000
        cfg.base_cooldown = 16
        cfg.max_active_factions = 500
        cfg.chaos_ticks = 120
        cfg.discovery_interval = 35
        cfg.epoch_interval = 130
        cfg.checkpoint_interval = 350
        cfg.void_min_tick = 1_000
        cfg.flash_raid_interval_ticks = 750
        cfg.enable_expansion = True
        cfg.enable_whiteout = True
        cfg.enable_timezones = True
        cfg.enable_flash_raids = True
        cfg.enable_burnout = True
    else:
        cfg.width = 512
        cfg.height = 512
        cfg.num_users = 36_000
        cfg.base_cooldown = 18
        cfg.max_active_factions = 200
        cfg.chaos_ticks = 120
        cfg.discovery_interval = 45
        cfg.epoch_interval = 120
    return cfg
