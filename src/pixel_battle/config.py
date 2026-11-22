"""Simulation configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimConfig:
    # Canvas & population (36k concurrent online GPU slots cycling through unique 32-bit user_ids; 20% one-shot)
    width: int = 512
    height: int = 512
    num_users: int = 36_000
    base_cooldown: int = 18  # Avg ticks between placements per online slot (~2,000 px/tick)

    # Simulation horizon & pacing
    max_ticks: int = 8_000          # Placement horizon (or overridden by max_hours)
    max_hours: float = 0.0          # If > 0, run for this many wall-clock hours cleanly
    chaos_ticks: int = 140          # Pure free-agent chaos before first pareidolia discovery
    discovery_interval: int = 45    # Ticks between multi-faction discovery waves
    epoch_interval: int = 110       # Ticks between faction VLM strategic/diplomacy turns
    max_active_factions: int = 200  # Dense r/place power-law ecosystem (~60% micro, 24% med, 12% macro, 4% ribbon)
    checkpoint_interval: int = 250  # Save full state every N ticks
    target_tps: float = 18.0        # Soft ticks/sec cap when watching live (0 = uncapped)

    # Void dynamics
    enable_void: bool = True
    void_min_tick: int = 900        # Earliest tick a Void eruption can emerge
    void_chance_per_epoch: float = 0.12
    void_max_duration_ticks: int = 450

    # Local AI models
    ollama_url: str = "http://localhost:11434"
    vlm_model: str = "qwen2.5vl:7b"
    text_model: str = "qwen2.5:7b-instruct"
    use_diffusion: bool = True
    use_flux_typography: bool = True
    flux_model_repo: str = "dhairyashil/FLUX.1-schnell-mflux-4bit"
    flux_steps: int = 4
    diffusion_base_model: str = "stabilityai/sdxl-turbo"
    diffusion_lora_repo: str = "nerijs/pixel-art-xl"
    diffusion_lora_weight: str = "pixel-art-xl.safetensors"
    diffusion_steps: int = 6

    # Pixel painting policy ("auto" -> neural if trained checkpoint exists, else stochastic)
    policy_mode: str = "auto"
    neural_model_path: Path = field(default_factory=lambda: Path("models/reddit_citizen.pt"))

    # Device & output paths
    device: str = "auto"            # auto -> mps if available, else cpu
    run_dir: Path = field(default_factory=lambda: Path("runs/latest"))
    seed: int = 42

    # Web server
    host: str = "127.0.0.1"
    port: int = 8765
