"""Faction state, blueprint edge-weighting, diplomacy board, and social dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pixel_battle.palette import COLOR_HEX, NUM_COLORS

# Peak UTC hour for evening prime time (~20:00 local) per Timezone Region
TIMEZONE_PEAK_UTC: dict[str, float] = {
    "EUROPE": 19.0,       # ~20:00-21:00 CET
    "AMERICAS": 1.0,      # ~20:00 EST / 17:00 PST
    "LATAM": 0.5,         # ~20:30 BRT / 18:30 CST
    "ASIA_PACIFIC": 12.0, # ~20:00 CST/SGT / 21:00 JST
    "GLOBAL": -1.0,       # Always evenly awake
}

FLAG_TIMEZONES: dict[str, str] = {
    "germany": "EUROPE", "france": "EUROPE", "ukraine": "EUROPE", "italy": "EUROPE",
    "uk": "EUROPE", "turkey": "EUROPE", "sweden": "EUROPE", "norway": "EUROPE",
    "finland": "EUROPE", "denmark": "EUROPE", "switzerland": "EUROPE", "netherlands": "EUROPE",
    "poland": "EUROPE", "ireland": "EUROPE", "spain": "EUROPE", "portugal": "EUROPE",
    "greece": "EUROPE", "belgium": "EUROPE",
    "usa": "AMERICAS", "canada": "AMERICAS",
    "brazil": "LATAM", "mexico": "LATAM", "argentina": "LATAM", "guatemala": "LATAM",
    "colombia": "LATAM", "chile": "LATAM",
    "japan": "ASIA_PACIFIC", "south_korea": "ASIA_PACIFIC", "india": "ASIA_PACIFIC", "vietnam": "ASIA_PACIFIC",
    "pride": "GLOBAL", "trans": "GLOBAL",
}


def timezone_awake_factor(timezone_region: str, virtual_hour_utc: float) -> float:
    """Return diurnal activity multiplier in [0.22, 1.35] for a Timezone Region at a UTC hour."""
    peak = TIMEZONE_PEAK_UTC.get(timezone_region.upper(), -1.0)
    if peak < 0.0:
        return 1.0
    hour_mod = float(virtual_hour_utc) % 24.0
    # Cosine wave: 1.0 at peak hour, -1.0 at peak+12h (local ~08:00/04:00 sleep trough)
    phase = 2.0 * math.pi * ((hour_mod - peak) / 24.0)
    wave = 0.5 * (1.0 + math.cos(phase))  # [0.0 .. 1.0]
    # Steepen sleep trough slightly so 2am-6am local is genuinely vulnerable (~0.22x)
    return 0.22 + 1.13 * (wave ** 1.25)


def compute_edge_weights(blueprint: torch.Tensor) -> torch.Tensor:
    """Compute outline/contrast priority weights for a [h, w] int16 blueprint (-1 = transparent)."""
    h, w = blueprint.shape
    device = blueprint.device
    valid = blueprint >= 0
    weights = torch.full((h, w), 0.55, dtype=torch.float32, device=device)

    # Check 4-neighbor color transitions -> high priority so builders sketch outlines first
    diff_count = torch.zeros((h, w), dtype=torch.float32, device=device)
    if h > 1:
        d_ud = (blueprint[1:, :] != blueprint[:-1, :]).to(torch.float32)
        diff_count[1:, :] += d_ud
        diff_count[:-1, :] += d_ud
    if w > 1:
        d_lr = (blueprint[:, 1:] != blueprint[:, :-1]).to(torch.float32)
        diff_count[:, 1:] += d_lr
        diff_count[:, :-1] += d_lr

    # Slight perimeter emphasis for crisp silhouette framing
    weights = weights + 0.85 * torch.clamp(diff_count, 0.0, 3.0)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return weights


@dataclass
class Faction:
    id: int
    name: str
    theme: str
    slogan: str
    art_mode: str  # "diffusion" | "procedural" | "banner" | "flag" | "micro_sprite" | "painting" | "clean_space" | "ribbon" | "fractal" | "black_hole" | "void_monster" | "void"
    art_prompt: str
    primary_color: int
    secondary_color: int
    accent_color: int
    x0: int
    y0: int
    w: int
    h: int
    blueprint: torch.Tensor           # [h, w] int16 (-1 = transparent)
    edge_weight: torch.Tensor         # [h, w] float32
    created_tick: int
    banner_text: str = ""
    color_theme: str = ""
    typography_spec: dict[str, Any] | None = None
    text_stamped: bool = False
    size_tier: str = "medium"         # "micro" | "medium" | "macro" | "ribbon"
    category: str = "art"             # "flag" | "game" | "anime" | "meme" | "painting" | "clean_space" | "ribbon" | "fractal" | "void" | "logo" | "art"
    subreddit: str = ""
    extra_key: str = ""               # flag key, sprite key, painting key, fractal kind, ribbon pattern
    ribbon_axis: str = ""             # "h" | "v" | "d1" | "d2"
    ribbon_thickness: int = 8
    void_variant: str = "tendrils"    # "tendrils" | "black_hole" | "fractal" | "monster"
    is_cameo: bool = False
    is_void: bool = False
    alive: bool = True

    # Rev 2.0 Timezone, Multilingual, AI-origin, Streamer Flash-Raid & Burnout fields
    timezone_region: str = "GLOBAL"   # "AMERICAS" | "EUROPE" | "ASIA_PACIFIC" | "LATAM" | "GLOBAL"
    language: str = "en"              # "en" | "zh" | "es" | "fr" | "de" | "ja" | "ko" | "pt" | "tr"
    ai_origin: bool = True            # True when invented/generated live by local AI
    is_streamer_raid: bool = False
    raid_end_tick: int = 0
    burnout_rate: float = 1.0

    # Strategy & military state
    role_weights: dict[str, float] = field(
        default_factory=lambda: {"builder": 0.62, "defender": 0.25, "expander": 0.13}
    )
    expand_dir: tuple[int, int] = (0, 0)
    raid_target_id: int | None = None
    last_action: str = "FOUND"
    last_morph_tick: int = 0
    evolution_stage: int = 1
    has_border_frame: bool = False

    # Diplomacy
    allies: set[int] = field(default_factory=set)
    ally_expiry: dict[int, int] = field(default_factory=dict)
    enemies: set[int] = field(default_factory=set)
    pending_ally_offers: set[int] = field(default_factory=set)

    # Live metrics
    morale: float = 0.75
    hype: float = 1.0
    member_count: int = 0
    integrity: float = 0.0
    owned_pixels: int = 0
    recent_churn: float = 0.0
    low_integrity_epochs: int = 0

    # Short narrative memory
    memory: list[str] = field(default_factory=list)

    # Cached CPU blueprint & silhouette halo coords for fast 1,200-faction updates without MPS stalls
    _bp_cpu_cache: np.ndarray | None = field(default=None, repr=False)
    _halo_xy_cache: torch.Tensor | None = field(default=None, repr=False)

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h

    @property
    def cx(self) -> float:
        return self.x0 + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y0 + self.h * 0.5

    def set_blueprint(
        self,
        bp_np: np.ndarray,
        x0: int | None = None,
        y0: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        dev = device or self.blueprint.device
        if x0 is not None:
            self.x0 = int(x0)
        if y0 is not None:
            self.y0 = int(y0)
        self.h, self.w = int(bp_np.shape[0]), int(bp_np.shape[1])
        self.blueprint = torch.from_numpy(bp_np.astype(np.int16)).to(dev)
        self._bp_cpu_cache = bp_np.astype(np.int16, copy=True)
        self._halo_xy_cache = None
        self.edge_weight = compute_edge_weights(self.blueprint)

    def remember(self, note: str) -> None:
        self.memory.append(note)
        if len(self.memory) > 8:
            self.memory = self.memory[-8:]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "theme": self.theme,
            "slogan": self.slogan,
            "art_mode": self.art_mode,
            "art_prompt": self.art_prompt,
            "size_tier": self.size_tier,
            "category": self.category,
            "subreddit": self.subreddit,
            "void_variant": self.void_variant if self.is_void else "",
            "primary_hex": COLOR_HEX[self.primary_color % NUM_COLORS],
            "secondary_hex": COLOR_HEX[self.secondary_color % NUM_COLORS],
            "bbox": [self.x0, self.y0, self.x1, self.y1],
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "is_void": self.is_void,
            "is_streamer_raid": self.is_streamer_raid,
            "timezone_region": self.timezone_region,
            "language": self.language,
            "ai_origin": self.ai_origin,
            "alive": self.alive,
            "members": self.member_count,
            "integrity": round(self.integrity, 3),
            "owned_pixels": self.owned_pixels,
            "morale": round(self.morale, 2),
            "stage": self.evolution_stage,
            "last_action": self.last_action,
            "allies": sorted(self.allies),
            "enemies": sorted(self.enemies),
            "role_weights": {k: round(v, 2) for k, v in self.role_weights.items()},
        }


@dataclass
class DiplomacyMessage:
    tick: int
    sender_id: int
    sender_name: str
    recipient_id: int | None  # None = public propaganda board
    kind: str  # "propaganda" | "proposal" | "accept" | "threat" | "taunt" | "report"
    text: str


class DiplomacyBoard:
    """Shared Reddit/Discord-style bulletin board + private neighbor inboxes."""

    def __init__(self) -> None:
        self.public_feed: list[DiplomacyMessage] = []
        self.inboxes: dict[int, list[DiplomacyMessage]] = {}

    def post_public(self, tick: int, sender: Faction, text: str, kind: str = "propaganda") -> None:
        msg = DiplomacyMessage(
            tick=tick,
            sender_id=sender.id,
            sender_name=sender.name,
            recipient_id=None,
            kind=kind,
            text=text.strip()[:220],
        )
        self.public_feed.append(msg)
        if len(self.public_feed) > 80:
            self.public_feed = self.public_feed[-80:]

    def send_direct(
        self, tick: int, sender: Faction, recipient_id: int, text: str, kind: str = "proposal"
    ) -> None:
        msg = DiplomacyMessage(
            tick=tick,
            sender_id=sender.id,
            sender_name=sender.name,
            recipient_id=recipient_id,
            kind=kind,
            text=text.strip()[:220],
        )
        box = self.inboxes.setdefault(recipient_id, [])
        box.append(msg)
        if len(box) > 20:
            self.inboxes[recipient_id] = box[-20:]

    def pop_inbox(self, faction_id: int, limit: int = 6) -> list[DiplomacyMessage]:
        box = self.inboxes.get(faction_id, [])
        if not box:
            return []
        recent = box[-limit:]
        self.inboxes[faction_id] = []
        return recent

    def recent_public(self, limit: int = 8) -> list[DiplomacyMessage]:
        return self.public_feed[-limit:]


def describe_neighbors(
    faction: Faction, all_factions: list[Faction], max_gap: int = 44
) -> list[dict[str, Any]]:
    """Find neighboring or overlapping factions with relative direction and overlap stats."""
    out: list[dict[str, Any]] = []
    for other in all_factions:
        if not other.alive or other.id == faction.id:
            continue
        dx_gap = max(0, max(faction.x0 - other.x1, other.x0 - faction.x1))
        dy_gap = max(0, max(faction.y0 - other.y1, other.y0 - faction.y1))
        dist = float((dx_gap * dx_gap + dy_gap * dy_gap) ** 0.5)
        if dist > max_gap:
            continue

        # Compute overlap pixels
        ox0 = max(faction.x0, other.x0)
        oy0 = max(faction.y0, other.y0)
        ox1 = min(faction.x1, other.x1)
        oy1 = min(faction.y1, other.y1)
        overlap = max(0, ox1 - ox0) * max(0, oy1 - oy0)

        # Cardinal direction from faction to other
        vx = other.cx - faction.cx
        vy = other.cy - faction.cy
        if abs(vx) >= abs(vy):
            direction = "EAST" if vx > 0 else "WEST"
        else:
            direction = "SOUTH" if vy > 0 else "NORTH"

        relation = (
            "ALLY"
            if other.id in faction.allies
            else "VOID_THREAT"
            if other.is_void
            else "ENEMY"
            if other.id in faction.enemies
            else "PENDING_ALLY_OFFER"
            if other.id in faction.pending_ally_offers
            else "NEUTRAL"
        )
        out.append(
            {
                "id": other.id,
                "name": other.name,
                "theme": other.theme,
                "direction": direction,
                "distance": round(dist, 1),
                "overlap_pixels": overlap,
                "members": other.member_count,
                "relation": relation,
            }
        )
    out.sort(key=lambda d: (-d["overlap_pixels"], d["distance"]))
    return out


def compute_shared_border_rect(a: Faction, b: Faction, pad: int = 5) -> tuple[int, int, int, int] | None:
    """Return a thin seam rectangle (x0, y0, x1, y1) if two allied factions touch or nearly touch."""
    ox0 = max(a.x0, b.x0)
    oy0 = max(a.y0, b.y0)
    ox1 = min(a.x1, b.x1)
    oy1 = min(a.y1, b.y1)

    # Direct overlap seam
    if ox1 > ox0 and oy1 > oy0:
        if (ox1 - ox0) <= (oy1 - oy0):
            mid_x = (ox0 + ox1) // 2
            return (max(0, mid_x - 3), oy0, mid_x + 4, oy1)
        mid_y = (oy0 + oy1) // 2
        return (ox0, max(0, mid_y - 3), ox1, mid_y + 4)

    # Near-touching vertical seam (East/West neighbors)
    if oy1 - oy0 >= 12:
        gap_x = max(a.x0 - b.x1, b.x0 - a.x1)
        if 0 <= gap_x <= pad * 2:
            mid_x = (a.x1 + b.x0) // 2 if a.x1 <= b.x0 else (b.x1 + a.x0) // 2
            return (max(0, mid_x - 3), oy0 + 2, mid_x + 4, oy1 - 2)

    # Near-touching horizontal seam (North/South neighbors)
    if ox1 - ox0 >= 12:
        gap_y = max(a.y0 - b.y1, b.y0 - a.y1)
        if 0 <= gap_y <= pad * 2:
            mid_y = (a.y1 + b.y0) // 2 if a.y1 <= b.y0 else (b.y1 + a.y0) // 2
            return (ox0 + 2, max(0, mid_y - 3), ox1 - 2, mid_y + 4)

    return None
