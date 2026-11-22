"""GPU-backed 512x512 canvas state and fast visual rendering helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw

from pixel_battle.palette import NUM_COLORS, PALETTE_RGB, WHITE_IDX, get_tier_colors

if TYPE_CHECKING:
    from pixel_battle.factions import Faction


class CanvasState:
    """Maintains the live pixel grid, active viewport, unlocked palette, ownership, recency, and activity on MPS/CPU."""

    def __init__(
        self,
        width: int,
        height: int,
        device: torch.device,
        active_w: int | None = None,
        active_h: int | None = None,
        palette_tier: int = 3,
    ) -> None:
        self.width = width
        self.height = height
        self.device = device

        self.colors = torch.full((height, width), WHITE_IDX, dtype=torch.uint8, device=device)
        self.prev_colors = torch.full((height, width), WHITE_IDX, dtype=torch.uint8, device=device)
        self.owner = torch.full((height, width), -1, dtype=torch.int16, device=device)
        self.last_edit_tick = torch.full((height, width), -10_000, dtype=torch.int32, device=device)
        self.activity = torch.zeros((height, width), dtype=torch.float32, device=device)

        self.active_w: int = min(width, max(64, active_w if active_w is not None else width))
        self.active_h: int = min(height, max(64, active_h if active_h is not None else height))
        self.palette_tier: int = 3
        self.allowed_colors: tuple[int, ...] = get_tier_colors(3)
        self.allowed_color_mask: torch.Tensor = torch.ones(NUM_COLORS, dtype=torch.bool, device=device)
        self.expand_viewport(self.active_w, self.active_h, palette_tier)

    def expand_viewport(self, new_w: int, new_h: int, new_palette_tier: int) -> None:
        """Update the unlocked active viewport dimensions and unlocked palette tier."""
        self.active_w = max(64, min(self.width, int(new_w)))
        self.active_h = max(64, min(self.height, int(new_h)))
        self.palette_tier = max(1, min(4, int(new_palette_tier)))
        self.allowed_colors = get_tier_colors(self.palette_tier)
        mask = torch.zeros(NUM_COLORS, dtype=torch.bool, device=self.device)
        idx = torch.tensor(self.allowed_colors, dtype=torch.long, device=self.device)
        mask[idx] = True
        self.allowed_color_mask = mask

    @property
    def is_whiteout(self) -> bool:
        return self.palette_tier >= 4

    def decay_activity(self, factor: float = 0.94) -> None:
        self.activity.mul_(factor)

    def apply_placements(
        self,
        tick: int,
        xs: torch.Tensor,
        ys: torch.Tensor,
        new_colors: torch.Tensor,
        factions: torch.Tensor,
    ) -> None:
        """Apply a batch of pixel placements in-place on GPU, bounded to the active viewport & palette."""
        if xs.numel() == 0:
            return
        xs_c = xs.clamp(0, self.active_w - 1)
        ys_c = ys.clamp(0, self.active_h - 1)
        if self.palette_tier >= 4:
            new_u8 = torch.full_like(new_colors, WHITE_IDX, dtype=torch.uint8)
        else:
            new_u8 = new_colors.to(torch.uint8)
        prev = self.colors[ys_c, xs_c]
        changed_bool = prev != new_u8
        changed = changed_bool.to(torch.float32)

        # Store prior color only when a pixel changes to preserve pre-griefing state
        if changed_bool.any():
            self.prev_colors[ys_c[changed_bool], xs_c[changed_bool]] = prev[changed_bool]

        self.colors[ys_c, xs_c] = new_u8
        self.owner[ys_c, xs_c] = factions.to(torch.int16)
        self.last_edit_tick[ys_c, xs_c] = int(tick)
        # Boost activity where pixels actually flipped (and small bump for reinforcement)
        self.activity[ys_c, xs_c] = torch.clamp(
            self.activity[ys_c, xs_c] + 0.25 + 1.0 * changed, max=12.0
        )

    def colors_numpy(self) -> np.ndarray:
        return self.colors.detach().cpu().numpy()

    def prev_colors_numpy(self) -> np.ndarray:
        return self.prev_colors.detach().cpu().numpy()

    def owner_numpy(self) -> np.ndarray:
        return self.owner.detach().cpu().numpy()

    def activity_numpy(self) -> np.ndarray:
        return self.activity.detach().cpu().numpy()

    def to_rgb_numpy(self, active_only: bool = False) -> np.ndarray:
        idx = self.colors_numpy()
        if active_only:
            idx = idx[: self.active_h, : self.active_w]
        return PALETTE_RGB[idx]

    def to_pil(self, scale: int = 1, active_only: bool = False) -> Image.Image:
        rgb = self.to_rgb_numpy(active_only=active_only)
        img = Image.fromarray(rgb, mode="RGB")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        return img

    def crop_rgb_numpy(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0 = max(0, min(self.active_w - 1, x0))
        y0 = max(0, min(self.active_h - 1, y0))
        x1 = max(x0 + 1, min(self.active_w, x1))
        y1 = max(y0 + 1, min(self.active_h, y1))
        idx = self.colors[y0:y1, x0:x1].detach().cpu().numpy()
        return PALETTE_RGB[idx]

    def crop_pil(self, x0: int, y0: int, x1: int, y1: int, scale: int = 4) -> Image.Image:
        rgb = self.crop_rgb_numpy(x0, y0, x1, y1)
        img = Image.fromarray(rgb, mode="RGB")
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        return img

    def render_faction_view_pil(
        self,
        faction: Faction,
        all_factions: list[Faction],
        margin: int = 36,
        target_size: int = 384,
    ) -> Image.Image:
        """Render a crisp upscaled crop around a faction with subtle neighbor bounding boxes."""
        vx0 = max(0, faction.x0 - margin)
        vy0 = max(0, faction.y0 - margin)
        vx1 = min(self.active_w, faction.x1 + margin)
        vy1 = min(self.active_h, faction.y1 + margin)

        rgb = self.crop_rgb_numpy(vx0, vy0, vx1, vy1)
        vw, vh = vx1 - vx0, vy1 - vy0
        scale = max(2, min(6, target_size // max(vw, vh, 1)))
        img = Image.fromarray(rgb, mode="RGB").resize(
            (vw * scale, vh * scale), Image.Resampling.NEAREST
        )
        draw = ImageDraw.Draw(img)

        # Draw own territory outline in bright cyan
        ox0 = (faction.x0 - vx0) * scale
        oy0 = (faction.y0 - vy0) * scale
        ox1 = (faction.x1 - vx0) * scale - 1
        oy1 = (faction.y1 - vy0) * scale - 1
        draw.rectangle([ox0, oy0, ox1, oy1], outline=(0, 220, 255), width=2)
        draw.rectangle([ox0, max(0, oy0 - 14), min(img.width, ox0 + 140), oy0], fill=(0, 40, 60))
        draw.text((ox0 + 3, max(0, oy0 - 13)), f"YOU: {faction.name[:18]}", fill=(180, 250, 255))

        # Draw visible neighbors
        for other in all_factions:
            if not other.alive or other.id == faction.id:
                continue
            if other.x1 <= vx0 or other.x0 >= vx1 or other.y1 <= vy0 or other.y0 >= vy1:
                continue
            nx0 = max(0, (other.x0 - vx0) * scale)
            ny0 = max(0, (other.y0 - vy0) * scale)
            nx1 = min(img.width - 1, (other.x1 - vx0) * scale - 1)
            ny1 = min(img.height - 1, (other.y1 - vy0) * scale - 1)
            if other.id in faction.allies:
                col = (50, 235, 110)
                tag = f"ALLY: {other.name[:16]}"
            elif other.is_void:
                col = (255, 40, 90)
                tag = "VOID"
            else:
                col = (255, 180, 40)
                tag = other.name[:18]
            draw.rectangle([nx0, ny0, nx1, ny1], outline=col, width=2)
            draw.rectangle([nx0, ny0, min(img.width, nx0 + 130), min(img.height, ny0 + 14)], fill=(20, 20, 25))
            draw.text((nx0 + 3, ny0 + 1), tag, fill=col)

        return img

    def render_heatmap_rgba(self) -> np.ndarray:
        """Render an [H, W, 4] RGBA heatmap overlay for the web dashboard."""
        act = self.activity_numpy()
        norm = np.clip(act / 4.5, 0.0, 1.0)
        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # Magma-like ramp: transparent -> orange -> bright yellow-white
        rgba[..., 0] = np.clip(norm * 2.2 * 255, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip((norm - 0.25) * 1.8 * 255, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip((norm - 0.65) * 2.8 * 255, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.where(norm > 0.06, np.clip(norm * 215, 0, 215), 0).astype(np.uint8)
        return rgba
