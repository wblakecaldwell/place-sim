"""Base protocol for pixel-painting policies.

Keeping the motor cortex behind this interface makes it a drop-in swap to
replace StochasticPolicy with a PyTorch behavioral-cloning policy trained on
real Reddit r/place logs later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from pixel_battle.canvas import CanvasState
from pixel_battle.factions import Faction


@dataclass
class PlacementBatch:
    xs: torch.Tensor        # [M] int32
    ys: torch.Tensor        # [M] int32
    colors: torch.Tensor    # [M] uint8
    factions: torch.Tensor  # [M] int16 (-1 for free agent)
    user_ids: torch.Tensor  # [M] int32


class PixelPolicy(Protocol):
    user_factions: torch.Tensor

    def step(
        self,
        tick: int,
        canvas: CanvasState,
        factions: list[Faction],
        shared_borders: list[tuple[int, int, tuple[int, int, int, int]]],
    ) -> PlacementBatch:
        """Compute one tick of pixel placements across all ready users."""
        ...

    def recruit_initial_members(
        self,
        faction: Faction,
        target_count: int,
    ) -> int:
        """Assign nearby/aligned free agents to a newly discovered faction."""
        ...

    def social_step(
        self,
        factions: list[Faction],
        canvas: CanvasState,
    ) -> None:
        """Update faction membership via recruitment, hype, and defection."""
        ...

    def release_faction_members(self, faction_id: int) -> None:
        """Release all members of a dissolved faction back into free agents."""
        ...
