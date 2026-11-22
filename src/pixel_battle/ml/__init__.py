"""Reddit r/place 2022 dataset preparation, neural policy architecture, and MPS training."""

from pixel_battle.ml.model import (
    ARCH_ARTIST,
    ARCH_BUILDER,
    ARCH_DEFENDER,
    ARCH_FREE,
    ARCH_VOID,
    NO_GOAL_IDX,
    PATCH_RADIUS,
    PATCH_SIZE,
    RedditCitizenNet,
)

__all__ = [
    "ARCH_ARTIST",
    "ARCH_BUILDER",
    "ARCH_DEFENDER",
    "ARCH_FREE",
    "ARCH_VOID",
    "NO_GOAL_IDX",
    "PATCH_RADIUS",
    "PATCH_SIZE",
    "RedditCitizenNet",
]
