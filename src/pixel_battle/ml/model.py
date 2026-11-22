"""RedditCitizenNet v2: wider 25x25 PyTorch MPS neural policy with temporal prev_color memory.

Predicts factorized joint action P(dy, dx | context) * P(color | context, dy, dx)
over a 25x25 local neighborhood (backward-compatible with v1 17x17 checkpoints), supporting:
  - Free-agent mode (goal = NO_GOAL_IDX): authentic Reddit chaos, autonomous counter-griefing
    via prev_color memory, Blue Corner colonies, stripes, lattices, and Void tendrils
  - Goal-conditioned mode (goal = blueprint crop): authentic human blueprint building & defense
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel_battle.palette import NUM_COLORS

PATCH_SIZE = 21
PATCH_RADIUS = PATCH_SIZE // 2  # 10
NO_GOAL_IDX = NUM_COLORS        # 32 -> transparent / no blueprint

# Ground-truth behavioral archetypes extracted from Reddit r/place logs
ARCH_BUILDER = 0   # Extends matching color clusters, stripes, fills
ARCH_DEFENDER = 1  # Rapidly reverts recent edits / defends pixels
ARCH_VOID = 2      # Spreads black / dark-red tendrils into colorful art
ARCH_ARTIST = 3    # Adds contrasting pixel details/outlines inside dense art
ARCH_FREE = 4      # Unconditioned general Redditor (also used for archetype dropout)
NUM_ARCHETYPES = 5


class _ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(x + h)


class RedditCitizenNet(nn.Module):
    """~760K-param (v2) dilated residual CNN for real-time batched r/place action prediction on MPS."""

    def __init__(
        self,
        hidden_dim: int = 96,
        patch_size: int = PATCH_SIZE,
        use_prev_color: bool = True,
        color_emb_dim: int = 16,
        goal_emb_dim: int = 12,
        arch_emb_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_radius = patch_size // 2
        self.use_prev_color = use_prev_color

        self.color_emb = nn.Embedding(NUM_COLORS, color_emb_dim)
        self.goal_emb = nn.Embedding(NUM_COLORS + 1, goal_emb_dim)
        self.arch_emb = nn.Embedding(NUM_ARCHETYPES, arch_emb_dim)

        # Input channels:
        # v2: cur(16) + prev(16) + goal(12) + recency(1) + just_changed(1) + mismatch(1) + has_goal(1) + arch(8) = 56
        # v1: cur(16) + goal(12) + recency(1) + mismatch(1) + has_goal(1) + arch(8) = 39
        extra_prev = (color_emb_dim + 1) if use_prev_color else 0
        in_ch = color_emb_dim + extra_prev + goal_emb_dim + 3 + arch_emb_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        # Receptive field: 3 + 4 + 8 + 12 + 4 = 31x31 (covers full 21x21 & 25x25 cleanly)
        d3 = 3 if patch_size >= 21 else 2
        self.blocks = nn.Sequential(
            _ResBlock(hidden_dim, dilation=1),
            _ResBlock(hidden_dim, dilation=2),
            _ResBlock(hidden_dim, dilation=d3),
            _ResBlock(hidden_dim, dilation=1),
        )

        # Head 1: spatial click logits over patch_size * patch_size (625 for 25x25)
        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

        # Head 2: color logits conditioned on spatial backbone features + direct local skip features
        col_in = hidden_dim + in_ch
        self.color_head = nn.Sequential(
            nn.Linear(col_in, 160 if use_prev_color else 128),
            nn.GELU(),
            nn.Linear(160 if use_prev_color else 128, NUM_COLORS),
        )

    def encode(
        self,
        cur_patch: torch.Tensor,            # [B, P, P] int64 (0..31)
        recency: torch.Tensor,              # [B, P, P] float32 (0..1)
        goal_patch: torch.Tensor,           # [B, P, P] int64 (0..32, 32=NO_GOAL)
        archetype: torch.Tensor,            # [B] int64 (0..4)
        prev_patch: torch.Tensor | None = None,  # [B, P, P] int64 (0..31)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run shared visual backbone and return (feat_map, raw_skip_map, spatial_logits)."""
        B, H, W = cur_patch.shape
        c_emb = self.color_emb(cur_patch).permute(0, 3, 1, 2)
        g_emb = self.goal_emb(goal_patch).permute(0, 3, 1, 2)
        has_goal = (goal_patch < NO_GOAL_IDX).to(torch.float32)
        mismatch = ((goal_patch < NO_GOAL_IDX) & (cur_patch != goal_patch)).to(torch.float32)
        a_emb = self.arch_emb(archetype)[:, :, None, None].expand(B, -1, H, W)

        parts = [c_emb]
        if self.use_prev_color:
            if prev_patch is None:
                prev_patch = cur_patch
            p_emb = self.color_emb(prev_patch).permute(0, 3, 1, 2)
            just_changed = ((cur_patch != prev_patch) & (recency > 0.05)).to(torch.float32)
            parts.extend([p_emb, just_changed.unsqueeze(1)])

        parts.extend(
            [
                g_emb,
                recency.unsqueeze(1),
                mismatch.unsqueeze(1),
                has_goal.unsqueeze(1),
                a_emb,
            ]
        )
        raw = torch.cat(parts, dim=1)
        feat = self.blocks(self.stem(raw))
        spatial_logits = self.spatial_head(feat).flatten(1)
        return feat, raw, spatial_logits

    def predict_color_at(
        self,
        feat: torch.Tensor,
        raw: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict [B, 32] color logits conditioned on chosen spatial index pos_idx."""
        B = feat.shape[0]
        feat_flat = feat.flatten(2)
        raw_flat = raw.flatten(2)
        idx_f = pos_idx[:, None, None].expand(B, feat_flat.shape[1], 1)
        idx_r = pos_idx[:, None, None].expand(B, raw_flat.shape[1], 1)
        f_at = feat_flat.gather(2, idx_f).squeeze(2)
        r_at = raw_flat.gather(2, idx_r).squeeze(2)
        return self.color_head(torch.cat([f_at, r_at], dim=1))

    def forward(
        self,
        cur_patch: torch.Tensor,
        recency: torch.Tensor,
        goal_patch: torch.Tensor,
        archetype: torch.Tensor,
        target_pos: torch.Tensor,
        prev_patch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass returning (spatial_logits [B, P*P], color_logits [B, 32])."""
        feat, raw, spatial_logits = self.encode(cur_patch, recency, goal_patch, archetype, prev_patch=prev_patch)
        color_logits = self.predict_color_at(feat, raw, target_pos)
        return spatial_logits, color_logits

    @torch.inference_mode()
    def sample_actions(
        self,
        cur_patch: torch.Tensor,
        recency: torch.Tensor,
        goal_patch: torch.Tensor,
        archetype: torch.Tensor,
        prev_patch: torch.Tensor | None = None,
        temp_pos: float = 1.0,
        temp_col: float = 0.9,
        valid_pos_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (dy, dx, color) from the learned joint distribution in one batched GPU pass."""
        P = self.patch_size
        R = self.patch_radius
        use_bf16 = cur_patch.device.type == "mps"
        if use_bf16:
            with torch.autocast("mps", dtype=torch.bfloat16):
                feat, raw, spatial_logits = self.encode(
                    cur_patch, recency, goal_patch, archetype, prev_patch=prev_patch
                )
        else:
            feat, raw, spatial_logits = self.encode(
                cur_patch, recency, goal_patch, archetype, prev_patch=prev_patch
            )

        logits_p = spatial_logits.float() / max(0.05, temp_pos)
        if valid_pos_mask is not None:
            logits_p = logits_p.masked_fill(~valid_pos_mask, -1e9)
        probs_p = F.softmax(logits_p, dim=-1)
        pos_idx = torch.multinomial(probs_p, num_samples=1).squeeze(-1)

        if use_bf16:
            with torch.autocast("mps", dtype=torch.bfloat16):
                color_logits = self.predict_color_at(feat, raw, pos_idx)
        else:
            color_logits = self.predict_color_at(feat, raw, pos_idx)

        probs_c = F.softmax(color_logits.float() / max(0.05, temp_col), dim=-1)
        color = torch.multinomial(probs_c, num_samples=1).squeeze(-1).to(torch.uint8)

        dy = (pos_idx // P).to(torch.int32) - R
        dx = (pos_idx % P).to(torch.int32) - R
        return dy, dx, color
