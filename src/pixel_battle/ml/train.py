"""PyTorch MPS bf16 trainer for RedditCitizenNet v2 (25x25 + prev_color memory)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pixel_battle.ml.dataset import build_reddit_training_dataset
from pixel_battle.ml.model import (
    ARCH_DEFENDER,
    ARCH_FREE,
    ARCH_VOID,
    NO_GOAL_IDX,
    PATCH_SIZE,
    RedditCitizenNet,
)
from pixel_battle.palette import NUM_COLORS

log = logging.getLogger(__name__)


def _augment_batch_gpu(
    cur: torch.Tensor,      # [B, P, P] int64
    prev: torch.Tensor,     # [B, P, P] int64
    rec: torch.Tensor,      # [B, P, P] float32
    goal: torch.Tensor,     # [B, P, P] int64
    arch: torch.Tensor,     # [B] int64
    pos: torch.Tensor,      # [B] int64
    col: torch.Tensor,      # [B] int64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU batch augmentations: dihedral symmetry + chroma color permutation + archetype dropout."""
    dev = cur.device
    B, P, _ = cur.shape
    py = pos // P
    px = pos % P

    # 1. Random horizontal flip
    if torch.rand(1).item() < 0.5:
        cur = torch.flip(cur, dims=[2])
        prev = torch.flip(prev, dims=[2])
        rec = torch.flip(rec, dims=[2])
        goal = torch.flip(goal, dims=[2])
        px = (P - 1) - px

    # 2. Random vertical flip
    if torch.rand(1).item() < 0.5:
        cur = torch.flip(cur, dims=[1])
        prev = torch.flip(prev, dims=[1])
        rec = torch.flip(rec, dims=[1])
        goal = torch.flip(goal, dims=[1])
        py = (P - 1) - py

    # 3. Random 90-deg transpose
    if torch.rand(1).item() < 0.5:
        cur = cur.transpose(1, 2)
        prev = prev.transpose(1, 2)
        rec = rec.transpose(1, 2)
        goal = goal.transpose(1, 2)
        py, px = px, py

    pos = py * P + px

    # 4. Random chroma-color permutation (45% of batches): permutes colors 2..31, keeps WHITE(0), BLACK(1), NO_GOAL(32)
    if torch.rand(1).item() < 0.45:
        perm = torch.arange(NUM_COLORS + 1, dtype=torch.long, device=dev)
        perm[2:NUM_COLORS] = torch.randperm(NUM_COLORS - 2, device=dev) + 2
        cur = perm[cur]
        prev = perm[prev]
        goal = perm[goal]
        col = perm[col]

    # 5. Archetype dropout (20% -> ARCH_FREE) only for non-Void samples
    drop = (torch.rand(B, device=dev) < 0.20) & (arch != ARCH_VOID)
    arch = torch.where(drop, torch.full_like(arch, ARCH_FREE), arch)

    return cur, prev, rec, goal, arch, pos, col


def train_reddit_citizen_model(
    data_dir: Path = Path("data/reddit_place"),
    out_model: Path = Path("models/reddit_citizen.pt"),
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 2.5e-3,
    hidden_dim: int = 80,
    patch_size: int = PATCH_SIZE,
    early_samples: int = 320_000,
    peak_samples: int = 160_000,
    device_spec: str = "auto",
    seed: int = 42,
) -> Path:
    """Build Reddit r/place 2022 v2 dataset (if needed) and train RedditCitizenNet v2 on MPS."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device_spec == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_spec)
    use_bf16 = device.type == "mps"

    dataset_path = build_reddit_training_dataset(
        data_dir=data_dir,
        out_npz=data_dir / f"train_dataset_v2_p{patch_size}.npz",
        early_samples=early_samples,
        peak_samples=peak_samples,
        patch_size=patch_size,
        seed=seed,
    )

    log.info("Loading %s into memory...", dataset_path)
    npz = np.load(dataset_path)
    cur_all = npz["cur_patches"]
    prev_all = npz["prev_patches"]
    rec_all = npz["recency"]
    goal_all = npz["goal_patches"]
    arch_all = npz["archetypes"]
    pos_all = npz["target_pos"]
    col_all = npz["target_col"]

    P = int(cur_all.shape[1])
    N = len(col_all)
    n_val = min(25_000, max(3_000, int(N * 0.04)))
    n_train = N - n_val
    log.info("Dataset ready (%dx%d): %d train, %d val on %s (bf16=%s).", P, P, n_train, n_val, device, use_bf16)

    model = RedditCitizenNet(hidden_dim=hidden_dim, patch_size=P, use_prev_color=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("RedditCitizenNet v2 parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, n_train // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-4)

    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(n_train)
        tr_loss_p = 0.0
        tr_loss_c = 0.0
        n_batches = 0

        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            cur = torch.from_numpy(cur_all[idx]).to(device=device, dtype=torch.long)
            prev = torch.from_numpy(prev_all[idx]).to(device=device, dtype=torch.long)
            rec = torch.from_numpy(rec_all[idx]).to(device=device, dtype=torch.float32) / 255.0
            goal = torch.from_numpy(goal_all[idx]).to(device=device, dtype=torch.long)
            arch = torch.from_numpy(arch_all[idx]).to(device=device, dtype=torch.long)
            pos = torch.from_numpy(pos_all[idx].astype(np.int64)).to(device=device)
            col = torch.from_numpy(col_all[idx].astype(np.int64)).to(device=device)

            cur, prev, rec, goal, arch, pos, col = _augment_batch_gpu(cur, prev, rec, goal, arch, pos, col)

            optimizer.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast("mps", dtype=torch.bfloat16):
                    logits_p, logits_c = model(cur, rec, goal, arch, pos, prev_patch=prev)
                    loss_p = F.cross_entropy(logits_p.float(), pos)
                    loss_c = F.cross_entropy(logits_c.float(), col)
                    loss = loss_p + loss_c
            else:
                logits_p, logits_c = model(cur, rec, goal, arch, pos, prev_patch=prev)
                loss_p = F.cross_entropy(logits_p, pos)
                loss_c = F.cross_entropy(logits_c, col)
                loss = loss_p + loss_c

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            tr_loss_p += float(loss_p.item())
            tr_loss_c += float(loss_c.item())
            n_batches += 1

        # Validation pass
        model.eval()
        val_loss_p = 0.0
        val_loss_c = 0.0
        val_pos_top5 = 0
        val_col_free_ok = 0
        val_col_free_n = 0
        val_col_def_free_ok = 0
        val_col_def_free_n = 0
        val_col_goal_ok = 0
        val_col_goal_n = 0
        val_n = 0

        with torch.inference_mode():
            for start in range(n_train, N, batch_size):
                end = min(N, start + batch_size)
                cur = torch.from_numpy(cur_all[start:end]).to(device=device, dtype=torch.long)
                prev = torch.from_numpy(prev_all[start:end]).to(device=device, dtype=torch.long)
                rec = torch.from_numpy(rec_all[start:end]).to(device=device, dtype=torch.float32) / 255.0
                goal = torch.from_numpy(goal_all[start:end]).to(device=device, dtype=torch.long)
                arch = torch.from_numpy(arch_all[start:end]).to(device=device, dtype=torch.long)
                pos = torch.from_numpy(pos_all[start:end].astype(np.int64)).to(device=device)
                col = torch.from_numpy(col_all[start:end].astype(np.int64)).to(device=device)

                if use_bf16:
                    with torch.autocast("mps", dtype=torch.bfloat16):
                        logits_p, logits_c = model(cur, rec, goal, arch, pos, prev_patch=prev)
                else:
                    logits_p, logits_c = model(cur, rec, goal, arch, pos, prev_patch=prev)
                logits_p = logits_p.float()
                logits_c = logits_c.float()

                b = end - start
                val_loss_p += float(F.cross_entropy(logits_p, pos, reduction="sum").item())
                val_loss_c += float(F.cross_entropy(logits_c, col, reduction="sum").item())
                top5 = logits_p.topk(5, dim=-1).indices
                val_pos_top5 += int((top5 == pos.unsqueeze(1)).any(dim=1).sum().item())

                pred_c = logits_c.argmax(dim=-1)
                has_g = (goal.flatten(1).gather(1, pos.unsqueeze(1)).squeeze(1)) < NO_GOAL_IDX
                ok_c = pred_c == col
                val_col_goal_ok += int((ok_c & has_g).sum().item())
                val_col_goal_n += int(has_g.sum().item())
                val_col_free_ok += int((ok_c & (~has_g)).sum().item())
                val_col_free_n += int((~has_g).sum().item())
                is_def_free = (~has_g) & (arch == ARCH_DEFENDER)
                val_col_def_free_ok += int((ok_c & is_def_free).sum().item())
                val_col_def_free_n += int(is_def_free.sum().item())
                val_n += b

        val_lp = val_loss_p / val_n
        val_lc = val_loss_c / val_n
        val_tot = val_lp + val_lc
        dt = time.perf_counter() - t0
        log.info(
            "Epoch %2d/%d (%.1fs) | train=%.3f (pos=%.3f, col=%.3f) | val=%.3f | "
            "pos@5=%.1f%% | free_col@1=%.1f%% | def_free@1=%.1f%% | goal_col@1=%.1f%%",
            ep,
            epochs,
            dt,
            (tr_loss_p + tr_loss_c) / max(1, n_batches),
            tr_loss_p / max(1, n_batches),
            tr_loss_c / max(1, n_batches),
            val_tot,
            100.0 * val_pos_top5 / max(1, val_n),
            100.0 * val_col_free_ok / max(1, val_col_free_n),
            100.0 * val_col_def_free_ok / max(1, val_col_def_free_n),
            100.0 * val_col_goal_ok / max(1, val_col_goal_n),
        )

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            torch.save(
                {
                    "version": 2,
                    "model_state": model.state_dict(),
                    "config": {
                        "hidden_dim": model.hidden_dim,
                        "patch_size": P,
                        "use_prev_color": True,
                        "num_colors": NUM_COLORS,
                    },
                    "metrics": {
                        "val_loss": round(val_tot, 4),
                        "pos_top5": round(val_pos_top5 / max(1, val_n), 4),
                        "free_col_top1": round(val_col_free_ok / max(1, val_col_free_n), 4),
                        "def_free_col_top1": round(val_col_def_free_ok / max(1, val_col_def_free_n), 4),
                        "goal_col_top1": round(val_col_goal_ok / max(1, val_col_goal_n), 4),
                    },
                },
                out_model,
            )

    log.info("Saved best RedditCitizenNet v2 weights -> %s (best val_loss=%.4f)", out_model, best_val_loss)
    return out_model
