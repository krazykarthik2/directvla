#!/usr/bin/env python3
"""
SmolVLA Policy Training Pipeline on Wallclock-Sharded Datasets.
Trains the Multimodal Flow-Matching Policy with genuine OpenAI CLIP ViT-B/32:
  - Fast batch loading across wallclock shards via WallclockShardedDataset
  - Optimal Transport Flow Matching (OT-FM) conditional vector field loss
  - Cosine annealing learning rate schedule with AdamW optimizer
  - Automatic checkpoint saving with atomic rename and frozen CLIP weight stripping
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_model import SmolVLAPolicy, ACTION_MEAN, ACTION_STD
from vla_dataset import WallclockShardedDataset, WallclockDatasetManager


def safe_save_checkpoint(state_dict, filepath):
    """Saves checkpoint atomically to prevent corruption."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    for attempt in range(5):
        try:
            torch.save(state_dict, tmp_path)
            if os.path.exists(filepath):
                try:
                    os.replace(tmp_path, filepath)
                except OSError:
                    time.sleep(0.05)
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    os.rename(tmp_path, filepath)
            else:
                os.rename(tmp_path, filepath)
            return
        except Exception as e:
            if attempt == 4:
                print(f"[WARN] Failed saving checkpoint: {e}", flush=True)
            time.sleep(0.1)


def train_vla(epochs=100, batch_size=16, lr=1.5e-3, device='cpu', 
              shards_dir=os.path.join(_CUR_DIR, "data", "shards"),
              checkpoint_dir=os.path.join(_CUR_DIR, "checkpoints")):
    
    device = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    print("=" * 72, flush=True)
    print("   SmolVLA Policy Training on Wallclock-Sharded Demonstrations", flush=True)
    print("   - Genuine OpenAI CLIP ViT-B/32 (49 Visual Patches + 77 Language Tokens)", flush=True)
    print("   - Multimodal Action Expert: 128-Step Flow-Matching Action Decoder", flush=True)
    print(f"   - Target Compute Hardware: {device}", flush=True)
    print("=" * 72, flush=True)

    # 1. Dataset & DataLoader
    dataset = WallclockShardedDataset(shards_dir=shards_dir, use_clip_cache=True)
    if len(dataset) == 0:
        print("[ERROR] No demonstrations found in shards directory. Run generate_demos.py first!", flush=True)
        sys.exit(1)

    print(f">> Loaded {len(dataset)} demonstrations across {len(dataset.shards_info)} wallclock shards.", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=(len(dataset) > batch_size))

    # Inspect sample action dimension from dataset
    sample_batch = dataset[0]
    sample_act = sample_batch[3] # norm_traj [128, action_dim]
    sample_prop = sample_batch[2] # proprio0
    act_dim = sample_act.shape[-1]
    prop_dim = sample_prop.shape[-1]
    print(f">> Detected Trajectory Action Dim: {act_dim}D, Proprioception Dim: {prop_dim}D", flush=True)

    # 2. Instantiate Policy
    model = SmolVLAPolicy(horizon=128, action_dim=act_dim, proprio_dim=prop_dim, d_model=128, num_layers=2, device=device).to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f">> Trainable Action Decoder Parameters: {param_count:,}", flush=True)

    # 3. Optimizer & LR Scheduler
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss(reduction='none')

    best_loss = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, "dobot_vla_best.pth")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "dobot_vla_latest.pth")

    if act_dim == 9:
        # [j1, j2, j3, j4, x, y, z, r, gripper]
        dim_weights = torch.tensor([1.5, 1.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.0, 2.5], device=device).view(1, 1, 9)
    else:
        # [x, y, z, gripper]
        dim_weights = torch.tensor([1.2, 1.2, 1.6, 2.5], device=device).view(1, 1, 4)

    print(f"\n>> Training SmolVLA Policy across {len(dataset)} demos ({epochs} epochs)...", flush=True)
    pbar = tqdm(range(1, epochs + 1), desc="Training SmolVLA")

    try:
        for epoch in pbar:
            total_epoch_loss = 0.0
            num_samples = 0

            for batch in dataloader:
                vis_patches, txt_feats, proprio0, x_1 = batch
                vis_patches = vis_patches.to(device)
                txt_feats = txt_feats.to(device)
                proprio0 = proprio0.to(device)
                x_1 = x_1.to(device)

                B = vis_patches.size(0)
                optimizer.zero_grad(set_to_none=True)

                # Flow Matching Target Velocity Formulation:
                # x_0 ~ N(0, I) standard Gaussian noise
                x_0 = torch.randn_like(x_1)
                # Continuous diffusion time t ~ U(0, 1)
                t = torch.rand(B, device=device)
                t_expanded = t.view(B, 1, 1)

                # Linear Optimal Transport Flow Interpolation:
                x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
                u_t = x_1 - x_0  # True target flow velocity vector field

                # Predict velocity
                v_pred = model.forward_from_embeddings(x_t, t, vis_patches, txt_feats, proprio=proprio0)

                # Weighted MSE loss
                loss_raw = loss_fn(v_pred, u_t)
                loss = (loss_raw * dim_weights).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_epoch_loss += loss.item() * B
                num_samples += B

            scheduler.step()
            avg_loss = total_epoch_loss / max(1, num_samples)

            # Strip frozen CLIP weights before saving to keep checkpoints lightweight (<15MB)
            filtered_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.vlm.clip.")}

            if avg_loss < best_loss:
                best_loss = avg_loss
                safe_save_checkpoint(filtered_state, best_checkpoint_path)

            if epoch % 5 == 0 or epoch == epochs:
                safe_save_checkpoint(filtered_state, latest_checkpoint_path)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "OT_Loss": f"{avg_loss:.5f}",
                "Best": f"{best_loss:.5f}",
                "LR": f"{current_lr:.6f}"
            })

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving latest checkpoint...", flush=True)
        filtered_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.vlm.clip.")}
        safe_save_checkpoint(filtered_state, latest_checkpoint_path)
        return

    filtered_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.vlm.clip.")}
    safe_save_checkpoint(filtered_state, latest_checkpoint_path)
    print(f"\n[SUCCESS] SmolVLA Training Complete! Best Loss: {best_loss:.5f}")
    print(f"          Best Model -> {best_checkpoint_path}")
    print(f"          Latest Model -> {latest_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy on Wallclock Shards")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shards_dir", type=str, default=os.path.join(_CUR_DIR, "data", "shards"))
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(_CUR_DIR, "checkpoints"))
    args = parser.parse_args()

    train_vla(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        shards_dir=args.shards_dir,
        checkpoint_dir=args.checkpoint_dir
    )
