#!/usr/bin/env python3
"""
Zero-Disk-Footprint Streaming VLA Pipeline for Dobot Magician.

Complete in-memory streaming:
  1. Generates 1 episode with physical kinematics and 20-distractor clutter.
  2. Every 10 frames (steps 0, 10, 20, ..., 120):
     - Renders the exact physical scene frame.
     - Encodes it directly with OpenAI CLIP ViT-B/32 (49 spatial patches).
     - Predicts the remaining action chunk from that exact frame and intermediate proprioception.
  3. Trains the SmolVLA Flow-Matching policy online in RAM without EVER writing 
     raw images, .npz, or intermediate shard files to disk.
  4. Only atomic model checkpoints (dobot_vla_best.pth) are saved to disk.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from tqdm import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_model import SmolVLAPolicy, ACTION_MEAN_9D, ACTION_STD_9D
from vla_camera import SyntheticDeskRenderer, preprocess_image_for_clip
from vla_robot_env import DobotKinematics
from train_vla import safe_save_checkpoint


def quintic(p_start, p_end, steps):
    """Minimum-jerk quintic polynomial: s(t) = 10t^3 - 15t^4 + 6t^5."""
    if steps <= 1:
        return [np.array(p_end, dtype=np.float32)]
    p_s = np.array(p_start, dtype=np.float32)
    p_e = np.array(p_end, dtype=np.float32)
    pts = []
    for i in range(steps):
        s = i / float(steps - 1)
        w = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        pts.append(p_s + w * (p_e - p_s))
    return pts


class StreamingTrajectoryGenerator:
    """Generates continuous physical demonstrations in-memory with equal 7-color ratios and 20 distractors."""
    def __init__(self):
        self.kin = DobotKinematics()
        self.renderer = SyntheticDeskRenderer()
        self.all_colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange"]
        self.nouns = ["cube", "thing", "block", "item", "object", "piece", "one"]
        self.dest_nouns = ["platform", "area", "zone", "target", "spot", "box", "plate", ""]

    def generate_single_episode(self, desk_tone="dark_mat"):
        for _ in range(200): # Retry until kinematically valid
            t_col = str(np.random.choice(self.all_colors))
            avail_p = [c for c in self.all_colors if c != t_col]
            p_col = str(np.random.choice(avail_p))
            occupied = []

            # Dense clutter: up to 20 distractors
            num_dist_cubes = int(np.random.choice([0, 1, 2, 4, 6, 8, 12, 16, 20],
                                                  p=[0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]))
            min_d = 0.060 if num_dist_cubes <= 4 else (0.042 if num_dist_cubes <= 10 else 0.028)

            def rand_pos(m_d=0.060):
                for _ in range(120):
                    a = np.random.uniform(-0.55, 0.55)
                    d = np.random.uniform(0.18, 0.28)
                    p = np.array([d * np.cos(a), d * np.sin(a), 0.012], dtype=np.float32)
                    if all(np.linalg.norm(p[:2] - o[:2]) > m_d for o in occupied):
                        occupied.append(p)
                        return p
                p = np.array([0.22 + np.random.uniform(-0.03, 0.03), 0.08 + np.random.uniform(-0.03, 0.03), 0.012], dtype=np.float32)
                occupied.append(p)
                return p

            cube_pos = rand_pos(min_d)
            plat_pos = rand_pos(0.065)
            plat_pos[2] = 0.005

            dist_cubes = []
            for _ in range(num_dist_cubes):
                c_col = str(np.random.choice(self.all_colors))
                dist_cubes.append((c_col, rand_pos(min_d)))

            num_dist_plats = int(np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20]))
            dist_plats = []
            for _ in range(num_dist_plats):
                dp_c = str(np.random.choice([c for c in self.all_colors if c != p_col]))
                dp = rand_pos(0.065)
                dp[2] = 0.005
                dist_plats.append((dp_c, dp))

            task_type = np.random.choice(["pick_place", "push", "reach", "lift"], p=[0.60, 0.20, 0.10, 0.10])
            hz = 0.045; csz = -0.012; psz = -0.005
            ps = np.array([0.22, 0.0, hz], dtype=np.float32)
            hc = np.array([cube_pos[0], cube_pos[1], hz], dtype=np.float32)
            cs = np.array([cube_pos[0], cube_pos[1], csz], dtype=np.float32)
            hp = np.array([plat_pos[0], plat_pos[1], hz], dtype=np.float32)
            pps = np.array([plat_pos[0], plat_pos[1], psz], dtype=np.float32)
            ret = np.array([plat_pos[0], plat_pos[1], hz], dtype=np.float32)

            R = np.random.randint
            n = str(np.random.choice(self.nouns))
            dn = str(np.random.choice(self.dest_nouns))
            dest_str = f"{p_col} {dn}".strip() if dn else p_col

            p_push_start = None
            p_push_end = None
            push_dir = None

            if task_type == "pick_place":
                patterns = [
                    f"pick up the {t_col} {n} and put it on {dest_str}",
                    f"pick up the {t_col} {n} and place it on the {dest_str}",
                    f"pick the {t_col} {n} and move to {dest_str}",
                    f"grab the {t_col} {n} and drop it on {dest_str}",
                    f"take the {t_col} {n} and put on {dest_str}",
                    f"lift the {t_col} {n} and set it down on {dest_str}",
                    f"transfer {t_col} {n} onto {dest_str}",
                    f"move the {t_col} {n} to {dest_str}",
                    f"pick {t_col} {n} and place on {dest_str}",
                    f"put the {t_col} {n} on {dest_str}",
                ]
                prompt = str(np.random.choice(patterns))
                stages = [
                    (ps, hc, 0.0, R(14, 18)),
                    (hc, cs, 0.0, R(10, 14)),
                    (cs, cs, 1.0, R(5, 8)),
                    (cs, hc, 1.0, R(10, 14)),
                    (hc, hp, 1.0, R(16, 22)),
                    (hp, pps, 1.0, R(10, 14)),
                    (pps, pps, 0.0, R(5, 8)),
                    (pps, ret, 0.0, R(12, 16)),
                ]
            elif task_type == "push":
                push_patterns = [
                    f"push the {t_col} {n} towards {dest_str}",
                    f"nudge {t_col} {n} forward to {dest_str}",
                    f"slide the {t_col} {n} to {dest_str}",
                    f"push {t_col} {n} into {dest_str}",
                    f"move {t_col} {n} toward {dest_str}",
                ]
                prompt = str(np.random.choice(push_patterns))
                push_dir = (plat_pos[:2] - cube_pos[:2])
                norm = np.linalg.norm(push_dir)
                push_dir = push_dir / norm if norm > 1e-4 else np.array([1.0, 0.0])
                push_dist = np.random.uniform(0.040, 0.065)
                p_push_approach = np.array([cube_pos[0] - push_dir[0] * 0.035, cube_pos[1] - push_dir[1] * 0.035, hz], dtype=np.float32)
                p_push_start = np.array([cube_pos[0] - push_dir[0] * 0.035, cube_pos[1] - push_dir[1] * 0.035, -0.010], dtype=np.float32)
                push_end_xy = cube_pos[:2] + push_dir * push_dist
                p_push_end = np.array([push_end_xy[0], push_end_xy[1], -0.010], dtype=np.float32)
                p_push_retract = np.array([push_end_xy[0], push_end_xy[1], hz], dtype=np.float32)
                stages = [
                    (ps, p_push_approach, 0.0, R(14, 18)),
                    (p_push_approach, p_push_start, 0.0, R(10, 14)),
                    (p_push_start, p_push_end, 0.0, R(24, 30)),
                    (p_push_end, p_push_retract, 0.0, R(12, 16)),
                    (p_push_retract, ps, 0.0, R(12, 16)),
                ]
            elif task_type == "reach":
                reach_templates = [
                    "reach towards the {t} cube",
                    "move gripper above the {t} cube",
                    "hover over the {t} object",
                ]
                prompt = np.random.choice(reach_templates).format(t=t_col)
                p_reach_hover = np.array([cube_pos[0], cube_pos[1], hz + 0.010], dtype=np.float32)
                stages = [
                    (ps, p_reach_hover, 0.0, R(20, 26)),
                    (p_reach_hover, p_reach_hover, 0.0, R(15, 20)),
                    (p_reach_hover, ps, 0.0, R(20, 26)),
                ]
            else: # lift
                lift_templates = [
                    "pick up the {t} cube and hold it",
                    "lift the {t} block into the air",
                    "elevate the {t} object above the desk",
                ]
                prompt = np.random.choice(lift_templates).format(t=t_col)
                stages = [
                    (ps, hc, 0.0, R(14, 18)),
                    (hc, cs, 0.0, R(10, 14)),
                    (cs, cs, 1.0, R(6, 10)),
                    (cs, hc, 1.0, R(14, 18)),
                    (hc, hc, 1.0, R(16, 22)),
                ]

            raw_pts = []
            for p_s, p_e, g, n_steps in stages:
                for pt in quintic(p_s, p_e, n_steps):
                    raw_pts.append((pt, g))

            idx = np.linspace(0, len(raw_pts) - 1, 128).astype(int)
            traj = [raw_pts[k] for k in idx]

            kin_valid = True
            for pt, _ in traj:
                j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
                deg = np.degrees(j)
                if not (-125 <= deg[0] <= 125 and -5 <= deg[1] <= 85 and -15 <= deg[2] <= 70 and -150 <= deg[3] <= 150):
                    kin_valid = False
                    break
            if not kin_valid:
                continue

            grasp_start = None
            release_step = None
            for si, (_, g) in enumerate(traj):
                if g > 0.5 and grasp_start is None:
                    grasp_start = si
                if grasp_start is not None and g < 0.5 and release_step is None:
                    release_step = si

            # Store states along the trajectory
            pts_array = np.array([p for p, _ in traj], dtype=np.float32)
            grippers = np.array([g for _, g in traj], dtype=np.float32)
            joints = np.zeros((128, 4), dtype=np.float32)
            for si in range(128):
                joints[si] = self.kin.inverse(pts_array[si, 0], pts_array[si, 1], pts_array[si, 2], 0.0)

            # 9D representation: [j1, j2, j3, j4, x, y, z, r, gripper]
            actions_9d = np.zeros((128, 9), dtype=np.float32)
            actions_9d[:, :4] = joints
            actions_9d[:, 4:7] = pts_array
            actions_9d[:, 7] = 0.0
            actions_9d[:, 8] = grippers

            return {
                "task_type": task_type,
                "prompt": prompt,
                "t_col": t_col,
                "p_col": p_col,
                "cube_pos": cube_pos,
                "plat_pos": plat_pos,
                "dist_cubes": dist_cubes,
                "dist_plats": dist_plats,
                "desk_tone": desk_tone,
                "traj": traj,
                "grasp_start": grasp_start,
                "release_step": release_step,
                "actions_9d": actions_9d,
                "push_dir": push_dir,
                "pps": pps
            }
        raise RuntimeError("Failed to generate kinematically valid episode after 200 attempts.")

    def render_frame_at_step(self, episode, step_idx):
        """Renders the exact physical desk frame at step_idx without saving anything to disk."""
        task_type = episode["task_type"]
        cube_pos = episode["cube_pos"]
        plat_pos = episode["plat_pos"]
        pt, g = episode["traj"][step_idx]
        grasp_start = episode["grasp_start"]
        release_step = episode["release_step"]

        if task_type == "pick_place":
            if grasp_start is None or step_idx < grasp_start:
                vis_cube = cube_pos.copy()
            elif release_step is None or step_idx < release_step:
                vis_cube = pt.copy()
                vis_cube[2] = max(pt[2] - 0.005, 0.010)
            else:
                vis_cube = episode["pps"].copy()
                vis_cube[2] = plat_pos[2] + 0.008
        elif task_type == "push":
            push_dir = episode["push_dir"]
            if pt[2] < 0.0 and np.linalg.norm(pt[:2] - cube_pos[:2]) < 0.055:
                vis_cube = np.array([pt[0] + push_dir[0] * 0.015, pt[1] + push_dir[1] * 0.015, 0.012], dtype=np.float32)
            else:
                vis_cube = cube_pos.copy()
        elif task_type == "lift":
            if grasp_start is None or step_idx < grasp_start:
                vis_cube = cube_pos.copy()
            else:
                vis_cube = pt.copy()
                vis_cube[2] = max(pt[2] - 0.005, 0.010)
        else:
            vis_cube = cube_pos.copy()

        img_bgr = self.renderer.render_scene(
            target_cube_pos=vis_cube,
            target_color=episode["t_col"],
            platform_pos=plat_pos,
            plat_color=episode["p_col"],
            ee_pos=pt,
            gripper_closed=bool(g > 0.5),
            distractors_cubes=episode["dist_cubes"],
            distractors_plats=episode["dist_plats"],
            desk_tone=episode["desk_tone"],
        )
        return img_bgr


def stream_train_vla(total_episodes=10000, 
                     batch_size=256, 
                     d_model=256, 
                     num_layers=3, 
                     lr=1.5e-3, 
                     frame_interval=10, 
                     device="cuda",
                     checkpoint_dir=os.path.join(_CUR_DIR, "checkpoints")):
    """
    Online Zero-Disk VLA Training:
    Generates 1 episode, extracts CLIP patch features every 10 frames, accumulates into
    in-memory micro-batches, and trains the SmolVLA policy with zero intermediate disk writes.
    """
    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    print("=" * 72, flush=True)
    print("   ZERO-DISK STREAMING SMOLVLA TRAINING PIPELINE", flush=True)
    print(f"   Target Episodes        : {total_episodes:,}", flush=True)
    print(f"   Batch Size             : {batch_size}", flush=True)
    print(f"   CLIP Conversion Rate   : Every {frame_interval} frames (0, 10, 20, ...)", flush=True)
    print(f"   Architecture           : d_model={d_model}, layers={num_layers}, 127 context tokens", flush=True)
    print(f"   Compute Device         : {device}", flush=True)
    print("   Disk Write Policy      : ZERO (pure in-memory streaming, only checkpoints saved)", flush=True)
    print("=" * 72, flush=True)

    generator = StreamingTrajectoryGenerator()

    # Load frozen OpenAI CLIP ViT-B/32 on device for in-memory extraction
    print(">> Initializing in-memory OpenAI CLIP ViT-B/32 encoder...", flush=True)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Instantiate SmolVLA Flow-Matching Policy
    model = SmolVLAPolicy(
        horizon=128, 
        action_dim=9, 
        proprio_dim=9, 
        d_model=d_model, 
        num_layers=num_layers, 
        device=device
    ).to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss(reduction='none')

    # Dobot 9D loss weights: [j1, j2, j3, j4, x, y, z, r, gripper]
    dim_weights = torch.tensor([1.5, 1.5, 1.5, 1.2, 1.0, 1.0, 1.2, 1.0, 2.5], device=device).view(1, 1, 9)

    best_loss = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, "dobot_vla_best.pth")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "dobot_vla_latest.pth")

    # In-memory streaming buffer
    buf_vis = []
    buf_txt = []
    buf_proprio = []
    buf_action = []

    mean_9d = ACTION_MEAN_9D.to(device)
    std_9d = ACTION_STD_9D.to(device)

    # Frame sampling steps (every 10 frames from 0 to 127)
    sample_steps = list(range(0, 128, frame_interval))

    pbar = tqdm(total=total_episodes, desc="Streaming Episodes")
    ep_count = 0
    train_step = 0
    total_loss_accum = 0.0
    accum_steps = 0

    while ep_count < total_episodes:
        episode = generator.generate_single_episode()
        prompt_text = episode["prompt"]
        actions_9d = episode["actions_9d"] # [128, 9]

        # Tokenize text once per episode
        tok_t = clip.tokenize([prompt_text]).to(device)
        with torch.no_grad():
            text_feat_77 = clip_model.encode_text_all_tokens(tok_t) if hasattr(clip_model, "encode_text_all_tokens") else None
            if text_feat_77 is None:
                # Direct token embedding projection matching SmolVLABackbone
                x_tok = clip_model.token_embedding(tok_t).type(clip_model.dtype)
                x_tok = x_tok + clip_model.positional_embedding.type(clip_model.dtype)
                x_tok = x_tok.permute(1, 0, 2)
                x_tok = clip_model.transformer(x_tok)
                x_tok = x_tok.permute(1, 0, 2)
                x_tok = clip_model.ln_final(x_tok).type(torch.float32)
                text_feat_77 = x_tok # [1, 77, 512]

        # For every 10 frames, render scene, extract CLIP visual patches, and build chunk
        for step_i in sample_steps:
            img_bgr = generator.render_frame_at_step(episode, step_i)
            img_t = preprocess_image_for_clip(img_bgr, device=device) # [1, 3, 224, 224]

            with torch.no_grad():
                # Extract 49 spatial patches (ViT-B/32)
                vis_patches_49, _ = model.backbone.vlm.encode_vision(img_t) # [1, 49, 512]

            proprio_i = torch.from_numpy(actions_9d[step_i]).float().to(device) # [9]

            # Future action trajectory chunk of horizon 128 (padded if step_i > 0)
            rem_actions = actions_9d[step_i:]
            if len(rem_actions) < 128:
                pad = np.repeat(rem_actions[-1:], 128 - len(rem_actions), axis=0)
                full_act = np.concatenate([rem_actions, pad], axis=0)
            else:
                full_act = rem_actions[:128]

            norm_traj = (torch.from_numpy(full_act).float().to(device) - mean_9d) / (std_9d + 1e-6) # [128, 9]

            buf_vis.append(vis_patches_49.squeeze(0)) # [49, 512]
            buf_txt.append(text_feat_77.squeeze(0))   # [77, 512]
            buf_proprio.append(proprio_i)             # [9]
            buf_action.append(norm_traj)              # [128, 9]

        ep_count += 1
        pbar.update(1)

        # Train whenever in-memory buffer reaches batch_size
        while len(buf_vis) >= batch_size:
            b_vis = torch.stack(buf_vis[:batch_size], dim=0)       # [B, 49, 512]
            b_txt = torch.stack(buf_txt[:batch_size], dim=0)       # [B, 77, 512]
            b_prop = torch.stack(buf_proprio[:batch_size], dim=0)  # [B, 9]
            b_act = torch.stack(buf_action[:batch_size], dim=0)    # [B, 128, 9]

            # Pop used samples from memory
            buf_vis = buf_vis[batch_size:]
            buf_txt = buf_txt[batch_size:]
            buf_proprio = buf_proprio[batch_size:]
            buf_action = buf_action[batch_size:]

            optimizer.zero_grad(set_to_none=True)

            # Flow-Matching Loss: u_t = x_1 - x_0
            B = b_vis.size(0)
            x_0 = torch.randn_like(b_act)
            t = torch.rand(B, device=device)
            t_exp = t.view(B, 1, 1)
            x_t = (1.0 - t_exp) * x_0 + t_exp * b_act
            u_t = b_act - x_0

            v_pred = model.forward_from_embeddings(x_t, t, b_vis, b_txt, proprio=b_prop)
            loss = (loss_fn(v_pred, u_t) * dim_weights).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_step += 1
            total_loss_accum += loss.item()
            accum_steps += 1

            if train_step % 20 == 0:
                cur_loss = total_loss_accum / max(1, accum_steps)
                total_loss_accum = 0.0
                accum_steps = 0
                pbar.set_postfix({"Loss": f"{cur_loss:.4f}", "Step": train_step})

                # Checkpoint saving
                filtered_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.vlm.clip.")}
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    safe_save_checkpoint(filtered_state, best_checkpoint_path)
                safe_save_checkpoint(filtered_state, latest_checkpoint_path)

    pbar.close()
    filtered_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.vlm.clip.")}
    safe_save_checkpoint(filtered_state, latest_checkpoint_path)
    print(f"\n[SUCCESS] Online Zero-Disk Streaming Training Complete! Best Loss: {best_loss:.5f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Disk Streaming VLA Pipeline")
    parser.add_argument("--num_demos", type=int, default=10000, help="Total episodes to stream and train")
    parser.add_argument("--batch_size", type=int, default=256, help="In-memory training batch size")
    parser.add_argument("--frame_interval", type=int, default=10, help="Frames interval to convert CLIP (e.g. 10)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Decoder layers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(_CUR_DIR, "checkpoints"))
    args = parser.parse_args()

    stream_train_vla(
        total_episodes=args.num_demos,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        lr=args.lr,
        frame_interval=args.frame_interval,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
