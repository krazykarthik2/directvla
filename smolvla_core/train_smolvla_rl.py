import os
import sys
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim
from train_smolvla import SmolVLAPolicy, SmolVLADataset, safe_save_model, ACTION_MEAN, ACTION_STD, MODEL_DIR, DATA_DIR, DEVICE

def train_smolvla_with_demo_anchored_rl(
    num_episodes=150,
    bc_epochs=50,
    batch_size=16,
    lr=1.5e-3,
    target_success_rate=90.0,
    update_every=4
):
    print("=" * 68, flush=True)
    print("   SMOLVLA: DEMO-ANCHORED RL & PRETRAINED CROSS-ATTENTION FLOW", flush=True)
    print("   - Frozen / Pre-trained Multimodal Vision-Language Backbone", flush=True)
    print("   - Trainable Cross-Attention Action Head + 70/30 Demo Replay Anchor", flush=True)
    print("   - Terminal Physical Verification & Continuous Receding Horizon", flush=True)
    print(f"   - Hardware Compute Engine: {DEVICE} ({'CUDA GPU Acceleration' if DEVICE.type == 'cuda' else 'Optimized Multi-core CPU'})", flush=True)
    print("=" * 68, flush=True)

    dataset = SmolVLADataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmolVLAPolicy(vocab_size=len(dataset.tokenizer.vocab), d_model=128, num_layers=2).to(DEVICE)
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    # Load existing checkpoint if available
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f">> Loaded existing SmolVLA checkpoint -> {model_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}", flush=True)

    # 1. Freeze Vision & Language Embeddings (Train ONLY Cross-Attention Action Head)
    print(">> Freezing Vision & Language Backbone (Training ONLY Cross-Attention Action Head)...", flush=True)
    for name, param in model.named_parameters():
        if "action" in name or "out_head" in name or "proprio_proj" in name or "time_embed" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">> Trainable Parameters (Cross-Attention Action Head): {trainable_params:,} / {total_params:,}", flush=True)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    sim = DobotPickPlaceSim()
    recent_successes = []
    best_success_rate = 0.0

    print(f"\n>> Starting Demo-Anchored RL Fine-Tuning ({num_episodes} episodes)...", flush=True)

    data_iter = iter(dataloader)

    ep = 0
    while True:
        ep += 1
        # Sample next demo anchor batch (cycling)
        try:
            d_img, d_tokens, d_proprio, d_norm_traj = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            d_img, d_tokens, d_proprio, d_norm_traj = next(data_iter)

        action_type = "pick_place"
        obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

        img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        token_ids = dataset.tokenizer.encode(sim.instruction, max_len=16).unsqueeze(0).to(DEVICE)
        proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 1. Sample trajectory from model
        with torch.no_grad():
            trajectory = model.sample(img_t, token_ids, proprio=proprio_t, num_steps=20).squeeze(0).cpu().numpy()

        # 2. Execute trajectory rollouts
        min_d = float("inf")
        grasped = False
        steps = len(trajectory)

        for step_idx in range(steps):
            target_pt = trajectory[step_idx]
            diff_xyz = target_pt[:3] - sim.ee_pos[:3]

            dist_to_plat_2d = np.linalg.norm(sim.ee_pos[:2] - sim.target_platform_pos[:2])
            if sim.grasped and dist_to_plat_2d < 0.035 and sim.ee_pos[2] <= 0.045:
                grip_cmd = 0.0 # Touchdown release
            else:
                grip_cmd = 1.0 if target_pt[3] > 0.35 else 0.0

            action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
            obs, _ = sim.step_delta(action, max_step=0.012)

            d_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
            min_d = min(min_d, d_plat)
            if sim.grasped:
                grasped = True

        # 3. Terminal physical success check
        final_d = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
        succ = bool(final_d < 0.040 and sim.target_cube_pos[2] <= 0.025 and not sim.gripper_closed)

        # 4. Dense RL reward function
        reward = -final_d * 50.0
        if grasped:
            reward += 30.0
        if succ:
            reward += 100.0

        recent_successes.append(1.0 if succ else 0.0)
        if len(recent_successes) > 30:
            recent_successes.pop(0)

        advantage = reward - 10.0 # Baseline center

        # 5. RL Flow-Matching loss
        B = 1
        x_1_env = torch.tensor(trajectory, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x_0_env = torch.randn_like(x_1_env)
        t_env = torch.rand(B, device=DEVICE)
        t_exp = t_env.view(B, 1, 1)

        x_t_env = (1.0 - t_exp) * x_0_env + t_exp * x_1_env
        u_t_env = x_1_env - x_0_env

        v_pred_env, _ = model.forward_flow(x_t_env, t_env, img_t, token_ids, proprio=proprio_t)
        base_rl_loss = loss_fn(v_pred_env, u_t_env)
        rl_loss = base_rl_loss * max(-2.0, min(2.0, -advantage * 0.05))

        # 6. Demo Anchor Loss (70% anchor to expert kinematic demonstrations)
        d_img = d_img.to(DEVICE)
        d_tokens = d_tokens.to(DEVICE)
        d_proprio = d_proprio.to(DEVICE)
        d_norm_traj = d_norm_traj.to(DEVICE)

        B_d = d_img.size(0)
        x_0_demo = torch.randn_like(d_norm_traj)
        t_demo = torch.rand(B_d, device=DEVICE)
        t_d_exp = t_demo.view(B_d, 1, 1)

        x_t_demo = (1.0 - t_d_exp) * x_0_demo + t_d_exp * d_norm_traj
        u_t_demo = d_norm_traj - x_0_demo

        v_pred_demo, _ = model.forward_flow(x_t_demo, t_demo, d_img, d_tokens, proprio=d_proprio)
        demo_loss = loss_fn(v_pred_demo, u_t_demo)

        # Combined 70/30 anchor loss
        total_loss = 0.70 * demo_loss + 0.30 * rl_loss
        total_loss.backward()

        if ep % update_every == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        succ_pct = (sum(recent_successes) / len(recent_successes)) * 100.0
        max_str = f"{num_episodes}" if num_episodes > 0 else "INF"
        print(f"EP {ep:04d}/{max_str} | Act: {action_type[:4].upper()} | MinDist: {min_d*1000:.1f}mm | Grasped: {grasped} | Succ: {succ} | R: {reward:+.1f} | Win30: {succ_pct:.1f}%", flush=True)

        if succ_pct > best_success_rate or ep % 20 == 0:
            best_success_rate = max(best_success_rate, succ_pct)
            safe_save_model(model.state_dict(), model_path)

        if len(recent_successes) >= 30 and succ_pct >= target_success_rate:
            print(f"\n[TARGET REACHED] Achieved {succ_pct:.1f}% success rate! Task solved.", flush=True)
            safe_save_model(model.state_dict(), model_path)
            break

        if num_episodes > 0 and ep >= num_episodes:
            print(f"\n[INFO] Reached requested episode limit ({num_episodes}). Finishing RL.", flush=True)
            break

    safe_save_model(model.state_dict(), model_path)
    print(f"\n[DONE] SmolVLA Demo-Anchored RL complete! Checkpoint saved -> {model_path}", flush=True)

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    target_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    train_smolvla_with_demo_anchored_rl(num_episodes=episodes, target_success_rate=target_pct)
