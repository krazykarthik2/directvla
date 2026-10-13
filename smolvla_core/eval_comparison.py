import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim
from smolvla_embedding import SmolVLMTokenizer
from train_smolvla import SmolVLAPolicy, MODEL_DIR, DEVICE

def evaluate_policy(model, tokenizer, num_episodes=50, base_seed=42):
    sim = DobotPickPlaceSim()
    successes = 0
    min_distances = []
    grasp_successes = 0

    for ep in range(num_episodes):
        np.random.seed(base_seed + ep)
        action_type = "pick_place"
        obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

        with torch.no_grad():
            img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            token_ids = tokenizer.encode(sim.instruction, max_len=16).unsqueeze(0).to(DEVICE)
            proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            trajectory = model.sample(img_t, token_ids, proprio=proprio_t, num_steps=20).squeeze(0).cpu().numpy()

        ep_min_dist = float("inf")
        grasped_in_ep = False

        for traj_step in range(len(trajectory)):
            target_point = trajectory[traj_step]
            diff_xyz = target_point[:3] - sim.ee_pos[:3]

            dist_to_plat_2d = np.linalg.norm(sim.ee_pos[:2] - sim.target_platform_pos[:2])
            if sim.grasped and dist_to_plat_2d < 0.035 and sim.ee_pos[2] <= 0.045:
                grip_cmd = 0.0 # Autonomous touchdown release
            else:
                grip_cmd = 1.0 if target_point[3] > 0.35 else 0.0

            delta_action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
            obs, _ = sim.step_delta(delta_action, max_step=0.015)

            dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
            if dist_to_plat < ep_min_dist:
                ep_min_dist = dist_to_plat
            if sim.grasped:
                grasped_in_ep = True

        min_distances.append(ep_min_dist)
        if grasped_in_ep:
            grasp_successes += 1

        # Strict physical completion criteria
        final_cube_dist = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
        is_success = bool(final_cube_dist < 0.040 and sim.target_cube_pos[2] <= 0.025 and not sim.gripper_closed)
        if is_success:
            successes += 1

    success_rate = (successes / num_episodes) * 100.0
    grasp_rate = (grasp_successes / num_episodes) * 100.0
    avg_min_dist = np.mean(min_distances)
    return {
        "success_rate": success_rate,
        "grasp_rate": grasp_rate,
        "avg_min_dist": avg_min_dist,
        "success_count": successes
    }

if __name__ == "__main__":
    tokenizer = SmolVLMTokenizer()
    print("=" * 60)
    print(" EVALUATION BENCHMARK: Pure RL vs Fasttrain Pipeline")
    print(f" Compute Engine: {DEVICE} ({'CUDA GPU Acceleration' if DEVICE.type == 'cuda' else 'CPU'})")
    print(" 50 Identical Evaluation Scenarios Each")
    print("=" * 60)

    # 1. Fasttrain Model (BC Policy)
    fasttrain_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    fasttrain_model = SmolVLAPolicy(vocab_size=len(tokenizer.vocab), d_model=128, num_layers=2).to(DEVICE)
    if os.path.exists(fasttrain_path):
        fasttrain_model.load_state_dict(torch.load(fasttrain_path, map_location=DEVICE))
        fasttrain_model.eval()
        print("Evaluating Fasttrain Model (dobot_bc_policy.pth)...")
        ft_res = evaluate_policy(fasttrain_model, tokenizer, num_episodes=50)
        print(f"-> Fasttrain: Success Rate = {ft_res['success_rate']:.1f}% ({ft_res['success_count']}/50), Grasp Rate = {ft_res['grasp_rate']:.1f}%, Avg Min Dist = {ft_res['avg_min_dist']:.4f}m")
    else:
        print(f"[INFO] No checkpoint at {fasttrain_path}")
        ft_res = {"success_rate": 0.0, "grasp_rate": 0.0, "avg_min_dist": 0.0, "success_count": 0}

    # 2. Pure RL Model
    pure_rl_path = os.path.join(MODEL_DIR, "dobot_rl_only.pth")
    pure_rl_model = SmolVLAPolicy(vocab_size=len(tokenizer.vocab), d_model=128, num_layers=2).to(DEVICE)
    if os.path.exists(pure_rl_path):
        pure_rl_model.load_state_dict(torch.load(pure_rl_path, map_location=DEVICE))
        pure_rl_model.eval()
        print("Evaluating Pure RL Model (dobot_rl_only.pth)...")
        rl_res = evaluate_policy(pure_rl_model, tokenizer, num_episodes=50)
    else:
        print("Evaluating Untrained Baseline Policy...")
        pure_rl_model.eval()
        rl_res = evaluate_policy(pure_rl_model, tokenizer, num_episodes=50)
    print(f"-> Baseline:  Success Rate = {rl_res['success_rate']:.1f}% ({rl_res['success_count']}/50), Grasp Rate = {rl_res['grasp_rate']:.1f}%, Avg Min Dist = {rl_res['avg_min_dist']:.4f}m")
    print("=" * 60)
