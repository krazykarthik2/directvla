#!/usr/bin/env python3
"""
Closed-Loop Real-World VLA Controller for Dobot Magician with Top-Angle Desk Camera.
Features:
  1. Live desk top-angle camera stream capture and perspective rectification (/dev/video0).
  2. Real ROS 2 Dobot Bridge & Direct PySerial Hardware communication.
  3. Pretrained SmolVLA multimodal flow-matching policy inference with CLIP ViT-B/32.
  4. Smooth trajectory chunking with closed-loop replanning.
  5. Live OpenCV visualizer showing desk workspace, trajectory overlay, and robot state.
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import torch
import clip

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_camera import DeskTopCamera, SyntheticDeskRenderer, preprocess_image_for_clip
from vla_robot_env import DobotRealEnv, DobotKinematics
from vla_model import SmolVLAPolicy, ACTION_MEAN, ACTION_STD


def parse_colors_from_prompt(prompt):
    """Extracts target cube color and target platform color from instruction prompt."""
    prompt_lower = prompt.lower()
    known_colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange"]
    target_cube_color = "red"
    target_plat_color = "green"

    for col in known_colors:
        if f"{col} cube" in prompt_lower or f"{col} block" in prompt_lower or f"{col} object" in prompt_lower or f"{col} pen" in prompt_lower or f"{col} item" in prompt_lower:
            target_cube_color = col
            break

    for col in known_colors:
        if col != target_cube_color and (f"{col} platform" in prompt_lower or f"{col} area" in prompt_lower or f"{col} target" in prompt_lower or f"{col} zone" in prompt_lower):
            target_plat_color = col
            break

    return target_cube_color, target_plat_color


def initialize_scene_objects(prompt, num_distractors=2):
    """
    Initializes scene layout with target objects and distractor objects matching training distribution.
    Guarantees non-overlapping positions on the desk.
    """
    target_cube_col, target_plat_col = parse_colors_from_prompt(prompt)
    cube_colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange"]
    plat_colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange"]

    occupied = []

    def get_pos(min_dist=0.065, default_pos=None):
        for _ in range(100):
            a = np.random.uniform(-0.55, 0.55)
            d = np.random.uniform(0.19, 0.27)
            p = np.array([d * np.cos(a), d * np.sin(a), 0.012], dtype=np.float32)
            if all(np.linalg.norm(p[:2] - o[:2]) > min_dist for o in occupied):
                occupied.append(p)
                return p
        p = np.array(default_pos if default_pos is not None else [0.22, 0.08, 0.012], dtype=np.float32)
        occupied.append(p)
        return p

    target_cube_pos = get_pos(default_pos=[0.22, 0.08, 0.012])
    target_plat_pos = get_pos(min_dist=0.075, default_pos=[0.22, -0.10, 0.005])
    target_plat_pos[2] = 0.005

    # Distractor cubes
    avail_cubes = [c for c in cube_colors if c != target_cube_col]
    np.random.shuffle(avail_cubes)
    dist_cubes = []
    for i in range(min(num_distractors, len(avail_cubes))):
        c_col = avail_cubes[i]
        c_pos = get_pos()
        dist_cubes.append((c_col, c_pos))

    # Distractor platforms
    dist_plats = []
    avail_plats = [c for c in plat_colors if c != target_plat_col]
    if num_distractors > 0 and avail_plats:
        p_col = avail_plats[0]
        dp_pos = get_pos(min_dist=0.075)
        dp_pos[2] = 0.005
        dist_plats.append((p_col, dp_pos))

    return {
        "target_cube_color": target_cube_col,
        "target_cube_pos": target_cube_pos,
        "target_plat_color": target_plat_col,
        "target_plat_pos": target_plat_pos,
        "distractors_cubes": dist_cubes,
        "distractors_plats": dist_plats,
        "is_grasped": False
    }


def run_real_dobot_vla(
    checkpoint_path=os.path.join(_CUR_DIR, "checkpoints", "dobot_vla_best.pth"),
    instruction="pick up the red cube and place it on the green platform",
    camera_index=0,
    use_ros2=True,
    port=None,
    device='cpu',
    sample_steps=12,
    headless=False,
    num_distractors=2
):
    device = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    print("=" * 72, flush=True)
    print("   Real Dobot Magician VLA Closed-Loop Controller", flush=True)
    print("   - Desk Top-Angle Camera: live feed + workspace homography", flush=True)
    print("   - Backbone: Genuine OpenAI CLIP ViT-B/32 Multimodal Policy", flush=True)
    print(f"   - Target Hardware Execution Device: {device}", flush=True)
    print(f"   - Initial Language Prompt: \"{instruction}\"", flush=True)
    print(f"   - Scene Distractors: {num_distractors} cubes + platforms enabled in output run", flush=True)
    print("=" * 72, flush=True)

    # 1. Initialize Robot Hardware & ROS 2 Environment
    env = DobotRealEnv(use_ros2=use_ros2, port=port)
    env.reset()

    # 2. Initialize Desk Top-Angle Camera & Scene Layout
    cam = DeskTopCamera(device_index=camera_index)
    syn_renderer = None
    scene = initialize_scene_objects(instruction, num_distractors=num_distractors)
    if not cam.is_connected:
        print(f"[VLA Controller] Notice: Physical camera not detected. Using high-fidelity synthetic desk camera view.", flush=True)
        print(f"[VLA Controller] Virtual Scene configured with target [{scene['target_cube_color']} cube -> {scene['target_plat_color']} plat] and {len(scene['distractors_cubes'])} distractor cubes, {len(scene['distractors_plats'])} distractor plat.", flush=True)
        syn_renderer = SyntheticDeskRenderer()

    # 3. Load Policy Checkpoint
    action_dim = 9
    proprio_dim = 9
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "out_head.3.weight" in state_dict:
                action_dim = state_dict["out_head.3.weight"].shape[0]
            if "proprio_proj.0.weight" in state_dict:
                proprio_dim = state_dict["proprio_proj.0.weight"].shape[1]
        except Exception:
            pass

    policy = SmolVLAPolicy(horizon=128, action_dim=action_dim, proprio_dim=proprio_dim, d_model=128, num_layers=2, device=device).to(device)
    if os.path.exists(checkpoint_path):
        try:
            policy.load_state_dict(state_dict, strict=False)
            print(f"[VLA Controller] Successfully loaded policy weights from {checkpoint_path} (action_dim={action_dim}, proprio_dim={proprio_dim})", flush=True)
        except Exception as e:
            print(f"[VLA Controller] Warning: Failed loading checkpoint ({e}). Using initialized policy.", flush=True)
    else:
        print(f"[VLA Controller] Warning: Checkpoint {checkpoint_path} not found. Running with base model.", flush=True)

    policy.eval()

    # 4. Tokenize Language Instruction
    tokens = clip.tokenize([instruction]).to(device)

    # Control loop variables
    current_trajectory = None
    traj_step = 0
    auto_execute = True
    running = True

    print("\nControls:")
    print("  [SPACE] : Pause / Resume policy execution")
    print("  [R]     : Reset robot to hover home pose & replan")
    print("  [Q/ESC] : Exit controller safely")
    print("=" * 72, flush=True)

    try:
        while running:
            proprio = env.get_proprioception(full_state=(proprio_dim == 9))

            # 1. Acquire current desk camera observation
            if cam.is_connected:
                raw_frame = cam.read_raw_frame()
                rectified_rgb = cam.read_rectified_rgb()
                if rectified_rgb is None:
                    continue
                img_tensor = preprocess_image_for_clip(rectified_rgb, device=device)
                vis_frame = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Synthetic desk camera frame
                # In 9D proprio: indices 4, 5, 6 are ee_x, ee_y, ee_z; index 8 is grip
                # In 5D proprio: indices 0, 1, 2 are ee_x, ee_y, ee_z; index 4 is grip
                ee_xyz = proprio[4:7] if proprio_dim == 9 else proprio[:3]
                ee_grip = proprio[8] if proprio_dim == 9 else proprio[4]

                # Simulate physical grasp & transport of the target cube
                dist_to_cube = np.linalg.norm(ee_xyz[:2] - scene["target_cube_pos"][:2])
                if ee_grip > 0.5 and (dist_to_cube < 0.035 or scene["is_grasped"]):
                    scene["is_grasped"] = True
                    # Target cube follows gripper
                    scene["target_cube_pos"] = np.array([ee_xyz[0], ee_xyz[1], max(0.012, ee_xyz[2] - 0.012)], dtype=np.float32)
                elif ee_grip <= 0.5 and scene["is_grasped"]:
                    scene["is_grasped"] = False
                    # Release cube onto platform or desk surface
                    scene["target_cube_pos"][2] = 0.012

                rectified_rgb = syn_renderer.render_scene(
                    target_cube_pos=scene["target_cube_pos"],
                    target_color=scene["target_cube_color"],
                    platform_pos=scene["target_plat_pos"],
                    plat_color=scene["target_plat_color"],
                    ee_pos=ee_xyz,
                    gripper_closed=bool(ee_grip > 0.5),
                    distractors_cubes=scene["distractors_cubes"],
                    distractors_plats=scene["distractors_plats"]
                )
                img_tensor = preprocess_image_for_clip(rectified_rgb, device=device)
                vis_frame = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)

            # 2. Check if replanning is needed
            need_replan = (current_trajectory is None) or (traj_step >= len(current_trajectory))

            if auto_execute and need_replan:
                with torch.no_grad():
                    proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(device)
                    # Sample full 128-step trajectory chunk with Euler ODE flow matching
                    t0 = time.time()
                    traj_t = policy.sample(
                        img_tensor, tokens, proprio=proprio_t, num_steps=sample_steps
                    )
                    inference_ms = (time.time() - t0) * 1000.0

                    current_trajectory = traj_t.squeeze(0).cpu().numpy()
                    traj_step = 0

            # 3. Advance execution step
            if auto_execute and current_trajectory is not None and traj_step < len(current_trajectory):
                target_pt = current_trajectory[traj_step]
                if action_dim == 9:
                    # target_pt is [j1, j2, j3, j4, x, y, z, r, gripper]
                    j_tgt = target_pt[:4]
                    g_tgt = target_pt[8]
                    # Step via joint space
                    env.step_joints(np.array([j_tgt[0], j_tgt[1], j_tgt[2], j_tgt[3], g_tgt], dtype=np.float32))
                else:
                    # target_pt is [x, y, z, gripper]
                    target_xyz = target_pt[:3]
                    target_grip = target_pt[3]
                    curr_xyz = proprio[4:7] if proprio_dim == 9 else proprio[:3]
                    diff = target_xyz - curr_xyz
                    dist = np.linalg.norm(diff)
                    max_step = 0.010
                    if dist > max_step:
                        step_xyz = curr_xyz + (diff / dist) * max_step
                    else:
                        step_xyz = target_xyz
                    target_5d = np.array([step_xyz[0], step_xyz[1], step_xyz[2], 0.0, target_grip], dtype=np.float32)
                    env.step_target(target_5d)

                traj_step += 1

            # 4. Rendering Visual HUD
            if not headless:
                hud = cv2.resize(vis_frame, (480, 480))
                # Draw planned trajectory overlay
                if current_trajectory is not None:
                    for i in range(0, len(current_trajectory) - 1, 2):
                        pt1 = current_trajectory[i]
                        pt2 = current_trajectory[i + 1]
                        # Map meter workspace to 480x480 HUD
                        # x: [0.12, 0.35] -> y: [420, 40]
                        # y: [-0.18, 0.18] -> x: [40, 440]
                        px1 = int(240 + (pt1[1] / 0.18) * 190)
                        py1 = int(420 - ((pt1[0] - 0.12) / 0.23) * 370)
                        px2 = int(240 + (pt2[1] / 0.18) * 190)
                        py2 = int(420 - ((pt2[0] - 0.12) / 0.23) * 370)
                        cv2.line(hud, (px1, py1), (px2, py2), (0, 230, 255), 2)

                # End-effector overlay
                ex = int(240 + (proprio[1] / 0.18) * 190)
                ey = int(420 - ((proprio[0] - 0.12) / 0.23) * 370)
                grip_val = proprio[8] if proprio_dim == 9 else proprio[4]
                grip_col = (0, 0, 255) if grip_val > 0.5 else (255, 180, 0)
                cv2.circle(hud, (ex, ey), 8, grip_col, -1)
                cv2.circle(hud, (ex, ey), 8, (255, 255, 255), 2)

                # Information header
                cv2.rectangle(hud, (0, 0), (480, 72), (25, 28, 35), -1)
                cv2.putText(hud, f"Prompt: {instruction[:45]}...", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 100), 1)
                state_str = f"EE: [{proprio[0]:.2f}, {proprio[1]:.2f}, {proprio[2]:.2f}] | Grip: {'ON' if grip_val > 0.5 else 'OFF'} | Step: {traj_step}/128"
                cv2.putText(hud, state_str, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)
                dist_str = f"Target: [{scene['target_cube_color']} cube -> {scene['target_plat_color']} plat] | Distractors: {len(scene['distractors_cubes'])} cubes, {len(scene['distractors_plats'])} plat"
                cv2.putText(hud, dist_str, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 180, 255), 1)
                hw_str = f"Hardware: {'CONNECTED' if env.hardware_connected else 'SIM BRIDGE'} | Exec: {'RUNNING' if auto_execute else 'PAUSED'}"
                cv2.putText(hud, hw_str, (10, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 255, 150), 1)

                cv2.imshow("Dobot Magician SmolVLA (Desk Top-Angle Camera)", hud)
                key = cv2.waitKey(20) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord(' '):
                    auto_execute = not auto_execute
                    print(f"[VLA Controller] Execution {'resumed' if auto_execute else 'paused'}.", flush=True)
                elif key == ord('r'):
                    env.reset()
                    current_trajectory = None
                    traj_step = 0
                    scene = initialize_scene_objects(instruction, num_distractors=num_distractors)
                    print("[VLA Controller] Reset robot and re-initialized scene objects & distractors.", flush=True)
            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[VLA Controller] Stopped by user.", flush=True)
    finally:
        env.reset()
        env.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[VLA Controller] Safely shut down Dobot VLA controller.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SmolVLA Controller on Real Dobot Magician")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_CUR_DIR, "checkpoints", "dobot_vla_best.pth"))
    parser.add_argument("--prompt", type=str, default="pick up the red cube and place it on the green platform")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--no_ros2", action="store_true", help="Disable ROS 2 bridge and use direct serial")
    parser.add_argument("--port", type=str, default=None, help="Serial port for Dobot (auto if None)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=12, help="Euler ODE sampling steps")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    parser.add_argument("--num_distractors", type=int, default=2, help="Number of distractor cubes/platforms in virtual desk scene")
    args = parser.parse_args()

    run_real_dobot_vla(
        checkpoint_path=args.checkpoint,
        instruction=args.prompt,
        camera_index=args.camera_index,
        use_ros2=not args.no_ros2,
        port=args.port,
        device=args.device,
        sample_steps=args.steps,
        headless=args.headless,
        num_distractors=args.num_distractors
    )
