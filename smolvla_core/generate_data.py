import numpy as np
import os
from sim import DobotPickPlaceSim
from model import build_intent_vector

def interpolate_waypoints(start_pos, end_pos, num_steps):
    return [start_pos + (end_pos - start_pos) * (t / float(num_steps)) for t in range(1, num_steps + 1)]

def generate_dataset(num_episodes=60):
    sim = DobotPickPlaceSim(render_display=False)
    os.makedirs('data', exist_ok=True)

    data_obs = []
    data_intent = []
    data_proprio = []
    data_actions = []

    print(f"Generating {num_episodes} demonstration episodes with analytical Dobot IK...")

    for ep in range(num_episodes):
        obs_dict = sim.reset(random_scene=True, num_distractors=2)
        task_mode = np.random.choice(["pick_place", "push"])
        
        target_cube_color = sim.target_color
        target_plat_color = sim.target_plat_color
        intent_vec = build_intent_vector(task_mode, target_cube_color, target_plat_color)

        c_pos = sim.target_cube_pos.copy()
        p_pos = sim.target_platform_pos.copy()
        
        # Build trajectory waypoints based on task
        waypoints = []
        # format: (target_xyz, gripper_val)

        if task_mode == "pick_place":
            # 1. Hover above target cube
            waypoints.append((np.array([c_pos[0], c_pos[1], 0.080]), 0.0, 10))
            # 2. Descend to cube height
            waypoints.append((np.array([c_pos[0], c_pos[1], 0.020]), 0.0, 8))
            # 3. Grasp cube
            waypoints.append((np.array([c_pos[0], c_pos[1], 0.020]), 1.0, 4))
            # 4. Lift cube up
            waypoints.append((np.array([c_pos[0], c_pos[1], 0.090]), 1.0, 8))
            # 5. Move hover over platform
            waypoints.append((np.array([p_pos[0], p_pos[1], 0.090]), 1.0, 12))
            # 6. Lower onto platform
            waypoints.append((np.array([p_pos[0], p_pos[1], 0.024]), 1.0, 8))
            # 7. Release gripper
            waypoints.append((np.array([p_pos[0], p_pos[1], 0.024]), 0.0, 4))
            # 8. Retract up
            waypoints.append((np.array([p_pos[0], p_pos[1], 0.090]), 0.0, 6))

        else: # "push" towards destination platform
            # Approach slightly behind the cube relative to platform
            push_dir = (p_pos[:2] - c_pos[:2])
            dist = np.linalg.norm(push_dir)
            if dist > 0.01:
                push_dir /= dist
            else:
                push_dir = np.array([1.0, 0.0])

            behind_pos = c_pos[:2] - push_dir * 0.035
            # 1. Hover behind cube
            waypoints.append((np.array([behind_pos[0], behind_pos[1], 0.070]), 0.0, 10))
            # 2. Drop down behind cube
            waypoints.append((np.array([behind_pos[0], behind_pos[1], 0.018]), 0.0, 8))
            # 3. Push cube forward toward platform
            push_target = c_pos[:2] + push_dir * 0.080
            waypoints.append((np.array([push_target[0], push_target[1], 0.018]), 0.0, 18))
            # 4. Retract upwards
            waypoints.append((np.array([push_target[0], push_target[1], 0.080]), 0.0, 6))

        # Execute trajectory and record state-action pairs
        curr_ee = sim.ee_pos[:3].copy()
        for target_xyz, grip_cmd, steps in waypoints:
            path = interpolate_waypoints(curr_ee, target_xyz, steps)
            for next_pos in path:
                # Delta displacement
                delta_xyz = next_pos - sim.ee_pos[:3]
                # Bound deltas
                delta_xyz = np.clip(delta_xyz, -0.006, 0.006)
                action = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0, grip_cmd], dtype=np.float32)

                # Record current observation before stepping
                data_obs.append(obs_dict["image"])
                data_intent.append(intent_vec)
                data_proprio.append(obs_dict["proprio"])
                data_actions.append(action)

                # Step environment
                obs_dict, _ = sim.step_delta(action)
                curr_ee = sim.ee_pos[:3].copy()

        if (ep + 1) % 10 == 0:
            print(f"Generated {ep + 1}/{num_episodes} episodes...")

    # Save to disk
    np.save('data/obs.npy', np.stack(data_obs))
    np.save('data/intent.npy', np.stack(data_intent))
    np.save('data/proprio.npy', np.stack(data_proprio))
    np.save('data/actions.npy', np.stack(data_actions))
    print(f"Finished. Total transitions recorded: {len(data_obs)}")

if __name__ == "__main__":
    generate_dataset(60)
