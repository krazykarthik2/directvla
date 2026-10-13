import os
import sys
import time
import math
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "demonstrations")
os.makedirs(DATA_DIR, exist_ok=True)

def count_saved_demos():
    return len([f for f in os.listdir(DATA_DIR) if f.startswith("demo_") and f.endswith(".npz")])

def get_next_demo_index():
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("demo_") and f.endswith(".npz")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            num = int(f.replace("demo_", "").replace(".npz", ""))
            indices.append(num)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0

def world_to_screen(x, y):
    sx = int(200 + (y / 0.28) * 160)
    sy = int(260 - ((x - 0.08) / 0.26) * 230)
    return sx, sy

def world_to_side_screen(x, z):
    sx = int(450 + ((x - 0.05) / 0.30) * 260)
    sy = int(260 - (z / 0.22) * 220)
    return sx, sy

def generate_smooth_trajectory(p_start, p_end, steps):
    """Quintic polynomial smooth interpolation between two 3D positions."""
    if steps <= 1:
        return [p_end.copy()]
    pts = []
    for i in range(steps):
        s = i / float(steps - 1)
        s_quintic = 10 * (s ** 3) - 15 * (s ** 4) + 6 * (s ** 5)
        pt = p_start + s_quintic * (p_end - p_start)
        pts.append(pt)
    return pts

def render_gui(screen, font, font_bold, sim, current_demo_num, total_in_batch, total_saved, stage_name, current_step, fps_mode, cam_img=None):
    screen.fill((22, 25, 32))

    # Top View Panel
    pygame.draw.rect(screen, (30, 33, 42), (20, 20, 360, 360), border_radius=8)
    bx, by = world_to_screen(0.08, 0.0)
    pygame.draw.circle(screen, (70, 75, 95), (bx, by), 12)

    for p_col, p_pos in sim.distractor_platforms:
        px, py = world_to_screen(p_pos[0], p_pos[1])
        pygame.draw.rect(screen, COLOR_PALETTE.get(p_col, (40, 210, 80)), (px - 18, py - 18, 36, 36), border_radius=3)

    tpx, tpy = world_to_screen(sim.target_platform_pos[0], sim.target_platform_pos[1])
    t_col = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
    pygame.draw.rect(screen, t_col, (tpx - 20, tpy - 20, 40, 40), border_radius=4)
    pygame.draw.rect(screen, (255, 255, 255), (tpx - 20, tpy - 20, 40, 40), 2, border_radius=4)
    
    for c_col, c_pos in sim.distractor_cubes:
        cx, cy = world_to_screen(c_pos[0], c_pos[1])
        col = COLOR_PALETTE.get(c_col, (45, 120, 240))
        pygame.draw.rect(screen, col, (cx - 9, cy - 9, 18, 18), border_radius=2)

    tcx, tcy = world_to_screen(sim.target_cube_pos[0], sim.target_cube_pos[1])
    tc_col = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
    pygame.draw.rect(screen, tc_col, (tcx - 10, tcy - 10, 20, 20), border_radius=2)
    pygame.draw.rect(screen, (255, 255, 255), (tcx - 10, tcy - 10, 20, 20), 2, border_radius=2)
    
    ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
    grip_color = (255, 80, 80) if sim.gripper_closed else (100, 200, 255)
    pygame.draw.circle(screen, grip_color, (ex, ey), 8)
    pygame.draw.line(screen, (160, 170, 190), (200, 350), (ex, ey), 3)

    act_lbl = "PICK & PLACE" if getattr(sim, "action_type", "pick_place") == "pick_place" else "PUSH"
    top_label = font.render(f"ACTION: {act_lbl} ({sim.target_color.upper()} -> {sim.target_plat_color.upper()})", True, (170, 180, 200))
    screen.blit(top_label, (30, 30))

    # Side Elevation Panel
    pygame.draw.rect(screen, (35, 38, 48), (400, 20, 360, 360), border_radius=8)
    pygame.draw.line(screen, (60, 65, 80), (410, 350), (750, 350), 2)
    
    psx, psy = world_to_side_screen(sim.target_platform_pos[0], sim.target_platform_pos[2])
    pygame.draw.rect(screen, t_col, (psx - 20, psy - 4, 40, 8), border_radius=2)

    csx, csy = world_to_side_screen(sim.target_cube_pos[0], sim.target_cube_pos[2])
    pygame.draw.rect(screen, tc_col, (csx - 8, csy - 8, 16, 16), border_radius=2)

    esx, esy = world_to_side_screen(sim.ee_pos[0], sim.ee_pos[2])
    pygame.draw.circle(screen, grip_color, (esx, esy), 8)

    side_label = font.render("SIDE ELEVATION VIEW (Dobot Kinematics)", True, (170, 180, 200))
    screen.blit(side_label, (410, 30))

    if cam_img is not None:
        img_hwc = (np.transpose(cam_img, (2, 1, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.surfarray.make_surface(img_hwc)
        cam_surf_scaled = pygame.transform.scale(cam_surf, (100, 100))
        screen.blit(cam_surf_scaled, (270, 270))
        pygame.draw.rect(screen, (100, 220, 255), (270, 270, 100, 100), 2)
        cam_tag = font.render("Top Camera Focus", True, (100, 220, 255))
        screen.blit(cam_tag, (260, 250))

    pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 115), border_radius=8)
    status_str = f"Generating Demo #{current_demo_num} (Batch: {total_in_batch} | Total Dataset: {total_saved})"
    screen.blit(font_bold.render(status_str, True, (100, 210, 255)), (35, 405))

    prompt_str = f"Instruction: \"{sim.instruction}\""
    screen.blit(font_bold.render(prompt_str, True, (255, 230, 120)), (35, 432))

    stage_str = f"Phase: {stage_name} | Clutter: {len(sim.distractor_cubes)} objects | Steps: {current_step}"
    screen.blit(font.render(stage_str, True, (200, 205, 220)), (35, 458))

    speed_info = font.render(f"Speed: {fps_mode} | [F] Fast/Normal | [Q] Stop", True, (140, 145, 160))
    screen.blit(speed_info, (35, 482))

    pygame.display.flip()

def run_auto_demonstrator(num_demos=60, base_delay=0.00005):
    print("=" * 68)
    print("   Dobot Dynamic Velocity Demonstration Generator (Adaptive Timesteps)")
    print("   - Non-choreographed, continuous distance-based motion generation")
    print("   - Multi-Color Grounding & Parameterized Goal Embeddings")
    print("=" * 68)
    print(f">> Existing Demos in Dataset: {count_saved_demos()}")
    print(f">> Generating Batch of {num_demos} demonstrations...")
    print("=" * 68)

    sim = DobotPickPlaceSim()
    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dynamic Velocity Demonstration Generator")
    font = pygame.font.SysFont("Arial", 14)
    font_bold = pygame.font.SysFont("Arial", 16, bold=True)

    delay = base_delay
    fps_label = "Turbo Speed"
    demos_completed = 0

    # Velocity-based step calculation helper with slight randomized jitter for natural variation
    def calc_velocity_steps(p1, p2, target_step_size=0.008, min_steps=8):
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        # Add slight natural jitter (+-10%) so model never overfits to exact step counts
        jitter = np.random.uniform(0.90, 1.10)
        steps = int(max(min_steps, round((dist / target_step_size) * jitter)))
        return steps

    while demos_completed < num_demos:
        demo_idx = get_next_demo_index()
        act_choice = "pick_place" # Pure Pick and Place demonstrations only
        obs_dict = sim.reset(random_scene=True, num_distractors=2, action_type=act_choice)
        
        target_start = sim.target_cube_pos.copy()
        platform_target = sim.target_platform_pos.copy()
        prompt_text = sim.instruction

        img_list = []
        proprio_list = []
        legacy_obs_list = []
        action_list = []
        aborted = False

        hover_cube_z = 0.12
        p_start = sim.ee_pos[:3].copy()
        p_hover_cube = np.array([target_start[0], target_start[1], hover_cube_z], dtype=np.float32)
        
        if act_choice == "pick_place":
            p_cube_surface = np.array([target_start[0], target_start[1], 0.026], dtype=np.float32)
            p_hover_plat = np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32)
            p_plat_surface = np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32)
            p_retract = np.array([platform_target[0], platform_target[1], 0.12], dtype=np.float32)

            stages = [
                (f"1. Approach {sim.target_color.upper()} Cube", p_start, p_hover_cube, 0.0, 0.0, calc_velocity_steps(p_start, p_hover_cube, 0.009, 14)),
                (f"2. Descend on {sim.target_color.upper()} Cube", p_hover_cube, p_cube_surface, 0.0, 0.0, calc_velocity_steps(p_hover_cube, p_cube_surface, 0.007, 12)),
                (f"3. Grasp {sim.target_color.upper()} Cube", p_cube_surface, p_cube_surface, 1.0, 0.0, int(np.random.randint(5, 8))),
                (f"4. Lift {sim.target_color.upper()} Cube", p_cube_surface, p_hover_cube, 1.0, 0.0, calc_velocity_steps(p_cube_surface, p_hover_cube, 0.007, 12)),
                (f"5. Carry to {sim.target_plat_color.upper()} Box", p_hover_cube, p_hover_plat, 1.0, 0.0, calc_velocity_steps(p_hover_cube, p_hover_plat, 0.009, 16)),
                (f"6. Lower to {sim.target_plat_color.upper()} Box", p_hover_plat, p_plat_surface, 1.0, 0.0, calc_velocity_steps(p_hover_plat, p_plat_surface, 0.007, 12)),
                (f"7. Release {sim.target_color.upper()} Cube", p_plat_surface, p_plat_surface, 0.0, 1.0, int(np.random.randint(5, 8))),
                ("8. Retract Arm (Task Done)", p_plat_surface, p_retract, 0.0, 1.0, calc_velocity_steps(p_plat_surface, p_retract, 0.008, 10)),
            ]
        else: # "push"
            push_dir = (platform_target[:2] - target_start[:2])
            dist = np.linalg.norm(push_dir)
            if dist > 0.01:
                push_dir /= dist
            else:
                push_dir = np.array([1.0, 0.0])

            # Behind position (aligned with push vector)
            behind_pos = target_start[:2] - push_dir * 0.038
            # Push destination targets directly onto the platform center
            push_dest = platform_target[:2] + push_dir * 0.010

            p_behind_high = np.array([behind_pos[0], behind_pos[1], 0.080], dtype=np.float32)
            p_behind_low = np.array([behind_pos[0], behind_pos[1], 0.015], dtype=np.float32)
            p_push_end = np.array([push_dest[0], push_dest[1], 0.015], dtype=np.float32)
            p_retract = np.array([push_dest[0], push_dest[1], 0.100], dtype=np.float32)
            p_home = np.array([0.20, 0.0, 0.12], dtype=np.float32)

            stages = [
                (f"1. Move Behind {sim.target_color.upper()} Cube", p_start, p_behind_high, 0.0, 0.0, calc_velocity_steps(p_start, p_behind_high, 0.009, 14)),
                (f"2. Lower Behind {sim.target_color.upper()} Cube", p_behind_high, p_behind_low, 0.0, 0.0, calc_velocity_steps(p_behind_high, p_behind_low, 0.007, 10)),
                (f"3. Push Toward {sim.target_plat_color.upper()}", p_behind_low, p_push_end, 0.0, 0.0, calc_velocity_steps(p_behind_low, p_push_end, 0.005, 25)),
                (f"4. Retract Gripper", p_push_end, p_retract, 0.0, 1.0, calc_velocity_steps(p_push_end, p_retract, 0.008, 10)),
                (f"5. Return Home", p_retract, p_home, 0.0, 1.0, calc_velocity_steps(p_retract, p_home, 0.009, 14)),
            ]

        for stage_name, start_pt, end_pt, grip_state, succ_state, steps in stages:
            pts = generate_smooth_trajectory(start_pt, end_pt, steps)
            for pt in pts:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            aborted = True
                        elif event.key == pygame.K_f:
                            if delay > 0.0005:
                                delay = 0.00005
                                fps_label = "Turbo Speed (Max FPS)"
                            else:
                                delay = 0.005
                                fps_label = "Standard Speed (120 FPS)"

                if aborted:
                    break

                vla_obs = sim.get_vla_observation()
                img_list.append(vla_obs["image"])
                proprio_list.append(vla_obs["proprio"])
                legacy_obs_list.append(sim.get_observation())
                
                current_ee = sim.ee_pos.copy()
                dx = pt[0] - current_ee[0]
                dy = pt[1] - current_ee[1]
                dz = pt[2] - current_ee[2]
                dyaw = 0.0
                
                delta_action = np.array([dx, dy, dz, dyaw, grip_state, succ_state], dtype=np.float32)
                action_list.append(delta_action)
                sim.step_delta(delta_action[:5])

                total_saved_now = count_saved_demos()
                render_gui(screen, font, font_bold, sim, demo_idx, num_demos, total_saved_now, stage_name, len(action_list), fps_label, vla_obs["image"])
                if delay > 0:
                    time.sleep(delay)

            if aborted:
                break

        if aborted:
            print("\n[INFO] Demonstration generation stopped by user.")
            break

        # Strict Demo Quality Verification Filter: Only save demos that ACTUALLY succeed!
        final_dist_to_plat = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
        is_demo_valid = (final_dist_to_plat < 0.040 and sim.target_cube_pos[2] <= 0.025)
        if not is_demo_valid:
            print(f"[REJECTED] Demo #{demo_idx:03d} failed to place/push object onto platform (dist: {final_dist_to_plat*1000:.1f}mm). Discarding...", flush=True)
            continue

        img_arr = np.array(img_list, dtype=np.float32)
        proprio_arr = np.array(proprio_list, dtype=np.float32)
        legacy_arr = np.array(legacy_obs_list, dtype=np.float32)
        acts_arr = np.array(action_list, dtype=np.float32)

        orig_len = len(acts_arr)
        target_len = 128
        indices = np.linspace(0, orig_len - 1, target_len).astype(int)

        filename = os.path.join(DATA_DIR, f"demo_{demo_idx:03d}.npz")
        
        np.savez_compressed(
            filename,
            images=img_arr[indices],
            proprioception=proprio_arr[indices],
            observations=legacy_arr[indices],
            actions=acts_arr[indices],
            prompt=np.array([prompt_text]),
            action_type=np.array([act_choice]),
            target_color=np.array([sim.target_color]),
            target_plat_color=np.array([sim.target_plat_color])
        )
        demos_completed += 1
        total_now = count_saved_demos()
        print(f"[SUCCESS] Saved Verified Demo #{demo_idx:03d} [{act_choice.upper()}] ({sim.target_color}->{sim.target_plat_color} | Total: {total_now} | Dist: {final_dist_to_plat*1000:.1f}mm)")

    pygame.quit()
    print(f"\n[DONE] Finished batch! Total {count_saved_demos()} demonstration datasets in: {DATA_DIR}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_auto_demonstrator(num_demos=n)
