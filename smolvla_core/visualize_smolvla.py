import os
import sys
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.dirname(__file__))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from smolvla_embedding import SmolVLMTokenizer
from train_smolvla import SmolVLAPolicy, MODEL_DIR, DEVICE

def visualize_all_smolvla_inputs():
    device = DEVICE
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")

    tokenizer = SmolVLMTokenizer()
    model = SmolVLAPolicy(d_model=128, num_layers=2).to(device)

    has_model = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.eval()
            has_model = True
            print(f"[INFO] SmolVLA Model loaded from {model_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load model: {e}", flush=True)
    else:
        print(f"[WARN] No checkpoint at {model_path} — running visualization without model execution.", flush=True)

    sim = DobotPickPlaceSim()
    pygame.init()

    screen = pygame.display.set_mode((1200, 720))
    pygame.display.set_caption("SmolVLA Live Visualizer (Pretrained CLIP ViT Patches + Subwords)")

    font_xs = pygame.font.SysFont("Consolas", 10)
    font_sm = pygame.font.SysFont("Consolas", 11)
    font = pygame.font.SysFont("Consolas", 12)
    font_bold = pygame.font.SysFont("Consolas", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 15, bold=True)

    selected_token_idx = 4
    clock = pygame.time.Clock()
    running = True

    # Speed modes: 0=normal(60fps), 1=fast(3x), 2=slow(0.5x)
    SPEED_NAMES = ["NORMAL (60fps)", "FAST (3x)", "SLOW (0.5x)"]
    speed_mode = 0

    action_type = "pick_place"
    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)

    current_trajectory = None
    traj_step = 0
    episode_total_ticks = 0
    MAX_EPISODE_TICKS = 220
    total_episodes = 0
    successful_episodes = 0
    failed_episodes = 0
    task_success_status = None
    success_banner_timer = 0

    # Scroll offset for token panel
    token_scroll = 0

    # Cache attention weights for display
    cached_cross_weights = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_UP:
                    selected_token_idx = max(0, selected_token_idx - 1)
                elif event.key == pygame.K_DOWN:
                    selected_token_idx = min(76, selected_token_idx + 1)
                elif event.key == pygame.K_1:
                    action_type = "pick_place"
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type="pick_place")
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_2:
                    action_type = "push"
                    obs = sim.reset(random_scene=True, num_distractors=2, action_type="push")
                    current_trajectory = None
                    task_success_status = None
                    episode_total_ticks = 0
                elif event.key == pygame.K_f:
                    speed_mode = (speed_mode + 1) % 3
                elif event.key == pygame.K_SPACE:
                    # Pause/resume (toggle between paused and current speed)
                    pass
            elif event.type == pygame.MOUSEWHEEL:
                token_scroll = max(0, token_scroll - event.y * 2)

        prompt_text = sim.instruction
        token_ids = tokenizer.encode(prompt_text, max_len=77)
        active_tokens = tokenizer.decode_active_tokens(token_ids)
        if selected_token_idx >= len(active_tokens):
            selected_token_idx = max(0, len(active_tokens) - 1)

        # Determine steps per frame based on speed
        if speed_mode == 0:
            steps_this_frame = 1
            fps_target = 60
        elif speed_mode == 1:
            steps_this_frame = 3
            fps_target = 60
        else:
            steps_this_frame = 1
            fps_target = 30

        # ---- MODEL EXECUTION ----
        if has_model:
            for _ in range(steps_this_frame):
                if task_success_status is None:
                    final_dist = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
                    if final_dist < 0.040 and sim.target_cube_pos[2] <= 0.025 and not sim.gripper_closed:
                        task_success_status = True
                        total_episodes += 1
                        successful_episodes += 1
                        success_banner_timer = 20 if speed_mode == 1 else 45
                        rate = (successful_episodes / total_episodes) * 100.0
                        print(f"[TRIAL #{total_episodes:03d}] SUCCESS! ({successful_episodes}/{total_episodes} = {rate:.1f}%)", flush=True)
                    elif episode_total_ticks >= MAX_EPISODE_TICKS:
                        task_success_status = False
                        total_episodes += 1
                        failed_episodes += 1
                        success_banner_timer = 20 if speed_mode == 1 else 45
                        rate = (successful_episodes / total_episodes) * 100.0
                        print(f"[TRIAL #{total_episodes:03d}] FAILED ({successful_episodes}/{total_episodes} = {rate:.1f}%)", flush=True)

                if task_success_status is not None:
                    if success_banner_timer > 0:
                        success_banner_timer -= 1
                    else:
                        obs = sim.reset(random_scene=True, num_distractors=2, action_type=action_type)
                        current_trajectory = None
                        task_success_status = None
                        episode_total_ticks = 0
                else:
                    episode_total_ticks += 1
                    need_replan = (current_trajectory is None) or (traj_step >= len(current_trajectory))

                    if need_replan:
                        with torch.no_grad():
                            img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(device)
                            tokens_t = token_ids.unsqueeze(0).to(device)
                            proprio_t = torch.tensor(obs["proprio"], dtype=torch.float32).unsqueeze(0).to(device)

                            # Get attention weights from backbone for visualization
                            vis_patches, _ = model.backbone.vlm.encode_vision(img_t)
                            text_feats = model.backbone.vlm.encode_text(tokens_t)

                            # Compute cross-attention weights
                            base_weights = torch.einsum('bld,bpd->blp', text_feats.float(), vis_patches.float())
                            cached_cross_weights = torch.softmax(base_weights * 5.0, dim=-1)[0].cpu().numpy()

                            sample_steps = 10 if speed_mode == 1 else 15
                            current_trajectory = model.sample(img_t, tokens_t, proprio=proprio_t, num_steps=sample_steps).squeeze(0).cpu().numpy()
                            traj_step = 0

                    if current_trajectory is not None and traj_step < len(current_trajectory):
                        target_point = current_trajectory[traj_step]
                        diff_xyz = target_point[:3] - sim.ee_pos[:3]
                        dist_to_pt = np.linalg.norm(diff_xyz)

                        advance_threshold = 0.012 if speed_mode == 1 else 0.008
                        if dist_to_pt < advance_threshold:
                            traj_step += 2 if speed_mode == 1 else 1
                            if traj_step < len(current_trajectory):
                                target_point = current_trajectory[traj_step]
                                diff_xyz = target_point[:3] - sim.ee_pos[:3]
                        else:
                            traj_step += 2 if speed_mode == 1 else 1

                        # Pure model-predicted gripper
                        grip_cmd = 1.0 if target_point[3] > 0.5 else 0.0

                        max_step_rate = 0.015 if speed_mode == 1 else 0.010
                        delta_action = np.array([diff_xyz[0], diff_xyz[1], diff_xyz[2], 0.0, grip_cmd], dtype=np.float32)
                        obs, _ = sim.step_delta(delta_action, max_step=max_step_rate)

        # If no model, still compute attention for static visualization
        if cached_cross_weights is None:
            with torch.no_grad():
                img_t = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(device)
                tokens_t = token_ids.unsqueeze(0).to(device)
                vis_patches, _ = model.backbone.vlm.encode_vision(img_t)
                text_feats = model.backbone.vlm.encode_text(tokens_t)
                base_weights = torch.einsum('bld,bpd->blp', text_feats.float(), vis_patches.float())
                cached_cross_weights = torch.softmax(base_weights * 5.0, dim=-1)[0].cpu().numpy()

        all_tokens_weights = cached_cross_weights  # [77, 49]
        cur_grid = all_tokens_weights[selected_token_idx].reshape(7, 7)

        # =================================================================
        # RENDERING
        # =================================================================
        screen.fill((16, 18, 24))

        # Title Bar
        screen.blit(font_title.render("SMOLVLA LIVE VISUALIZER — ATTENTION + MODEL EXECUTION", True, (240, 245, 255)), (25, 10))
        speed_str = SPEED_NAMES[speed_mode]
        sub_title = f'Task: "{prompt_text}" | Patches: 49 (7x7) | Tokens: {len(active_tokens)} | Speed: {speed_str}'
        screen.blit(font_sm.render(sub_title, True, (130, 140, 160)), (25, 31))

        # =============================================================
        # PANEL 1: Language Token Selector (Left: 220px) — ALL tokens, scrollable
        # =============================================================
        panel1_rect = pygame.Rect(20, 50, 220, 530)
        pygame.draw.rect(screen, (24, 27, 36), panel1_rect, border_radius=6)
        screen.blit(font_bold.render("1. LANGUAGE TOKENS", True, (255, 200, 90)), (30, 60))
        screen.blit(font_xs.render(f"[UP/DOWN] Select | {len(active_tokens)} active", True, (140, 150, 170)), (30, 78))

        # Clip rendering to panel
        token_area_y = 96
        token_area_h = 480
        max_visible = token_area_h // 22
        max_scroll = max(0, len(active_tokens) - max_visible)
        token_scroll = min(token_scroll, max_scroll)

        # Auto-scroll to keep selected token visible
        if selected_token_idx < token_scroll:
            token_scroll = selected_token_idx
        elif selected_token_idx >= token_scroll + max_visible:
            token_scroll = selected_token_idx - max_visible + 1

        y_off = token_area_y
        for draw_idx in range(token_scroll, min(token_scroll + max_visible, len(active_tokens))):
            tok_str = active_tokens[draw_idx]
            is_selected = (draw_idx == selected_token_idx)
            btn_rect = pygame.Rect(28, y_off, 204, 20)
            if is_selected:
                pygame.draw.rect(screen, (45, 85, 170), btn_rect, border_radius=3)
                pygame.draw.rect(screen, (100, 180, 255), btn_rect, 1, border_radius=3)
                text_col = (255, 255, 255)
                tag = ">"
            else:
                pygame.draw.rect(screen, (30, 34, 46), btn_rect, border_radius=3)
                text_col = (170, 180, 200)
                tag = " "
            screen.blit(font_xs.render(f"{tag}[{draw_idx:02d}] {tok_str}", True, text_col), (34, y_off + 4))
            y_off += 22

        # Scroll indicator
        if len(active_tokens) > max_visible:
            bar_h = max(20, int(token_area_h * max_visible / len(active_tokens)))
            bar_y = token_area_y + int((token_area_h - bar_h) * token_scroll / max(1, max_scroll))
            pygame.draw.rect(screen, (60, 70, 90), (232, bar_y, 4, bar_h), border_radius=2)

        # =============================================================
        # PANEL 2: Focused 7x7 Patch Heatmap for Selected Token
        # =============================================================
        pygame.draw.rect(screen, (24, 27, 36), (250, 50, 300, 530), border_radius=6)
        sel_word = active_tokens[selected_token_idx] if selected_token_idx < len(active_tokens) else "N/A"
        screen.blit(font_bold.render(f"2. ATTN: '{sel_word.upper()}'", True, (100, 220, 255)), (260, 60))
        screen.blit(font_xs.render("7x7 ViT-B/32 Patch Grid", True, (140, 150, 170)), (260, 78))

        g_min, g_max = cur_grid.min(), cur_grid.max()
        cur_grid_norm = (cur_grid - g_min) / (g_max - g_min + 1e-6) if g_max > g_min else cur_grid

        cell_sz = 34
        start_x, start_y = 277, 100
        for r in range(7):
            for c in range(7):
                val = cur_grid_norm[r, c]
                col = (int(25 + val * 230), int(35 + val * 170), int(55 + (1 - val) * 45))
                pygame.draw.rect(screen, col, (start_x + c * cell_sz, start_y + r * cell_sz, cell_sz - 2, cell_sz - 2))
        pygame.draw.rect(screen, (80, 95, 125), (start_x - 2, start_y - 2, 7 * cell_sz + 1, 7 * cell_sz + 1), 1)

        # Overlay camera image on heatmap (small inset)
        img_hwc = (np.transpose(obs["image"], (1, 2, 0)) * 255).astype(np.uint8)
        cam_mini = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (7 * cell_sz - 3, 7 * cell_sz - 3))
        cam_mini.set_alpha(60)
        screen.blit(cam_mini, (start_x, start_y))

        screen.blit(font_xs.render(f"Token [{selected_token_idx}] '{sel_word}'", True, (180, 190, 210)), (265, start_y + 7 * cell_sz + 10))
        screen.blit(font_xs.render(f"Range: [{g_min:.4f} .. {g_max:.4f}]", True, (180, 190, 210)), (265, start_y + 7 * cell_sz + 28))

        # =============================================================
        # PANEL 3: ALL Token Heatmaps Grid — show all active tokens
        # =============================================================
        panel3_x, panel3_y = 560, 50
        panel3_w, panel3_h = 370, 530
        pygame.draw.rect(screen, (24, 27, 36), (panel3_x, panel3_y, panel3_w, panel3_h), border_radius=6)
        screen.blit(font_bold.render("3. ALL TOKEN HEATMAPS", True, (255, 160, 100)), (panel3_x + 10, 60))
        screen.blit(font_xs.render(f"All {len(active_tokens)} active tokens", True, (140, 150, 170)), (panel3_x + 10, 78))

        # Dynamic grid: fit as many as possible
        mini_cell_sz = 7  # 7px * 7 = 49px per token heatmap
        mini_pad_x = 8
        mini_pad_y = 18
        mini_total_w = 7 * mini_cell_sz + mini_pad_x
        mini_total_h = 7 * mini_cell_sz + mini_pad_y + 2
        mini_cols = max(1, (panel3_w - 20) // mini_total_w)
        mini_rows = max(1, (panel3_h - 50) // mini_total_h)
        max_display = mini_cols * mini_rows

        m_start_x = panel3_x + 10
        m_start_y = panel3_y + 42

        for t_idx in range(min(len(active_tokens), max_display)):
            col_i = t_idx % mini_cols
            row_i = t_idx // mini_cols
            tx = m_start_x + col_i * mini_total_w
            ty = m_start_y + row_i * mini_total_h

            if ty + mini_total_h > panel3_y + panel3_h:
                break

            tok_str = active_tokens[t_idx]
            is_cur = (t_idx == selected_token_idx)
            header_col = (100, 220, 255) if is_cur else (150, 160, 180)
            label = f"{t_idx}:{tok_str[:6]}"
            screen.blit(font_xs.render(label, True, header_col), (tx, ty))

            t_grid = all_tokens_weights[t_idx].reshape(7, 7)
            tg_min, tg_max = t_grid.min(), t_grid.max()
            tg_norm = (t_grid - tg_min) / (tg_max - tg_min + 1e-6) if tg_max > tg_min else t_grid

            for r in range(7):
                for c in range(7):
                    v = tg_norm[r, c]
                    c_col = (int(20 + v * 235), int(30 + v * 170), int(50 + (1 - v) * 40))
                    pygame.draw.rect(screen, c_col, (tx + c * mini_cell_sz, ty + 13 + r * mini_cell_sz, mini_cell_sz - 1, mini_cell_sz - 1))

            border_col = (100, 220, 255) if is_cur else (50, 60, 80)
            pygame.draw.rect(screen, border_col, (tx - 1, ty + 12, 7 * mini_cell_sz + 1, 7 * mini_cell_sz + 1), 1)

        # =============================================================
        # PANEL 4: Sensor Inputs + Live Stats (Right Column)
        # =============================================================
        p4_x = 940
        pygame.draw.rect(screen, (24, 27, 36), (p4_x, 50, 240, 530), border_radius=6)
        screen.blit(font_bold.render("4. SENSOR + LIVE STATS", True, (120, 240, 150)), (p4_x + 10, 60))

        # Camera feed
        cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.transpose(img_hwc, (1, 0, 2))), (120, 120))
        screen.blit(cam_surf, (p4_x + 60, 85))
        pygame.draw.rect(screen, (90, 110, 140), (p4_x + 59, 84, 122, 122), 1)
        screen.blit(font_xs.render("64x64 Camera Feed", True, (150, 160, 180)), (p4_x + 60, 210))

        # Proprioception
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 10, 230, 220, 100), border_radius=4)
        screen.blit(font_bold.render("Proprioception [5D]:", True, (255, 210, 110)), (p4_x + 20, 238))
        p = obs["proprio"]
        screen.blit(font_xs.render(f"X: {p[0]:+.4f}m  Y: {p[1]:+.4f}m", True, (200, 210, 230)), (p4_x + 20, 258))
        screen.blit(font_xs.render(f"Z: {p[2]:+.4f}m  Yaw: {p[3]:+.3f}", True, (200, 210, 230)), (p4_x + 20, 276))
        grip_str = "CLOSED" if sim.gripper_closed else "OPEN"
        grip_col = (255, 120, 120) if sim.gripper_closed else (100, 220, 255)
        screen.blit(font_xs.render(f"Gripper: {p[4]:.1f} ({grip_str})", True, grip_col), (p4_x + 20, 294))

        # Live trial stats
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 10, 340, 220, 100), border_radius=4)
        screen.blit(font_bold.render("Live Trial Stats:", True, (255, 180, 100)), (p4_x + 20, 348))
        win_pct = (successful_episodes / total_episodes * 100.0) if total_episodes > 0 else 0.0
        screen.blit(font_xs.render(f"Trial:   #{total_episodes + 1}", True, (200, 210, 230)), (p4_x + 20, 370))
        screen.blit(font_xs.render(f"Success: {successful_episodes}", True, (80, 240, 120)), (p4_x + 20, 388))
        screen.blit(font_xs.render(f"Failed:  {failed_episodes}", True, (255, 100, 100)), (p4_x + 20, 406))
        screen.blit(font_xs.render(f"Rate:    {win_pct:.1f}%", True, (200, 210, 230)), (p4_x + 130, 388))
        screen.blit(font_xs.render(f"Tick:    {episode_total_ticks}/{MAX_EPISODE_TICKS}", True, (160, 170, 190)), (p4_x + 130, 406))

        # Kinematic side view
        pygame.draw.rect(screen, (30, 34, 46), (p4_x + 10, 450, 220, 120), border_radius=4)
        screen.blit(font_bold.render("Kinematic View:", True, (180, 200, 240)), (p4_x + 20, 458))

        j1, j2, j3, j4 = sim.kin.inverse(sim.ee_pos[0], sim.ee_pos[1], sim.ee_pos[2], sim.ee_pos[3])
        r0, z0 = 0.08, 0.0
        r1, z1 = 0.08, sim.kin.L1
        r2 = r1 + sim.kin.L2 * np.cos(j2)
        z2 = z1 + sim.kin.L2 * np.sin(j2)
        r3 = r2 + sim.kin.L3 * np.cos(j2 + j3)
        z3 = z2 + sim.kin.L3 * np.sin(j2 + j3)
        r4 = np.sqrt(sim.ee_pos[0]**2 + sim.ee_pos[1]**2)
        z4 = sim.ee_pos[2]

        def w_side(r, z):
            sx = int(p4_x + 35 + ((r - 0.05) / 0.30) * 160)
            sy = int(555 - (z / 0.22) * 80)
            return sx, sy

        pts = [w_side(r, z) for r, z in [(r0, z0), (r1, z1), (r2, z2), (r3, z3), (r4, z4)]]
        for i in range(len(pts) - 1):
            pygame.draw.line(screen, (160, 180, 220), pts[i], pts[i+1], 2)
        for pt in pts[:-1]:
            pygame.draw.circle(screen, (255, 160, 40), pt, 3)
        grip_c = (255, 80, 80) if sim.gripper_closed else (80, 220, 255)
        pygame.draw.circle(screen, grip_c, pts[-1], 5)

        # Success/fail banner
        if task_success_status is not None and success_banner_timer > 0:
            if task_success_status:
                banner_text = f"SUCCESS! ({successful_episodes}/{total_episodes})"
                banner_col = (50, 240, 100)
            else:
                banner_text = f"TIMEOUT ({successful_episodes}/{total_episodes})"
                banner_col = (255, 90, 90)
            pygame.draw.rect(screen, (30, 30, 30), (p4_x + 10, 580, 220, 25), border_radius=4)
            screen.blit(font_bold.render(banner_text, True, banner_col), (p4_x + 30, 585))

        # =============================================================
        # CONTROLS HUD (Bottom)
        # =============================================================
        pygame.draw.rect(screen, (20, 23, 30), (20, 590, 910, 120), border_radius=6)
        screen.blit(font_bold.render("CONTROLS:", True, (120, 210, 255)), (35, 598))
        screen.blit(font.render(f"[F] Speed: {speed_str}  |  [R] Randomize  |  [1] Pick&Place  |  [2] Push  |  [UP/DOWN] Token  |  [Scroll] Token List", True, (200, 210, 230)), (130, 598))

        # Model status indicator
        model_status = "MODEL: ACTIVE" if has_model else "MODEL: NOT LOADED (visualization only)"
        model_col = (80, 240, 120) if has_model else (255, 150, 50)
        screen.blit(font_bold.render(model_status, True, model_col), (35, 620))

        pygame.display.flip()
        clock.tick(fps_target)

    pygame.quit()

    print("\n" + "=" * 55, flush=True)
    print("       SMOLVLA VISUALIZER SESSION SUMMARY", flush=True)
    print("=" * 55, flush=True)
    print(f"  Total Trials     : {total_episodes}", flush=True)
    print(f"  Successful       : {successful_episodes}", flush=True)
    print(f"  Failed           : {failed_episodes}", flush=True)
    if total_episodes > 0:
        print(f"  Success Rate     : {(successful_episodes / total_episodes) * 100:.1f}%", flush=True)
    print("=" * 55 + "\n", flush=True)

if __name__ == "__main__":
    visualize_all_smolvla_inputs()
