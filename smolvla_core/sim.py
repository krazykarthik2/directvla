import os
import numpy as np
import pygame

class DobotKinematics:
    """Exact Analytical Forward and Inverse Kinematics for Dobot Magician 4-DOF."""
    def __init__(self):
        self.L1 = 0.138    # Base to Rear Arm joint height
        self.L2 = 0.135    # Rear arm length
        self.L3 = 0.147    # Forearm length
        self.L4 = 0.060    # Tool end-effector offset

    def forward(self, j1, j2, j3, j4):
        r = self.L2 * np.cos(j2) + self.L3 * np.cos(j2 + j3) + self.L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = self.L1 + self.L2 * np.sin(j2) + self.L3 * np.sin(j2 + j3)
        yaw = j1 + j4
        return np.array([x, y, z, yaw], dtype=np.float32)

    def inverse(self, x, y, z, yaw=0.0):
        j1 = np.arctan2(y, x)
        r_total = np.sqrt(x**2 + y**2)
        r = r_total - self.L4
        dz = z - self.L1

        D_sq = r**2 + dz**2
        D = np.sqrt(D_sq)

        cos_j3 = (D_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_j3 = np.clip(cos_j3, -1.0, 1.0)
        j3 = -np.arccos(cos_j3)

        alpha = np.arctan2(dz, r)
        beta = np.arctan2(self.L3 * np.sin(-j3), self.L2 + self.L3 * np.cos(-j3))
        j2 = alpha + beta

        j4 = yaw - j1
        return np.array([j1, j2, j3, j4], dtype=np.float32)

COLOR_PALETTE = {
    "red": (240, 45, 45),
    "blue": (45, 120, 240),
    "yellow": (240, 220, 45),
    "green": (40, 210, 80),
    "purple": (180, 50, 230),
    "orange": (245, 140, 30),
    "cyan": (35, 220, 225)
}

class DobotPickPlaceSim:
    """
    Simulation Environment for Vision-Language-Action (VLA) Learning with Visual Distractors:
    - Multi-object visual scene (Target Object + Distractor Objects + Target Platform + Distractor Platforms)
    - Instruction-conditioned tasks (e.g. "pick red cube and place on green platform", "pick blue cube...", etc.)
    - Modalities:
        * RGB Camera Image: [3, 64, 64] float32 in [0, 1]
        * Proprioception Robot State: [ee_x, ee_y, ee_z, ee_yaw, gripper_status] (5 dims)
        * Language Instruction: prompt string
    """
    def __init__(self, render_display=False):
        self.kin = DobotKinematics()
        self.cube_size = 0.022
        self.gripper_closed = False
        self.grasped = False
        
        # State
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.target_color = "red"
        self.target_cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
        self.target_plat_color = "green"
        self.target_platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)
        
        # Distractors
        self.distractor_cubes = []      # list of (color_name, np.array([x, y, z]))
        self.distractor_platforms = []  # list of (color_name, np.array([x, y, z]))

        self.instruction = f"pick up the {self.target_color} cube and place it on the {self.target_plat_color} platform"
        
        # Pygame surface for camera rendering
        pygame.init()
        self.cam_surface = pygame.Surface((64, 64))
        
        # Visualization screen (Top View + Side View)
        self.render_display = render_display
        self.window = None
        self.clock = None
        if self.render_display:
            self.window = pygame.display.set_mode((800, 400))
            pygame.display.set_caption("Dobot 4-DOF VLA Simulation (Top View & Side View)")
            self.clock = pygame.time.Clock()

        self.reset()

    # Aliases for backward compatibility
    @property
    def cube_pos(self):
        return self.target_cube_pos

    @cube_pos.setter
    def cube_pos(self, val):
        self.target_cube_pos = np.array(val, dtype=np.float32)

    @property
    def platform_pos(self):
        return self.target_platform_pos

    @platform_pos.setter
    def platform_pos(self, val):
        self.target_platform_pos = np.array(val, dtype=np.float32)

    def reset(self, random_scene=True, prompt=None, num_distractors=2):
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False

        if random_scene:
            # 1. Randomize Target Object and Platform Colors
            cube_colors = ["red", "blue", "yellow", "purple"]
            plat_colors = ["green", "cyan", "orange"]

            self.target_color = str(np.random.choice(cube_colors))
            self.target_plat_color = str(np.random.choice(plat_colors))

            # 2. Spawn Positions ensuring non-overlapping clutter
            occupied_positions = []

            def get_non_overlapping_pos(min_dist=0.065):
                for _ in range(100):
                    angle = np.random.uniform(-0.75, 0.75)
                    dist = np.random.uniform(0.17, 0.28)
                    pos = np.array([dist * np.cos(angle), dist * np.sin(angle), 0.011], dtype=np.float32)
                    if all(np.linalg.norm(pos[:2] - p[:2]) > min_dist for p in occupied_positions):
                        occupied_positions.append(pos)
                        return pos
                return np.array([0.22, 0.10, 0.011], dtype=np.float32)

            self.target_cube_pos = get_non_overlapping_pos()
            plat_pos = get_non_overlapping_pos(min_dist=0.08)
            plat_pos[2] = 0.005
            self.target_platform_pos = plat_pos

            # 3. Spawn Distractor Objects
            self.distractor_cubes = []
            avail_cube_colors = [c for c in cube_colors if c != self.target_color]
            np.random.shuffle(avail_cube_colors)
            for i in range(min(num_distractors, len(avail_cube_colors))):
                d_pos = get_non_overlapping_pos()
                self.distractor_cubes.append((avail_cube_colors[i], d_pos))

            # 4. Spawn Distractor Platform
            self.distractor_platforms = []
            avail_plat_colors = [c for c in plat_colors if c != self.target_plat_color]
            if avail_plat_colors:
                dp_pos = get_non_overlapping_pos(min_dist=0.08)
                dp_pos[2] = 0.005
                self.distractor_platforms.append((avail_plat_colors[0], dp_pos))

            # 5. Language Instruction Prompt
            if prompt is not None:
                self.instruction = prompt
            else:
                prompt_templates = [
                    f"pick up the {self.target_color} cube and place it on the {self.target_plat_color} platform",
                    f"grasp the {self.target_color} cube and move to {self.target_plat_color} box",
                    f"transfer {self.target_color} block onto {self.target_plat_color} platform",
                    f"pick {self.target_color} object and place on {self.target_plat_color} target"
                ]
                self.instruction = str(np.random.choice(prompt_templates))
        else:
            self.target_color = "red"
            self.target_plat_color = "green"
            self.target_cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
            self.target_platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)
            self.distractor_cubes = [("blue", np.array([0.20, -0.05, 0.011], dtype=np.float32))]
            self.distractor_platforms = [("cyan", np.array([0.18, 0.18, 0.005], dtype=np.float32))]
            self.instruction = "pick up the red cube and place it on the green platform"

        return self.get_vla_observation()

    def render_camera_rgb(self):
        """
        Renders an overhead RGB camera view with clutter/distractors as [3, 64, 64] float32 in [0, 1].
        """
        surf = self.cam_surface
        surf.fill((30, 32, 40)) # Dark workspace table mat

        def to_cam_px(x, y):
            px = int(32 + (y / 0.28) * 28)
            py = int(58 - ((x - 0.10) / 0.25) * 52)
            return px, py

        # 1. Draw Distractor Platforms
        for plat_color, p_pos in self.distractor_platforms:
            gx, gy = to_cam_px(p_pos[0], p_pos[1])
            col = COLOR_PALETTE.get(plat_color, (40, 210, 80))
            pygame.draw.rect(surf, col, (gx - 5, gy - 5, 10, 10), border_radius=1)

        # 2. Draw Target Platform
        tgx, tgy = to_cam_px(self.target_platform_pos[0], self.target_platform_pos[1])
        t_col = COLOR_PALETTE.get(self.target_plat_color, (40, 210, 80))
        pygame.draw.rect(surf, t_col, (tgx - 5, tgy - 5, 10, 10), border_radius=1)

        # 3. Draw Distractor Cubes
        for cube_color, c_pos in self.distractor_cubes:
            cx, cy = to_cam_px(c_pos[0], c_pos[1])
            col = COLOR_PALETTE.get(cube_color, (45, 120, 240))
            pygame.draw.rect(surf, col, (cx - 3, cy - 3, 6, 6))

        # 4. Draw Target Cube
        tcx, tcy = to_cam_px(self.target_cube_pos[0], self.target_cube_pos[1])
        tc_col = COLOR_PALETTE.get(self.target_color, (240, 45, 45))
        pygame.draw.rect(surf, tc_col, (tcx - 3, tcy - 3, 6, 6))

        # 5. Draw Robot Base Origin
        bx, by = to_cam_px(0.08, 0.0)
        pygame.draw.circle(surf, (90, 95, 115), (bx, by), 5)

        # 6. Draw Robot Arm Linkage
        ex, ey = to_cam_px(self.ee_pos[0], self.ee_pos[1])
        pygame.draw.line(surf, (170, 175, 195), (bx, by), (ex, ey), 2)

        # 7. Draw Gripper End-Effector
        grip_color = (255, 90, 90) if self.gripper_closed else (90, 200, 255)
        pygame.draw.circle(surf, grip_color, (ex, ey), 3)

        rgb_hwc = pygame.surfarray.array3d(surf) # [64, 64, 3]
        rgb_chw = np.transpose(rgb_hwc, (2, 1, 0)).astype(np.float32) / 255.0
        return rgb_chw

    def render_viewer(self):
        """Renders overhead Top View (left) and Side View (right) for human monitoring."""
        if not self.render_display or self.window is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.window.fill((22, 24, 30))

        # Helper conversions
        # Top view (0 to 400 width, 0 to 400 height)
        # x range: [0.0, 0.35], y range: [-0.25, 0.25]
        def to_top_view(x, y):
            sx = int(200 + (y / 0.28) * 170)
            sy = int(360 - ((x - 0.05) / 0.32) * 330)
            return sx, sy

        # Side view (400 to 800 width, 0 to 400 height)
        # radial r range: [0.0, 0.35], z range: [0.0, 0.25]
        def to_side_view(r, z):
            sx = int(450 + (r / 0.35) * 300)
            sy = int(340 - (z / 0.25) * 280)
            return sx, sy

        # Divider and Backgrounds
        pygame.draw.rect(self.window, (30, 32, 40), (10, 10, 380, 380), border_radius=6)
        pygame.draw.rect(self.window, (30, 32, 40), (410, 10, 380, 380), border_radius=6)
        
        # Ground plane in side view
        pygame.draw.line(self.window, (70, 75, 90), (420, 340), (780, 340), 2)

        # 1. Platforms
        all_plats = [(self.target_plat_color, self.target_platform_pos)] + self.distractor_platforms
        for p_col, p_pos in all_plats:
            col = COLOR_PALETTE.get(p_col, (40, 210, 80))
            # Top
            tx, ty = to_top_view(p_pos[0], p_pos[1])
            pygame.draw.rect(self.window, col, (tx - 16, ty - 16, 32, 32), border_radius=3)
            # Side
            r = np.sqrt(p_pos[0]**2 + p_pos[1]**2)
            sx, sy = to_side_view(r, p_pos[2])
            pygame.draw.rect(self.window, col, (sx - 16, sy - 4, 32, 8), border_radius=2)

        # 2. Cubes
        all_cubes = [(self.target_color, self.target_cube_pos)] + self.distractor_cubes
        for c_col, c_pos in all_cubes:
            col = COLOR_PALETTE.get(c_col, (240, 45, 45))
            # Top
            cx, cy = to_top_view(c_pos[0], c_pos[1])
            pygame.draw.rect(self.window, col, (cx - 10, cy - 10, 20, 20), border_radius=2)
            # Side
            r = np.sqrt(c_pos[0]**2 + c_pos[1]**2)
            sx, sy = to_side_view(r, c_pos[2])
            pygame.draw.rect(self.window, col, (sx - 10, sy - 10, 20, 20), border_radius=2)

        # 3. Kinematics for Side View Joints: Base -> Shoulder -> Elbow -> Wrist -> EE
        j1, j2, j3, j4 = self.kin.inverse(self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3])
        p0 = (0.0, 0.0) # r=0, z=0
        p1 = (0.0, self.kin.L1)
        r_shoulder = self.kin.L2 * np.cos(j2)
        z_shoulder = self.kin.L1 + self.kin.L2 * np.sin(j2)
        p2 = (r_shoulder, z_shoulder)
        r_elbow = r_shoulder + self.kin.L3 * np.cos(j2 + j3)
        z_elbow = z_shoulder + self.kin.L3 * np.sin(j2 + j3)
        p3 = (r_elbow, z_elbow)
        p4 = (r_elbow + self.kin.L4, z_elbow) # EE

        # Draw Side View Arm
        pts_side = [to_side_view(*pt) for pt in [p0, p1, p2, p3, p4]]
        for i in range(len(pts_side) - 1):
            pygame.draw.line(self.window, (190, 200, 220), pts_side[i], pts_side[i+1], 4)
        for pt in pts_side:
            pygame.draw.circle(self.window, (250, 160, 40), pt, 5)

        # Draw Top View Arm
        bx, by = to_top_view(0.0, 0.0)
        ex, ey = to_top_view(self.ee_pos[0], self.ee_pos[1])
        pygame.draw.line(self.window, (190, 200, 220), (bx, by), (ex, ey), 4)
        pygame.draw.circle(self.window, (90, 95, 115), (bx, by), 8)
        
        grip_color = (255, 90, 90) if self.gripper_closed else (90, 200, 255)
        pygame.draw.circle(self.window, grip_color, (ex, ey), 8)
        pygame.draw.circle(self.window, grip_color, pts_side[-1], 8)

        # Text Overlay
        font = pygame.font.SysFont("Arial", 14)
        lbl_top = font.render("TOP VIEW (Model Camera Focus)", True, (180, 190, 210))
        lbl_side = font.render("SIDE VIEW (Analytical 4-DOF Dobot IK)", True, (180, 190, 210))
        lbl_task = font.render(f"Instruction: {self.instruction}", True, (240, 240, 240))
        self.window.blit(lbl_top, (20, 20))
        self.window.blit(lbl_side, (420, 20))
        self.window.blit(lbl_task, (20, 365))

        pygame.display.flip()
        self.clock.tick(30)

    def get_proprioception(self):
        """Returns 5-dim robot state: [x, y, z, yaw, grip_state]"""
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3], grip_val
        ], dtype=np.float32)

    def get_vla_observation(self):
        """Returns complete Vision-Language-Action observation dictionary."""
        return {
            "image": self.render_camera_rgb(),
            "proprio": self.get_proprioception(),
            "prompt": self.instruction
        }

    def get_observation(self):
        """Legacy observation format."""
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3],
            grip_val,
            self.target_cube_pos[0], self.target_cube_pos[1], self.target_cube_pos[2],
            self.target_platform_pos[0], self.target_platform_pos[1], self.target_platform_pos[2]
        ], dtype=np.float32)

    def step_delta(self, delta_action, max_step=0.006):
        """Applies bounded velocity delta displacement."""
        dx = np.clip(delta_action[0], -max_step, max_step)
        dy = np.clip(delta_action[1], -max_step, max_step)
        dz = np.clip(delta_action[2], -max_step, max_step)
        dyaw = np.clip(delta_action[3], -0.1, 0.1)

        new_x = np.clip(self.ee_pos[0] + dx, 0.14, 0.30)
        new_y = np.clip(self.ee_pos[1] + dy, -0.22, 0.22)
        new_z = np.clip(self.ee_pos[2] + dz, 0.012, 0.20)
        new_yaw = self.ee_pos[3] + dyaw

        target_ee = [new_x, new_y, new_z, new_yaw, delta_action[4]]
        return self.step(target_ee)

    def step(self, target_ee):
        self.ee_pos = np.array(target_ee[:4], dtype=np.float32)
        self.gripper_closed = bool(target_ee[4] > 0.5)

        ee_xyz = self.ee_pos[:3]
        dist_to_cube = np.linalg.norm(ee_xyz - self.target_cube_pos)

        # Grasping physics for target object
        if self.gripper_closed:
            if dist_to_cube < 0.032:
                self.grasped = True
        else:
            self.grasped = False

        if self.grasped:
            self.target_cube_pos = ee_xyz.copy()
            self.target_cube_pos[2] = max(self.target_cube_pos[2] - 0.015, 0.011)
        else:
            if self.target_cube_pos[2] > 0.011:
                self.target_cube_pos[2] = max(self.target_cube_pos[2] - 0.01, 0.011)

        if self.render_display:
            self.render_viewer()

        obs = self.get_vla_observation()
        
        dist_to_goal = np.linalg.norm(self.target_cube_pos[:2] - self.target_platform_pos[:2])
        is_success = bool(dist_to_goal < 0.04 and self.target_cube_pos[2] <= 0.025 and not self.gripper_closed)

        return obs, is_success
