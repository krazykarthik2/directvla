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
    - Modalities:
        * RGB Camera Image: [3, 64, 64] float32 in [0, 1]
        * Proprioception Robot State: [ee_x, ee_y, ee_z, ee_yaw, gripper_status] (5 dims)
        * Language Instruction: prompt string
    """
    def __init__(self):
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
        self.action_type = "pick_place"
        
        # Pygame surface for camera rendering
        pygame.init()
        self.cam_surface = pygame.Surface((64, 64))
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

    def reset(self, random_scene=True, prompt=None, num_distractors=2, action_type=None):
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False

        if action_type is not None:
            self.action_type = action_type
        else:
            self.action_type = str(np.random.choice(["pick_place", "push"]))

        if random_scene:
            cube_colors = ["red", "blue", "yellow", "purple"]
            plat_colors = ["green", "cyan", "orange"]

            self.target_color = str(np.random.choice(cube_colors))
            self.target_plat_color = str(np.random.choice(plat_colors))

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

            # Distractor Cubes
            self.distractor_cubes = []
            avail_cube_colors = [c for c in cube_colors if c != self.target_color]
            np.random.shuffle(avail_cube_colors)
            for i in range(min(num_distractors, len(avail_cube_colors))):
                d_pos = get_non_overlapping_pos()
                self.distractor_cubes.append((avail_cube_colors[i], d_pos))

            # Distractor Platforms
            self.distractor_platforms = []
            avail_plat_colors = [c for c in plat_colors if c != self.target_plat_color]
            if avail_plat_colors:
                dp_pos = get_non_overlapping_pos(min_dist=0.08)
                dp_pos[2] = 0.005
                self.distractor_platforms.append((avail_plat_colors[0], dp_pos))

            if prompt is not None:
                self.instruction = prompt
            else:
                if self.action_type == "pick_place":
                    self.instruction = f"pick up the {self.target_color} cube and place it on the {self.target_plat_color} platform"
                else:
                    self.instruction = f"push the {self.target_color} cube towards the {self.target_plat_color} platform"
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
        surf = self.cam_surface
        surf.fill((30, 32, 40))

        def to_cam_px(x, y):
            px = int(32 + (y / 0.28) * 28)
            py = int(58 - ((x - 0.10) / 0.25) * 52)
            return px, py

        # Physical-to-pixel scaling: y-axis ~100 px/m, x-axis ~208 px/m
        # Platform (0.035m): ~7px wide x ~7px tall
        # Cube (0.022m): ~5px wide x ~5px tall

        # Platforms (flat landing pads)
        for plat_color, p_pos in self.distractor_platforms:
            gx, gy = to_cam_px(p_pos[0], p_pos[1])
            col = COLOR_PALETTE.get(plat_color, (40, 210, 80))
            pygame.draw.rect(surf, col, (gx - 3, gy - 3, 7, 7), border_radius=1)

        tgx, tgy = to_cam_px(self.target_platform_pos[0], self.target_platform_pos[1])
        t_col = COLOR_PALETTE.get(self.target_plat_color, (40, 210, 80))
        pygame.draw.rect(surf, t_col, (tgx - 3, tgy - 3, 7, 7), border_radius=1)
        pygame.draw.rect(surf, (255, 255, 255), (tgx - 3, tgy - 3, 7, 7), 1, border_radius=1)

        # Cubes (small objects)
        for cube_color, c_pos in self.distractor_cubes:
            cx, cy = to_cam_px(c_pos[0], c_pos[1])
            col = COLOR_PALETTE.get(cube_color, (45, 120, 240))
            pygame.draw.rect(surf, col, (cx - 2, cy - 2, 5, 5))

        tcx, tcy = to_cam_px(self.target_cube_pos[0], self.target_cube_pos[1])
        tc_col = COLOR_PALETTE.get(self.target_color, (240, 45, 45))
        pygame.draw.rect(surf, tc_col, (tcx - 2, tcy - 2, 5, 5))

        # Base & Linkage
        bx, by = to_cam_px(0.08, 0.0)
        pygame.draw.circle(surf, (90, 95, 115), (bx, by), 5)

        ex, ey = to_cam_px(self.ee_pos[0], self.ee_pos[1])
        pygame.draw.line(surf, (170, 175, 195), (bx, by), (ex, ey), 2)

        grip_color = (255, 90, 90) if self.gripper_closed else (90, 200, 255)
        pygame.draw.circle(surf, grip_color, (ex, ey), 3)

        rgb_hwc = pygame.surfarray.array3d(surf)
        rgb_chw = np.transpose(rgb_hwc, (2, 1, 0)).astype(np.float32) / 255.0
        return rgb_chw

    def get_proprioception(self):
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3], grip_val
        ], dtype=np.float32)

    def get_vla_observation(self):
        return {
            "image": self.render_camera_rgb(),
            "proprio": self.get_proprioception(),
            "prompt": self.instruction
        }

    def get_observation(self):
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3],
            grip_val,
            self.target_cube_pos[0], self.target_cube_pos[1], self.target_cube_pos[2],
            self.target_platform_pos[0], self.target_platform_pos[1], self.target_platform_pos[2]
        ], dtype=np.float32)

    def step_delta(self, delta_action, max_step=0.008):
        dx = np.clip(delta_action[0], -max_step, max_step)
        dy = np.clip(delta_action[1], -max_step, max_step)
        dz = np.clip(delta_action[2], -max_step, max_step)
        dyaw = np.clip(delta_action[3], -0.1, 0.1)

        new_x = np.clip(self.ee_pos[0] + dx, 0.13, 0.32)
        new_y = np.clip(self.ee_pos[1] + dy, -0.25, 0.25)
        new_z = np.clip(self.ee_pos[2] + dz, 0.012, 0.22)
        new_yaw = self.ee_pos[3] + dyaw

        target_ee = [new_x, new_y, new_z, new_yaw, delta_action[4]]
        return self.step(target_ee)

    def step(self, target_ee):
        self.ee_pos = np.array(target_ee[:4], dtype=np.float32)
        self.gripper_closed = bool(target_ee[4] > 0.5)

        ee_xyz = self.ee_pos[:3]
        dist_to_cube = np.linalg.norm(ee_xyz - self.target_cube_pos)

        # Grasping logic
        if self.gripper_closed:
            if dist_to_cube < 0.035:
                self.grasped = True
        else:
            self.grasped = False

        if self.grasped:
            self.target_cube_pos = ee_xyz.copy()
            self.target_cube_pos[2] = max(self.target_cube_pos[2] - 0.015, 0.011)
        else:
            # Pushing logic when gripper is not holding
            if dist_to_cube < 0.025:
                push_vec = self.target_cube_pos[:2] - ee_xyz[:2]
                n_dist = np.linalg.norm(push_vec)
                if n_dist > 1e-4:
                    self.target_cube_pos[:2] += (push_vec / n_dist) * 0.005

            if self.target_cube_pos[2] > 0.011:
                self.target_cube_pos[2] = max(self.target_cube_pos[2] - 0.01, 0.011)

        obs = self.get_vla_observation()
        dist_to_goal = np.linalg.norm(self.target_cube_pos[:2] - self.target_platform_pos[:2])
        is_success = bool(dist_to_goal < 0.04 and self.target_cube_pos[2] <= 0.025 and not self.gripper_closed)

        return obs, is_success
