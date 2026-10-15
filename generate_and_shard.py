#!/usr/bin/env python3
"""
PHASE 1: Generate + Shard Demonstrations in One Pass.
Generates high-quality pick-and-place demonstrations with:
  - Intermediate camera renders at EVERY trajectory step (cube follows EE when grasped)
  - Full joint/proprioception recordings per timestep
  - Wallclock-sharded .npz output (50 episodes per shard)
  - Also imports real hardware recordings from saved_learnings/*.json

Every shard lands on disk as shard_YYYYMMDD_HHMMSS_XXXX.npz.
Phase 2 (extract_clip_embeddings.py) can run IN PARALLEL picking these up.
"""

import os
import sys
import glob
import json
import math
import argparse
import datetime
import numpy as np
from tqdm import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

from vla_camera import SyntheticDeskRenderer, COLOR_PALETTE
from vla_robot_env import DobotKinematics
from vla_dataset import WallclockDatasetManager


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





def generate_demos(num_demos, mgr, renderer, desk_tone="dark_mat"):
    """
    Generate verified pick-and-place demos with intermediate camera renders.
    The cube physically follows the EE when grasped - every frame is accurate.
    """
    all_colors = ["red", "green", "blue", "yellow", "purple", "cyan", "orange"]
    nouns = ["cube", "thing", "block", "item", "object", "piece", "one"]
    dest_nouns = ["platform", "area", "zone", "target", "spot", "box", "plate", ""]

    kin = DobotKinematics()
    completed = 0
    rejected = 0
    pbar = tqdm(total=num_demos, desc="[Phase1] Generating+Sharding")

    while completed < num_demos:
        # Equal color ratios across all 7 colors
        t_col = str(np.random.choice(all_colors))
        avail_p = [c for c in all_colors if c != t_col]
        p_col = str(np.random.choice(avail_p))
        occupied = []

        # High distractor scaling: supports clean (0) all the way up to dense clutter (20)
        num_dist_cubes = int(np.random.choice([0, 1, 2, 4, 6, 8, 12, 16, 20], 
                                              p=[0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]))
        min_d = 0.060 if num_dist_cubes <= 4 else (0.042 if num_dist_cubes <= 10 else 0.028)

        def rand_pos(min_dist=0.060):
            for _ in range(120):
                a = np.random.uniform(-0.55, 0.55)
                d = np.random.uniform(0.18, 0.28)
                p = np.array([d * np.cos(a), d * np.sin(a), 0.012], dtype=np.float32)
                if all(np.linalg.norm(p[:2] - o[:2]) > min_dist for o in occupied):
                    occupied.append(p)
                    return p
            p = np.array([0.22 + np.random.uniform(-0.03, 0.03), 0.08 + np.random.uniform(-0.03, 0.03), 0.012], dtype=np.float32)
            occupied.append(p)
            return p

        cube_pos = rand_pos(min_d)
        plat_pos = rand_pos(0.065); plat_pos[2] = 0.005

        # Distractor cubes (supports up to 20 with varied colors)
        dist_cubes = []
        for _ in range(num_dist_cubes):
            c_col = str(np.random.choice(all_colors))
            dist_cubes.append((c_col, rand_pos(min_d)))

        # Distractor platforms
        num_dist_plats = int(np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20]))
        dist_plats = []
        for _ in range(num_dist_plats):
            dp_c = str(np.random.choice([c for c in all_colors if c != p_col]))
            dp = rand_pos(0.065); dp[2] = 0.005
            dist_plats.append((dp_c, dp))

        # Task type distribution: 60% Pick-Place, 20% Push, 10% Reach, 10% Lift
        task_type = np.random.choice(["pick_place", "push", "reach", "lift"], p=[0.60, 0.20, 0.10, 0.10])

        hz = 0.045; csz = -0.012; psz = -0.005
        ps = np.array([0.22, 0.0, hz], dtype=np.float32)
        hc = np.array([cube_pos[0], cube_pos[1], hz], dtype=np.float32)
        cs = np.array([cube_pos[0], cube_pos[1], csz], dtype=np.float32)
        hp = np.array([plat_pos[0], plat_pos[1], hz], dtype=np.float32)
        pps = np.array([plat_pos[0], plat_pos[1], psz], dtype=np.float32)
        ret = np.array([plat_pos[0], plat_pos[1], hz], dtype=np.float32)

        R = np.random.randint

        # Dynamic natural phrasing covering "thing", "item", "put on orange", "place on", etc.
        n = str(np.random.choice(nouns))
        dn = str(np.random.choice(dest_nouns))
        dest_str = f"{p_col} {dn}".strip() if dn else p_col

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
                (ps, hc, 0.0, R(14, 18)),   # approach hover
                (hc, cs, 0.0, R(10, 14)),   # descend to cube
                (cs, cs, 1.0, R(5, 8)),     # grasp
                (cs, hc, 1.0, R(10, 14)),   # lift
                (hc, hp, 1.0, R(16, 22)),   # carry to platform
                (hp, pps, 1.0, R(10, 14)),  # lower to platform
                (pps, pps, 0.0, R(5, 8)),   # release
                (pps, ret, 0.0, R(12, 16)), # retract to hover
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
            # Push vector along direction from cube towards platform
            push_dir = (plat_pos[:2] - cube_pos[:2])
            norm = np.linalg.norm(push_dir)
            if norm > 1e-4:
                push_dir = push_dir / norm
            else:
                push_dir = np.array([1.0, 0.0])
            push_dist = np.random.uniform(0.040, 0.065)
            p_push_approach = np.array([cube_pos[0] - push_dir[0] * 0.035, cube_pos[1] - push_dir[1] * 0.035, hz], dtype=np.float32)
            p_push_start = np.array([cube_pos[0] - push_dir[0] * 0.035, cube_pos[1] - push_dir[1] * 0.035, -0.010], dtype=np.float32)
            push_end_xy = cube_pos[:2] + push_dir * push_dist
            p_push_end = np.array([push_end_xy[0], push_end_xy[1], -0.010], dtype=np.float32)
            p_push_retract = np.array([push_end_xy[0], push_end_xy[1], hz], dtype=np.float32)
            stages = [
                (ps, p_push_approach, 0.0, R(14, 18)), # hover behind cube
                (p_push_approach, p_push_start, 0.0, R(10, 14)), # descend behind cube
                (p_push_start, p_push_end, 0.0, R(24, 30)), # push cube along desk
                (p_push_end, p_push_retract, 0.0, R(12, 16)), # lift up
                (p_push_retract, ps, 0.0, R(12, 16)), # return home
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
        else: # "lift"
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
                (hc, hc, 1.0, R(16, 22)), # hold stationary in air
            ]

        raw_pts = []
        for p_s, p_e, g, n in stages:
            for pt in quintic(p_s, p_e, n):
                raw_pts.append((pt, g))

        # resample to H=128
        idx = np.linspace(0, len(raw_pts) - 1, 128).astype(int)
        traj = [raw_pts[k] for k in idx]

        # Verify kinematic feasibility of all 128 steps
        kin_valid = True
        for pt, _ in traj:
            j = kin.inverse(pt[0], pt[1], pt[2], 0.0)
            deg = np.degrees(j)
            if not (-125 <= deg[0] <= 125 and -5 <= deg[1] <= 85 and -15 <= deg[2] <= 70 and -150 <= deg[3] <= 150):
                kin_valid = False
                break
        if not kin_valid:
            rejected += 1
            continue

        # --- find grasp/release boundaries for physically accurate cube tracking ---
        grasp_start = None
        release_step = None
        for si, (_, g) in enumerate(traj):
            if g > 0.5 and grasp_start is None:
                grasp_start = si
            if grasp_start is not None and g < 0.5 and release_step is None:
                release_step = si

        # --- render camera at EVERY step with correct cube physics ---
        image_seq = np.zeros((128, 224, 224, 3), dtype=np.uint8)
        proprio_seq = np.zeros((128, 9), dtype=np.float32)
        action_seq = np.zeros((128, 9), dtype=np.float32)
        joints_seq = np.zeros((128, 4), dtype=np.float32)

        p_push_start_pt = p_push_start if task_type == "push" else None
        p_push_end_pt = p_push_end if task_type == "push" else None

        for si, (pt, g) in enumerate(traj):
            # Cube position logic depending on task type:
            if task_type == "pick_place":
                if grasp_start is None or si < grasp_start:
                    vis_cube = cube_pos.copy()
                elif release_step is None or si < release_step:
                    vis_cube = pt.copy()
                    vis_cube[2] = max(pt[2] - 0.005, 0.010)
                else:
                    vis_cube = pps.copy()
                    vis_cube[2] = plat_pos[2] + 0.008
            elif task_type == "push":
                # Cube slides when gripper contacts it at z < 0
                if pt[2] < 0.0 and np.linalg.norm(pt[:2] - cube_pos[:2]) < 0.055:
                    vis_cube = np.array([pt[0] + push_dir[0] * 0.015, pt[1] + push_dir[1] * 0.015, 0.012], dtype=np.float32)
                else:
                    vis_cube = cube_pos.copy()
            elif task_type == "lift":
                if grasp_start is None or si < grasp_start:
                    vis_cube = cube_pos.copy()
                else:
                    vis_cube = pt.copy()
                    vis_cube[2] = max(pt[2] - 0.005, 0.010)
            else: # reach
                vis_cube = cube_pos.copy()

            image_seq[si] = renderer.render_scene(
                target_cube_pos=vis_cube,
                target_color=t_col,
                platform_pos=plat_pos,
                plat_color=p_col,
                ee_pos=pt,
                gripper_closed=bool(g > 0.5),
                distractors_cubes=dist_cubes,
                distractors_plats=dist_plats,
                desk_tone=desk_tone,
            )
            j = kin.inverse(pt[0], pt[1], pt[2], 0.0)
            joints_seq[si] = j
            # 9D format: [j1, j2, j3, j4, x, y, z, r, gripper]
            proprio_seq[si] = [j[0], j[1], j[2], j[3], pt[0], pt[1], pt[2], 0.0, g]
            action_seq[si] = [j[0], j[1], j[2], j[3], pt[0], pt[1], pt[2], 0.0, g]

        mgr.add_demonstration(
            image_seq, proprio_seq, action_seq, prompt,
            metadata={"target": t_col, "platform": p_col, "task": task_type, "joints": joints_seq.tolist()},
        )
        completed += 1
        pbar.update(1)

    mgr.commit_shard()   # flush any remaining partial shard
    pbar.close()
    print(f"[Phase1] Done: {completed} verified demos sharded ({rejected} rejected).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Generate + Shard Demonstrations")
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--desk_tone", type=str, default="dark_mat", choices=["dark_mat", "wood"])
    parser.add_argument("--episodes_per_shard", type=int, default=50, help="Episodes per shard")
    args = parser.parse_args()

    mgr = WallclockDatasetManager(episodes_per_shard=args.episodes_per_shard)
    renderer = SyntheticDeskRenderer()

    if args.num_demos > 0:
        generate_demos(args.num_demos, mgr, renderer, args.desk_tone)

    mgr.update_manifest()
    print(f"[Phase1] Manifest written. Shards ready for CLIP extraction.", flush=True)
