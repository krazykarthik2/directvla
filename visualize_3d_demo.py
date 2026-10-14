#!/usr/bin/env python3
"""
Interactive 3D ROS 2 & RViz Demonstration Player for Dobot Magician.
Features:
  - Generates or plays back demonstration pick-and-place trajectories in 3D.
  - Publishes full 3D robot joint states (/joint_states) with realistic parallelogram physics.
  - Publishes 3D RViz Visualization Markers (/dobot/trajectory_markers):
      * Small target cube on top of the desk (following gripper when grasped).
      * Target drop platform on the desk.
      * Visual distractors (extra blocks).
      * 3D end-effector motion trail / path.
  - Supports continuous looping, speed adjustment, and live terminal telemetry.
"""

import os
import sys
import time
import math
import argparse
import numpy as np

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from vla_robot_env import DobotKinematics
from generate_and_shard import quintic


class Dobot3DDemoVisualizer(Node):
    def __init__(self, fps=25.0, loop=True, task_prompt="pick up the red cube and place it on the green platform"):
        super().__init__('dobot_3d_demo_visualizer')
        self.kin = DobotKinematics()
        self.fps = fps
        self.loop = loop
        self.task_prompt = task_prompt

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dobot/trajectory_markers', 10)

        # Plan the demonstration
        self.setup_demo_trajectory()

        # Step counter
        self.step_idx = 0
        self.trail_points = []
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.get_logger().info(f"Dobot 3D Demo Player ready! Task: '{self.task_prompt}' ({len(self.traj)} steps @ {self.fps} FPS)")

    def setup_demo_trajectory(self):
        # Desk is at z = 0. Small cube (~20mm) top surface at z = 0.015m
        self.cube_pos = np.array([0.22, 0.08, 0.012], dtype=np.float32)
        self.plat_pos = np.array([0.22, -0.10, 0.005], dtype=np.float32)

        self.dist_cubes = [
            ("blue", np.array([0.25, -0.04, 0.012], dtype=np.float32)),
            ("yellow", np.array([0.19, -0.05, 0.012], dtype=np.float32))
        ]

        hz = 0.065
        csz = 0.015
        psz = 0.022

        ps = np.array([0.22, 0.0, hz], dtype=np.float32)
        hc = np.array([self.cube_pos[0], self.cube_pos[1], hz])
        cs = np.array([self.cube_pos[0], self.cube_pos[1], csz])
        hp = np.array([self.plat_pos[0], self.plat_pos[1], hz])
        pps = np.array([self.plat_pos[0], self.plat_pos[1], psz])
        ret = np.array([self.plat_pos[0], self.plat_pos[1], hz])

        stages = [
            ("1. Approach Hover", ps, hc, 0.0, 16),
            ("2. Descend to Cube Top", hc, cs, 0.0, 12),
            ("3. Grasp Cube", cs, cs, 1.0, 6),
            ("4. Lift Cube", cs, hc, 1.0, 12),
            ("5. Transfer to Platform", hc, hp, 1.0, 18),
            ("6. Lower to Platform", hp, pps, 1.0, 12),
            ("7. Release Cube", pps, pps, 0.0, 6),
            ("8. Retract to Hover", pps, ret, 0.0, 14),
        ]

        raw_pts = []
        for phase_name, p_s, p_e, g, n in stages:
            for pt in quintic(p_s, p_e, n):
                raw_pts.append((pt, g, phase_name))

        idx = np.linspace(0, len(raw_pts) - 1, 128).astype(int)
        self.traj = [raw_pts[k] for k in idx]

        # Grasp boundaries
        self.grasp_start = 30
        self.release_step = 108

    def timer_callback(self):
        pt, g, phase = self.traj[self.step_idx]

        # 1. Compute exact Dobot joint angles
        j = self.kin.inverse(pt[0], pt[1], pt[2], 0.0)
        j1_rad, j2_rad, j3_rel_rad, j4_rad = j[0], j[1], j[2], j[3]
        gripper_pos = 0.001 if g > 0.5 else 0.012

        # 2. Publish /joint_states for robot_state_publisher and RViz
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = "magician_root_link"
        js.name = [
            'magician_joint_1',
            'magician_joint_2',
            'magician_joint_3',
            'magician_joint_4',
            'magician_joint_prismatic_l'
        ]
        js.position = [float(j1_rad), float(j2_rad), float(j3_rel_rad), float(j4_rad), float(gripper_pos)]
        self.joint_pub.publish(js)

        # 3. Publish 3D Markers to RViz
        self.publish_markers(pt, g, phase)

        # 4. Telemetry output
        deg = np.degrees(j)
        sys.stdout.write(
            f"\r[3D Player] Step {self.step_idx:03d}/128 | Phase: {phase:<25} | "
            f"XYZ: ({pt[0]*100:4.1f}, {pt[1]*100:4.1f}, {pt[2]*100:4.1f})cm | "
            f"Joints: [J1:{deg[0]:4.0f}°, J2:{deg[1]:3.0f}°, J3:{deg[2]:3.0f}°, J4:{deg[3]:3.0f}°] | "
            f"Grip: {'CLOSED' if g > 0.5 else 'OPEN '} "
        )
        sys.stdout.flush()

        self.step_idx += 1
        if self.step_idx >= len(self.traj):
            if self.loop:
                self.step_idx = 0
                self.trail_points.clear()
                print("\n[3D Player] Episode finished. Looping to start...")
            else:
                print("\n[3D Player] Episode finished.")
                self.timer.cancel()
                raise SystemExit(0)

    def publish_markers(self, pt, g, phase):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # Cube position tracking
        if self.step_idx < self.grasp_start:
            vis_cube = self.cube_pos.copy()
        elif self.step_idx < self.release_step:
            vis_cube = pt.copy()
            vis_cube[2] = max(pt[2] - 0.005, 0.010)
        else:
            vis_cube = np.array([self.plat_pos[0], self.plat_pos[1], self.plat_pos[2] + 0.008])

        # 1. Target Red Cube (aligned directly inside gripper jaw contact point)
        # Z_OFFSET = 0.088 aligns with gripper jaws at desk level
        Z_OFFSET = 0.088
        m_cube = Marker()
        m_cube.header.stamp = now
        m_cube.header.frame_id = "magician_root_link"
        m_cube.ns = "cube"
        m_cube.id = 1
        m_cube.type = Marker.CUBE
        m_cube.action = Marker.ADD
        m_cube.pose.position.x = float(vis_cube[0])
        m_cube.pose.position.y = float(vis_cube[1])
        m_cube.pose.position.z = float(vis_cube[2] + Z_OFFSET)
        m_cube.pose.orientation.w = 1.0
        m_cube.scale.x = 0.022
        m_cube.scale.y = 0.022
        m_cube.scale.z = 0.022
        m_cube.color = ColorRGBA(r=0.95, g=0.15, b=0.15, a=0.95)
        ma.markers.append(m_cube)

        # 2. Target Green Drop Platform (resting on desk)
        m_plat = Marker()
        m_plat.header.stamp = now
        m_plat.header.frame_id = "magician_root_link"
        m_plat.ns = "platform"
        m_plat.id = 2
        m_plat.type = Marker.CUBE
        m_plat.action = Marker.ADD
        m_plat.pose.position.x = float(self.plat_pos[0])
        m_plat.pose.position.y = float(self.plat_pos[1])
        m_plat.pose.position.z = float(self.plat_pos[2] + Z_OFFSET - 0.005)
        m_plat.pose.orientation.w = 1.0
        m_plat.scale.x = 0.050
        m_plat.scale.y = 0.050
        m_plat.scale.z = 0.008
        m_plat.color = ColorRGBA(r=0.15, g=0.85, b=0.30, a=0.95)
        ma.markers.append(m_plat)

        # 3. Distractor Cubes
        for idx, (c_name, c_pos) in enumerate(self.dist_cubes):
            m_dist = Marker()
            m_dist.header.stamp = now
            m_dist.header.frame_id = "magician_root_link"
            m_dist.ns = "cube"
            m_dist.id = 10 + idx
            m_dist.type = Marker.CUBE
            m_dist.action = Marker.ADD
            m_dist.pose.position.x = float(c_pos[0])
            m_dist.pose.position.y = float(c_pos[1])
            m_dist.pose.position.z = float(c_pos[2] + Z_OFFSET)
            m_dist.pose.orientation.w = 1.0
            m_dist.scale.x = 0.020
            m_dist.scale.y = 0.020
            m_dist.scale.z = 0.020
            if c_name == "blue":
                m_dist.color = ColorRGBA(r=0.15, g=0.45, b=0.95, a=0.9)
            else:
                m_dist.color = ColorRGBA(r=0.95, g=0.85, b=0.15, a=0.9)
            ma.markers.append(m_dist)

        # 4. Trajectory Path Trail (LINE_STRIP)
        p = Point()
        p.x = float(pt[0])
        p.y = float(pt[1])
        p.z = float(pt[2] + Z_OFFSET)
        self.trail_points.append(p)
        if len(self.trail_points) > 128:
            self.trail_points.pop(0)

        m_trail = Marker()
        m_trail.header.stamp = now
        m_trail.header.frame_id = "magician_root_link"
        m_trail.ns = "trajectory"
        m_trail.id = 3
        m_trail.type = Marker.LINE_STRIP
        m_trail.action = Marker.ADD
        m_trail.scale.x = 0.003
        m_trail.color = ColorRGBA(r=0.0, g=1.0, b=0.8, a=0.85)
        m_trail.points = list(self.trail_points)
        ma.markers.append(m_trail)

        self.marker_pub.publish(ma)


def main():
    parser = argparse.ArgumentParser(description="3D ROS 2 RViz Dobot Demo Visualizer")
    parser.add_argument("--fps", type=float, default=25.0, help="Playback speed in frames per second")
    parser.add_argument("--no-loop", action="store_true", help="Play demonstration once without looping")
    args = parser.parse_args()

    rclpy.init()
    node = Dobot3DDemoVisualizer(fps=args.fps, loop=not args.no_loop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
