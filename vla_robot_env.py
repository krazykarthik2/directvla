#!/usr/bin/env python3
"""
Real ROS 2 Dobot Environment & Hardware Bridge for VLA Policy Execution.
Integrates:
  - Exact Dobot Magician analytical kinematics (4-DOF).
  - ROS 2 Topic Communication (/dobot/state, /dobot/target_coord, /joint_states).
  - Direct PySerial Hardware interface (pydobot.Dobot) with auto-detection.
  - Safe kinematic simulation fallback when hardware is offline.
  - Metric coordinate conversions and physical safety guards.
"""

import os
import sys
import json
import math
import time
import threading
import numpy as np

# Add local path to import dobot_driver.pydobot
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)

try:
    from dobot_driver.pydobot import Dobot, MODE_PTP
    PYDOBOT_AVAILABLE = True
except ImportError:
    PYDOBOT_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Bool
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class DobotKinematics:
    """Exact Analytical Forward and Inverse Kinematics for Dobot Magician 4-DOF."""
    def __init__(self):
        self.L1 = 0.138    # Base to Rear Arm joint height (m)
        self.L2 = 0.135    # Rear arm length (m)
        self.L3 = 0.147    # Forearm length (m)
        self.L4 = 0.060    # Tool end-effector offset (m)

    def forward(self, j1, j2, j3, j4):
        """
        Joint angles in radians -> [x, y, z, yaw] in meters.
        j1: base yaw (Z axis)
        j2: rear arm pitch (Y axis)
        j3: forearm pitch relative to rear arm (Y axis)
        j4: end-effector roll/yaw offset
        The parallelogram link keeps the end-effector base level (net pitch = 0).
        """
        r = self.L2 * np.sin(j2) + self.L3 * np.cos(j2 + j3) + self.L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = self.L2 * np.cos(j2) - self.L3 * np.sin(j2 + j3)
        yaw = j1 + j4
        return np.array([x, y, z, yaw], dtype=np.float32)

    def inverse(self, x, y, z, yaw=0.0):
        """
        [x, y, z, yaw] in meters -> [j1, j2, j3, j4] in radians.
        Analytical closed-form inverse kinematics strictly obeying Dobot limits.
        """
        j1 = np.arctan2(y, x)
        r_total = np.sqrt(x**2 + y**2)
        r_t = r_total - self.L4
        z_t = z

        D_sq = r_t**2 + z_t**2
        D = np.sqrt(np.maximum(D_sq, 1e-8))
        K = (D_sq + self.L2**2 - self.L3**2) / (2.0 * self.L2)
        cos_val = np.clip(K / D, -1.0, 1.0)
        phi = np.arctan2(r_t, z_t)
        j2 = phi - np.arccos(cos_val)

        theta = np.arctan2(self.L2 * np.cos(j2) - z_t, r_t - self.L2 * np.sin(j2))
        j3 = theta - j2

        j4 = yaw - j1
        return np.array([j1, j2, j3, j4], dtype=np.float32)


class DobotRealEnv:
    """
    Real Dobot Magician Robotic Environment for VLA.
    Provides standard unified robotics API:
      - reset(): moves to standard hover pose [0.20, 0.0, 0.12, 0.0, 0.0]
      - step_target(target_pose_5d): moves EE to metric target [x, y, z, yaw, gripper]
      - step_delta(delta_5d): moves EE by incremental delta [dx, dy, dz, dyaw, gripper]
      - get_proprioception(): returns current 5D state [x, y, z, yaw, gripper] in meters
    """
    # Safe metric workspace boundaries (in meters) matching physical Dobot limits
    # Desk level is at z ~= -0.015 to 0.000 m. Objects rest on top of desk.
    X_BOUNDS = (0.170, 0.285)
    Y_BOUNDS = (-0.160, 0.160)
    Z_BOUNDS = (-0.020, 0.090)
    YAW_BOUNDS = (-np.pi / 2.0, np.pi / 2.0)

    def __init__(self, use_ros2=True, port=None, baudrate=115200):
        self.kin = DobotKinematics()
        self.lock = threading.Lock()
        
        # State:
        # Cartesian: [ee_x, ee_y, ee_z, ee_yaw] in meters (hover inspection pose above desk)
        self.ee_pos = np.array([0.22, 0.0, 0.065, 0.0], dtype=np.float32)
        # Joint space: [j1, j2, j3, j4] in radians
        init_j = self.kin.inverse(0.22, 0.0, 0.065, 0.0)
        self.joint_pos = np.array(init_j, dtype=np.float32)
        self.gripper_closed = False
        self.hardware_connected = False
        self.arm = None

        self.use_ros2 = use_ros2 and ROS2_AVAILABLE
        self.ros_node = None
        self.ros_spin_thread = None

        # 1. Try connecting to hardware directly via PySerial
        if PYDOBOT_AVAILABLE:
            try:
                self.arm = Dobot(port=port, baudrate=baudrate)
                self.hardware_connected = True
                print(f"[DobotEnv] Connected directly to Dobot Magician on port {self.arm.port}", flush=True)
                self.arm.speed(velocity=100.0, acceleration=100.0)
                pose = self.arm.get_pose()
                self.ee_pos = np.array([
                    pose.position.x / 1000.0,
                    pose.position.y / 1000.0,
                    pose.position.z / 1000.0,
                    math.radians(pose.position.r)
                ], dtype=np.float32)
                self.joint_pos = np.array([
                    math.radians(pose.joints.j1),
                    math.radians(pose.joints.j2),
                    math.radians(pose.joints.j3 - pose.joints.j2),
                    math.radians(pose.joints.j4)
                ], dtype=np.float32)
            except Exception as e:
                print(f"[DobotEnv] Direct hardware connection: {e}. Checking ROS 2 or kinematic mode...", flush=True)

        # 2. Setup ROS 2 Node if requested
        if self.use_ros2:
            self._init_ros2_node()

    def _init_ros2_node(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = rclpy.create_node('dobot_vla_bridge')
            self.coord_pub = self.ros_node.create_publisher(Twist, '/dobot/target_coord', 10)
            self.joints_pub = self.ros_node.create_publisher(Twist, '/dobot/target_joints', 10)
            self.tool_pub = self.ros_node.create_publisher(Bool, '/dobot/tool_state', 10)
            self.cmd_pub = self.ros_node.create_publisher(String, '/dobot/command', 10)
            self.state_sub = self.ros_node.create_subscription(
                String, '/dobot/state', self._ros_state_callback, 10
            )
            # Spin node in background thread
            self.ros_spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self.ros_node), daemon=True
            )
            self.ros_spin_thread.start()
            print("[DobotEnv] ROS 2 Dobot VLA Bridge initialized successfully.", flush=True)
        except Exception as e:
            print(f"[DobotEnv] Failed initializing ROS 2: {e}", flush=True)

    def _ros_state_callback(self, msg):
        try:
            data = json.loads(msg.data)
            with self.lock:
                self.ee_pos[0] = data["x"] / 1000.0
                self.ee_pos[1] = data["y"] / 1000.0
                self.ee_pos[2] = data["z"] / 1000.0
                self.ee_pos[3] = math.radians(data["r"])
                # Extract joint angles: j1, j2, j3, j4
                if "j1" in data and "j2" in data and "j3" in data and "j4" in data:
                    j1_rad = math.radians(data["j1"])
                    j2_rad = math.radians(data["j2"])
                    # URDF joint_3 is relative to joint_2
                    j3_rad = math.radians(data["j3"] - data["j2"])
                    j4_rad = math.radians(data["j4"])
                    self.joint_pos = np.array([j1_rad, j2_rad, j3_rad, j4_rad], dtype=np.float32)
                self.gripper_closed = bool(data["tool_active"])
                if not data.get("mock", True):
                    self.hardware_connected = True
        except Exception:
            pass

    def get_proprioception(self, full_state=True):
        """
        Returns robot proprioception state.
        If full_state=True: returns 9-dim array [j1, j2, j3, j4, ee_x, ee_y, ee_z, ee_yaw, gripper_status]
        If full_state=False: returns 5-dim array [ee_x, ee_y, ee_z, ee_yaw, gripper_status]
        """
        with self.lock:
            grip_val = 1.0 if self.gripper_closed else 0.0
            if full_state:
                return np.array([
                    self.joint_pos[0],
                    self.joint_pos[1],
                    self.joint_pos[2],
                    self.joint_pos[3],
                    self.ee_pos[0],
                    self.ee_pos[1],
                    self.ee_pos[2],
                    self.ee_pos[3],
                    grip_val
                ], dtype=np.float32)
            else:
                return np.array([
                    self.ee_pos[0],
                    self.ee_pos[1],
                    self.ee_pos[2],
                    self.ee_pos[3],
                    grip_val
                ], dtype=np.float32)

    def clip_workspace(self, target_xyz_yaw):
        """Enforces physical Dobot Magician workspace boundaries."""
        x = np.clip(target_xyz_yaw[0], self.X_BOUNDS[0], self.X_BOUNDS[1])
        y = np.clip(target_xyz_yaw[1], self.Y_BOUNDS[0], self.Y_BOUNDS[1])
        z = np.clip(target_xyz_yaw[2], self.Z_BOUNDS[0], self.Z_BOUNDS[1])
        yaw = np.clip(target_xyz_yaw[3], self.YAW_BOUNDS[0], self.YAW_BOUNDS[1])
        return np.array([x, y, z, yaw], dtype=np.float32)

    def step_target(self, target_5d, wait=False):
        """
        Commands the end-effector to an absolute target:
        target_5d: [x_m, y_m, z_m, yaw_rad, gripper_val]
        Updates Cartesian and calculates joint targets for ROS 2 and hardware.
        """
        bounded_xyzyaw = self.clip_workspace(target_5d[:4])
        target_grip = bool(target_5d[4] > 0.5)

        # Compute inverse kinematics for joint targets
        j_target = self.kin.inverse(bounded_xyzyaw[0], bounded_xyzyaw[1], bounded_xyzyaw[2], bounded_xyzyaw[3])

        with self.lock:
            self.ee_pos = bounded_xyzyaw.copy()
            self.joint_pos = np.array(j_target, dtype=np.float32)
            self.gripper_closed = target_grip

        x_mm = float(bounded_xyzyaw[0] * 1000.0)
        y_mm = float(bounded_xyzyaw[1] * 1000.0)
        z_mm = float(bounded_xyzyaw[2] * 1000.0)
        r_deg = float(math.degrees(bounded_xyzyaw[3]))

        # Dobot firmware reported joint angles in degrees
        j1_deg = float(math.degrees(j_target[0]))
        j2_deg = float(math.degrees(j_target[1]))
        j3_firmware_deg = float(math.degrees(j_target[1] + j_target[2]))
        j4_deg = float(math.degrees(j_target[3]))

        # Send to physical hardware if connected directly
        if self.hardware_connected and self.arm is not None:
            try:
                self.arm.move_to(x_mm, y_mm, z_mm, r_deg, mode=MODE_PTP.MOVJ_XYZ, wait=wait)
                self.arm.grip(target_grip)
            except Exception as e:
                print(f"[DobotEnv] Hardware command error: {e}", flush=True)

        # Send to ROS 2 topics if active
        if self.ros_node is not None and ROS2_AVAILABLE:
            msg = Twist()
            msg.linear.x = x_mm
            msg.linear.y = y_mm
            msg.linear.z = z_mm
            msg.angular.z = r_deg
            msg.angular.x = 1.0 if target_grip else 0.0
            self.coord_pub.publish(msg)

            j_msg = Twist()
            j_msg.linear.x = j1_deg
            j_msg.linear.y = j2_deg
            j_msg.linear.z = j3_firmware_deg
            j_msg.angular.z = j4_deg
            self.joints_pub.publish(j_msg)

            b_msg = Bool()
            b_msg.data = target_grip
            self.tool_pub.publish(b_msg)

        return self.get_proprioception()

    def step_joints(self, target_joints_5d, wait=False):
        """
        Commands the robot directly in joint space:
        target_joints_5d: [j1_rad, j2_rad, j3_rad, j4_rad, gripper_val]
        """
        j1, j2, j3, j4 = target_joints_5d[:4]
        target_grip = bool(target_joints_5d[4] > 0.5)

        # Compute forward kinematics
        fwd_xyzyaw = self.kin.forward(j1, j2, j3, j4)
        bounded_xyzyaw = self.clip_workspace(fwd_xyzyaw)

        with self.lock:
            self.joint_pos = np.array([j1, j2, j3, j4], dtype=np.float32)
            self.ee_pos = bounded_xyzyaw.copy()
            self.gripper_closed = target_grip

        j1_deg = float(math.degrees(j1))
        j2_deg = float(math.degrees(j2))
        j3_firmware_deg = float(math.degrees(j2 + j3))
        j4_deg = float(math.degrees(j4))

        if self.hardware_connected and self.arm is not None:
            try:
                self.arm.rotate_joint(j1_deg, j2_deg, j3_firmware_deg, j4_deg, mode=MODE_PTP.MOVJ_ANGLE, wait=wait)
                self.arm.grip(target_grip)
            except Exception as e:
                print(f"[DobotEnv] Hardware rotate joint error: {e}", flush=True)

        if self.ros_node is not None and ROS2_AVAILABLE:
            j_msg = Twist()
            j_msg.linear.x = j1_deg
            j_msg.linear.y = j2_deg
            j_msg.linear.z = j3_firmware_deg
            j_msg.angular.z = j4_deg
            self.joints_pub.publish(j_msg)

            msg = Twist()
            msg.linear.x = float(bounded_xyzyaw[0] * 1000.0)
            msg.linear.y = float(bounded_xyzyaw[1] * 1000.0)
            msg.linear.z = float(bounded_xyzyaw[2] * 1000.0)
            msg.angular.z = float(math.degrees(bounded_xyzyaw[3]))
            msg.angular.x = 1.0 if target_grip else 0.0
            self.coord_pub.publish(msg)

            b_msg = Bool()
            b_msg.data = target_grip
            self.tool_pub.publish(b_msg)

        return self.get_proprioception()

    def step_delta(self, delta_5d, max_step=0.012):
        """
        Commands an incremental delta displacement:
        delta_5d: [dx, dy, dz, dyaw, gripper_val]
        """
        curr = self.get_proprioception()
        dx = np.clip(delta_5d[0], -max_step, max_step)
        dy = np.clip(delta_5d[1], -max_step, max_step)
        dz = np.clip(delta_5d[2], -max_step, max_step)
        dyaw = np.clip(delta_5d[3], -0.10, 0.10)
        new_grip = delta_5d[4]

        new_target = np.array([
            curr[0] + dx,
            curr[1] + dy,
            curr[2] + dz,
            curr[3] + dyaw,
            new_grip
        ], dtype=np.float32)

        return self.step_target(new_target)

    def reset(self):
        """Resets the robot to standard hover inspection pose."""
        hover_pose = np.array([0.22, 0.0, 0.065, 0.0, 0.0], dtype=np.float32)
        return self.step_target(hover_pose, wait=False)

    def close(self):
        if self.arm is not None:
            try:
                self.arm.close()
            except Exception:
                pass
            self.arm = None


if __name__ == "__main__":
    print("[TEST] Initializing DobotRealEnv...")
    env = DobotRealEnv(use_ros2=False)
    p0 = env.get_proprioception()
    print(f"       Initial Proprioception (5D): {p0}")
    kin = DobotKinematics()
    j = kin.inverse(p0[0], p0[1], p0[2], p0[3])
    print(f"       Inverse Kinematics Joints (deg): {np.degrees(j)}")
    fwd = kin.forward(j[0], j[1], j[2], j[3])
    print(f"       Forward Kinematics Verification: {fwd}")
    assert np.allclose(p0[:4], fwd, atol=1e-3), "Kinematics check failed!"

    # Step test
    p1 = env.step_target(np.array([0.22, 0.05, 0.08, 0.0, 1.0]))
    print(f"       Stepped to Target: {p1}")
    env.close()
    print("[TEST] DobotRealEnv test passed successfully!")
