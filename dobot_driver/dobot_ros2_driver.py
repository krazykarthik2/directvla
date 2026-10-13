#!/usr/bin/env python3
import json
import math
import os
import sys
import threading
import time
from typing import Optional

_WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool

from dobot_driver.pydobot import Dobot, DobotException, MODE_PTP
from vla_robot_env import DobotKinematics


class DobotRealDriver(Node):
    def __init__(self):
        super().__init__('dobot_real_driver')
        self.kin = DobotKinematics()

        self.declare_parameter('port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('tool', 'gripper')  # gripper, suction_cup, pen, none
        self.declare_parameter('mock', False)  # auto-enable mock if no device

        port = self.get_parameter('port').get_parameter_value().string_value
        baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.tool = self.get_parameter('tool').get_parameter_value().string_value
        mock_requested = self.get_parameter('mock').get_parameter_value().bool_value

        self.arm: Optional[Dobot] = None
        self.is_mock = False
        self.lock = threading.Lock()

        # Simulated joint state for mock fallback
        self.mock_j1 = 0.0
        self.mock_j2 = 0.0
        self.mock_j3 = 30.0
        self.mock_j4 = 0.0
        self.mock_x = 200.0
        self.mock_y = 0.0
        self.mock_z = 0.0
        self.mock_r = 0.0
        self.tool_active = False

        if not mock_requested:
            try:
                self.get_logger().info(f"Connecting to Dobot (port: '{port or 'auto'}')...")
                self.arm = Dobot(port=port if port else None, baudrate=baudrate)
                self.get_logger().info(f"Successfully connected to Dobot on port {self.arm.port}")
                self.arm.speed(velocity=100.0, acceleration=100.0)
            except Exception as e:
                self.get_logger().warn(f"Dobot hardware connection failed: {e}. Switching to Mock Driver mode for testing!")
                self.is_mock = True
        else:
            self.is_mock = True
            self.get_logger().info("Starting in Mock Driver mode.")

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.state_pub = self.create_publisher(String, '/dobot/state', 10)

        # Subscribers for control
        # 1. Target coordinates (linear x, y, z and angular z = r; angular x = tool trigger 1/0)
        self.create_subscription(Twist, '/dobot/target_coord', self.target_coord_callback, 10)
        # 2. Target joints (Twist linear.x=j1, y=j2, z=j3, angular.z=j4 in degrees)
        self.create_subscription(Twist, '/dobot/target_joints', self.target_joints_callback, 10)
        # 3. Tool command (suction or gripper trigger)
        self.create_subscription(Bool, '/dobot/tool_state', self.tool_state_callback, 10)
        # 4. Raw JSON command string e.g. {"cmd": "home"} or {"cmd": "move_to", "x": 200, "y": 50, "z": 20, "r": 0}
        self.create_subscription(String, '/dobot/command', self.command_callback, 10)

        # Timer for publishing joint states to RViz
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_states)

    def publish_states(self):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        # URDF joint names matching magician.urdf.xacro
        js_msg.name = [
            'magician_joint_1',
            'magician_joint_2',
            'magician_joint_3',
            'magician_joint_4',
            'magician_joint_prismatic_l'
        ]

        with self.lock:
            if not self.is_mock and self.arm:
                try:
                    pose = self.arm.get_pose()
                    j1_deg = pose.joints.j1
                    j2_deg = pose.joints.j2
                    j3_deg = pose.joints.j3
                    j4_deg = pose.joints.j4
                    x, y, z, r = pose.position.x, pose.position.y, pose.position.z, pose.position.r
                except Exception as e:
                    self.get_logger().error(f"Error reading robot pose: {e}")
                    return
            else:
                j1_deg = self.mock_j1
                j2_deg = self.mock_j2
                j3_deg = self.mock_j3
                j4_deg = self.mock_j4
                x, y, z, r = self.mock_x, self.mock_y, self.mock_z, self.mock_r

        # Joint angle conversion for Dobot Magician URDF
        # joint_1 (base yaw): degrees to radians
        j1_rad = math.radians(j1_deg)
        # joint_2 (rear arm pitch)
        j2_rad = math.radians(j2_deg)
        # joint_3 (forearm pitch relative to rear arm)
        # In Dobot kinematics, j3_reported is angle relative to horizontal / ground or rear arm.
        # Magician URDF expects joint_3 relative to joint_2 link:
        j3_rad = math.radians(j3_deg - j2_deg)
        # joint_4 (end-effector rotation / yaw)
        j4_rad = math.radians(j4_deg)
        # Gripper jaw translation (0.00 to 0.0135)
        gripper_pos = 0.012 if self.tool_active else 0.001

        js_msg.position = [j1_rad, j2_rad, j3_rad, j4_rad, gripper_pos]
        self.joint_pub.publish(js_msg)

        # Publish state json
        state_data = {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "r": round(r, 2),
            "j1": round(j1_deg, 2),
            "j2": round(j2_deg, 2),
            "j3": round(j3_deg, 2),
            "j4": round(j4_deg, 2),
            "tool_active": self.tool_active,
            "mock": self.is_mock
        }
        str_msg = String()
        str_msg.data = json.dumps(state_data)
        self.state_pub.publish(str_msg)

    def update_mock_from_xyz(self, x, y, z, r):
        self.mock_x = float(x)
        self.mock_y = float(y)
        self.mock_z = float(z)
        self.mock_r = float(r)
        try:
            j1, j2, j3, j4 = self.kin.inverse(x / 1000.0, y / 1000.0, z / 1000.0, math.radians(r))
            self.mock_j1 = math.degrees(j1)
            self.mock_j2 = math.degrees(j2)
            self.mock_j3 = math.degrees(j2 + j3)  # Dobot reported pitch from horizontal
            self.mock_j4 = math.degrees(j4)
        except Exception:
            pass

    def update_mock_from_joints(self, j1, j2, j3, j4):
        self.mock_j1 = float(j1)
        self.mock_j2 = float(j2)
        self.mock_j3 = float(j3)
        self.mock_j4 = float(j4)
        try:
            # j3_urdf = j3_reported - j2
            j1_r = math.radians(j1)
            j2_r = math.radians(j2)
            j3_rel_r = math.radians(j3 - j2)
            j4_r = math.radians(j4)
            xyz_yaw = self.kin.forward(j1_r, j2_r, j3_rel_r, j4_r)
            self.mock_x = float(xyz_yaw[0] * 1000.0)
            self.mock_y = float(xyz_yaw[1] * 1000.0)
            self.mock_z = float(xyz_yaw[2] * 1000.0)
            self.mock_r = float(math.degrees(xyz_yaw[3]))
        except Exception:
            pass

    def target_coord_callback(self, msg: Twist):
        x = msg.linear.x
        y = msg.linear.y
        z = msg.linear.z
        r = msg.angular.z
        tool_val = msg.angular.x > 0.5
        self.set_tool(tool_val)

        with self.lock:
            if not self.is_mock and self.arm:
                try:
                    self.arm.move_to(x, y, z, r, mode=MODE_PTP.MOVJ_XYZ, wait=False)
                except Exception as e:
                    self.get_logger().error(f"Move error: {e}")
            else:
                self.update_mock_from_xyz(x, y, z, r)

    def target_joints_callback(self, msg: Twist):
        j1 = msg.linear.x
        j2 = msg.linear.y
        j3 = msg.linear.z
        j4 = msg.angular.z

        with self.lock:
            if not self.is_mock and self.arm:
                try:
                    self.arm.rotate_joint(j1, j2, j3, j4, mode=MODE_PTP.MOVJ_ANGLE, wait=False)
                except Exception as e:
                    self.get_logger().error(f"Rotate joint error: {e}")
            else:
                self.update_mock_from_joints(j1, j2, j3, j4)

    def tool_state_callback(self, msg: Bool):
        self.set_tool(msg.data)

    def set_tool(self, active: bool):
        with self.lock:
            self.tool_active = active
            if not self.is_mock and self.arm:
                try:
                    if self.tool in ['gripper', 'extended_gripper']:
                        self.arm.grip(active)
                    elif self.tool == 'suction_cup':
                        self.arm.suck(active)
                except Exception as e:
                    self.get_logger().error(f"Tool actuation error: {e}")

    def command_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            cmd = data.get("cmd")
            if cmd == "home":
                with self.lock:
                    if not self.is_mock and self.arm:
                        self.arm.home()
                    else:
                        self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4 = 0.0, 0.0, 30.0, 0.0
                        self.mock_x, self.mock_y, self.mock_z, self.mock_r = 200.0, 0.0, 0.0, 0.0
            elif cmd == "move_to":
                x = data.get("x", 200.0)
                y = data.get("y", 0.0)
                z = data.get("z", 0.0)
                r = data.get("r", 0.0)
                wait = data.get("wait", False)
                mode = MODE_PTP.MOVJ_XYZ if data.get("mode") != "MOVL" else MODE_PTP.MOVL_XYZ
                with self.lock:
                    if not self.is_mock and self.arm:
                        self.arm.move_to(x, y, z, r, mode=mode, wait=wait)
                    else:
                        self.mock_x, self.mock_y, self.mock_z, self.mock_r = x, y, z, r
            elif cmd == "rotate_joint":
                j1 = data.get("j1", 0.0)
                j2 = data.get("j2", 0.0)
                j3 = data.get("j3", 0.0)
                j4 = data.get("j4", 0.0)
                wait = data.get("wait", False)
                with self.lock:
                    if not self.is_mock and self.arm:
                        self.arm.rotate_joint(j1, j2, j3, j4, mode=MODE_PTP.MOVJ_ANGLE, wait=wait)
                    else:
                        self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4 = j1, j2, j3, j4
            elif cmd == "tool":
                self.set_tool(bool(data.get("state", False)))
            elif cmd == "clear_alarms":
                with self.lock:
                    if not self.is_mock and self.arm:
                        self.arm.clear_alarms()
        except Exception as e:
            self.get_logger().error(f"Failed to execute command '{msg.data}': {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DobotRealDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.arm:
            node.arm.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
