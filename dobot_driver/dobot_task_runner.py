#!/usr/bin/env python3
import argparse
import atexit
import json
import math
import os
import readline
import sys
import threading
import time
from typing import Optional, Union

_WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from dobot_driver.pydobot import Dobot, MODE_PTP

LEARNINGS_DIR = os.path.expanduser("~/dobot/saved_learnings")
HISTORY_FILE = os.path.expanduser("~/.dobot_history")

# Configure terminal history (Up/Down arrow support)
try:
    if os.path.exists(HISTORY_FILE):
        readline.read_history_file(HISTORY_FILE)
    readline.set_history_length(1000)
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))
except Exception:
    pass


class SimpleCompleter:
    def __init__(self, options):
        self.options = options

    def complete(self, text, state):
        matches = [o for o in self.options if o.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None



def ensure_learnings_dir():
    os.makedirs(LEARNINGS_DIR, exist_ok=True)


class DobotBridge(Node):
    """
    Direct Dobot hardware interface with integrated ROS2 publishing so RViz
    and hardware share a single serial connection without conflicts.
    """
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, default_tool: str = "gripper"):
        super().__init__('dobot_bridge_runner')
        self.arm: Optional[Dobot] = None
        self.is_mock = False
        self.lock = threading.Lock()
        self.tool_type = default_tool
        self.tool_state = "stop"   # "on"/"in", "off"/"blow", "stop"

        # Current live/mock coordinates and joints
        self.mock_x, self.mock_y, self.mock_z, self.mock_r = 200.0, 0.0, 0.0, 0.0
        self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4 = 0.0, 0.0, 30.0, 0.0
        self.curr_j1, self.curr_j2, self.curr_j3, self.curr_j4 = 0.0, 0.0, 30.0, 0.0
        self.curr_x, self.curr_y, self.curr_z, self.curr_r = 200.0, 0.0, 0.0, 0.0

        try:
            print(f"[>>] Connecting to Dobot Magician on port '{port or 'auto-detect'}'...")
            self.arm = Dobot(port=port if port else None, baudrate=baudrate)
            print(f"[✓] Connected to Dobot on port: {self.arm.port}")
            self.arm.speed(velocity=100.0, acceleration=100.0)
            # Read initial pose from hardware
            init_p = self.arm.get_pose()
            self.curr_j1, self.curr_j2, self.curr_j3, self.curr_j4 = (
                init_p.joints.j1, init_p.joints.j2, init_p.joints.j3, init_p.joints.j4
            )
            self.curr_x, self.curr_y, self.curr_z, self.curr_r = (
                init_p.position.x, init_p.position.y, init_p.position.z, init_p.position.r
            )
        except Exception as e:
            print(f"[!] Warning: Could not connect to physical Dobot: {e}")
            print("[!] Running in offline / simulation mode.\n")
            self.is_mock = True

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.03, self.publish_joint_states)  # ~33 Hz live streaming

    def publish_joint_states(self):
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = [
            'magician_joint_1',
            'magician_joint_2',
            'magician_joint_3',
            'magician_joint_4',
            'magician_joint_prismatic_l'
        ]

        if not self.is_mock and self.arm:
            try:
                p = self.arm.get_pose()
                self.curr_j1 = p.joints.j1
                self.curr_j2 = p.joints.j2
                self.curr_j3 = p.joints.j3
                self.curr_j4 = p.joints.j4
                self.curr_x = p.position.x
                self.curr_y = p.position.y
                self.curr_z = p.position.z
                self.curr_r = p.position.r
            except Exception:
                pass
            j1, j2, j3, j4 = self.curr_j1, self.curr_j2, self.curr_j3, self.curr_j4
        else:
            with self.lock:
                j1, j2, j3, j4 = self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4

        j1_rad = math.radians(j1)
        j2_rad = math.radians(j2)
        j3_rad = math.radians(j3 - j2)
        j4_rad = math.radians(j4)
        
        # In RViz URDF (magician.urdf.xacro):
        # magician_joint_prismatic_l 0.000 = closed/clamped, 0.0135 = open/blown
        # tool_state "on"/"in"/"clamp" -> jaws closed (0.000)
        # tool_state "off"/"blow"/"open" -> jaws open (0.0135)
        # tool_state "stop" -> relaxed (0.007)
        if self.tool_state in ["on", "in", "clamp"]:
            gripper_pos = 0.000
        elif self.tool_state in ["off", "blow", "open"]:
            gripper_pos = 0.0135
        else:
            gripper_pos = 0.007

        js_msg.position = [j1_rad, j2_rad, j3_rad, j4_rad, gripper_pos]
        self.joint_pub.publish(js_msg)

    def _inverse_kinematics(self, x: float, y: float, z: float, r: float):
        l1 = 135.0
        l2 = 147.0
        j1 = math.degrees(math.atan2(y, x)) if (x != 0 or y != 0) else 0.0
        dist_xy = math.sqrt(x * x + y * y)
        d_sq = dist_xy * dist_xy + z * z
        d = math.sqrt(d_sq)
        d = min(max(d, 10.0), l1 + l2 - 1.0)

        cos_alpha = (l1 * l1 + d_sq - l2 * l2) / (2.0 * l1 * d)
        cos_alpha = min(max(cos_alpha, -1.0), 1.0)
        alpha = math.acos(cos_alpha)

        beta = math.atan2(z, dist_xy)
        j2 = math.degrees(beta + alpha)

        cos_gamma = (l1 * l1 + l2 * l2 - d_sq) / (2.0 * l1 * l2)
        cos_gamma = min(max(cos_gamma, -1.0), 1.0)
        gamma = math.acos(cos_gamma)
        j3 = j2 - (180.0 - math.degrees(gamma))
        j4 = r - j1
        return j1, j2, j3, j4

    def _mock_rotate_joint(self, target_j1: float, target_j2: float, target_j3: float, target_j4: float, wait: bool = True):
        if not wait:
            with self.lock:
                self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4 = target_j1, target_j2, target_j3, target_j4
            return

        with self.lock:
            start_j1, start_j2, start_j3, start_j4 = self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4

        max_delta = max(
            abs(target_j1 - start_j1),
            abs(target_j2 - start_j2),
            abs(target_j3 - start_j3),
            abs(target_j4 - start_j4)
        )
        steps = max(10, int(max_delta / 2.5))
        dt = 0.03
        for i in range(1, steps + 1):
            alpha = i / steps
            with self.lock:
                self.mock_j1 = start_j1 + (target_j1 - start_j1) * alpha
                self.mock_j2 = start_j2 + (target_j2 - start_j2) * alpha
                self.mock_j3 = start_j3 + (target_j3 - start_j3) * alpha
                self.mock_j4 = start_j4 + (target_j4 - start_j4) * alpha
            time.sleep(dt)

    def _mock_move_to(self, target_x: float, target_y: float, target_z: float, target_r: float, wait: bool = True):
        j1, j2, j3, j4 = self._inverse_kinematics(target_x, target_y, target_z, target_r)
        if wait:
            self._mock_rotate_joint(j1, j2, j3, j4, wait=True)
        else:
            with self.lock:
                self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4 = j1, j2, j3, j4
        with self.lock:
            self.mock_x, self.mock_y, self.mock_z, self.mock_r = target_x, target_y, target_z, target_r

    def move_to(self, x: float, y: float, z: float, r: float = 0.0, mode: int = MODE_PTP.MOVJ_XYZ, wait: bool = True):
        if not self.is_mock and self.arm:
            self.arm.move_to(x, y, z, r, mode=mode, wait=wait)
        else:
            self._mock_move_to(x, y, z, r, wait=wait)

    def rotate_joint(self, j1: float, j2: float, j3: float, j4: float = 0.0, mode: int = MODE_PTP.MOVJ_ANGLE, wait: bool = True):
        if not self.is_mock and self.arm:
            self.arm.rotate_joint(j1, j2, j3, j4, mode=mode, wait=wait)
        else:
            self._mock_rotate_joint(j1, j2, j3, j4, wait=wait)

    def home(self):
        if not self.is_mock and self.arm:
            self.arm.home()
        else:
            self._mock_rotate_joint(0.0, 0.0, 30.0, 0.0, wait=True)
            with self.lock:
                self.mock_x, self.mock_y, self.mock_z, self.mock_r = 200.0, 0.0, 0.0, 0.0

    def grip(self, mode: Union[str, bool, int]):
        m = str(mode).lower()
        if m in ["on", "in", "close", "1", "true"]:
            self.tool_state = "on"
        elif m in ["off", "blow", "open", "2"]:
            self.tool_state = "off"
        else:
            self.tool_state = "stop"
        self.tool_type = "gripper"

        if not self.is_mock and self.arm:
            self.arm.grip(mode)

    def suck(self, mode: Union[str, bool, int]):
        m = str(mode).lower()
        if m in ["in", "on", "suck", "1", "true"]:
            self.tool_state = "in"
        elif m in ["off", "blow", "out", "release", "2"]:
            self.tool_state = "off"
        else:
            self.tool_state = "stop"
        self.tool_type = "suction_cup"

        if not self.is_mock and self.arm:
            self.arm.suck(mode)

    def speed(self, velocity: float = 100.0, acceleration: float = 100.0):
        if not self.is_mock and self.arm:
            self.arm.speed(velocity=velocity, acceleration=acceleration)

    def clear_alarms(self):
        if not self.is_mock and self.arm:
            self.arm.clear_alarms()

    def get_pose(self):
        if not self.is_mock and self.arm:
            try:
                return self.arm.get_pose()
            except Exception:
                pass
        from dobot_driver.pydobot.dobot import Pose, Position, Joints
        with self.lock:
            return Pose(
                Position(self.mock_x, self.mock_y, self.mock_z, self.mock_r),
                Joints(self.mock_j1, self.mock_j2, self.mock_j3, self.mock_j4)
            )

    def close(self):
        if self.arm:
            self.arm.close()


def start_bridge_thread(bridge: DobotBridge):
    def spin():
        try:
            rclpy.spin(bridge)
        except Exception:
            pass
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    return t


def run_doasisay(port: Optional[str] = None, tool: str = "gripper"):
    bridge = DobotBridge(port=port, default_tool=tool)
    start_bridge_thread(bridge)

    try:
        readline.set_completer(SimpleCompleter([
            "move", "joint", "grip on", "grip off", "grip stop",
            "suck in", "suck off", "suck stop", "home", "speed",
            "pose", "clear", "exit", "quit"
        ]).complete)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    print("=" * 60)
    print("           DOBOT 'DO AS I SAY' INTERACTIVE CONSOLE")
    print("=" * 60)
    print(f"Active Attachment: {tool.upper()}")
    print("Available Commands:")
    print("  move <x> <y> <z> [r]       -> Move end-effector to coordinates (mm)")
    print("  joint <j1> <j2> <j3> [j4]  -> Rotate joints to angles (degrees)")
    print("  grip on / grip off / grip stop -> Gripper: close (clamp) / open (blow) / stop")
    print("  suck in / suck off / suck stop -> Suction: suck in / blow out / stop")
    print("  home                       -> Return Dobot to Home position")
    print("  speed <v> <a>              -> Set velocity and acceleration")
    print("  pose                       -> Print current robot pose & joint angles")
    print("  clear                      -> Clear any active alarms / errors")
    print("  exit / quit                -> Exit console")
    print("=" * 60)

    while True:
        try:
            cmd_line = input("\n[Dobot-Exec]> ").strip()
            if not cmd_line:
                continue
            parts = cmd_line.split()
            cmd = parts[0].lower()

            if cmd in ["exit", "quit", "q"]:
                print("Exiting DoAsISay...")
                break
            elif cmd == "home":
                print(">> Homing Dobot...")
                bridge.home()
                print(">> Done.")
            elif cmd == "pose":
                p = bridge.get_pose()
                print(f"Position: X={p.position.x:.2f} Y={p.position.y:.2f} Z={p.position.z:.2f} R={p.position.r:.2f}")
                print(f"Joints:   J1={p.joints.j1:.2f}° J2={p.joints.j2:.2f}° J3={p.joints.j3:.2f}° J4={p.joints.j4:.2f}°")
                print(f"Tool:     Type={bridge.tool_type}, State={bridge.tool_state}")
            elif cmd == "move":
                if len(parts) < 4:
                    print("Usage: move <x> <y> <z> [r]")
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r = float(parts[4]) if len(parts) > 4 else 0.0
                print(f">> Moving to X={x}, Y={y}, Z={z}, R={r}...")
                bridge.move_to(x, y, z, r, mode=MODE_PTP.MOVJ_XYZ, wait=True)
                print(">> Reached target position.")
            elif cmd == "joint":
                if len(parts) < 4:
                    print("Usage: joint <j1> <j2> <j3> [j4]")
                    continue
                j1, j2, j3 = float(parts[1]), float(parts[2]), float(parts[3])
                j4 = float(parts[4]) if len(parts) > 4 else 0.0
                print(f">> Rotating joints to J1={j1}°, J2={j2}°, J3={j3}°, J4={j4}°...")
                bridge.rotate_joint(j1, j2, j3, j4, mode=MODE_PTP.MOVJ_ANGLE, wait=True)
                print(">> Reached target joints.")
            elif cmd == "grip":
                sub = parts[1].lower() if len(parts) > 1 else "on"
                if sub in ["on", "in", "clamp", "1"]:
                    print(">> Setting Gripper: ON (Clamp / Suck)")
                    bridge.grip("on")
                elif sub in ["off", "blow", "open", "out", "2"]:
                    print(">> Setting Gripper: OFF (Blow / Open)")
                    bridge.grip("off")
                elif sub in ["stop", "0"]:
                    print(">> Setting Gripper: STOP (Valve & Pump Off)")
                    bridge.grip("stop")
                else:
                    print("Usage: grip on | grip off | grip stop")
            elif cmd == "suck":
                sub = parts[1].lower() if len(parts) > 1 else "in"
                if sub in ["in", "on", "suck", "1"]:
                    print(">> Setting Suction: IN (Suck In)")
                    bridge.suck("in")
                elif sub in ["off", "blow", "out", "release", "2"]:
                    print(">> Setting Suction: OFF (Blow Out / Release)")
                    bridge.suck("off")
                elif sub in ["stop", "0"]:
                    print(">> Setting Suction: STOP (Valve & Pump Off)")
                    bridge.suck("stop")
                else:
                    print("Usage: suck in | suck off | suck stop")
            elif cmd == "speed":
                if len(parts) < 3:
                    print("Usage: speed <velocity> <acceleration>")
                    continue
                v, a = float(parts[1]), float(parts[2])
                print(f">> Setting speed: velocity={v}, accel={a}")
                bridge.speed(velocity=v, acceleration=a)
            elif cmd == "clear":
                print(">> Clearing alarms...")
                bridge.clear_alarms()
                print(">> Alarms cleared.")
            else:
                print(f"Unknown command: '{cmd}'. Type 'exit' to quit.")
        except KeyboardInterrupt:
            print("\nExiting DoAsISay...")
            break
        except Exception as e:
            print(f"[!] Error executing command: {e}")

    bridge.close()


def run_learn(port: Optional[str] = None, tool: str = "gripper"):
    ensure_learnings_dir()
    bridge = DobotBridge(port=port, default_tool=tool)
    start_bridge_thread(bridge)

    try:
        readline.set_completer(SimpleCompleter([
            "record", "r", "grip on", "grip off", "grip stop",
            "suck in", "suck off", "suck stop", "show", "list",
            "save", "done", "exit", "quit"
        ]).complete)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    print("=" * 60)
    print("            DOBOT TEACH & LEARN (WAYPOINT RECORDER)")
    print("=" * 60)
    print(f"Active Attachment: {tool.upper()}")
    print("Instructions:")
    print("  1. Press the unlocking button on the Dobot forearm to freely move the arm.")
    print("  2. Position the arm at your desired waypoint.")
    print("  3. Set tool state if needed:")
    print("     - Gripper: 'grip on' (clamp), 'grip off' (blow/open), 'grip stop'")
    print("     - Suction: 'suck in' (vacuum), 'suck off' (blow out), 'suck stop'")
    print("  4. Type 'record' (or 'r') to capture the current waypoint.")
    print("  5. Type 'save <name>' or 'done' to save your learning trajectory.")
    print("=" * 60)

    trajectory = []
    current_tool_type = "suction_cup" if tool == "suction_cup" else "gripper"
    current_tool_action = "stop"

    while True:
        try:
            cmd_line = input("\n[Learn-Mode]> ").strip()
            if not cmd_line:
                continue
            parts = cmd_line.split()
            cmd = parts[0].lower()

            if cmd in ["r", "record", "rec"]:
                p = bridge.get_pose()
                wp = {
                    "type": "coord",
                    "x": round(p.position.x, 2),
                    "y": round(p.position.y, 2),
                    "z": round(p.position.z, 2),
                    "r": round(p.position.r, 2),
                    "j1": round(p.joints.j1, 2),
                    "j2": round(p.joints.j2, 2),
                    "j3": round(p.joints.j3, 2),
                    "j4": round(p.joints.j4, 2),
                    "tool_type": current_tool_type,
                    "tool_action": current_tool_action,
                    "pause": 1.0
                }
                trajectory.append(wp)
                print(f"[✓] Recorded Waypoint #{len(trajectory)}: X={wp['x']}, Y={wp['y']}, Z={wp['z']}, R={wp['r']} | Tool={wp['tool_type']}:{wp['tool_action']}")

            elif cmd == "grip":
                sub = parts[1].lower() if len(parts) > 1 else "on"
                current_tool_type = "gripper"
                if sub in ["on", "in", "clamp", "1"]:
                    current_tool_action = "on"
                    print(">> Tool set to: GRIP ON (Clamp)")
                    bridge.grip("on")
                elif sub in ["off", "blow", "open", "out", "2"]:
                    current_tool_action = "off"
                    print(">> Tool set to: GRIP OFF (Blow / Open)")
                    bridge.grip("off")
                else:
                    current_tool_action = "stop"
                    print(">> Tool set to: GRIP STOP")
                    bridge.grip("stop")

            elif cmd == "suck":
                sub = parts[1].lower() if len(parts) > 1 else "in"
                current_tool_type = "suction_cup"
                if sub in ["in", "on", "suck", "1"]:
                    current_tool_action = "in"
                    print(">> Tool set to: SUCK IN (Vacuum)")
                    bridge.suck("in")
                elif sub in ["off", "blow", "out", "release", "2"]:
                    current_tool_action = "off"
                    print(">> Tool set to: SUCK OFF (Blow out / Release)")
                    bridge.suck("off")
                else:
                    current_tool_action = "stop"
                    print(">> Tool set to: SUCK STOP")
                    bridge.suck("stop")

            elif cmd in ["show", "list"]:
                print(f"\nCurrently recorded {len(trajectory)} waypoints:")
                for i, wp in enumerate(trajectory, 1):
                    ttype = wp.get("tool_type", "gripper")
                    taction = wp.get("tool_action", "on" if wp.get("tool") else "stop")
                    print(f"  #{i}: X={wp['x']}, Y={wp['y']}, Z={wp['z']}, R={wp['r']} | Tool={ttype}:{taction}")

            elif cmd in ["save", "done"]:
                if not trajectory:
                    print("[!] No waypoints recorded yet! Move arm and type 'record'.")
                    continue
                if len(parts) > 1:
                    name = parts[1].strip()
                else:
                    name = input("Enter a name for this learning session: ").strip()
                if not name:
                    name = f"learning_{int(time.time())}"
                if not name.endswith(".json"):
                    name += ".json"
                filepath = os.path.join(LEARNINGS_DIR, name)
                with open(filepath, "w") as f:
                    json.dump({"name": name, "timestamp": time.time(), "waypoints": trajectory}, f, indent=2)
                print(f"\n[✓] Successfully saved {len(trajectory)} waypoints to: {filepath}")
                break
            elif cmd in ["exit", "quit", "q"]:
                print("Exiting without saving.")
                break
            else:
                print("Commands: 'record', 'grip on/off/stop', 'suck in/off/stop', 'show', 'save <name>', 'exit'")
        except KeyboardInterrupt:
            print("\nExiting Learn session...")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

    bridge.close()


def run_replay(port: Optional[str] = None, replay_name: Optional[str] = None, tool: str = "gripper"):
    ensure_learnings_dir()
    files = [f for f in os.listdir(LEARNINGS_DIR) if f.endswith(".json")]

    if not files:
        print(f"[!] No saved learnings found in {LEARNINGS_DIR}.")
        print("[!] Run ./learn.sh first to record movements from the Dobot.")
        return

    selected_file = None
    if replay_name:
        if not replay_name.endswith(".json"):
            replay_name += ".json"
        if replay_name in files:
            selected_file = replay_name
        else:
            print(f"[!] Learning '{replay_name}' not found.")

    if not selected_file:
        print("=" * 60)
        print("              DOBOT REPLAY LEARNED TRAJECTORIES")
        print("=" * 60)
        print(f"Active Attachment: {tool.upper()}")
        print("Available Saved Learnings:")
        for idx, fname in enumerate(files, 1):
            print(f"  [{idx}] {fname}")
        print("=" * 60)

        while True:
            try:
                choice = input(f"Select trajectory to replay [1-{len(files)}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(files):
                    selected_file = files[int(choice) - 1]
                    break
                elif choice in files:
                    selected_file = choice
                    break
                elif choice.lower() in ["exit", "quit", "q"]:
                    return
                else:
                    print("Invalid selection. Try again.")
            except KeyboardInterrupt:
                return

    filepath = os.path.join(LEARNINGS_DIR, selected_file)
    with open(filepath, "r") as f:
        data = json.load(f)

    waypoints = data.get("waypoints", [])
    print(f"\n>> Loaded trajectory '{selected_file}' with {len(waypoints)} waypoints.")

    loops_input = input("How many times to loop replay? (default: 1): ").strip()
    loops = int(loops_input) if loops_input.isdigit() else 1

    bridge = DobotBridge(port=port, default_tool=tool)
    start_bridge_thread(bridge)

    print("\n>> Starting Replay in 2 seconds (Press Ctrl+C to abort)...")
    time.sleep(2.0)

    try:
        for loop in range(1, loops + 1):
            print(f"\n--- Executing Loop {loop}/{loops} ---")
            for idx, wp in enumerate(waypoints, 1):
                x, y, z, r = wp.get("x", 200.0), wp.get("y", 0.0), wp.get("z", 0.0), wp.get("r", 0.0)
                pause = wp.get("pause", 0.5)

                # Backwards compatible tool state resolution
                tool_type = wp.get("tool_type", tool)
                tool_action = wp.get("tool_action")
                if tool_action is None:
                    # Legacy boolean
                    tool_action = "on" if wp.get("tool", False) else "stop"

                print(f"  [Waypoint {idx}/{len(waypoints)}] Moving to X={x}, Y={y}, Z={z}, R={r} | Tool={tool_type}:{tool_action}")

                # 1. Move to target coordinate
                bridge.move_to(x, y, z, r, mode=MODE_PTP.MOVJ_XYZ, wait=True)

                # 2. Actuate tool
                if tool_type == "suction_cup":
                    bridge.suck(tool_action)
                else:
                    bridge.grip(tool_action)

                time.sleep(pause)
        print("\n[✓] Trajectory replay completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Replay interrupted by user.")
    except Exception as e:
        print(f"[!] Replay error: {e}")
    finally:
        bridge.close()


def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="Dobot Task Runner")
    parser.add_argument("action", choices=["doasisay", "learn", "replay"], help="Action to perform")
    parser.add_argument("--port", type=str, default="", help="Serial port for Dobot")
    parser.add_argument("--name", type=str, default="", help="Learning file name for replay")
    parser.add_argument("--tool", type=str, default="gripper", help="End-effector attachment type")
    args = parser.parse_args()

    port = args.port if args.port else None
    try:
        if args.action == "doasisay":
            run_doasisay(port=port, tool=args.tool)
        elif args.action == "learn":
            run_learn(port=port, tool=args.tool)
        elif args.action == "replay":
            run_replay(port=port, replay_name=args.name, tool=args.tool)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
