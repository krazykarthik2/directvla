import math
import struct
import threading
import time
from enum import IntEnum
from typing import NamedTuple, Optional, Set, Union

import serial
from serial.tools import list_ports

from .message import Message


class MODE_PTP(IntEnum):
    JUMP_XYZ = 0x00
    MOVJ_XYZ = 0x01
    MOVL_XYZ = 0x02
    JUMP_ANGLE = 0x03
    MOVJ_ANGLE = 0x04
    MOVL_ANGLE = 0x05
    MOVJ_INC = 0x06
    MOVL_INC = 0x07
    MOVJ_XYZ_INC = 0x08
    JUMP_MOVL_XYZ = 0x09


STEP_PER_CIRCLE = 360.0 / 1.8 * 10.0 * 16.0
MM_PER_CIRCLE = 3.1415926535898 * 36.0


class DobotException(Exception):
    pass


class Position(NamedTuple):
    x: float
    y: float
    z: float
    r: float


class Joints(NamedTuple):
    j1: float
    j2: float
    j3: float
    j4: float

    def in_radians(self) -> "Joints":
        return Joints(
            math.radians(self.j1),
            math.radians(self.j2),
            math.radians(self.j3),
            math.radians(self.j4),
        )


class Pose(NamedTuple):
    position: Position
    joints: Joints


class Alarm(IntEnum):
    COMMON_RESETTING = 0x00
    COMMON_UNDEFINED_INSTRUCTION = 0x01
    COMMON_FILE_SYSTEM = 0x02
    COMMON_MCU_FPGA_COMM = 0x03
    COMMON_ANGLE_SENSOR = 0x04

    PLAN_INV_SINGULARITY = 0x10
    PLAN_INV_CALC = 0x11
    PLAN_INV_LIMIT = 0x12
    PLAN_PUSH_DATA_REPEAT = 0x13
    PLAN_ARC_INPUT_PARAM = 0x14
    PLAN_JUMP_PARAM = 0x15
    PLAN_LINE_HAND = 0x16
    PLAN_LINE_OUT_SPACE = 0x17
    PLAN_ARC_OUT_SPACE = 0x18
    PLAN_MOTIONTYPE = 0x19
    PLAN_SPEED_INPUT_PARAM = 0x1A
    PLAN_CP_CALC = 0x1B

    MOVE_INV_SINGULARITY = 0x20
    MOVE_INV_CALC = 0x21
    MOVE_INV_LIMIT = 0x22

    OVERSPEED_AXIS1 = 0x30
    OVERSPEED_AXIS2 = 0x31
    OVERSPEED_AXIS3 = 0x32
    OVERSPEED_AXIS4 = 0x33

    LIMIT_AXIS1_POS = 0x40
    LIMIT_AXIS1_NEG = 0x41
    LIMIT_AXIS2_POS = 0x42
    LIMIT_AXIS2_NEG = 0x43
    LIMIT_AXIS3_POS = 0x44
    LIMIT_AXIS3_NEG = 0x45
    LIMIT_AXIS4_POS = 0x46
    LIMIT_AXIS4_NEG = 0x47
    LIMIT_AXIS23_POS = 0x48
    LIMIT_AXIS23_NEG = 0x49

    LOSE_STEP_AXIS1 = 0x50
    LOSE_STEP_AXIS2 = 0x51
    LOSE_STEP_AXIS3 = 0x52
    LOSE_STEP_AXIS4 = 0x53

    OTHER_AXIS1_DRV_ALARM = 0x60
    OTHER_AXIS1_OVERFLOW = 0x61
    OTHER_AXIS1_FOLLOW = 0x62
    OTHER_AXIS2_DRV_ALARM = 0x63
    OTHER_AXIS2_OVERFLOW = 0x64
    OTHER_AXIS2_FOLLOW = 0x65
    OTHER_AXIS3_DRV_ALARM = 0x66
    OTHER_AXIS3_OVERFLOW = 0x67
    OTHER_AXIS3_FOLLOW = 0x68
    OTHER_AXIS4_DRV_ALARM = 0x69
    OTHER_AXIS4_OVERFLOW = 0x6A
    OTHER_AXIS4_FOLLOW = 0x6B


class Dobot:
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.5):
        if port is None:
            ports = list_ports.comports()
            for p in ports:
                if "ttyUSB" in p.device or "ttyACM" in p.device:
                    port = p.device
                    break
            if port is None:
                raise DobotException("No Dobot serial port found (checked ttyUSB*, ttyACM*).")

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._lock = threading.Lock()
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=self.timeout,
            )
        except Exception as e:
            raise DobotException(f"Failed to open port {self.port}: {e}")

        time.sleep(0.5)
        self.clear_alarms()
        self._set_queued_cmd_start_exec()
        self._set_queued_cmd_clear()
        self.speed(velocity=100.0, acceleration=100.0)

    def close(self):
        with self._lock:
            if hasattr(self, "_ser") and self._ser.is_open:
                self._ser.close()

    def _send_command(self, msg: Message) -> Optional[Message]:
        with self._lock:
            self._send_message(msg)
            return self._read_message()

    def _send_message(self, msg: Message):
        time.sleep(0.005)
        if self._ser.is_open:
            self._ser.write(msg.bytes())

    def _read_message(self) -> Optional[Message]:
        tries = 10
        begin_found = False
        last_byte = None

        while not begin_found and tries > 0:
            raw = self._ser.read(1)
            if not raw:
                tries -= 1
                continue
            current_byte = ord(raw)
            if last_byte == 0xAA and current_byte == 0xAA:
                begin_found = True
            last_byte = current_byte

        if not begin_found:
            return None

        raw_len = self._ser.read(1)
        if not raw_len:
            return None
        payload_length = ord(raw_len)
        payload_checksum = self._ser.read(payload_length + 1)
        if len(payload_checksum) < payload_length + 1:
            return None

        b = bytearray([0xAA, 0xAA, payload_length])
        b.extend(payload_checksum)
        msg = Message(b)
        return msg

    def get_pose(self) -> Pose:
        msg = Message()
        msg.id = 10
        msg.ctrl = 0
        response = self._send_command(msg)
        if response is None or len(response.params) < 32:
            raise DobotException("Invalid get_pose response")

        params = response.params
        x = struct.unpack_from("f", params, 0)[0]
        y = struct.unpack_from("f", params, 4)[0]
        z = struct.unpack_from("f", params, 8)[0]
        r = struct.unpack_from("f", params, 12)[0]
        j1 = struct.unpack_from("f", params, 16)[0]
        j2 = struct.unpack_from("f", params, 20)[0]
        j3 = struct.unpack_from("f", params, 24)[0]
        j4 = struct.unpack_from("f", params, 28)[0]
        return Pose(Position(x, y, z, r), Joints(j1, j2, j3, j4))

    def get_alarms(self) -> Set[Alarm]:
        msg = Message()
        msg.id = 20
        msg.ctrl = 0
        response = self._send_command(msg)
        ret: Set[Alarm] = set()
        if response is None:
            return ret
        for idx in range(min(len(response.params), 32)):
            alarm_byte = struct.unpack_from("B", response.params, idx)[0]
            if alarm_byte != 0:
                for bit in range(8):
                    if (alarm_byte >> bit) & 0x01:
                        try:
                            ret.add(Alarm(idx * 8 + bit))
                        except ValueError:
                            pass
        return ret

    def clear_alarms(self):
        msg = Message()
        msg.id = 20
        msg.ctrl = 1
        self._send_command(msg)

    def _set_queued_cmd_start_exec(self):
        msg = Message()
        msg.id = 240
        msg.ctrl = 1
        return self._send_command(msg)

    def _set_queued_cmd_stop_exec(self):
        msg = Message()
        msg.id = 241
        msg.ctrl = 1
        return self._send_command(msg)

    def _set_queued_cmd_clear(self):
        msg = Message()
        msg.id = 245
        msg.ctrl = 1
        return self._send_command(msg)

    def _get_queued_cmd_current_index(self) -> int:
        msg = Message()
        msg.id = 246
        msg.ctrl = 0
        response = self._send_command(msg)
        if response is None or len(response.params) < 8:
            return 0
        return struct.unpack_from("Q", response.params, 0)[0]

    def _extract_cmd_index(self, response: Optional[Message]) -> int:
        if response is None or len(response.params) < 8:
            return 0
        return struct.unpack_from("Q", response.params, 0)[0]

    def wait_for_cmd(self, cmd_id: int, timeout: float = 15.0):
        if cmd_id <= 0:
            time.sleep(1.0)
            return True
        start = time.time()
        while time.time() - start < timeout:
            current_id = self._get_queued_cmd_current_index()
            if current_id >= cmd_id:
                time.sleep(0.1)
                return True
            time.sleep(0.05)
        return False

    def speed(self, velocity: float = 100.0, acceleration: float = 100.0):
        # Joint params (ID 80)
        msg = Message()
        msg.id = 80
        msg.ctrl = 1
        msg.params = bytearray()
        for _ in range(4):
            msg.params.extend(struct.pack("f", float(velocity)))
        for _ in range(4):
            msg.params.extend(struct.pack("f", float(acceleration)))
        self._send_command(msg)

        # Coordinate params (ID 81)
        msg2 = Message()
        msg2.id = 81
        msg2.ctrl = 1
        msg2.params = bytearray()
        for _ in range(2):
            msg2.params.extend(struct.pack("f", float(velocity)))
        for _ in range(2):
            msg2.params.extend(struct.pack("f", float(acceleration)))
        self._send_command(msg2)

        # Common params (ID 83)
        msg3 = Message()
        msg3.id = 83
        msg3.ctrl = 1
        msg3.params = bytearray()
        msg3.params.extend(struct.pack("f", float(velocity)))
        msg3.params.extend(struct.pack("f", float(acceleration)))
        self._send_command(msg3)

    def move_to(self, x: float, y: float, z: float, r: float = 0.0, mode: int = MODE_PTP.MOVJ_XYZ, wait: bool = False):
        self.clear_alarms()
        self._set_queued_cmd_start_exec()
        msg = Message()
        msg.id = 84
        msg.ctrl = 0x03
        msg.params = bytearray([mode])
        msg.params.extend(struct.pack("f", float(x)))
        msg.params.extend(struct.pack("f", float(y)))
        msg.params.extend(struct.pack("f", float(z)))
        msg.params.extend(struct.pack("f", float(r)))
        res = self._send_command(msg)
        cmd_id = self._extract_cmd_index(res)
        if wait:
            self.wait_for_cmd(cmd_id)
        return cmd_id

    def rotate_joint(self, j1: float, j2: float, j3: float, j4: float = 0.0, mode: int = MODE_PTP.MOVJ_ANGLE, wait: bool = False):
        self.clear_alarms()
        self._set_queued_cmd_start_exec()
        msg = Message()
        msg.id = 84
        msg.ctrl = 0x03
        msg.params = bytearray([mode])
        msg.params.extend(struct.pack("f", float(j1)))
        msg.params.extend(struct.pack("f", float(j2)))
        msg.params.extend(struct.pack("f", float(j3)))
        msg.params.extend(struct.pack("f", float(j4)))
        res = self._send_command(msg)
        cmd_id = self._extract_cmd_index(res)
        if wait:
            self.wait_for_cmd(cmd_id)
        return cmd_id

    def grip(self, mode: Union[str, bool, int] = "on"):
        """
        Dobot Pneumatic Gripper control:
        - "on" / "in" / "clamp" / True / 1: Close/Grip (Pump ON, Valve ON [1, 1])
        - "off" / "blow" / "open" / 2: Open/Release (Pump ON, Valve OFF [1, 0])
        - "stop" / False / 0: Stop pump & valve (Pump OFF, Valve OFF [0, 0])
        """
        if isinstance(mode, bool):
            enable_ctrl, valve_state = (1, 1) if mode else (0, 0)
        elif isinstance(mode, int):
            if mode == 1:
                enable_ctrl, valve_state = 1, 1
            elif mode == 2:
                enable_ctrl, valve_state = 1, 0
            else:
                enable_ctrl, valve_state = 0, 0
        else:
            m = str(mode).lower()
            if m in ["on", "in", "clamp", "close", "1", "true"]:
                # Grip on (Close): [1, 1]
                enable_ctrl, valve_state = 1, 1
            elif m in ["off", "blow", "open", "release", "2"]:
                # Grip off (Open): [1, 0]
                enable_ctrl, valve_state = 1, 0
            elif m in ["stop", "none", "0", "false"]:
                # Stop: [0, 0]
                enable_ctrl, valve_state = 0, 0
            else:
                enable_ctrl, valve_state = (1, 1) if m else (0, 0)

        msg = Message()
        msg.id = 63
        msg.ctrl = 0x03
        msg.params = bytearray([enable_ctrl, valve_state])
        self._send_command(msg)

    def suck(self, mode: Union[str, bool, int] = "in"):
        """
        Dobot Suction Cup control:
        Uses the shared pneumatic pump & valve interface (Command ID 63):
        - suck in / on:  Pump ON, Valve ON  [1, 1]
        - suck off / blow: Pump ON, Valve OFF [1, 0]
        - suck stop:     Pump OFF, Valve OFF [0, 0]
        """
        if isinstance(mode, bool):
            enable_ctrl, valve_state = (1, 1) if mode else (0, 0)
        elif isinstance(mode, int):
            if mode == 1:
                enable_ctrl, valve_state = 1, 1
            elif mode == 2:
                enable_ctrl, valve_state = 1, 0
            else:
                enable_ctrl, valve_state = 0, 0
        else:
            m = str(mode).lower()
            if m in ["in", "on", "suck", "1", "true"]:
                # Suck in / on: [1, 1]
                enable_ctrl, valve_state = 1, 1
            elif m in ["off", "blow", "out", "release", "2"]:
                # Suck off / blow: [1, 0]
                enable_ctrl, valve_state = 1, 0
            elif m in ["stop", "none", "0", "false"]:
                # Stop: [0, 0]
                enable_ctrl, valve_state = 0, 0
            else:
                enable_ctrl, valve_state = (1, 1) if m else (0, 0)

        msg = Message()
        msg.id = 63
        msg.ctrl = 0x03
        msg.params = bytearray([enable_ctrl, valve_state])
        self._send_command(msg)

    def home(self):
        self.clear_alarms()
        self._set_queued_cmd_start_exec()
        msg = Message()
        msg.id = 31
        msg.ctrl = 0x03
        msg.params = bytearray([0])
        res = self._send_command(msg)
        cmd_id = self._extract_cmd_index(res)
        if cmd_id > 0:
            self.wait_for_cmd(cmd_id, timeout=30.0)
        return cmd_id
