"""Safe, finger-only adapter for a physical 16-DoF LEAP Hand.

The vision mapper uses the MuJoCo Menagerie joint order, while the physical
hand follows the official LEAP motor order.  This module is the only place
where that permutation and the physical Dynamixel offset are applied.
Importing this module never opens a serial port or enables torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

MAPPING_JOINT_ORDER = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

MOTOR_JOINT_ORDER = (
    "if_rot",
    "if_mcp",
    "if_pip",
    "if_dip",
    "mf_rot",
    "mf_mcp",
    "mf_pip",
    "mf_dip",
    "rf_rot",
    "rf_mcp",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_mcp",
    "th_axl",
    "th_ipl",
)

# motor_sim[i] = mapping[MOTOR_FROM_MAPPING[i]]
MOTOR_FROM_MAPPING = np.asarray(
    (1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 14, 13, 15),
    dtype=np.int64,
)
MAPPING_FROM_MOTOR = np.argsort(MOTOR_FROM_MAPPING)

# Official LEAPsim limits in physical motor order.
MOTOR_SIM_LOWER_RAD = np.asarray(
    (
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -1.047,
        -0.314,
        -0.506,
        -0.366,
        -0.349,
        -0.470,
        -1.200,
        -1.340,
    ),
    dtype=np.float64,
)
MOTOR_SIM_UPPER_RAD = np.asarray(
    (
        1.047,
        2.230,
        1.885,
        2.042,
        1.047,
        2.230,
        1.885,
        2.042,
        1.047,
        2.230,
        1.885,
        2.042,
        2.094,
        2.443,
        1.900,
        1.880,
    ),
    dtype=np.float64,
)
MAPPING_LOWER_RAD = MOTOR_SIM_LOWER_RAD[MAPPING_FROM_MOTOR]
MAPPING_UPPER_RAD = MOTOR_SIM_UPPER_RAD[MAPPING_FROM_MOTOR]

PROTOCOL_VERSION = 2.0
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_CURRENT_LIMIT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
POSITION_CURRENT_MODE = 5
POSITION_BYTES = 4
TICKS_PER_REVOLUTION = 4096.0


def _joint_vector(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain 16 finite values")
    return result


def mapping_to_motor_sim(values: Any, *, clip: bool = True) -> np.ndarray:
    """Convert mapper order to official physical motor order."""

    mapping = _joint_vector(values, "mapping joint targets")
    if clip:
        mapping = np.clip(mapping, MAPPING_LOWER_RAD, MAPPING_UPPER_RAD)
    return mapping[MOTOR_FROM_MAPPING]


def motor_sim_to_mapping(values: Any) -> np.ndarray:
    """Convert official physical motor order back to mapper order."""

    motor = _joint_vector(values, "motor joint values")
    return motor[MAPPING_FROM_MOTOR]


def motor_sim_to_real(values: Any) -> np.ndarray:
    """Apply the official +pi physical LEAP motor zero offset."""

    motor = _joint_vector(values, "motor joint values")
    return motor + math.pi


def motor_real_to_sim(values: Any) -> np.ndarray:
    motor = _joint_vector(values, "real motor positions")
    return motor - math.pi


def real_radians_to_ticks(values: Any) -> np.ndarray:
    real = _joint_vector(values, "real motor positions")
    return np.rint(real * TICKS_PER_REVOLUTION / math.tau).astype(np.int64)


def ticks_to_real_radians(values: Any) -> np.ndarray:
    ticks = _joint_vector(values, "motor encoder ticks")
    return ticks * math.tau / TICKS_PER_REVOLUTION


@dataclass(frozen=True)
class HardwareSettings:
    motor_ids: tuple[int, ...]
    baudrate: int
    kp: int
    ki: int
    kd: int
    current_limit: int
    side_gain_scale: float
    maximum_step_rad: float
    stable_tracking_frames: int
    disable_torque_on_exit: bool

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> HardwareSettings:
        if not isinstance(value, dict):
            raise ValueError("hardware config must be a mapping")
        motor_ids_raw = value.get("motor_ids")
        if not isinstance(motor_ids_raw, (list, tuple)):
            raise ValueError("hardware.motor_ids must be a list")
        settings = cls(
            motor_ids=tuple(int(item) for item in motor_ids_raw),
            baudrate=int(value["baudrate"]),
            kp=int(value["kp"]),
            ki=int(value["ki"]),
            kd=int(value["kd"]),
            current_limit=int(value["current_limit"]),
            side_gain_scale=float(value["side_gain_scale"]),
            maximum_step_rad=float(value["maximum_step_rad"]),
            stable_tracking_frames=int(value["stable_tracking_frames"]),
            disable_torque_on_exit=bool(value["disable_torque_on_exit"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if len(self.motor_ids) != 16 or len(set(self.motor_ids)) != 16:
            raise ValueError("hardware.motor_ids must contain 16 unique IDs")
        if any(motor_id < 0 or motor_id > 252 for motor_id in self.motor_ids):
            raise ValueError("hardware.motor_ids must be in [0, 252]")
        if self.baudrate <= 0:
            raise ValueError("hardware.baudrate must be positive")
        if not 0 <= self.kp <= 16383 or not 0 <= self.ki <= 16383:
            raise ValueError("hardware kp/ki gains must be in [0, 16383]")
        if not 0 <= self.kd <= 16383:
            raise ValueError("hardware.kd must be in [0, 16383]")
        if not 1 <= self.current_limit <= 1000:
            raise ValueError("hardware.current_limit must be in [1, 1000]")
        if not 0.0 < self.side_gain_scale <= 1.0:
            raise ValueError("hardware.side_gain_scale must be in (0, 1]")
        if not 0.0 < self.maximum_step_rad <= 0.25:
            raise ValueError("hardware.maximum_step_rad must be in (0, 0.25]")
        if not 1 <= self.stable_tracking_frames <= 120:
            raise ValueError("hardware.stable_tracking_frames must be in [1, 120]")


class MockLeapHand:
    """In-memory hand used to verify the camera/mapping path without motors."""

    def __init__(
        self,
        settings: HardwareSettings,
        initial_mapping_rad: Any | None = None,
    ) -> None:
        self.settings = settings
        initial = (
            np.zeros(16, dtype=np.float64)
            if initial_mapping_rad is None
            else _joint_vector(initial_mapping_rad, "initial mapping pose")
        )
        self.last_mapping_rad = np.clip(initial, MAPPING_LOWER_RAD, MAPPING_UPPER_RAD)
        self.command_count = 0
        self.torque_enabled = False

    def connect_and_enable(self) -> np.ndarray:
        self.torque_enabled = True
        return self.last_mapping_rad.copy()

    def command_mapping(self, targets: Any) -> np.ndarray:
        if not self.torque_enabled:
            raise RuntimeError("mock LEAP torque is not enabled")
        desired = np.clip(
            _joint_vector(targets, "mapping joint targets"),
            MAPPING_LOWER_RAD,
            MAPPING_UPPER_RAD,
        )
        delta = np.clip(
            desired - self.last_mapping_rad,
            -self.settings.maximum_step_rad,
            self.settings.maximum_step_rad,
        )
        self.last_mapping_rad += delta
        self.command_count += 1
        return self.last_mapping_rad.copy()

    def read_mapping_positions(self) -> np.ndarray:
        return self.last_mapping_rad.copy()

    def close(self) -> None:
        self.torque_enabled = False


class DynamixelLeapHand:
    """Minimal Protocol 2.0 driver matching the official LEAP Hand v1 API."""

    def __init__(
        self,
        settings: HardwareSettings,
        port: str,
        *,
        sdk_module: ModuleType | Any | None = None,
    ) -> None:
        if not port.strip():
            raise ValueError("a serial port is required for the real LEAP Hand")
        self.settings = settings
        self.port_name = port.strip()
        self._sdk = sdk_module
        self._port: Any | None = None
        self._packet: Any | None = None
        self._port_open = False
        self._torque_may_be_enabled = False
        self._last_motor_sim_rad: np.ndarray | None = None

    @property
    def torque_enabled(self) -> bool:
        return self._torque_may_be_enabled

    def connect_and_enable(self) -> np.ndarray:
        if self._port_open:
            raise RuntimeError("LEAP serial port is already open")
        sdk = self._load_sdk()
        self._port = sdk.PortHandler(self.port_name)
        self._packet = sdk.PacketHandler(PROTOCOL_VERSION)
        if not self._port.openPort():
            raise RuntimeError(f"cannot open LEAP serial port {self.port_name}")
        self._port_open = True
        try:
            if not self._port.setBaudRate(self.settings.baudrate):
                raise RuntimeError(
                    f"cannot set {self.port_name} to {self.settings.baudrate} baud"
                )
            self._verify_all_motors()
            self._sync_write(ADDR_TORQUE_ENABLE, 1, [0] * 16)
            self._configure_motors()

            current_real = self._read_real_positions()
            if np.any(current_real < -0.10) or np.any(current_real > math.tau + 0.10):
                raise RuntimeError(
                    "a motor reported a position outside the expected single-turn range"
                )
            # Seed the goal with the measured pose before torque is enabled.  This
            # prevents the hand from jumping to an arbitrary startup posture.
            self._write_real_positions(current_real)
            self._torque_may_be_enabled = True
            self._sync_write(ADDR_TORQUE_ENABLE, 1, [1] * 16)
            self._last_motor_sim_rad = motor_real_to_sim(current_real)
            return motor_sim_to_mapping(self._last_motor_sim_rad)
        except Exception:
            self.close()
            raise

    def command_mapping(self, targets: Any) -> np.ndarray:
        if not self._port_open or not self._torque_may_be_enabled:
            raise RuntimeError("real LEAP Hand is not enabled")
        if self._last_motor_sim_rad is None:
            raise RuntimeError("real LEAP Hand has no startup position")
        desired = mapping_to_motor_sim(targets, clip=True)
        delta = np.clip(
            desired - self._last_motor_sim_rad,
            -self.settings.maximum_step_rad,
            self.settings.maximum_step_rad,
        )
        next_motor_sim = self._last_motor_sim_rad + delta
        self._write_real_positions(motor_sim_to_real(next_motor_sim))
        self._last_motor_sim_rad = next_motor_sim
        return motor_sim_to_mapping(next_motor_sim)

    def read_mapping_positions(self) -> np.ndarray:
        return motor_sim_to_mapping(motor_real_to_sim(self._read_real_positions()))

    def close(self) -> None:
        if self._port_open and self._port is not None:
            if self.settings.disable_torque_on_exit:
                try:
                    self._sync_write(ADDR_TORQUE_ENABLE, 1, [0] * 16)
                except Exception:
                    pass
            self._torque_may_be_enabled = False
            try:
                self._port.closePort()
            finally:
                self._port_open = False

    def _load_sdk(self) -> ModuleType | Any:
        if self._sdk is None:
            try:
                import dynamixel_sdk as sdk
            except ImportError as error:
                raise RuntimeError(
                    "dynamixel_sdk is required only for --device real; install "
                    "teleoperation/requirements-leaptele.txt into leaptele"
                ) from error
            self._sdk = sdk
        return self._sdk

    def _verify_all_motors(self) -> None:
        assert self._packet is not None and self._port is not None
        missing: list[str] = []
        for motor_id in self.settings.motor_ids:
            _model, communication, error = self._packet.ping(self._port, motor_id)
            if communication != self._sdk.COMM_SUCCESS or error != 0:
                missing.append(str(motor_id))
        if missing:
            raise RuntimeError("LEAP motor ping failed for IDs: " + ", ".join(missing))

    def _configure_motors(self) -> None:
        settings = self.settings
        self._sync_write(ADDR_OPERATING_MODE, 1, [POSITION_CURRENT_MODE] * 16)

        kp = [settings.kp] * 16
        kd = [settings.kd] * 16
        for motor_index in (0, 4, 8):
            kp[motor_index] = round(settings.kp * settings.side_gain_scale)
            kd[motor_index] = round(settings.kd * settings.side_gain_scale)
        self._sync_write(ADDR_POSITION_P_GAIN, 2, kp)
        self._sync_write(ADDR_POSITION_I_GAIN, 2, [settings.ki] * 16)
        self._sync_write(ADDR_POSITION_D_GAIN, 2, kd)
        self._sync_write(ADDR_CURRENT_LIMIT, 2, [settings.current_limit] * 16)

    def _write_real_positions(self, real_radians: Any) -> None:
        ticks = real_radians_to_ticks(real_radians)
        if np.any(ticks < 0) or np.any(ticks >= int(TICKS_PER_REVOLUTION)):
            raise RuntimeError("refusing a LEAP goal outside the single-turn range")
        self._sync_write(ADDR_GOAL_POSITION, POSITION_BYTES, ticks.tolist())

    def _read_real_positions(self) -> np.ndarray:
        if not self._port_open or self._port is None or self._packet is None:
            raise RuntimeError("LEAP serial port is not open")
        group = self._sdk.GroupSyncRead(
            self._port,
            self._packet,
            ADDR_PRESENT_POSITION,
            POSITION_BYTES,
        )
        for motor_id in self.settings.motor_ids:
            if not group.addParam(motor_id):
                raise RuntimeError(f"cannot add motor {motor_id} to position read")
        communication = group.txRxPacket()
        if communication != self._sdk.COMM_SUCCESS:
            raise RuntimeError(
                "LEAP position read failed: "
                + self._packet.getTxRxResult(communication)
            )
        ticks: list[int] = []
        for motor_id in self.settings.motor_ids:
            if not group.isAvailable(motor_id, ADDR_PRESENT_POSITION, POSITION_BYTES):
                raise RuntimeError(f"motor {motor_id} returned no position")
            ticks.append(
                int(group.getData(motor_id, ADDR_PRESENT_POSITION, POSITION_BYTES))
            )
        group.clearParam()
        return ticks_to_real_radians(ticks)

    def _sync_write(self, address: int, size: int, values: list[int]) -> None:
        if not self._port_open or self._port is None or self._packet is None:
            raise RuntimeError("LEAP serial port is not open")
        if len(values) != 16:
            raise ValueError("a LEAP sync write requires 16 values")
        group = self._sdk.GroupSyncWrite(self._port, self._packet, address, size)
        for motor_id, value in zip(self.settings.motor_ids, values, strict=True):
            integer = int(value)
            if integer < 0 or integer >= 1 << (8 * size):
                raise ValueError(
                    f"register value {integer} does not fit in {size} bytes"
                )
            encoded = list(integer.to_bytes(size, byteorder="little", signed=False))
            if not group.addParam(motor_id, encoded):
                raise RuntimeError(f"cannot add motor {motor_id} to sync write")
        communication = group.txPacket()
        group.clearParam()
        if communication != self._sdk.COMM_SUCCESS:
            raise RuntimeError(
                "LEAP sync write failed: " + self._packet.getTxRxResult(communication)
            )
