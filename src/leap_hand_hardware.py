"""Safe, finger-only adapter for a physical 16-DoF LEAP Hand.

The vision mapper uses the MuJoCo Menagerie joint order, while the physical
hand follows the official LEAP motor order.  This module is the only place
where that permutation and the physical Dynamixel offset are applied.
Importing this module never opens a serial port or enables torque.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

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
# In current-based position mode, Goal Current is the current/torque cap used by
# the internal position controller.  The EEPROM Current Limit is address 38;
# this driver intentionally does not modify that persistent device setting.
ADDR_GOAL_CURRENT = 102
# Backward-compatible name for callers that imported the old, imprecise label.
ADDR_CURRENT_LIMIT = ADDR_GOAL_CURRENT
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
POSITION_CURRENT_MODE = 5
CURRENT_BYTES = 2
VELOCITY_BYTES = 4
POSITION_BYTES = 4
PRESENT_FEEDBACK_START = ADDR_PRESENT_CURRENT
PRESENT_FEEDBACK_BYTES = CURRENT_BYTES + VELOCITY_BYTES + POSITION_BYTES
TICKS_PER_REVOLUTION = 4096.0
MOCK_MODEL_NUMBER = -1


def _joint_vector(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain 16 finite values")
    return result


def _signed_register(value: int, size: int) -> int:
    """Decode a little-endian register value returned as an unsigned integer."""

    integer = int(value)
    sign_bit = 1 << (8 * size - 1)
    modulus = 1 << (8 * size)
    if integer < 0 or integer >= modulus:
        raise ValueError(f"register value {integer} does not fit in {size} bytes")
    return integer - modulus if integer & sign_bit else integer


def _readonly_vector(values: Any, label: str, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != (16,):
        raise ValueError(f"{label} must contain 16 values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LeapHandFeedback:
    """One synchronized LEAP feedback sample in mapper joint order.

    ``present_velocity_raw`` and ``present_current_raw`` preserve the signed
    Dynamixel register values.  For XC330/XL330 LEAP motors their units are
    0.229 rpm/count and 1 mA/count, respectively.
    """

    actual_position_rad: np.ndarray
    present_velocity_raw: np.ndarray
    present_current_raw: np.ndarray
    monotonic_s: float

    def __post_init__(self) -> None:
        position = _readonly_vector(
            self.actual_position_rad,
            "actual positions",
            dtype=np.float64,
        )
        if not np.isfinite(position).all():
            raise ValueError("actual positions must be finite")
        velocity = _readonly_vector(
            self.present_velocity_raw,
            "present velocities",
            dtype=np.int64,
        )
        current = _readonly_vector(
            self.present_current_raw,
            "present currents",
            dtype=np.int64,
        )
        timestamp = float(self.monotonic_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "feedback monotonic timestamp must be finite and non-negative"
            )
        object.__setattr__(self, "actual_position_rad", position)
        object.__setattr__(self, "present_velocity_raw", velocity)
        object.__setattr__(self, "present_current_raw", current)
        object.__setattr__(self, "monotonic_s", timestamp)


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
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._clock = clock
        initial = (
            np.zeros(16, dtype=np.float64)
            if initial_mapping_rad is None
            else _joint_vector(initial_mapping_rad, "initial mapping pose")
        )
        self.last_mapping_rad = np.clip(initial, MAPPING_LOWER_RAD, MAPPING_UPPER_RAD)
        self.command_count = 0
        self.torque_enabled = False
        self._model_numbers = (MOCK_MODEL_NUMBER,) * 16

    @property
    def model_numbers(self) -> tuple[int, ...]:
        """Return explicit mock sentinels in configured motor order."""

        return self._model_numbers

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
        return self.read_feedback().actual_position_rad.copy()

    def read_feedback(self) -> LeapHandFeedback:
        """Return deterministic zero-effort feedback with the real API shape."""

        return LeapHandFeedback(
            actual_position_rad=self.last_mapping_rad,
            present_velocity_raw=np.zeros(16, dtype=np.int64),
            present_current_raw=np.zeros(16, dtype=np.int64),
            monotonic_s=self._clock(),
        )

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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not port.strip():
            raise ValueError("a serial port is required for the real LEAP Hand")
        self.settings = settings
        self.port_name = port.strip()
        self._sdk = sdk_module
        self._clock = clock
        self._port: Any | None = None
        self._packet: Any | None = None
        self._port_open = False
        self._torque_may_be_enabled = False
        self._last_motor_sim_rad: np.ndarray | None = None
        self._feedback_reader: Any | None = None
        self._model_numbers: tuple[int, ...] = ()

    @property
    def torque_enabled(self) -> bool:
        return self._torque_may_be_enabled

    @property
    def model_numbers(self) -> tuple[int, ...]:
        """Return model numbers reported by ping in configured motor order."""

        return self._model_numbers

    def connect_and_enable(self) -> np.ndarray:
        if self._port_open:
            raise RuntimeError("LEAP serial port is already open")
        sdk = self._load_sdk()
        self._port = sdk.PortHandler(self.port_name)
        self._packet = sdk.PacketHandler(PROTOCOL_VERSION)
        if not self._port.openPort():
            raise RuntimeError(f"cannot open LEAP serial port {self.port_name}")
        self._port_open = True
        # Opening an existing bus does not prove that its motors are currently
        # torque-disabled.  Keep the state conservative until every ID has
        # acknowledged and read back Torque Enable = 0.
        self._torque_may_be_enabled = True
        try:
            if not self._port.setBaudRate(self.settings.baudrate):
                raise RuntimeError(
                    f"cannot set {self.port_name} to {self.settings.baudrate} baud"
                )
            self._verify_all_motors()
            self._disable_torque_verified()
            self._configure_motors()

            current_real = self._read_real_positions()
            if np.any(current_real < -0.10) or np.any(current_real > math.tau + 0.10):
                raise RuntimeError(
                    "a motor reported a position outside the expected single-turn range"
                )
            # Seed the goal with the measured pose before torque is enabled.  This
            # prevents the hand from jumping to an arbitrary startup posture.
            startup_ticks = self._write_real_positions(current_real)
            self._verify_register_values(
                ADDR_GOAL_POSITION,
                POSITION_BYTES,
                startup_ticks.tolist(),
                "startup Goal Position",
            )
            self._torque_may_be_enabled = True
            self._sync_write(ADDR_TORQUE_ENABLE, 1, [1] * 16)
            self._verify_register_values(
                ADDR_TORQUE_ENABLE,
                1,
                [1] * 16,
                "Torque Enable",
            )
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
        return self.read_feedback().actual_position_rad.copy()

    def read_feedback(self) -> LeapHandFeedback:
        """Read all 16 motors' position, velocity, and current in one packet."""

        (
            motor_position_ticks,
            motor_velocity_raw,
            motor_current_raw,
            timestamp_s,
        ) = self._read_motor_feedback_raw()
        motor_position_rad = motor_real_to_sim(
            ticks_to_real_radians(motor_position_ticks)
        )
        return LeapHandFeedback(
            actual_position_rad=motor_sim_to_mapping(motor_position_rad),
            present_velocity_raw=motor_velocity_raw[MAPPING_FROM_MOTOR],
            present_current_raw=motor_current_raw[MAPPING_FROM_MOTOR],
            monotonic_s=timestamp_s,
        )

    def close(self) -> None:
        if self._port_open and self._port is not None:
            torque_disable_error: Exception | None = None
            port_close_error: Exception | None = None
            if self.settings.disable_torque_on_exit:
                try:
                    self._disable_torque_verified()
                except Exception as error:
                    torque_disable_error = error
            try:
                self._port.closePort()
            except Exception as error:
                port_close_error = error
            finally:
                self._port_open = False
                self._feedback_reader = None
            if torque_disable_error is not None:
                message = (
                    "LEAP torque-disable command failed; torque may still be enabled. "
                    "Motor power must be cut before touching the hand."
                )
                if port_close_error is not None:
                    message += " Closing the serial port also failed."
                raise RuntimeError(message) from torque_disable_error
            if port_close_error is not None:
                raise RuntimeError("failed to close the LEAP serial port") from (
                    port_close_error
                )

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
        model_numbers: list[int] = []
        for motor_id in self.settings.motor_ids:
            model, communication, error = self._packet.ping(self._port, motor_id)
            if communication != self._sdk.COMM_SUCCESS or error != 0:
                missing.append(str(motor_id))
            else:
                model_numbers.append(int(model))
        if missing:
            raise RuntimeError("LEAP motor ping failed for IDs: " + ", ".join(missing))
        self._model_numbers = tuple(model_numbers)

    def _configure_motors(self) -> None:
        settings = self.settings
        operating_modes = [POSITION_CURRENT_MODE] * 16
        self._sync_write(ADDR_OPERATING_MODE, 1, operating_modes)

        kp = [settings.kp] * 16
        kd = [settings.kd] * 16
        ki = [settings.ki] * 16
        goal_current = [settings.current_limit] * 16
        for motor_index in (0, 4, 8):
            kp[motor_index] = round(settings.kp * settings.side_gain_scale)
            kd[motor_index] = round(settings.kd * settings.side_gain_scale)
        self._sync_write(ADDR_POSITION_P_GAIN, 2, kp)
        self._sync_write(ADDR_POSITION_I_GAIN, 2, ki)
        self._sync_write(ADDR_POSITION_D_GAIN, 2, kd)
        self._sync_write(ADDR_GOAL_CURRENT, 2, goal_current)

        self._verify_register_values(
            ADDR_OPERATING_MODE,
            1,
            operating_modes,
            "Operating Mode",
        )
        self._verify_register_values(
            ADDR_GOAL_CURRENT,
            2,
            goal_current,
            "Goal Current",
        )
        self._verify_register_values(
            ADDR_POSITION_P_GAIN,
            2,
            kp,
            "Position P Gain",
        )
        self._verify_register_values(
            ADDR_POSITION_I_GAIN,
            2,
            ki,
            "Position I Gain",
        )
        self._verify_register_values(
            ADDR_POSITION_D_GAIN,
            2,
            kd,
            "Position D Gain",
        )

    def _write_real_positions(self, real_radians: Any) -> np.ndarray:
        ticks = real_radians_to_ticks(real_radians)
        if np.any(ticks < 0) or np.any(ticks >= int(TICKS_PER_REVOLUTION)):
            raise RuntimeError("refusing a LEAP goal outside the single-turn range")
        self._sync_write(ADDR_GOAL_POSITION, POSITION_BYTES, ticks.tolist())
        return ticks

    def _disable_torque_verified(self) -> None:
        """Broadcast torque-off, then require a zero readback from every ID."""

        # A failed or partially applied broadcast must never make the public
        # state look safer than the hardware can be proven to be.
        self._torque_may_be_enabled = True
        self._sync_write(ADDR_TORQUE_ENABLE, 1, [0] * 16)
        self._verify_register_values(
            ADDR_TORQUE_ENABLE,
            1,
            [0] * 16,
            "Torque Disable",
        )
        self._torque_may_be_enabled = False

    def _read_register(self, motor_id: int, address: int, size: int) -> int:
        if not self._port_open or self._port is None or self._packet is None:
            raise RuntimeError("LEAP serial port is not open")
        readers = {
            1: "read1ByteTxRx",
            2: "read2ByteTxRx",
            4: "read4ByteTxRx",
        }
        method_name = readers.get(size)
        if method_name is None:
            raise ValueError(f"unsupported LEAP register width: {size}")
        method = getattr(self._packet, method_name)
        value, communication, device_error = method(
            self._port,
            motor_id,
            address,
        )
        if communication != self._sdk.COMM_SUCCESS:
            raise RuntimeError(
                f"motor {motor_id} register {address} read failed: "
                + self._packet.getTxRxResult(communication)
            )
        if device_error != 0:
            error_text = (
                self._packet.getRxPacketError(device_error)
                if hasattr(self._packet, "getRxPacketError")
                else str(device_error)
            )
            raise RuntimeError(
                f"motor {motor_id} register {address} returned status error: "
                f"{error_text}"
            )
        return int(value)

    def _verify_register_values(
        self,
        address: int,
        size: int,
        expected_values: list[int],
        label: str,
    ) -> None:
        if len(expected_values) != 16:
            raise ValueError("LEAP register verification requires 16 values")
        for motor_id, expected in zip(
            self.settings.motor_ids,
            expected_values,
            strict=True,
        ):
            actual = self._read_register(motor_id, address, size)
            if actual != int(expected):
                raise RuntimeError(
                    f"{label} verification failed for motor {motor_id}: "
                    f"read {actual}, expected {int(expected)}"
                )

    def _read_real_positions(self) -> np.ndarray:
        positions, _velocities, _currents, _timestamp = self._read_motor_feedback_raw()
        return ticks_to_real_radians(positions)

    def _read_motor_feedback_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if not self._port_open or self._port is None or self._packet is None:
            raise RuntimeError("LEAP serial port is not open")
        if self._feedback_reader is None:
            group = self._sdk.GroupSyncRead(
                self._port,
                self._packet,
                PRESENT_FEEDBACK_START,
                PRESENT_FEEDBACK_BYTES,
            )
            for motor_id in self.settings.motor_ids:
                if not group.addParam(motor_id):
                    raise RuntimeError(
                        f"cannot add motor {motor_id} to LEAP feedback read"
                    )
            self._feedback_reader = group
        group = self._feedback_reader
        communication = group.txRxPacket()
        if communication != self._sdk.COMM_SUCCESS:
            raise RuntimeError(
                "LEAP feedback read failed: "
                + self._packet.getTxRxResult(communication)
            )
        timestamp_s = self._clock()
        position_ticks: list[int] = []
        velocity_raw: list[int] = []
        current_raw: list[int] = []
        for motor_id in self.settings.motor_ids:
            if not group.isAvailable(
                motor_id,
                PRESENT_FEEDBACK_START,
                PRESENT_FEEDBACK_BYTES,
            ):
                raise RuntimeError(f"motor {motor_id} returned no feedback")
            current_raw.append(
                _signed_register(
                    group.getData(motor_id, ADDR_PRESENT_CURRENT, CURRENT_BYTES),
                    CURRENT_BYTES,
                )
            )
            velocity_raw.append(
                _signed_register(
                    group.getData(motor_id, ADDR_PRESENT_VELOCITY, VELOCITY_BYTES),
                    VELOCITY_BYTES,
                )
            )
            position_ticks.append(
                _signed_register(
                    group.getData(motor_id, ADDR_PRESENT_POSITION, POSITION_BYTES),
                    POSITION_BYTES,
                )
            )
        return (
            np.asarray(position_ticks, dtype=np.int64),
            np.asarray(velocity_raw, dtype=np.int64),
            np.asarray(current_raw, dtype=np.int64),
            timestamp_s,
        )

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
