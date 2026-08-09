from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from src.leap_hand_hardware import (
    MAPPING_LOWER_RAD,
    MAPPING_UPPER_RAD,
    MOTOR_SIM_LOWER_RAD,
    MOTOR_SIM_UPPER_RAD,
    DynamixelLeapHand,
    HardwareSettings,
    MockLeapHand,
    mapping_to_motor_sim,
    motor_real_to_sim,
    motor_sim_to_mapping,
    motor_sim_to_real,
    real_radians_to_ticks,
    ticks_to_real_radians,
)

ROOT = Path(__file__).resolve().parents[1]


class LeapHandHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))
        cls.settings = HardwareSettings.from_config(config["hardware"])

    def test_mapping_order_is_permuted_to_official_motor_order(self) -> None:
        mapping = np.arange(16, dtype=np.float64)
        motor = mapping_to_motor_sim(mapping, clip=False)
        np.testing.assert_array_equal(
            motor,
            [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 14, 13, 15],
        )
        np.testing.assert_array_equal(motor_sim_to_mapping(motor), mapping)

    def test_mapping_limits_match_official_motor_limits(self) -> None:
        np.testing.assert_allclose(
            mapping_to_motor_sim(MAPPING_LOWER_RAD, clip=False), MOTOR_SIM_LOWER_RAD
        )
        np.testing.assert_allclose(
            mapping_to_motor_sim(MAPPING_UPPER_RAD, clip=False), MOTOR_SIM_UPPER_RAD
        )

    def test_real_motor_convention_adds_pi_and_round_trips_ticks(self) -> None:
        motor_sim = np.zeros(16, dtype=np.float64)
        real = motor_sim_to_real(motor_sim)
        np.testing.assert_allclose(real, np.pi)
        np.testing.assert_allclose(motor_real_to_sim(real), motor_sim)
        ticks = real_radians_to_ticks(real)
        np.testing.assert_array_equal(ticks, np.full(16, 2048))
        np.testing.assert_allclose(ticks_to_real_radians(ticks), real, atol=1e-12)

    def test_mock_hand_applies_hardware_slew_and_limits(self) -> None:
        hand = MockLeapHand(self.settings)
        initial = hand.connect_and_enable()
        target = np.full(16, 100.0, dtype=np.float64)
        applied = hand.command_mapping(target)
        self.assertTrue(
            np.all(np.abs(applied - initial) <= self.settings.maximum_step_rad)
        )
        for _ in range(100):
            applied = hand.command_mapping(target)
        np.testing.assert_allclose(applied, MAPPING_UPPER_RAD)
        hand.close()
        self.assertFalse(hand.torque_enabled)

    def test_mock_feedback_matches_real_feedback_shape_deterministically(self) -> None:
        timestamps = iter((12.5, 13.0))
        hand = MockLeapHand(
            self.settings,
            clock=lambda: next(timestamps),
        )
        self.assertEqual(hand.model_numbers, (-1,) * 16)
        with self.assertRaises(AttributeError):
            hand.model_numbers = (1200,) * 16
        hand.connect_and_enable()

        initial = hand.read_feedback()
        np.testing.assert_array_equal(initial.actual_position_rad, np.zeros(16))
        np.testing.assert_array_equal(initial.present_velocity_raw, np.zeros(16))
        np.testing.assert_array_equal(initial.present_current_raw, np.zeros(16))
        self.assertEqual(initial.monotonic_s, 12.5)

        applied = hand.command_mapping(np.ones(16))
        updated = hand.read_feedback()
        np.testing.assert_allclose(updated.actual_position_rad, applied)
        np.testing.assert_array_equal(updated.present_velocity_raw, np.zeros(16))
        np.testing.assert_array_equal(updated.present_current_raw, np.zeros(16))
        self.assertEqual(updated.monotonic_s, 13.0)

    def test_real_driver_holds_measured_pose_before_enabling_torque(self) -> None:
        writes: list[tuple[int, list[int]]] = []
        registers: dict[tuple[int, int], int] = {}
        sync_read_shapes: list[tuple[int, int]] = []
        sync_read_transactions = 0

        class FakePort:
            closed = False

            def openPort(self) -> bool:
                return True

            def setBaudRate(self, _baudrate: int) -> bool:
                return True

            def closePort(self) -> None:
                self.closed = True

        class FakePacket:
            def ping(self, _port: FakePort, motor_id: int) -> tuple[int, int, int]:
                return 1200 + motor_id, 0, 0

            def getTxRxResult(self, result: int) -> str:
                return str(result)

            def _read(self, motor_id: int, address: int) -> tuple[int, int, int]:
                return registers.get((motor_id, address), 0), 0, 0

            def read1ByteTxRx(
                self, _port: FakePort, motor_id: int, address: int
            ) -> tuple[int, int, int]:
                return self._read(motor_id, address)

            def read2ByteTxRx(
                self, _port: FakePort, motor_id: int, address: int
            ) -> tuple[int, int, int]:
                return self._read(motor_id, address)

            def read4ByteTxRx(
                self, _port: FakePort, motor_id: int, address: int
            ) -> tuple[int, int, int]:
                return self._read(motor_id, address)

        class FakeSyncWrite:
            def __init__(
                self,
                _port: FakePort,
                _packet: FakePacket,
                address: int,
                _size: int,
            ) -> None:
                self.address = address
                self.entries: list[tuple[int, int]] = []

            def addParam(self, motor_id: int, encoded: list[int]) -> bool:
                self.entries.append((motor_id, int.from_bytes(encoded, "little")))
                return True

            def txPacket(self) -> int:
                for motor_id, value in self.entries:
                    registers[(motor_id, self.address)] = value
                writes.append(
                    (self.address, [value for _motor_id, value in self.entries])
                )
                return 0

            def clearParam(self) -> None:
                pass

        class FakeSyncRead:
            def __init__(
                self,
                _port: FakePort,
                _packet: FakePacket,
                address: int,
                size: int,
            ) -> None:
                sync_read_shapes.append((address, size))

            def addParam(self, _motor_id: int) -> bool:
                return True

            def txRxPacket(self) -> int:
                nonlocal sync_read_transactions
                sync_read_transactions += 1
                return 0

            def isAvailable(self, *_args: object) -> bool:
                return True

            def getData(self, motor_id: int, address: int, _size: int) -> int:
                if address == 126:
                    return (-(motor_id + 1)) & 0xFFFF
                if address == 128:
                    return (-(100 + motor_id)) & 0xFFFFFFFF
                if address == 132:
                    return 2048
                raise AssertionError(f"unexpected feedback address {address}")

            def clearParam(self) -> None:
                pass

        fake_port = FakePort()
        sdk = SimpleNamespace(
            COMM_SUCCESS=0,
            PortHandler=lambda _name: fake_port,
            PacketHandler=lambda _version: FakePacket(),
            GroupSyncWrite=FakeSyncWrite,
            GroupSyncRead=FakeSyncRead,
        )
        hand = DynamixelLeapHand(self.settings, "COM_TEST", sdk_module=sdk)
        self.assertEqual(hand.model_numbers, ())
        initial = hand.connect_and_enable()
        self.assertEqual(hand.model_numbers, tuple(range(1200, 1216)))
        with self.assertRaises(AttributeError):
            hand.model_numbers = (1200,) * 16
        np.testing.assert_allclose(initial, np.zeros(16), atol=1e-12)
        self.assertEqual(sync_read_shapes, [(126, 10)])
        self.assertEqual(sync_read_transactions, 1)

        feedback = hand.read_feedback()
        np.testing.assert_allclose(feedback.actual_position_rad, np.zeros(16))
        np.testing.assert_array_equal(
            feedback.present_velocity_raw,
            motor_sim_to_mapping(-np.arange(100, 116)),
        )
        np.testing.assert_array_equal(
            feedback.present_current_raw,
            motor_sim_to_mapping(-np.arange(1, 17)),
        )
        self.assertGreaterEqual(feedback.monotonic_s, 0.0)
        self.assertEqual(sync_read_shapes, [(126, 10)])
        self.assertEqual(sync_read_transactions, 2)
        torque_writes = [values for address, values in writes if address == 64]
        self.assertEqual(torque_writes[0], [0] * 16)
        self.assertEqual(torque_writes[-1], [1] * 16)
        goal_write_index = next(
            index for index, (address, _values) in enumerate(writes) if address == 116
        )
        enable_index = max(
            index
            for index, (address, values) in enumerate(writes)
            if address == 64 and values == [1] * 16
        )
        self.assertLess(goal_write_index, enable_index)

        applied = hand.command_mapping(np.ones(16))
        self.assertTrue(np.all(np.abs(applied) <= self.settings.maximum_step_rad))
        hand.close()
        self.assertFalse(hand.torque_enabled)
        self.assertTrue(fake_port.closed)
        self.assertEqual(writes[-1], (64, [0] * 16))

    def test_configuration_rejects_successful_broadcast_with_bad_readback(
        self,
    ) -> None:
        class FakePort:
            pass

        class FakePacket:
            def getTxRxResult(self, result: int) -> str:
                return str(result)

            def read1ByteTxRx(
                self, _port: FakePort, _motor_id: int, _address: int
            ) -> tuple[int, int, int]:
                return 0, 0, 0

            def read2ByteTxRx(
                self, _port: FakePort, _motor_id: int, _address: int
            ) -> tuple[int, int, int]:
                return 0, 0, 0

            def read4ByteTxRx(
                self, _port: FakePort, _motor_id: int, _address: int
            ) -> tuple[int, int, int]:
                return 0, 0, 0

        class SuccessfulButIgnoredSyncWrite:
            def __init__(self, *_args: object) -> None:
                pass

            def addParam(self, _motor_id: int, _encoded: list[int]) -> bool:
                return True

            def txPacket(self) -> int:
                return 0

            def clearParam(self) -> None:
                pass

        fake_port = FakePort()
        sdk = SimpleNamespace(
            COMM_SUCCESS=0,
            GroupSyncWrite=SuccessfulButIgnoredSyncWrite,
        )
        hand = DynamixelLeapHand(self.settings, "COM_TEST", sdk_module=sdk)
        hand._port = fake_port
        hand._packet = FakePacket()
        hand._port_open = True

        with self.assertRaisesRegex(
            RuntimeError,
            "Operating Mode verification failed for motor 0",
        ):
            hand._configure_motors()

    def test_torque_enable_requires_matching_readback_from_every_motor(self) -> None:
        class FakePort:
            pass

        class FakePacket:
            def getTxRxResult(self, result: int) -> str:
                return str(result)

            def read1ByteTxRx(
                self, _port: FakePort, motor_id: int, _address: int
            ) -> tuple[int, int, int]:
                return (0 if motor_id == 7 else 1), 0, 0

        class SuccessfulSyncWrite:
            def __init__(self, *_args: object) -> None:
                pass

            def addParam(self, _motor_id: int, _encoded: list[int]) -> bool:
                return True

            def txPacket(self) -> int:
                return 0

            def clearParam(self) -> None:
                pass

        fake_port = FakePort()
        sdk = SimpleNamespace(
            COMM_SUCCESS=0,
            GroupSyncWrite=SuccessfulSyncWrite,
        )
        hand = DynamixelLeapHand(self.settings, "COM_TEST", sdk_module=sdk)
        hand._port = fake_port
        hand._packet = FakePacket()
        hand._port_open = True
        hand._torque_may_be_enabled = True

        hand._sync_write(64, 1, [1] * 16)
        with self.assertRaisesRegex(
            RuntimeError,
            "Torque Enable verification failed for motor 7",
        ):
            hand._verify_register_values(64, 1, [1] * 16, "Torque Enable")
        self.assertTrue(hand.torque_enabled)

    def test_close_reports_torque_disable_readback_failure_after_closing_port(
        self,
    ) -> None:
        class FakePort:
            closed = False

            def closePort(self) -> None:
                self.closed = True

        class FakePacket:
            def getTxRxResult(self, result: int) -> str:
                return f"failure-{result}"

            def read1ByteTxRx(
                self, _port: FakePort, _motor_id: int, _address: int
            ) -> tuple[int, int, int]:
                # The broadcast transmission succeeded, but the motor still
                # reports torque enabled.
                return 1, 0, 0

        class SuccessfulSyncWrite:
            def __init__(self, *_args: object) -> None:
                pass

            def addParam(self, _motor_id: int, _encoded: list[int]) -> bool:
                return True

            def txPacket(self) -> int:
                return 0

            def clearParam(self) -> None:
                pass

        fake_port = FakePort()
        sdk = SimpleNamespace(
            COMM_SUCCESS=0,
            GroupSyncWrite=SuccessfulSyncWrite,
        )
        hand = DynamixelLeapHand(self.settings, "COM_TEST", sdk_module=sdk)
        hand._port = fake_port
        hand._packet = FakePacket()
        hand._port_open = True
        hand._torque_may_be_enabled = True

        with self.assertRaisesRegex(RuntimeError, "torque may still be enabled"):
            hand.close()

        self.assertTrue(fake_port.closed)
        self.assertFalse(hand._port_open)
        self.assertTrue(hand.torque_enabled)


if __name__ == "__main__":
    unittest.main()
