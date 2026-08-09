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

    def test_real_driver_holds_measured_pose_before_enabling_torque(self) -> None:
        writes: list[tuple[int, list[int]]] = []

        class FakePort:
            closed = False

            def openPort(self) -> bool:
                return True

            def setBaudRate(self, _baudrate: int) -> bool:
                return True

            def closePort(self) -> None:
                self.closed = True

        class FakePacket:
            def ping(self, _port: FakePort, _motor_id: int) -> tuple[int, int, int]:
                return 1000, 0, 0

            def getTxRxResult(self, result: int) -> str:
                return str(result)

        class FakeSyncWrite:
            def __init__(
                self,
                _port: FakePort,
                _packet: FakePacket,
                address: int,
                _size: int,
            ) -> None:
                self.address = address
                self.values: list[int] = []

            def addParam(self, _motor_id: int, encoded: list[int]) -> bool:
                self.values.append(int.from_bytes(encoded, "little"))
                return True

            def txPacket(self) -> int:
                writes.append((self.address, self.values.copy()))
                return 0

            def clearParam(self) -> None:
                pass

        class FakeSyncRead:
            def __init__(self, *_args: object) -> None:
                pass

            def addParam(self, _motor_id: int) -> bool:
                return True

            def txRxPacket(self) -> int:
                return 0

            def isAvailable(self, *_args: object) -> bool:
                return True

            def getData(self, *_args: object) -> int:
                return 2048

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
        initial = hand.connect_and_enable()
        np.testing.assert_allclose(initial, np.zeros(16), atol=1e-12)
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
        self.assertTrue(fake_port.closed)
        self.assertEqual(writes[-1], (64, [0] * 16))


if __name__ == "__main__":
    unittest.main()
