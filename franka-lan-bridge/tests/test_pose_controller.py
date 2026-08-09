from __future__ import annotations

import importlib
import math
import sys
import types
import unittest
from threading import RLock
from unittest.mock import patch

import numpy as np


class FakeAffine:
    def __init__(self, matrix: object) -> None:
        self.matrix = np.asarray(matrix, dtype=float)


class FakeCartesianMotion:
    def __init__(
        self,
        affine: FakeAffine,
        reference_type: object,
        *,
        relative_dynamics_factor: float,
    ) -> None:
        self.affine = affine
        self.reference_type = reference_type
        self.relative_dynamics_factor = relative_dynamics_factor


class FakeRobot:
    def __init__(self) -> None:
        pose = types.SimpleNamespace(matrix=np.eye(4))
        self.current_pose = types.SimpleNamespace(end_effector_pose=pose)
        self.last_motion: FakeCartesianMotion | None = None

    def move(self, motion: FakeCartesianMotion, *, asynchronous: bool) -> None:
        del asynchronous
        self.last_motion = motion


class FakeFrankaController:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self._lock = RLock()

    @staticmethod
    def _vector3(value: object, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"{name} must contain three values")
        return vector

    @staticmethod
    def _validate_dynamics_factor(value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("invalid dynamics factor")

    def get_state_snapshot(self) -> dict[str, object]:
        return {"end_effector": {"position": [0.0, 0.0, 0.0]}}


def load_pose_controller_module():
    fake_franky = types.ModuleType("franky")
    fake_franky.Affine = FakeAffine
    fake_franky.CartesianMotion = FakeCartesianMotion
    fake_franky.ReferenceType = types.SimpleNamespace(Absolute="absolute")
    fake_utils = types.ModuleType("utils")
    fake_controller = types.ModuleType("utils.franka_controller")
    fake_controller.FrankaController = FakeFrankaController
    with patch.dict(
        sys.modules,
        {
            "franky": fake_franky,
            "utils": fake_utils,
            "utils.franka_controller": fake_controller,
        },
    ):
        sys.modules.pop("franka_bridge.pose_controller", None)
        module = importlib.import_module("franka_bridge.pose_controller")
    sys.modules.pop("franka_bridge.pose_controller", None)
    return module


class PoseControllerTests(unittest.TestCase):
    def test_adapter_adds_rotation_to_old_snapshot_and_builds_absolute_pose(
        self,
    ) -> None:
        module = load_pose_controller_module()
        controller = module.PoseFrankaController()
        angle = 0.1
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        controller.robot.current_pose.end_effector_pose.matrix[:3, :3] = rotation
        snapshot = controller.get_state_snapshot()
        np.testing.assert_allclose(
            snapshot["end_effector"]["rotation_matrix"],
            rotation,
        )

        position = (0.41, -0.01, 0.23)
        controller.move_global_pose(
            position,
            rotation,
            dynamics_factor=0.05,
            asynchronous=True,
        )
        target = controller.robot.last_motion.affine.matrix
        np.testing.assert_allclose(target[:3, :3], rotation)
        np.testing.assert_allclose(target[:3, 3], position)


if __name__ == "__main__":
    unittest.main()
