"""Rotation-capable adapter for the owner's ``utils.franka_controller``."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from franky import Affine, CartesianMotion, ReferenceType
from utils.franka_controller import FrankaController


class PoseFrankaController(FrankaController):
    """Add absolute XYZ + rotation planning and normalize older state snapshots."""

    def _current_pose_array(self) -> np.ndarray:
        pose = getattr(self, "pose_matrix", None)
        if pose is None:
            pose = self.robot.current_pose.end_effector_pose.matrix
        matrix = np.asarray(pose, dtype=float).copy()
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("current end-effector pose is not a finite 4x4 matrix")
        return matrix

    def get_state_snapshot(self) -> dict[str, object]:
        snapshot = super().get_state_snapshot()
        end_effector = snapshot.get("end_effector")
        if not isinstance(end_effector, dict):
            raise RuntimeError("state snapshot is missing end_effector")
        if end_effector.get("rotation_matrix") is None:
            pose = self._current_pose_array()
            end_effector["position"] = pose[:3, 3].copy()
            end_effector["rotation_matrix"] = pose[:3, :3].copy()
            end_effector["pose_matrix"] = pose
        return snapshot

    def move_global_pose(
        self,
        position: Sequence[float],
        rotation_matrix: Sequence[Sequence[float]],
        *,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> CartesianMotion:
        target_position = self._vector3(position, "position")
        rotation = np.asarray(rotation_matrix, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_matrix must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("rotation_matrix must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            raise ValueError("rotation_matrix determinant must be +1")
        self._validate_dynamics_factor(dynamics_factor)

        target = np.eye(4, dtype=float)
        target[:3, :3] = rotation
        target[:3, 3] = target_position
        motion = CartesianMotion(
            Affine(target),
            ReferenceType.Absolute,
            relative_dynamics_factor=dynamics_factor,
        )
        with self._lock:
            self.robot.move(motion, asynchronous=asynchronous)
        return motion
