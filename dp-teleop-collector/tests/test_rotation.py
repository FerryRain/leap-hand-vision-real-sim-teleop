from __future__ import annotations

import numpy as np
import pytest
from dp_collector.rotation import (
    matrix_to_rotation_6d,
    project_to_rotation_matrix,
    rotation_6d_to_matrix,
    rotation_matrix_is_valid,
)


def _random_rotations(count: int, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrices = []
    for _ in range(count):
        q, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        matrices.append(q)
    return np.asarray(matrices)


def test_identity_uses_pytorch3d_first_two_rows_convention() -> None:
    encoded = matrix_to_rotation_6d(np.eye(3))
    np.testing.assert_array_equal(encoded, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_rotation_6d_round_trip_supports_batches() -> None:
    expected = _random_rotations(20)
    reconstructed = rotation_6d_to_matrix(matrix_to_rotation_6d(expected))
    np.testing.assert_allclose(reconstructed, expected, atol=1e-7)
    assert np.all(rotation_matrix_is_valid(reconstructed))


def test_rotation_6d_rejects_degenerate_rows() -> None:
    with pytest.raises(ValueError, match="degenerate first"):
        rotation_6d_to_matrix(np.zeros(6))
    with pytest.raises(ValueError, match="parallel"):
        rotation_6d_to_matrix(np.array([1, 0, 0, 2, 0, 0], dtype=float))


def test_projection_returns_proper_rotation_for_reflection() -> None:
    noisy_reflection = np.diag([1.0, 1.0, -1.0])
    noisy_reflection[0, 1] = 0.03
    projected = project_to_rotation_matrix(noisy_reflection)
    assert bool(rotation_matrix_is_valid(projected))
    assert np.linalg.det(projected) > 0
