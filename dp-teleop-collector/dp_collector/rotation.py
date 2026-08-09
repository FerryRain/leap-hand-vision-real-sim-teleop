"""Rotation representations used by the teleoperation dataset.

The 6D convention intentionally matches PyTorch3D: flatten the first two
*rows* of a 3x3 rotation matrix.  This is worth stating explicitly because
some robotics libraries use the first two columns instead.
"""

from __future__ import annotations

import numpy as np


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two rows of ``matrix`` flattened to six values.

    Batched inputs with shape ``(..., 3, 3)`` are supported.  The function
    checks shape and finiteness but deliberately does not project a noisy
    robot rotation onto SO(3); call :func:`project_to_rotation_matrix` first
    when projection is desired.
    """

    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in shape (3, 3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("rotation matrix contains a non-finite value")
    return np.array(value[..., :2, :].reshape(value.shape[:-2] + (6,)), copy=True)


def rotation_6d_to_matrix(rotation_6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Convert PyTorch3D-style 6D rotations into orthonormal matrices.

    Gram--Schmidt orthogonalization is applied to the two row vectors.  A
    degenerate first vector, or two nearly parallel input vectors, is rejected
    instead of silently returning NaNs.
    """

    value = np.asarray(rotation_6d)
    if value.shape[-1:] != (6,):
        raise ValueError(f"6D rotation must end in shape (6,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("6D rotation contains a non-finite value")
    if eps <= 0:
        raise ValueError("eps must be positive")

    work = value.astype(np.result_type(value.dtype, np.float64), copy=False)
    first = work[..., 0:3]
    second = work[..., 3:6]

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= eps):
        raise ValueError("6D rotation has a degenerate first row")
    row_0 = first / first_norm

    second_orthogonal = second - np.sum(row_0 * second, axis=-1, keepdims=True) * row_0
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm <= eps):
        raise ValueError("6D rotation rows are parallel or degenerate")
    row_1 = second_orthogonal / second_norm
    row_2 = np.cross(row_0, row_1)
    return np.stack((row_0, row_1, row_2), axis=-2)


def project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project one or more finite 3x3 matrices to the nearest proper rotation."""

    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in shape (3, 3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("rotation matrix contains a non-finite value")

    work = value.astype(np.result_type(value.dtype, np.float64), copy=False)
    u, _, vh = np.linalg.svd(work)
    projected = u @ vh
    determinant = np.linalg.det(projected)
    if np.any(determinant < 0):
        u = np.array(u, copy=True)
        u[..., :, -1] *= np.where(determinant < 0, -1.0, 1.0)[..., None]
        projected = u @ vh
    return projected


def rotation_matrix_is_valid(
    matrix: np.ndarray,
    *,
    atol: float = 1e-5,
) -> np.ndarray | np.bool_:
    """Return a scalar or batched boolean indicating membership in SO(3)."""

    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        return np.bool_(False)
    finite = np.isfinite(value).all(axis=(-2, -1))
    identity = np.eye(3, dtype=np.result_type(value.dtype, np.float64))
    orthogonal = np.all(
        np.isclose(value @ np.swapaxes(value, -1, -2), identity, atol=atol),
        axis=(-2, -1),
    )
    proper = np.isclose(np.linalg.det(value), 1.0, atol=atol)
    return finite & orthogonal & proper
