from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from dp_collector.proprio_episode import ProprioEpisodeWriter
from dp_collector.proprio_exporter import (
    ProprioExportError,
    build_proprio_manifest,
    collect_proprio_bundles,
    export_proprio_to_zarr,
)
from dp_collector.proprio_schema import ProprioEpisodeSpec

JOINT_NAMES = tuple(f"joint_{index}" for index in range(16))
FINGERPRINT_FIELDS = (
    "leap_device",
    "operating_mode",
    "goal_current_raw",
    "kp",
    "ki",
    "kd",
    "maximum_goal_step_rad",
    "motor_ids",
    "motor_model_numbers",
    "present_current_unit",
    "teleop_config_sha256",
)


def fingerprint_extra(
    *,
    leap_device: str = "real",
    overrides: dict[str, object] | None = None,
    omit: set[str] | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "leap_device": leap_device,
        "operating_mode": 5,
        "goal_current_raw": 350,
        "kp": 600,
        "ki": 0,
        "kd": 200,
        "maximum_goal_step_rad": 0.06,
        "motor_ids": list(range(16)),
        "motor_model_numbers": (
            ["mock"] * 16 if leap_device == "mock" else [1200] * 16
        ),
        "present_current_unit": "signed_raw_register_count_model_dependent",
        "teleop_config_sha256": "a" * 64,
    }
    values.update(overrides or {})
    for field in omit or set():
        values.pop(field, None)
    return values


def record(
    root: Path,
    episode_id: str,
    *,
    period: float = 0.05,
    accepted: bool = True,
    leap_device: str = "real",
    fingerprint_overrides: dict[str, object] | None = None,
    omit_fingerprint_fields: set[str] | None = None,
    unrelated_metadata: object | None = None,
) -> None:
    extra = fingerprint_extra(
        leap_device=leap_device,
        overrides=fingerprint_overrides,
        omit=omit_fingerprint_fields,
    )
    extra.update(
        {
            "motor_model": "record_raw_only",
            "unrelated_metadata": unrelated_metadata,
        }
    )
    spec = ProprioEpisodeSpec(
        sample_period_s=period,
        sample_period_tolerance_s=period * 0.2,
        joint_names=JOINT_NAMES,
        extra=extra,
    )
    writer = ProprioEpisodeWriter(
        root,
        spec,
        initial_timestamp_s=0.0,
        initial_actual_position_rad=np.zeros(16),
        episode_id=episode_id,
    )
    for index in range(2):
        writer.append(
            timestamp_s=period * (index + 1),
            actual_position_rad=np.full(16, 0.1 * (index + 1)),
            present_current_raw=np.arange(-8, 8) + index,
            goal_position_rad=np.full(16, 1.3),
        )
    if accepted:
        writer.accept()
    else:
        writer.reject("test rejection")


def test_collect_defaults_to_accepted_and_rejects_mixed_periods(
    tmp_path: Path,
) -> None:
    record(tmp_path, "b_episode")
    record(tmp_path, "a_rejected", accepted=False)
    bundles = collect_proprio_bundles(tmp_path)
    assert [bundle.path.name for bundle in bundles] == ["b_episode"]
    assert len(collect_proprio_bundles(tmp_path, include_rejected=True)) == 2
    record(tmp_path, "c_period", period=0.10)
    with pytest.raises(ProprioExportError, match="incompatible"):
        collect_proprio_bundles(tmp_path)


def test_manifest_is_low_dimensional_and_explains_action_semantics(
    tmp_path: Path,
) -> None:
    record(tmp_path, "episode_grasp")
    bundles = collect_proprio_bundles(tmp_path)
    manifest = build_proprio_manifest(
        bundles,
        episode_ends=[2],
        num_steps=2,
        include_rejected=False,
    )
    assert manifest["image_observations"] is False
    assert manifest["franka_observations_or_actions"] is False
    assert manifest["observation"]["dim"] == 48
    assert manifest["action"]["dim"] == 16
    assert "constant position command" in manifest["action"]["warning"]
    assert "not_force_calibrated" in manifest["units"]["present_current_raw"]
    fingerprint = manifest["dynamics_fingerprint"]
    assert len(fingerprint["sha256"]) == 64
    # Only required fingerprint fields are copied here; unrelated metadata
    # remains episode-local.
    assert fingerprint["fields"] == fingerprint_extra()
    assert fingerprint["fields"]["motor_model_numbers"] == [1200] * 16
    assert not any("camera" in key or "depth" in key for key in manifest["arrays"])
    serialized = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in serialized


@pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None, reason="zarr<3 not installed"
)
def test_zarr_export_contains_only_proprio_arrays(tmp_path: Path) -> None:
    import zarr

    record(tmp_path, "episode_a")
    record(tmp_path, "episode_b")
    output = tmp_path / "proprio.zarr"
    summary = export_proprio_to_zarr(tmp_path, output)
    assert summary.num_episodes == 2
    assert summary.num_steps == 4

    root = zarr.open_group(str(output), mode="r")
    expected = {
        "action",
        "actual_position",
        "goal_position",
        "position_error",
        "present_current_raw",
        "robot_state",
        "sample_dt",
        "timestamp",
        "valid",
        "velocity",
    }
    assert set(root["data"].array_keys()) == expected
    assert "camera_0" not in root["data"]
    assert "depth_0" not in root["data"]
    assert root["data/robot_state"].shape == (4, 48)
    assert root["data/action"].shape == (4, 16)
    assert root["data/present_current_raw"].dtype == np.dtype("int16")
    assert root["data/present_current_raw"][0].tolist() == list(range(-8, 8))
    np.testing.assert_allclose(root["data/velocity"][:], 2.0)
    assert root["meta/episode_ends"][:].tolist() == [2, 4]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["episodes"][0]["source"] == "accepted/episode_a"
    assert (
        root.attrs["dynamics_fingerprint"] == manifest["dynamics_fingerprint"]["fields"]
    )
    assert (
        root.attrs["dynamics_fingerprint_sha256"]
        == manifest["dynamics_fingerprint"]["sha256"]
    )


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_export_rejects_episode_missing_required_dynamics_field(
    tmp_path: Path,
    field: str,
) -> None:
    record(
        tmp_path,
        "missing_metadata",
        omit_fingerprint_fields={field},
    )
    with pytest.raises(ProprioExportError, match=field):
        collect_proprio_bundles(tmp_path)


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("operating_mode", 3),
        ("goal_current_raw", 300),
        ("kp", 500),
        ("ki", 1),
        ("kd", 150),
        ("maximum_goal_step_rad", 0.04),
        ("motor_ids", list(reversed(range(16)))),
        ("motor_model_numbers", [1190] * 16),
        ("present_current_unit", "milliampere"),
        ("teleop_config_sha256", "b" * 64),
    ],
)
def test_export_rejects_mixed_hardware_or_control_dynamics(
    tmp_path: Path,
    field: str,
    different_value: object,
) -> None:
    record(tmp_path, "baseline")
    record(
        tmp_path,
        "different",
        fingerprint_overrides={field: different_value},
    )
    with pytest.raises(ProprioExportError, match=field):
        collect_proprio_bundles(tmp_path)


def test_export_rejects_mixed_mock_and_real_episodes(tmp_path: Path) -> None:
    record(tmp_path, "real_episode", leap_device="real")
    record(tmp_path, "mock_episode", leap_device="mock")
    with pytest.raises(ProprioExportError, match="leap_device"):
        collect_proprio_bundles(tmp_path)


def test_unrelated_episode_metadata_does_not_split_compatible_dynamics(
    tmp_path: Path,
) -> None:
    record(tmp_path, "episode_a", unrelated_metadata="session one")
    record(tmp_path, "episode_b", unrelated_metadata="session two")
    assert len(collect_proprio_bundles(tmp_path)) == 2


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("leap_device", "unknown", "leap_device"),
        ("maximum_goal_step_rad", 0.0, "maximum_goal_step_rad"),
        ("motor_ids", [0] * 16, "unique"),
        ("motor_model_numbers", [0] * 16, "motor_model_numbers"),
        ("present_current_unit", "", "present_current_unit"),
        ("teleop_config_sha256", "not-a-digest", "teleop_config_sha256"),
    ],
)
def test_export_rejects_malformed_dynamics_fingerprint(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    record(
        tmp_path,
        "malformed",
        fingerprint_overrides={field: invalid_value},
    )
    with pytest.raises(ProprioExportError, match=message):
        collect_proprio_bundles(tmp_path)
