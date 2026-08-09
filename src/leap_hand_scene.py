"""Build the fixed-wrist MuJoCo LEAP Hand mirror."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .leap_hand_mapping import JOINT_NAMES

ACTUATOR_NAMES = tuple(f"{name}_act" for name in JOINT_NAMES)


@dataclass(frozen=True)
class LeapDemoScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    palm_mocap_id: int
    palm_down_quaternion_wxyz: tuple[float, float, float, float]

    def set_palm_position(self, position_m: tuple[float, float, float]) -> None:
        self.data.mocap_pos[self.palm_mocap_id] = position_m
        self.data.mocap_quat[self.palm_mocap_id] = self.palm_down_quaternion_wxyz


def build_leap_demo_scene(
    root: Path,
    config: dict[str, Any],
    *,
    open_joint_targets: np.ndarray,
) -> LeapDemoScene:
    """Build only the LEAP Hand: no object, floor, wrist motion, or robot arm."""

    scene_cfg = _section(config, "scene")
    control_cfg = _section(config, "control")
    wrist_cfg = _section(config, "wrist")
    hand_path = (root / str(scene_cfg["hand_model"])).resolve()
    if not hand_path.is_file():
        raise FileNotFoundError(f"LEAP Hand model not found: {hand_path}")

    spec = mujoco.MjSpec.from_file(str(hand_path))
    spec.option.timestep = float(control_cfg["simulation_timestep_s"])
    spec.option.gravity = (0.0, 0.0, 0.0)
    palm = spec.body("palm")
    palm.mocap = True
    palm.pos = _vector(wrist_cfg["neutral_position_m"], 3, "neutral_position_m")
    palm.quat = (1.0, 0.0, 0.0, 0.0)
    spec.worldbody.add_light(
        name="demo_key_light",
        pos=(0.0, -0.4, 0.8),
        dir=(0.0, 0.2, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=(0.9, 0.9, 0.9),
    )

    model = spec.compile()
    finger_kp = float(scene_cfg["finger_kp"])
    if finger_kp <= 0.0:
        raise ValueError("scene.finger_kp must be positive")
    model.actuator_gainprm[:, 0] = finger_kp
    model.actuator_biasprm[:, 1] = -finger_kp
    data = mujoco.MjData(model)
    actual_actuators = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if actual_actuators != ACTUATOR_NAMES:
        raise RuntimeError(f"unexpected LEAP actuator order: {actual_actuators!r}")
    if model.nmocap != 1:
        raise RuntimeError(f"expected one palm mocap body, got {model.nmocap}")

    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    palm_mocap_id = int(model.body_mocapid[palm_id])
    data.ctrl[:] = np.clip(
        np.asarray(open_joint_targets, dtype=np.float64),
        model.actuator_ctrlrange[:, 0],
        model.actuator_ctrlrange[:, 1],
    )
    data.mocap_pos[palm_mocap_id] = wrist_cfg["neutral_position_m"]
    data.mocap_quat[palm_mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return LeapDemoScene(
        model=model,
        data=data,
        palm_mocap_id=palm_mocap_id,
        palm_down_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing config section {name!r}")
    return value


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result
