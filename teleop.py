"""Drive a physical LEAP Hand and a fixed-wrist MuJoCo mirror together."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from src.leap_hand_hardware import DynamixelLeapHand, HardwareSettings, MockLeapHand
from src.leap_hand_mapping import JOINT_NAMES
from src.leap_hand_scene import LeapDemoScene, build_leap_demo_scene
from src.runtime import (
    MediaPipeCamera,
    load_config,
    mock_landmarks,
    new_mapper,
    section,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yml"
DEFAULT_OUTPUT = ROOT / "logs" / "session.json"


def set_simulated_hand_target(
    scene: LeapDemoScene,
    target_rad: np.ndarray,
    fixed_palm_position_m: tuple[float, float, float],
) -> np.ndarray:
    """Apply exactly one 16-DoF command while keeping the palm fixed."""

    target = np.asarray(target_rad, dtype=np.float64)
    if target.shape != (16,) or not np.isfinite(target).all():
        raise ValueError("simulation target must contain 16 finite values")
    applied = np.clip(
        target,
        scene.model.actuator_ctrlrange[:, 0],
        scene.model.actuator_ctrlrange[:, 1],
    )
    scene.data.ctrl[:] = applied
    scene.set_palm_position(fixed_palm_position_m)
    return applied


def simulated_joint_positions(scene: LeapDemoScene) -> np.ndarray:
    positions: list[float] = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        positions.append(float(scene.data.qpos[scene.model.jnt_qposadr[joint_id]]))
    return np.asarray(positions, dtype=np.float64)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    hardware_settings = HardwareSettings.from_config(section(config, "hardware"))
    run_cfg = section(config, "run")

    source = str(run_cfg["source"] if args.source is None else args.source)
    device = str(run_cfg["device"] if args.device is None else args.device)
    duration_s = float(
        run_cfg["duration_s"] if args.duration is None else args.duration
    )
    if duration_s < 0.0:
        raise ValueError("duration must be non-negative; zero runs until stopped")
    if args.no_realtime and (source != "mock" or device != "mock" or not args.headless):
        raise ValueError(
            "--no-realtime requires mock source, mock device, and --headless"
        )
    if args.no_realtime and duration_s == 0.0:
        raise ValueError("a non-realtime mock run needs a finite duration")
    if device == "real" and not args.enable_torque:
        raise RuntimeError(
            "real hardware is disarmed; add --enable-torque after checking the port"
        )
    if device == "real" and not args.port:
        raise RuntimeError("--port COMx is required for the real LEAP Hand")
    if args.enable_torque and device != "real":
        raise ValueError("--enable-torque is only valid with --device real")

    diagnostic_hz = float(run_cfg["diagnostic_hz"])
    max_samples = int(run_cfg["max_samples"])
    max_read_failures = int(run_cfg["max_read_failures"])
    if diagnostic_hz <= 0.0 or max_samples <= 0 or max_read_failures <= 0:
        raise ValueError("run diagnostic settings must be positive")
    output_path = (
        DEFAULT_OUTPUT.resolve() if args.output is None else args.output.resolve()
    )

    settings, mapper, landmark_filter = new_mapper(config)
    mapping_cfg = section(config, "mapping")
    scene = build_leap_demo_scene(
        ROOT,
        config,
        open_joint_targets=np.asarray(mapping_cfg["joint_open_rad"], dtype=np.float64),
    )
    fixed_palm_position_m = tuple(
        float(value) for value in section(config, "wrist")["neutral_position_m"]
    )
    if len(fixed_palm_position_m) != 3:
        raise ValueError("wrist.neutral_position_m must contain three values")
    update_hz = float(section(config, "control")["update_hz"])
    simulation_timestep_s = float(section(config, "control")["simulation_timestep_s"])
    if update_hz <= 0.0 or simulation_timestep_s <= 0.0:
        raise ValueError("mapping update rate and simulation timestep must be positive")
    period_s = 1.0 / update_hz
    steps_per_update = max(1, round(period_s / simulation_timestep_s))

    camera = (
        MediaPipeCamera(
            config,
            args.camera_index,
            disable_preview=args.no_preview,
        )
        if source == "camera"
        else None
    )
    driver: MockLeapHand | DynamixelLeapHand
    if device == "real":
        driver = DynamixelLeapHand(hardware_settings, args.port)
    else:
        driver = MockLeapHand(
            hardware_settings,
            initial_mapping_rad=settings.joint_open_rad,
        )

    wall_start = time.monotonic()
    update_count = 0
    tracked_frames = 0
    lost_frames = 0
    consecutive_tracked = 0
    reload_count = 0
    command_paused = False
    motion_started = False
    stop_reason = "duration_complete"
    modes: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    dropped_samples = 0
    read_failures = 0
    consecutive_read_failures = 0
    last_diagnostic_s = -float("inf")
    last_target = np.asarray(settings.joint_open_rad, dtype=np.float64)
    last_applied = last_target.copy()
    last_hardware_actual = last_target.copy()
    last_simulated_actual = simulated_joint_positions(scene)
    last_simulated_target = last_target.copy()
    maximum_hardware_error_rad = np.zeros(16, dtype=np.float64)
    maximum_simulation_error_rad = np.zeros(16, dtype=np.float64)
    maximum_command_step_rad = np.zeros(16, dtype=np.float64)
    mirror_target_mismatch_count = 0
    initial_hardware_pose = last_target.copy()
    fatal_error: BaseException | None = None
    viewer_context: Any = nullcontext(None)

    try:
        if not args.headless:
            viewer_context = mujoco.viewer.launch_passive(scene.model, scene.data)
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = fixed_palm_position_m
                viewer.cam.distance = 0.38
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -18.0

            # Viewer and camera are ready before real torque can be enabled.
            initial_hardware_pose = driver.connect_and_enable()
            last_applied = initial_hardware_pose.copy()
            last_hardware_actual = initial_hardware_pose.copy()
            previous_applied = last_applied.copy()
            last_simulated_target = set_simulated_hand_target(
                scene, last_applied, fixed_palm_position_m
            )
            if not np.allclose(last_simulated_target, last_applied):
                mirror_target_mismatch_count += 1
            mujoco.mj_forward(scene.model, scene.data)

            while True:
                if viewer is not None and not viewer.is_running():
                    stop_reason = "simulation_viewer_closed"
                    break
                iteration_start = time.monotonic()
                elapsed_s = (
                    update_count * period_s
                    if args.no_realtime
                    else iteration_start - wall_start
                )
                if duration_s > 0.0 and elapsed_s >= duration_s:
                    break

                frame: np.ndarray | None = None
                handedness = "right"
                confidence = 1.0
                if camera is None:
                    landmarks = mock_landmarks(elapsed_s)
                else:
                    landmarks, frame, handedness, confidence = camera.read()

                if landmarks is None:
                    lost_frames += 1
                    consecutive_tracked = 0
                    landmark_filter.reset()
                    command_allowed = motion_started
                    if motion_started:
                        command = mapper.tracking_lost(now_s=elapsed_s)
                        last_target = np.asarray(
                            command.joint_target_rad, dtype=np.float64
                        )
                        mode = command.mode
                    else:
                        mode = "waiting_for_hand"
                else:
                    tracked_frames += 1
                    consecutive_tracked += 1
                    filtered = landmark_filter.update(landmarks)
                    command = mapper.update(filtered, now_s=elapsed_s)
                    last_target = np.asarray(command.joint_target_rad, dtype=np.float64)
                    if consecutive_tracked < hardware_settings.stable_tracking_frames:
                        command_allowed = False
                        mode = "waiting_stable_tracking"
                    else:
                        motion_started = True
                        command_allowed = True
                        mode = "tracking"

                if command_allowed and not command_paused:
                    last_applied = driver.command_mapping(last_target)
                    step = np.abs(last_applied - previous_applied)
                    maximum_command_step_rad = np.maximum(
                        maximum_command_step_rad, step
                    )
                    previous_applied = last_applied.copy()
                elif command_paused:
                    mode = "paused_hold"
                modes[mode] += 1

                # The exact post-safety hardware command drives the simulator.
                last_simulated_target = set_simulated_hand_target(
                    scene, last_applied, fixed_palm_position_m
                )
                if not np.allclose(last_simulated_target, last_applied):
                    mirror_target_mismatch_count += 1
                for _ in range(steps_per_update):
                    mujoco.mj_step(scene.model, scene.data)
                last_simulated_actual = simulated_joint_positions(scene)

                if elapsed_s - last_diagnostic_s >= 1.0 / diagnostic_hz:
                    try:
                        last_hardware_actual = driver.read_mapping_positions()
                        consecutive_read_failures = 0
                    except Exception:
                        read_failures += 1
                        consecutive_read_failures += 1
                        if consecutive_read_failures >= max_read_failures:
                            raise RuntimeError(
                                "too many consecutive LEAP position read failures"
                            )
                    hardware_error = last_applied - last_hardware_actual
                    simulation_error = last_simulated_target - last_simulated_actual
                    maximum_hardware_error_rad = np.maximum(
                        maximum_hardware_error_rad, np.abs(hardware_error)
                    )
                    maximum_simulation_error_rad = np.maximum(
                        maximum_simulation_error_rad, np.abs(simulation_error)
                    )
                    sample = {
                        "time_s": float(elapsed_s),
                        "mode": mode,
                        "vision_target_rad": last_target.tolist(),
                        "shared_applied_target_rad": last_applied.tolist(),
                        "hardware_actual_rad": last_hardware_actual.tolist(),
                        "simulation_actual_rad": last_simulated_actual.tolist(),
                        "hardware_error_rad": hardware_error.tolist(),
                        "simulation_error_rad": simulation_error.tolist(),
                    }
                    if len(samples) < max_samples:
                        samples.append(sample)
                    else:
                        dropped_samples += 1
                    last_diagnostic_s = elapsed_s

                if viewer is not None:
                    viewer.sync()
                if camera is not None and frame is not None:
                    color = (80, 225, 80) if mode == "tracking" else (80, 165, 255)
                    key = camera.show(
                        frame,
                        mode,
                        color,
                        _preview_details(
                            device=device,
                            handedness=handedness,
                            confidence=confidence,
                            gate=min(
                                consecutive_tracked,
                                hardware_settings.stable_tracking_frames,
                            ),
                            required_gate=hardware_settings.stable_tracking_frames,
                            applied=last_applied,
                        ),
                    )
                    if key in (ord("q"), ord("Q"), ord("e"), ord("E")):
                        stop_reason = "operator_emergency_stop"
                        break
                    if key == ord(" "):
                        command_paused = not command_paused
                    elif key in (ord("l"), ord("L")):
                        reloaded = load_config(config_path)
                        settings, mapper, landmark_filter = new_mapper(
                            reloaded,
                            seed_joints=last_target,
                        )
                        config = reloaded
                        reload_count += 1

                update_count += 1
                if not args.no_realtime:
                    remaining = period_s - (time.monotonic() - iteration_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            stop_reason = "keyboard_interrupt"
        else:
            fatal_error = error
            stop_reason = "runtime_error"
    finally:
        try:
            last_hardware_actual = driver.read_mapping_positions()
        except Exception:
            pass
        driver.close()
        if camera is not None:
            camera.close()

    report = {
        "version": 1,
        "entry": "real_leap_hand_with_mujoco_mirror",
        "source": source,
        "device": device,
        "serial_port": args.port if device == "real" else None,
        "finger_only": True,
        "controlled_joint_count": 16,
        "simulation_mirror": True,
        "same_target_sent_to_hardware_and_simulation": (
            mirror_target_mismatch_count == 0
        ),
        "mirror_target_mismatch_count": mirror_target_mismatch_count,
        "wrist_commanded": False,
        "fixed_simulation_palm_position_m": fixed_palm_position_m,
        "arm_loaded": False,
        "object_loaded": False,
        "floor_loaded": False,
        "config": str(config_path),
        "joint_order": JOINT_NAMES,
        "motor_ids": hardware_settings.motor_ids,
        "initial_hardware_pose_rad": initial_hardware_pose.tolist(),
        "final_vision_target_rad": last_target.tolist(),
        "final_shared_applied_target_rad": last_applied.tolist(),
        "final_hardware_actual_rad": last_hardware_actual.tolist(),
        "final_simulation_actual_rad": last_simulated_actual.tolist(),
        "maximum_hardware_error_rad": maximum_hardware_error_rad.tolist(),
        "maximum_simulation_error_rad": maximum_simulation_error_rad.tolist(),
        "maximum_command_step_rad": maximum_command_step_rad.tolist(),
        "maximum_step_limit_rad": hardware_settings.maximum_step_rad,
        "stable_tracking_frames": hardware_settings.stable_tracking_frames,
        "update_count": update_count,
        "tracked_frames": tracked_frames,
        "lost_frames": lost_frames,
        "motion_started": motion_started,
        "reload_count": reload_count,
        "read_failures": read_failures,
        "mode_counts": dict(modes),
        "diagnostic_samples": samples,
        "dropped_diagnostic_samples": dropped_samples,
        "elapsed_wall_s": max(0.0, time.monotonic() - wall_start),
        "stop_reason": stop_reason,
        "error": None
        if fatal_error is None
        else f"{type(fatal_error).__name__}: {fatal_error}",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "stop_reason": stop_reason,
                "device": device,
                "simulation_mirror": True,
                "wrist_commanded": False,
                "update_count": update_count,
                "error": report["error"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if fatal_error is not None:
        raise fatal_error


def _preview_details(
    *,
    device: str,
    handedness: str,
    confidence: float,
    gate: int,
    required_gate: int,
    applied: np.ndarray,
) -> tuple[str, ...]:
    degrees = np.degrees(applied)
    return (
        f"device={device}  simulation=ON  hand={handedness} {confidence:.2f}",
        f"tracking gate={gate}/{required_gate}  wrist/arm=DISABLED",
        "IF  [mcp rot pip dip] " + _format_degrees(degrees[0:4]),
        "MF  [mcp rot pip dip] " + _format_degrees(degrees[4:8]),
        "RF  [mcp rot pip dip] " + _format_degrees(degrees[8:12]),
        "TH [cmc axl mcp ipl] " + _format_degrees(degrees[12:16]),
    )


def _format_degrees(values: np.ndarray) -> str:
    return "[" + " ".join(f"{value:6.1f}" for value in values) + "] deg"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror camera-controlled LEAP fingers to MuJoCo and hardware"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("camera", "mock"))
    parser.add_argument("--device", choices=("mock", "real"))
    parser.add_argument("--port", default="")
    parser.add_argument("--enable-torque", action="store_true")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
