"""Camera-to-LEAP teleoperation with synchronized FR3 demonstration recording."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from franka_bridge.client import FrankaBridgeClient
from franka_bridge.terminal_keys import TerminalKeys
from src.d455 import D455MediaPipeCamera
from src.leap_hand_hardware import DynamixelLeapHand, HardwareSettings, MockLeapHand
from src.leap_hand_mapping import JOINT_NAMES
from src.runtime import load_config, new_mapper, section

from .camera import MockVisionSource
from .config import AppSettings, load_app_settings
from .episode import EpisodeWriter
from .franka import (
    FRANKA_STATE_NAMES,
    FrankaStateCache,
    PalmVelocityMapper,
    ParsedFrankaState,
)
from .queued_writer import QueuedEpisodeWriter
from .schema import STAGE_NAMES, EpisodeSpec, Stage
from .ui import PreviewUI

COLLECTOR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = COLLECTOR_ROOT.parent
DEFAULT_CONFIG = COLLECTOR_ROOT / "config.yml"
DEFAULT_TELEOP_CONFIG = REPOSITORY_ROOT / "config.yml"
DEFAULT_OUTPUT = COLLECTOR_ROOT / "datasets" / "grasp_release"
LEAP_STATE_NAMES = tuple(f"leap_{name}_actual_rad" for name in JOINT_NAMES)


@dataclass
class ActiveEpisode:
    writer: QueuedEpisodeWriter
    task: str
    stage: int
    franka_watchdog_start_count: int | None = None
    franka_workspace_guard_start_count: int | None = None
    invalid_steps: int = 0
    last_sample_s: float = -math.inf


@dataclass(frozen=True)
class RuntimeSummary:
    stop_reason: str
    accepted_episodes: int
    rejected_episodes: int
    aborted_episodes: int
    control_frames: int
    recorded_steps: int
    franka_states: int
    franka_sequence_gaps: int


async def run(args: argparse.Namespace) -> RuntimeSummary:
    app_settings = load_app_settings(args.config.resolve())
    teleop_config = load_config(args.teleop_config.resolve())
    hardware_settings = HardwareSettings.from_config(section(teleop_config, "hardware"))
    update_hz = float(section(teleop_config, "control")["update_hz"])
    if not 1.0 <= update_hz <= 60.0:
        raise ValueError("teleoperation control.update_hz must be in [1, 60]")
    _validate_args(args, app_settings)

    mapper_settings, mapper, landmark_filter = new_mapper(teleop_config)
    source: D455MediaPipeCamera | MockVisionSource
    if args.source == "d455":
        source = D455MediaPipeCamera(
            teleop_config,
            args.d455_serial,
            disable_preview=True,
        )
    else:
        source = MockVisionSource()

    driver: DynamixelLeapHand | MockLeapHand
    if args.leap_device == "real":
        driver = DynamixelLeapHand(hardware_settings, args.leap_port)
    else:
        driver = MockLeapHand(
            hardware_settings,
            initial_mapping_rad=mapper_settings.joint_open_rad,
        )

    ui = PreviewUI(enabled=not args.headless)
    velocity_mapper = PalmVelocityMapper(app_settings.franka_teleop)
    franka_client: FrankaBridgeClient | None = None
    franka_control_acquired = False
    franka_cache = FrankaStateCache()
    franka_state_task: asyncio.Task[None] | None = None
    terminal: TerminalKeys | None = None
    active: ActiveEpisode | None = None
    initial_leap = np.asarray(mapper_settings.joint_open_rad, dtype=np.float64)
    last_vision_target = initial_leap.copy()
    last_leap_applied = initial_leap.copy()
    last_leap_actual = initial_leap.copy()
    last_leap_read_s = time.monotonic()
    last_leap_command_s = last_leap_read_s
    last_franka_command_s: float | None = None
    last_franka_ack_s: float | None = None
    motion_started = False
    consecutive_tracked = 0
    accepted_episodes = 0
    rejected_episodes = 0
    aborted_episodes = 0
    control_frames = 0
    recorded_steps = 0
    read_failures = 0
    leap_read_failed = False
    stop_reason = "operator_stop"
    run_start_s = time.monotonic()
    last_status_print_s = -math.inf
    auto_tasks = _parse_auto_tasks(args.mock_auto_episodes)
    auto_task_index = 0
    auto_recorded_frames = 0
    fatal_error: BaseException | None = None

    try:
        if args.franka_mode != "off":
            franka_client = FrankaBridgeClient(
                args.franka_uri,
                client_name="dp-teleop-collector",
            )
            await franka_client.connect()
            franka_state_task = asyncio.create_task(
                franka_cache.consume(franka_client),
                name="dp-collector-franka-state",
            )
            await _wait_for_franka_state(franka_cache, timeout_s=3.0)
            if args.franka_mode == "teleop":
                _validate_franka_motion_server(
                    franka_client.safety,
                    app_settings,
                )
                await franka_client.acquire_control()
                franka_control_acquired = True

        initial_leap = await asyncio.to_thread(driver.connect_and_enable)
        last_leap_applied = initial_leap.copy()
        last_leap_actual = initial_leap.copy()
        last_vision_target = initial_leap.copy()
        mapper.last_joints = initial_leap.copy()
        print(
            json.dumps(
                {
                    "status": "ready",
                    "output": str(args.output.resolve()),
                    "source": args.source,
                    "leap_device": args.leap_device,
                    "franka_mode": args.franka_mode,
                    "controls": (
                        "G grasp | R release | SPACE accept | D reject | "
                        "1-7 stage | Q/E/Esc stop"
                    ),
                    "franka_deadman": (
                        "hold LEFT MOUSE in preview"
                        if args.franka_mode == "teleop"
                        else "disabled"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        if not args.headless:
            try:
                terminal = TerminalKeys()
                terminal.__enter__()
            except RuntimeError as error:
                terminal = None
                print(
                    f"warning: terminal hotkeys unavailable ({error}); use preview keys"
                )

        while True:
            iteration_start_s = time.monotonic()
            elapsed_s = iteration_start_s - run_start_s
            if args.duration > 0.0 and elapsed_s >= args.duration:
                stop_reason = "duration_complete"
                break
            if not ui.window_open():
                stop_reason = "preview_window_closed"
                break
            if franka_state_task is not None and franka_state_task.done():
                exception = franka_state_task.exception()
                if exception is not None:
                    raise RuntimeError("FR3 state receiver stopped") from exception
                raise RuntimeError("FR3 state receiver stopped unexpectedly")
            if active is not None:
                active.writer.raise_if_failed()

            landmarks, preview, handedness, confidence = await asyncio.to_thread(
                source.read
            )
            rgbd = source.latest_rgbd()
            training_fov_valid = _hand_inside_training_crop(
                source.latest_hand_display_uv,
                source_width=preview.shape[1],
                source_height=preview.shape[0],
                target_width=app_settings.collector.image_width,
                target_height=app_settings.collector.image_height,
            )
            now_s = time.monotonic()
            if landmarks is None:
                consecutive_tracked = 0
                landmark_filter.reset()
                tracking_valid = False
                if motion_started:
                    command = mapper.tracking_lost(now_s=elapsed_s)
                    last_vision_target = np.asarray(
                        command.joint_target_rad, dtype=np.float64
                    )
                    last_leap_applied = await asyncio.to_thread(
                        driver.command_mapping, last_vision_target
                    )
                    last_leap_command_s = time.monotonic()
                    mapping_mode = command.mode
                else:
                    mapping_mode = "waiting_for_hand"
            else:
                consecutive_tracked += 1
                if not training_fov_valid:
                    consecutive_tracked = 0
                filtered = landmark_filter.update(landmarks)
                command = mapper.update(filtered, now_s=elapsed_s)
                last_vision_target = np.asarray(
                    command.joint_target_rad, dtype=np.float64
                )
                tracking_valid = (
                    training_fov_valid
                    and consecutive_tracked >= hardware_settings.stable_tracking_frames
                )
                if tracking_valid:
                    motion_started = True
                    last_leap_applied = await asyncio.to_thread(
                        driver.command_mapping, last_vision_target
                    )
                    last_leap_command_s = time.monotonic()
                    mapping_mode = "tracking"
                else:
                    mapping_mode = (
                        "outside_training_fov"
                        if not training_fov_valid
                        else "waiting_stable_tracking"
                    )

            parsed_franka = _latest_franka(
                franka_cache,
                now_s=now_s,
                app_settings=app_settings,
                required=args.franka_mode != "off",
                require_control_lease=args.franka_mode == "teleop",
            )
            franka_valid = parsed_franka is None or parsed_franka.valid
            deadman_requested = (
                args.franka_mode == "teleop"
                and ui.deadman_down
                and tracking_valid
                and franka_valid
            )
            linear_velocity = velocity_mapper.update(
                source.latest_palm_position_m,
                deadman_down=deadman_requested,
            )
            angular_velocity = np.zeros(3, dtype=np.float64)
            franka_twist = np.concatenate((linear_velocity, angular_velocity))
            linear_speed = _vector_norm(linear_velocity)
            deadman_commanded = linear_speed > 1.0e-9

            if args.franka_mode == "teleop":
                assert franka_client is not None
                command_started_s = time.monotonic()
                await franka_client.send_velocity(
                    linear_velocity,
                    angular_velocity,
                    frame="global",
                    wait_ack=True,
                )
                last_franka_command_s = command_started_s
                last_franka_ack_s = time.monotonic()

            keys: set[str] = set()
            preview_key = ui.show(
                _crop_preview_to_training_fov(
                    preview,
                    target_width=app_settings.collector.image_width,
                    target_height=app_settings.collector.image_height,
                ),
                _preview_lines(
                    active=active,
                    mapping_mode=mapping_mode,
                    franka_mode=args.franka_mode,
                    franka_valid=franka_valid,
                    deadman_requested=deadman_requested,
                    deadman_commanded=deadman_commanded,
                    franka_speed=linear_speed,
                    consecutive_tracked=consecutive_tracked,
                    stable_frames=hardware_settings.stable_tracking_frames,
                ),
            )
            decoded = _decode_cv_key(preview_key)
            if decoded is not None:
                keys.add(decoded)
            if terminal is not None:
                terminal_key = terminal.poll()
                if terminal_key is not None:
                    keys.add(terminal_key)

            normalized_keys = {key.lower() for key in keys}
            if normalized_keys.intersection({"q", "e", "\x1b"}):
                stop_reason = "operator_emergency_stop"
                break
            for normalized in sorted(normalized_keys):
                if normalized in {"g", "r"}:
                    task = "grasp" if normalized == "g" else "release"
                    if not tracking_valid or not franka_valid:
                        print("cannot start: hand tracking or FR3 state is not valid")
                    elif active is None:
                        if args.franka_mode == "teleop":
                            assert franka_client is not None
                            previous_state_count = franka_cache.received_count
                            (
                                last_franka_command_s,
                                last_franka_ack_s,
                            ) = await _zero_franka_for_dataset_transition(
                                franka_client,
                                ui,
                                velocity_mapper,
                            )
                            deadman_requested = False
                            deadman_commanded = False
                            linear_speed = 0.0
                            franka_twist = np.zeros(6, dtype=np.float64)
                            await _wait_for_franka_update(
                                franka_cache,
                                after_count=previous_state_count,
                                timeout_s=0.5,
                            )
                            parsed_franka = _latest_franka(
                                franka_cache,
                                now_s=time.monotonic(),
                                app_settings=app_settings,
                                required=True,
                                require_control_lease=True,
                            )
                            assert parsed_franka is not None
                            if not parsed_franka.valid:
                                raise RuntimeError(
                                    "FR3 became invalid while starting an episode: "
                                    + ", ".join(parsed_franka.invalid_reasons)
                                )
                        active = await _start_episode(
                            task=task,
                            args=args,
                            app_settings=app_settings,
                            source=source,
                            franka_client=franka_client,
                            franka_state=parsed_franka,
                        )
                        auto_recorded_frames = 0
                        print(f"recording {task}: {active.writer.episode_id}")
                    else:
                        print(
                            "an episode is already recording; accept or reject it first"
                        )
                elif normalized == " " and active is not None:
                    if args.franka_mode == "teleop":
                        assert franka_client is not None
                        previous_state_count = franka_cache.received_count
                        await _zero_franka_for_dataset_transition(
                            franka_client,
                            ui,
                            velocity_mapper,
                        )
                        await _wait_for_franka_update(
                            franka_cache,
                            after_count=previous_state_count,
                            timeout_s=0.5,
                        )
                        final_franka = _latest_franka(
                            franka_cache,
                            now_s=time.monotonic(),
                            app_settings=app_settings,
                            required=True,
                            require_control_lease=True,
                        )
                        assert final_franka is not None
                        final_reasons = set(final_franka.invalid_reasons)
                        final_reasons.update(
                            _episode_bridge_counter_reasons(active, final_franka)
                        )
                        if final_reasons:
                            active.invalid_steps += 1
                            print(
                                "episode final FR3 check failed: "
                                + ", ".join(sorted(final_reasons))
                            )
                    outcome = await _accept_episode(active, args, app_settings)
                    if outcome:
                        accepted_episodes += 1
                    else:
                        rejected_episodes += 1
                    active = None
                    auto_recorded_frames = 0
                elif normalized == "d" and active is not None:
                    if args.franka_mode == "teleop":
                        assert franka_client is not None
                        await _zero_franka_for_dataset_transition(
                            franka_client,
                            ui,
                            velocity_mapper,
                        )
                    path = await asyncio.to_thread(
                        active.writer.reject,
                        "operator_rejected",
                    )
                    rejected_episodes += 1
                    print(f"rejected: {path}")
                    active = None
                    auto_recorded_frames = 0
                elif normalized in {str(index) for index in range(1, 8)}:
                    if active is not None:
                        active.stage = int(normalized)
                        print(f"stage={STAGE_NAMES.get(active.stage, active.stage)}")
            if auto_task_index < len(auto_tasks) and active is None and tracking_valid:
                active = await _start_episode(
                    task=auto_tasks[auto_task_index],
                    args=args,
                    app_settings=app_settings,
                    source=source,
                    franka_client=franka_client,
                    franka_state=parsed_franka,
                )
                auto_recorded_frames = 0

            sample_period_s = 1.0 / app_settings.collector.sample_hz
            if active is not None and now_s - active.last_sample_s >= sample_period_s:
                try:
                    last_leap_actual = await asyncio.to_thread(
                        driver.read_mapping_positions
                    )
                    last_leap_read_s = time.monotonic()
                    read_failures = 0
                    leap_read_failed = False
                except Exception:
                    read_failures += 1
                    leap_read_failed = True
                    if read_failures >= 3:
                        raise RuntimeError(
                            "three consecutive LEAP position reads failed"
                        )
                sample_s = time.monotonic()
                parsed_franka = _latest_franka(
                    franka_cache,
                    now_s=sample_s,
                    app_settings=app_settings,
                    required=args.franka_mode != "off",
                    require_control_lease=args.franka_mode == "teleop",
                )
                state, action, ages, invalid_reasons, extra = _build_step(
                    args=args,
                    app_settings=app_settings,
                    source=source,
                    rgbd=rgbd,
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                    tracking_valid=tracking_valid,
                    mapping_mode=mapping_mode,
                    leap_vision_target=last_vision_target,
                    leap_applied=last_leap_applied,
                    leap_actual=last_leap_actual,
                    leap_read_s=last_leap_read_s,
                    leap_read_failed=leap_read_failed,
                    leap_command_s=last_leap_command_s,
                    franka=parsed_franka,
                    franka_twist=franka_twist,
                    franka_command_s=last_franka_command_s,
                    franka_ack_s=last_franka_ack_s,
                    franka_watchdog_start_count=(active.franka_watchdog_start_count),
                    franka_workspace_guard_start_count=(
                        active.franka_workspace_guard_start_count
                    ),
                    sample_s=sample_s,
                    run_start_s=run_start_s,
                    deadman_requested=deadman_requested,
                )
                rgb, depth = _resize_rgbd(
                    rgbd.color_bgr,
                    rgbd.depth_units,
                    width=app_settings.collector.image_width,
                    height=app_settings.collector.image_height,
                )
                active.writer.append_frame(
                    rgb=rgb,
                    depth=depth,
                    timestamp_s=sample_s - run_start_s,
                    robot_state=state,
                    action=action,
                    stage=active.stage,
                    state_ages_s=ages,
                    valid=not invalid_reasons,
                    invalid_reasons=invalid_reasons,
                    extra=extra,
                )
                active.last_sample_s = sample_s
                active.invalid_steps += int(bool(invalid_reasons))
                recorded_steps += 1
                auto_recorded_frames += 1

                if active.writer.step_count >= int(
                    math.ceil(
                        app_settings.collector.maximum_episode_s
                        * app_settings.collector.sample_hz
                    )
                ):
                    if args.franka_mode == "teleop":
                        assert franka_client is not None
                        await _zero_franka_for_dataset_transition(
                            franka_client,
                            ui,
                            velocity_mapper,
                        )
                    path = await asyncio.to_thread(
                        active.writer.reject,
                        "maximum_episode_duration_exceeded",
                    )
                    rejected_episodes += 1
                    print(f"automatically rejected overlong episode: {path}")
                    active = None

                if (
                    active is not None
                    and auto_task_index < len(auto_tasks)
                    and auto_recorded_frames >= args.mock_frames_per_episode
                ):
                    outcome = await _accept_episode(active, args, app_settings)
                    if outcome:
                        accepted_episodes += 1
                    else:
                        rejected_episodes += 1
                    active = None
                    auto_task_index += 1
                    auto_recorded_frames = 0
                    if auto_task_index >= len(auto_tasks):
                        stop_reason = "mock_auto_complete"
                        break

            control_frames += 1
            if now_s - last_status_print_s >= 5.0:
                last_status_print_s = now_s
                print(
                    "status "
                    f"tracking={mapping_mode} franka={args.franka_mode} "
                    f"episode={active.writer.episode_id if active else 'idle'}"
                )

            remaining_s = 1.0 / update_hz - (time.monotonic() - iteration_start_s)
            if remaining_s > 0.0:
                await asyncio.sleep(remaining_s)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
            stop_reason = "keyboard_interrupt"
        else:
            fatal_error = error
            stop_reason = "runtime_error"
    finally:
        ui.deadman_down = False
        velocity_mapper.reset()
        # Physical stop must precede any potentially slow disk flush/encoding.
        if franka_control_acquired and franka_client is not None:
            with contextlib.suppress(Exception):
                await franka_client.send_velocity(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    frame="global",
                    wait_ack=True,
                )
            with contextlib.suppress(Exception):
                await franka_client.stop()
        # Disable LEAP torque before waiting for any queued disk flush.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(driver.close)
        if active is not None and active.writer.active:
            aborted_episodes += 1
            try:
                path = await asyncio.to_thread(
                    active.writer.close_partial,
                    reason=stop_reason,
                )
                print(f"partial episode retained: {path}")
            except Exception as error:
                print(
                    "partial episode persistence failed; the recoverable .partial "
                    f"directory was retained: {error}"
                )
        if franka_state_task is not None:
            franka_state_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await franka_state_task
        if franka_client is not None:
            with contextlib.suppress(Exception):
                await franka_client.close()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(source.close)
        with contextlib.suppress(Exception):
            ui.close()
        if terminal is not None:
            with contextlib.suppress(Exception):
                terminal.__exit__(None, None, None)

    summary = RuntimeSummary(
        stop_reason=stop_reason,
        accepted_episodes=accepted_episodes,
        rejected_episodes=rejected_episodes,
        aborted_episodes=aborted_episodes,
        control_frames=control_frames,
        recorded_steps=recorded_steps,
        franka_states=franka_cache.received_count,
        franka_sequence_gaps=franka_cache.sequence_gap_count,
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))
    if fatal_error is not None:
        raise fatal_error
    return summary


def _validate_args(args: argparse.Namespace, settings: AppSettings) -> None:
    if args.duration < 0.0:
        raise ValueError("--duration must be non-negative")
    if args.leap_device == "real":
        if not args.enable_leap_torque:
            raise RuntimeError(
                "real LEAP Hand is disarmed; add --enable-leap-torque after checks"
            )
        if not args.leap_port.strip():
            raise RuntimeError("--leap-port COMx is required for real LEAP Hand")
    elif args.enable_leap_torque:
        raise ValueError("--enable-leap-torque is only valid for --leap-device real")
    if args.franka_mode == "teleop":
        if not args.enable_franka_motion:
            raise RuntimeError(
                "FR3 teleop is disarmed; add --enable-franka-motion only after "
                "mapping checks"
            )
        if args.headless:
            raise ValueError("FR3 teleop requires the preview mouse deadman")
        if not settings.franka_teleop.mapping_confirmed:
            raise RuntimeError(
                "franka_teleop.mapping_confirmed=false; validate the camera "
                "mapping first"
            )
    elif args.enable_franka_motion:
        raise ValueError("--enable-franka-motion requires --franka-mode teleop")
    auto_tasks = _parse_auto_tasks(args.mock_auto_episodes)
    if auto_tasks:
        if not (
            args.source == "mock"
            and args.leap_device == "mock"
            and args.franka_mode == "off"
            and args.headless
        ):
            raise ValueError(
                "mock auto episodes require mock source, mock LEAP, FR3 off, "
                "and headless"
            )
        if args.mock_frames_per_episode < settings.collector.minimum_episode_steps:
            raise ValueError(
                "--mock-frames-per-episode is below collector.minimum_episode_steps"
            )
    elif args.headless:
        raise ValueError("headless collection requires --mock-auto-episodes")


def _validate_franka_motion_server(
    safety: dict[str, Any],
    settings: AppSettings,
) -> None:
    if safety.get("continuous_velocity_workspace_guard") is not True:
        raise RuntimeError(
            "FR3 server is missing the continuous velocity workspace guard; "
            "update and restart franka-lan-bridge on the Linux controller computer"
        )
    advertised = float(safety.get("max_linear_speed", float("nan")))
    if not math.isfinite(advertised) or advertised <= 0.0:
        raise RuntimeError("FR3 server did not advertise a valid max_linear_speed")
    if settings.franka_teleop.maximum_linear_speed_m_s > advertised:
        raise RuntimeError(
            "collector FR3 speed exceeds the server-advertised limit: "
            f"{settings.franka_teleop.maximum_linear_speed_m_s:.4f} > "
            f"{advertised:.4f} m/s"
        )


async def _start_episode(
    *,
    task: str,
    args: argparse.Namespace,
    app_settings: AppSettings,
    source: D455MediaPipeCamera | MockVisionSource,
    franka_client: FrankaBridgeClient | None,
    franka_state: ParsedFrankaState | None,
) -> ActiveEpisode:
    connected_arm = args.franka_mode != "off"
    robot_state_names = (
        FRANKA_STATE_NAMES + LEAP_STATE_NAMES if connected_arm else LEAP_STATE_NAMES
    )
    action_space = "global_twist_leap" if args.franka_mode == "teleop" else "hand_only"
    age_limits = {
        "camera": app_settings.collector.maximum_camera_age_s,
        "leap": app_settings.collector.maximum_leap_state_age_s,
    }
    if connected_arm:
        age_limits["franka"] = app_settings.collector.maximum_franka_state_age_s
    camera_diagnostics = dict(source.diagnostics())
    camera_diagnostics["raw_rgbd_saved"] = True
    camera_diagnostics["training_image_size"] = [
        app_settings.collector.image_width,
        app_settings.collector.image_height,
    ]
    camera_diagnostics["training_color_intrinsics"] = _scaled_intrinsics(
        camera_diagnostics.get("color_intrinsics"),
        width=app_settings.collector.image_width,
        height=app_settings.collector.image_height,
        horizontal_flip=bool(camera_diagnostics.get("flip_horizontal", False)),
    )
    leap_hardware_config = section(
        load_config(args.teleop_config.resolve()),
        "hardware",
    )
    spec = EpisodeSpec(
        task=task,
        action_space=action_space,
        robot_state_dim=len(robot_state_names),
        image_shape=(
            app_settings.collector.image_height,
            app_settings.collector.image_width,
        ),
        state_age_limits_s=age_limits,
        robot_state_names=robot_state_names,
        timestamp_clock="run_relative_monotonic",
        extra={
            "franka_mode": args.franka_mode,
            "franka_command_frame": "global",
            "franka_orientation_controlled": False,
            "franka_angular_velocity_fixed_zero": True,
            "leap_device": args.leap_device,
            "leap_serial_port": (
                args.leap_port if args.leap_device == "real" else None
            ),
            "leap_joint_order": list(JOINT_NAMES),
            "leap_hardware": _json_safe(leap_hardware_config),
            "camera": _json_safe(camera_diagnostics),
            "franka_server_safety": (
                {} if franka_client is None else _json_safe(franka_client.safety)
            ),
            "collector_config": str(args.config.resolve()),
            "collector_config_sha256": _file_sha256(args.config.resolve()),
            "teleop_config": str(args.teleop_config.resolve()),
            "teleop_config_sha256": _file_sha256(args.teleop_config.resolve()),
            "git_commit": _git_commit(REPOSITORY_ROOT),
            "sample_hz": app_settings.collector.sample_hz,
            "max_pending_frames": app_settings.collector.max_pending_frames,
            "session_note": args.session_note,
        },
    )
    base_writer = await asyncio.to_thread(
        EpisodeWriter,
        args.output.resolve(),
        spec,
        jpeg_quality=app_settings.collector.jpeg_quality,
    )
    writer = QueuedEpisodeWriter(
        base_writer,
        max_pending_frames=app_settings.collector.max_pending_frames,
    )
    return ActiveEpisode(
        writer=writer,
        task=task,
        stage=int(Stage.APPROACH),
        franka_watchdog_start_count=(
            None if franka_state is None else franka_state.watchdog_stop_count
        ),
        franka_workspace_guard_start_count=(
            None if franka_state is None else franka_state.workspace_guard_stop_count
        ),
        # The camera sample captured before directory creation can be stale if
        # Windows metadata/fsync is slow.  Start sampling from the next fresh
        # control frame instead of recording that pre-episode observation.
        last_sample_s=time.monotonic(),
    )


async def _accept_episode(
    active: ActiveEpisode,
    args: argparse.Namespace,
    app_settings: AppSettings,
) -> bool:
    reasons: list[str] = []
    if active.writer.step_count < app_settings.collector.minimum_episode_steps:
        reasons.append("too_few_steps")
    if active.invalid_steps > app_settings.collector.maximum_invalid_steps:
        reasons.append("too_many_invalid_steps")
    if reasons:
        reason = "+".join(reasons)
        path = await asyncio.to_thread(active.writer.reject, reason)
        print(f"cannot accept ({reason}); retained under rejected: {path}")
        return False
    path = await asyncio.to_thread(
        active.writer.accept,
        notes=args.session_note or None,
    )
    print(f"accepted: {path}")
    return True


def _build_step(
    *,
    args: argparse.Namespace,
    app_settings: AppSettings,
    source: D455MediaPipeCamera | MockVisionSource,
    rgbd: Any,
    landmarks: np.ndarray | None,
    handedness: str,
    confidence: float,
    tracking_valid: bool,
    mapping_mode: str,
    leap_vision_target: np.ndarray,
    leap_applied: np.ndarray,
    leap_actual: np.ndarray,
    leap_read_s: float,
    leap_read_failed: bool,
    leap_command_s: float,
    franka: ParsedFrankaState | None,
    franka_twist: np.ndarray,
    franka_command_s: float | None,
    franka_ack_s: float | None,
    franka_watchdog_start_count: int | None,
    franka_workspace_guard_start_count: int | None,
    sample_s: float,
    run_start_s: float,
    deadman_requested: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], tuple[str, ...], dict[str, Any]]:
    invalid_reasons: list[str] = []
    if not tracking_valid:
        invalid_reasons.append("hand_tracking_not_stable")
    if leap_read_failed:
        invalid_reasons.append("leap_state_read_failed")
    camera_age_s = max(0.0, sample_s - float(rgbd.capture_monotonic_s))
    leap_age_s = max(0.0, sample_s - float(leap_read_s))
    ages = {"camera": camera_age_s, "leap": leap_age_s}
    if args.franka_mode == "off":
        robot_state = np.asarray(leap_actual, dtype=np.float32)
    else:
        if franka is None:
            raise RuntimeError("FR3 state is unavailable while collection requires it")
        robot_state = np.concatenate(
            (franka.low_dim, np.asarray(leap_actual, dtype=np.float32))
        ).astype(np.float32)
        ages["franka"] = franka.age_s
        invalid_reasons.extend(franka.invalid_reasons)
        _append_bridge_counter_reason(
            invalid_reasons,
            current=franka.watchdog_stop_count,
            baseline=franka_watchdog_start_count,
            increased_reason="franka_velocity_watchdog_stopped",
        )
        _append_bridge_counter_reason(
            invalid_reasons,
            current=franka.workspace_guard_stop_count,
            baseline=franka_workspace_guard_start_count,
            increased_reason="franka_workspace_guard_triggered",
        )
    if args.franka_mode == "teleop":
        action = np.concatenate((franka_twist, leap_applied)).astype(np.float32)
    else:
        action = np.asarray(leap_applied, dtype=np.float32)
    for source_name, age in ages.items():
        limit = {
            "camera": app_settings.collector.maximum_camera_age_s,
            "leap": app_settings.collector.maximum_leap_state_age_s,
            "franka": app_settings.collector.maximum_franka_state_age_s,
        }[source_name]
        if age > limit:
            invalid_reasons.append(f"{source_name}_state_stale")

    palm = source.latest_palm_position_m
    extra: dict[str, Any] = {
        "sample_monotonic_s": float(sample_s),
        "sample_time_from_run_start_s": float(sample_s - run_start_s),
        "camera_timestamp_ms": rgbd.camera_timestamp_ms,
        "camera_capture_monotonic_s": rgbd.capture_monotonic_s,
        "depth_scale_m": rgbd.depth_scale_m,
        "handedness": handedness,
        "hand_confidence": float(confidence),
        "hand_landmarks": None if landmarks is None else landmarks.tolist(),
        "palm_position_camera_m": palm,
        "mapping_mode": mapping_mode,
        "leap_vision_target_rad": leap_vision_target.tolist(),
        "leap_applied_target_rad": leap_applied.tolist(),
        "leap_actual_rad": leap_actual.tolist(),
        "leap_state_read_monotonic_s": float(leap_read_s),
        "leap_action_sent_monotonic_s": float(leap_command_s),
        "leap_action_age_s": max(0.0, sample_s - float(leap_command_s)),
        "franka_command_global_twist": franka_twist.tolist(),
        "franka_action_sent_monotonic_s": franka_command_s,
        "franka_action_ack_monotonic_s": franka_ack_s,
        "franka_action_ack_latency_s": (
            None
            if franka_command_s is None or franka_ack_s is None
            else max(0.0, float(franka_ack_s) - float(franka_command_s))
        ),
        "franka_action_age_s": (
            None
            if franka_command_s is None
            else max(0.0, sample_s - float(franka_command_s))
        ),
        "franka_deadman_requested": bool(deadman_requested),
    }
    if franka is not None:
        extra["franka"] = {
            "sequence": franka.sequence,
            "robot_timestamp_s": franka.robot_timestamp_s,
            "received_monotonic_s": franka.received_monotonic_s,
            "robot": franka.raw_robot,
            "bridge": franka.raw_bridge,
        }
    return (
        robot_state,
        action,
        ages,
        tuple(sorted(set(invalid_reasons))),
        _json_safe(extra),
    )


def _append_bridge_counter_reason(
    reasons: list[str],
    *,
    current: int,
    baseline: int | None,
    increased_reason: str,
) -> None:
    if baseline is None:
        reasons.append("franka_bridge_counter_baseline_missing")
    elif current > baseline:
        reasons.append(increased_reason)
    elif current < baseline:
        reasons.append("franka_bridge_counter_reset")


def _episode_bridge_counter_reasons(
    active: ActiveEpisode,
    franka: ParsedFrankaState,
) -> tuple[str, ...]:
    reasons: list[str] = []
    _append_bridge_counter_reason(
        reasons,
        current=franka.watchdog_stop_count,
        baseline=active.franka_watchdog_start_count,
        increased_reason="franka_velocity_watchdog_stopped",
    )
    _append_bridge_counter_reason(
        reasons,
        current=franka.workspace_guard_stop_count,
        baseline=active.franka_workspace_guard_start_count,
        increased_reason="franka_workspace_guard_triggered",
    )
    return tuple(sorted(set(reasons)))


def _latest_franka(
    cache: FrankaStateCache,
    *,
    now_s: float,
    app_settings: AppSettings,
    required: bool,
    require_control_lease: bool,
) -> ParsedFrankaState | None:
    if not required:
        return None
    parsed = cache.parse_latest(
        now_s=now_s,
        maximum_age_s=app_settings.collector.maximum_franka_state_age_s,
        require_control_lease=require_control_lease,
    )
    if parsed is None:
        raise RuntimeError("FR3 state stream has no sample")
    return parsed


async def _wait_for_franka_state(cache: FrankaStateCache, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while cache.latest is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the first FR3 state")
        await asyncio.sleep(0.02)


async def _wait_for_franka_update(
    cache: FrankaStateCache,
    *,
    after_count: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while cache.received_count <= after_count:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for a fresh FR3 state")
        await asyncio.sleep(0.01)


async def _zero_franka_for_dataset_transition(
    client: FrankaBridgeClient,
    ui: PreviewUI,
    velocity_mapper: PalmVelocityMapper,
) -> tuple[float, float]:
    """Stop/recenter FR3 before starting or flushing episode files."""

    ui.deadman_down = False
    velocity_mapper.reset()
    sent_s = time.monotonic()
    await client.send_velocity(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        frame="global",
        wait_ack=True,
    )
    return sent_s, time.monotonic()


def _resize_rgbd(
    color_bgr: np.ndarray,
    depth_units: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    color = np.asarray(color_bgr)
    depth = np.asarray(depth_units)
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError("camera color must be BGR uint8 HxWx3")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("camera depth must be uint16 HxW")
    if color.shape[:2] != depth.shape:
        raise ValueError("camera color and aligned depth shapes must match")
    y_slice, x_slice = _center_crop_slices(
        source_width=color.shape[1],
        source_height=color.shape[0],
        target_width=width,
        target_height=height,
    )
    cropped_color = color[y_slice, x_slice]
    cropped_depth = depth[y_slice, x_slice]
    resized_bgr = cv2.resize(
        cropped_color,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    resized_depth = cv2.resize(
        cropped_depth,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(resized_depth)


def _crop_preview_to_training_fov(
    frame_bgr: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("preview frame must have shape HxWx3")
    y_slice, x_slice = _center_crop_slices(
        source_width=frame.shape[1],
        source_height=frame.shape[0],
        target_width=target_width,
        target_height=target_height,
    )
    return np.ascontiguousarray(frame[y_slice, x_slice])


def _hand_inside_training_crop(
    display_uv: np.ndarray | None,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> bool:
    if display_uv is None:
        return False
    points = np.asarray(display_uv, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return False
    y_slice, x_slice = _center_crop_slices(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
    )
    x_min = float(x_slice.start or 0) / float(source_width)
    x_max = float(x_slice.stop or source_width) / float(source_width)
    y_min = float(y_slice.start or 0) / float(source_height)
    y_max = float(y_slice.stop or source_height) / float(source_height)
    return bool(
        np.all(points[:, 0] >= x_min)
        and np.all(points[:, 0] < x_max)
        and np.all(points[:, 1] >= y_min)
        and np.all(points[:, 1] < y_max)
    )


def _preview_lines(
    *,
    active: ActiveEpisode | None,
    mapping_mode: str,
    franka_mode: str,
    franka_valid: bool,
    deadman_requested: bool,
    deadman_commanded: bool,
    franka_speed: float,
    consecutive_tracked: int,
    stable_frames: int,
) -> tuple[str, ...]:
    if active is None:
        episode_line = "TASK=IDLE"
    else:
        stage_name = STAGE_NAMES.get(active.stage, active.stage)
        episode_line = (
            f"TASK={active.task.upper()} EP={active.writer.episode_id} "
            f"N={active.writer.step_count} STAGE={stage_name}"
        )
    if franka_mode == "off":
        franka_line = "FR3=OFF"
    elif not franka_valid:
        franka_line = "FR3=INVALID ZERO"
    elif franka_mode == "observe":
        franka_line = "FR3=OBSERVE ONLY"
    elif deadman_commanded:
        franka_line = f"FR3=MOVING {1000.0 * franka_speed:.1f} mm/s"
    elif deadman_requested:
        franka_line = "FR3=DEADMAN CENTER/DEADBAND ZERO"
    else:
        franka_line = "FR3=ARMED ZERO | HOLD LEFT MOUSE TO MOVE"
    return (
        episode_line,
        f"LEAP={mapping_mode} "
        f"gate={min(consecutive_tracked, stable_frames)}/{stable_frames}",
        franka_line,
        "G grasp | R release | SPACE accept | D reject | 1-7 stage | Q/E stop",
    )


def _parse_auto_tasks(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    tasks = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = [task for task in tasks if task not in {"grasp", "release"}]
    if invalid:
        raise ValueError(f"unknown mock auto tasks: {invalid}")
    return tasks


def _decode_cv_key(value: int) -> str | None:
    if value in {-1, 255}:
        return None
    if value == 27:
        return "\x1b"
    if 0 <= value <= 255:
        return chr(value)
    return None


def _vector_norm(value: np.ndarray) -> float:
    return math.sqrt(math.fsum(float(item) ** 2 for item in value.flat))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scaled_intrinsics(
    value: Any,
    *,
    width: int,
    height: int,
    horizontal_flip: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        source_width = float(value["width"])
        source_height = float(value["height"])
        if source_width <= 0.0 or source_height <= 0.0:
            return None
        y_slice, x_slice = _center_crop_slices(
            source_width=int(source_width),
            source_height=int(source_height),
            target_width=width,
            target_height=height,
        )
        crop_x = float(x_slice.start or 0)
        crop_y = float(y_slice.start or 0)
        crop_width = float((x_slice.stop or int(source_width)) - crop_x)
        crop_height = float((y_slice.stop or int(source_height)) - crop_y)
        scale_x = float(width) / crop_width
        scale_y = float(height) / crop_height
        source_ppx = float(value["ppx"])
        if horizontal_flip:
            source_ppx = source_width - 1.0 - source_ppx
        return {
            "width": int(width),
            "height": int(height),
            "ppx": (source_ppx - crop_x) * scale_x,
            "ppy": (float(value["ppy"]) - crop_y) * scale_y,
            "fx": float(value["fx"]) * scale_x,
            "fy": float(value["fy"]) * scale_y,
            "source_crop_xywh": [crop_x, crop_y, crop_width, crop_height],
            "horizontal_flip": bool(horizontal_flip),
            "model": str(value.get("model", "unknown")),
            "coeffs": [float(item) for item in value.get("coeffs", ())],
        }
    except (KeyError, TypeError, ValueError):
        return None


def _center_crop_slices(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[slice, slice]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_width = max(1, int(round(source_height * target_aspect)))
        x0 = (source_width - crop_width) // 2
        return slice(0, source_height), slice(x0, x0 + crop_width)
    crop_height = max(1, int(round(source_width / target_aspect)))
    y0 = (source_height - crop_height) // 2
    return slice(y0, y0 + crop_height), slice(0, source_width)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect synchronized grasp/release demonstrations for Diffusion Policy"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teleop-config", type=Path, default=DEFAULT_TELEOP_CONFIG)
    parser.add_argument("--source", choices=("d455", "mock"), default="d455")
    parser.add_argument("--d455-serial")
    parser.add_argument(
        "--leap-device",
        choices=("mock", "real"),
        default="mock",
    )
    parser.add_argument("--leap-port", default="")
    parser.add_argument("--enable-leap-torque", action="store_true")
    parser.add_argument(
        "--franka-mode",
        choices=("off", "observe", "teleop"),
        default="off",
    )
    parser.add_argument("--franka-uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--enable-franka-motion", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-note", default="")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mock-auto-episodes", default="")
    parser.add_argument("--mock-frames-per-episode", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))
