"""Camera teleoperation and image-free LEAP grasp demonstration collection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import cv2
import numpy as np
from src.d455 import D455MediaPipeCamera
from src.leap_hand_hardware import (
    POSITION_CURRENT_MODE,
    DynamixelLeapHand,
    HardwareSettings,
    LeapHandFeedback,
    MockLeapHand,
)
from src.leap_hand_mapping import JOINT_NAMES
from src.runtime import MediaPipeCamera, load_config, new_mapper, section

from .camera import MockVisionSource
from .proprio_episode import ProprioEpisodeWriter
from .proprio_queued_writer import QueuedProprioEpisodeWriter
from .proprio_schema import ProprioEpisodeSpec

COLLECTOR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = COLLECTOR_ROOT.parent
DEFAULT_CONFIG = COLLECTOR_ROOT / "config.yml"
DEFAULT_TELEOP_CONFIG = REPOSITORY_ROOT / "config.yml"
DEFAULT_OUTPUT = COLLECTOR_ROOT / "datasets" / "leap_proprio_grasp"
WINDOW_NAME = "LEAP candy grasp proprio demonstration collector"


class TerminalKeys:
    """Dependency-free, non-blocking single-key input with safe restoration."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stdin if stream is None else stream
        self._msvcrt: ModuleType | None = None
        self._termios: ModuleType | None = None
        self._original_settings: list[object] | None = None

    def __enter__(self) -> "TerminalKeys":
        if not self.stream.isatty():
            raise RuntimeError("single-key control requires an interactive terminal")
        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt
        else:
            import termios
            import tty

            descriptor = self.stream.fileno()
            self._termios = termios
            self._original_settings = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._termios is not None and self._original_settings is not None:
            self._termios.tcsetattr(
                self.stream.fileno(),
                self._termios.TCSADRAIN,
                self._original_settings,
            )

    def poll(self) -> str | None:
        if self._msvcrt is not None:
            if not self._msvcrt.kbhit():
                return None
            key = self._msvcrt.getwch()
            if key in {"\x00", "\xe0"}:
                if self._msvcrt.kbhit():
                    self._msvcrt.getwch()
                return None
            return key
        readable, _writable, _exceptional = select.select([self.stream], [], [], 0.0)
        return self.stream.read(1) if readable else None


@dataclass(frozen=True)
class CollectorSettings:
    sample_hz: float
    minimum_episode_steps: int
    maximum_episode_s: float
    sample_period_tolerance_s: float
    maximum_feedback_bracket_s: float
    maximum_consecutive_feedback_failures: int
    max_pending_steps: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CollectorSettings":
        raw = section(config, "leap_proprio_collector")
        settings = cls(
            sample_hz=float(raw["sample_hz"]),
            minimum_episode_steps=int(raw["minimum_episode_steps"]),
            maximum_episode_s=float(raw["maximum_episode_s"]),
            sample_period_tolerance_s=float(raw["sample_period_tolerance_s"]),
            maximum_feedback_bracket_s=float(raw["maximum_feedback_bracket_s"]),
            maximum_consecutive_feedback_failures=int(
                raw["maximum_consecutive_feedback_failures"]
            ),
            max_pending_steps=int(raw["max_pending_steps"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not math.isfinite(self.sample_hz) or not 1.0 <= self.sample_hz <= 60.0:
            raise ValueError("sample_hz must be finite and in [1, 60]")
        if self.minimum_episode_steps < 2:
            raise ValueError("minimum_episode_steps must be at least 2")
        if not math.isfinite(self.maximum_episode_s) or self.maximum_episode_s <= 0:
            raise ValueError("maximum_episode_s must be finite and positive")
        period = 1.0 / self.sample_hz
        if (
            not math.isfinite(self.sample_period_tolerance_s)
            or self.sample_period_tolerance_s < 0.0
            or self.sample_period_tolerance_s > period
        ):
            raise ValueError(
                "sample_period_tolerance_s must be finite and in [0, sample period]"
            )
        if (
            not math.isfinite(self.maximum_feedback_bracket_s)
            or self.maximum_feedback_bracket_s <= 0.0
            or self.maximum_feedback_bracket_s > period
        ):
            raise ValueError(
                "maximum_feedback_bracket_s must be finite and in (0, sample period]"
            )
        if self.maximum_consecutive_feedback_failures < 1:
            raise ValueError("maximum_consecutive_feedback_failures must be positive")
        if self.max_pending_steps < 1:
            raise ValueError("max_pending_steps must be positive")


@dataclass
class ActiveEpisode:
    writer: QueuedProprioEpisodeWriter
    started_monotonic_s: float
    next_sample_due_s: float
    last_sample_timestamp_s: float
    previous_feedback: LeapHandFeedback
    previous_goal_position_rad: np.ndarray
    invalid_steps: int = 0
    sticky_invalid_reasons: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RuntimeSummary:
    stop_reason: str
    accepted_episodes: int
    rejected_episodes: int
    interrupted_episodes: int
    control_frames: int
    submitted_steps: int
    feedback_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "accepted_episodes": self.accepted_episodes,
            "rejected_episodes": self.rejected_episodes,
            "interrupted_episodes": self.interrupted_episodes,
            "control_frames": self.control_frames,
            "submitted_steps": self.submitted_steps,
            "feedback_failures": self.feedback_failures,
        }


def run(args: argparse.Namespace) -> RuntimeSummary:
    collector_config_path = args.config.resolve()
    teleop_config_path = args.teleop_config.resolve()
    collector_config = load_config(collector_config_path)
    teleop_config = load_config(teleop_config_path)
    settings = CollectorSettings.from_config(collector_config)
    settings = _apply_setting_overrides(settings, args)
    hardware_settings = HardwareSettings.from_config(section(teleop_config, "hardware"))
    if args.leap_device == "real" and not hardware_settings.disable_torque_on_exit:
        raise RuntimeError(
            "real proprio collection requires hardware.disable_torque_on_exit=true"
        )
    control_hz = float(section(teleop_config, "control")["update_hz"])
    if not math.isfinite(control_hz) or not 1.0 <= control_hz <= 60.0:
        raise ValueError("teleoperation control.update_hz must be in [1, 60]")
    if settings.sample_hz > control_hz:
        raise ValueError("proprio sample_hz cannot exceed control.update_hz")
    _validate_args(args, settings)

    mapper_settings, mapper, landmark_filter = new_mapper(teleop_config)
    source: MediaPipeCamera | D455MediaPipeCamera | MockVisionSource
    if args.source == "camera":
        source = MediaPipeCamera(
            teleop_config,
            args.camera_index,
            disable_preview=True,
        )
    elif args.source == "d455":
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

    output = args.output.resolve()
    sample_period_s = 1.0 / settings.sample_hz
    control_period_s = 1.0 / control_hz
    run_start_s = time.monotonic()
    last_goal = np.asarray(mapper_settings.joint_open_rad, dtype=np.float64)
    last_vision_target = last_goal.copy()
    last_feedback: LeapHandFeedback | None = None
    active: ActiveEpisode | None = None
    consecutive_tracked = 0
    motion_started = False
    consecutive_feedback_failures = 0
    feedback_failures = 0
    accepted_episodes = 0
    rejected_episodes = 0
    interrupted_episodes = 0
    control_frames = 0
    submitted_steps = 0
    stop_reason = "operator_stop"
    fatal_error: BaseException | None = None
    next_control_due_s = time.monotonic()
    auto_completed = 0
    window_created = False

    terminal_context: Any = contextlib.nullcontext(None)
    if sys.stdin.isatty():
        terminal_context = TerminalKeys()

    try:
        initial_position = driver.connect_and_enable()
        mapper.last_joints = np.asarray(initial_position, dtype=np.float64).copy()
        last_goal = np.asarray(initial_position, dtype=np.float64).copy()
        last_vision_target = last_goal.copy()
        last_feedback = driver.read_feedback()
        print(
            json.dumps(
                {
                    "status": "ready",
                    "collector": "leap_proprio_grasp",
                    "output": str(output),
                    "source": args.source,
                    "leap_device": args.leap_device,
                    "observation_dim": 48,
                    "action_dim": 16,
                    "rgbd_saved": False,
                    "franka_state_saved": False,
                    "controls": "G start grasp | SPACE accept | D reject | Q/E stop",
                    "control_semantics": (
                        "Mode 5 current-based position; not outer-loop force control"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        with terminal_context as terminal:
            while True:
                iteration_start_s = time.monotonic()
                if (
                    args.duration_s > 0.0
                    and iteration_start_s - run_start_s >= args.duration_s
                ):
                    stop_reason = "duration_complete"
                    break
                if active is not None:
                    active.writer.raise_if_failed()

                landmarks, frame, handedness, confidence = source.read()
                tracking_ready = False
                status = "waiting_for_hand"
                if landmarks is None:
                    consecutive_tracked = 0
                    landmark_filter.reset()
                    status = "tracking_lost_hold" if motion_started else status
                    if active is not None:
                        active.sticky_invalid_reasons.add("hand_tracking_lost")
                else:
                    consecutive_tracked += 1
                    filtered = landmark_filter.update(landmarks)
                    mapped = mapper.update(
                        filtered,
                        now_s=iteration_start_s - run_start_s,
                    )
                    last_vision_target = np.asarray(
                        mapped.joint_target_rad, dtype=np.float64
                    )
                    tracking_ready = (
                        consecutive_tracked >= hardware_settings.stable_tracking_frames
                    )
                    if tracking_ready:
                        motion_started = True
                        status = "tracking"
                    else:
                        status = (
                            "stabilizing_tracking "
                            f"{consecutive_tracked}/"
                            f"{hardware_settings.stable_tracking_frames}"
                        )

                feedback_this_frame: LeapHandFeedback | None = None
                try:
                    feedback_this_frame = driver.read_feedback()
                    last_feedback = feedback_this_frame
                    consecutive_feedback_failures = 0
                except Exception:
                    feedback_failures += 1
                    consecutive_feedback_failures += 1
                    if active is not None:
                        active.sticky_invalid_reasons.add("leap_feedback_read_failed")
                    if (
                        consecutive_feedback_failures
                        >= settings.maximum_consecutive_feedback_failures
                    ):
                        raise RuntimeError(
                            "too many consecutive LEAP feedback read failures"
                        )

                # Pair the just-measured pre-action proprioception with the
                # post-slew goal that is successfully sent immediately after
                # it. This preserves the causal obs_t -> action_t contract.
                if tracking_ready and feedback_this_frame is not None:
                    last_goal = driver.command_mapping(last_vision_target)

                if active is not None and feedback_this_frame is not None:
                    recorded = _record_resampled_samples(
                        active,
                        feedback=feedback_this_frame,
                        run_start_s=run_start_s,
                        sample_period_s=sample_period_s,
                        sample_period_tolerance_s=(settings.sample_period_tolerance_s),
                        maximum_feedback_bracket_s=(
                            settings.maximum_feedback_bracket_s
                        ),
                        tracking_ready=tracking_ready,
                        goal_position_rad=last_goal,
                        vision_target_rad=last_vision_target,
                    )
                    submitted_steps += recorded

                keys: set[str] = set()
                if not args.headless:
                    window_created = True
                    preview_key = _show_preview(
                        frame,
                        status=status,
                        active=active,
                        feedback=last_feedback,
                        goal_position_rad=last_goal,
                        handedness=handedness,
                        confidence=confidence,
                        accepted=accepted_episodes,
                        rejected=rejected_episodes,
                    )
                    decoded = _decode_key(preview_key)
                    if decoded is not None:
                        keys.add(decoded)
                    if not _preview_window_open():
                        keys.add("q")
                if terminal is not None:
                    terminal_key = terminal.poll()
                    if terminal_key is not None:
                        keys.add(terminal_key)
                normalized_keys = {key.lower() for key in keys}

                if normalized_keys.intersection({"q", "e", "\x1b"}):
                    stop_reason = "operator_stop"
                    break

                if "d" in normalized_keys and active is not None:
                    path = active.writer.reject("operator_rejected")
                    print(f"rejected: {path}")
                    rejected_episodes += 1
                    active = None

                if " " in normalized_keys and active is not None:
                    if active.writer.step_count < settings.minimum_episode_steps:
                        print(
                            "episode is too short: "
                            f"{active.writer.step_count}/"
                            f"{settings.minimum_episode_steps} steps"
                        )
                    else:
                        active, accepted_delta, rejected_delta = _finish_episode(active)
                        accepted_episodes += accepted_delta
                        rejected_episodes += rejected_delta

                if "g" in normalized_keys:
                    if active is not None:
                        print("a grasp episode is already recording")
                    elif not tracking_ready or feedback_this_frame is None:
                        print(
                            "cannot start: wait for stable hand tracking "
                            "and LEAP feedback"
                        )
                    else:
                        active = _start_episode(
                            output,
                            feedback=feedback_this_frame,
                            run_start_s=run_start_s,
                            sample_period_s=sample_period_s,
                            settings=settings,
                            hardware_settings=hardware_settings,
                            collector_config_path=collector_config_path,
                            teleop_config_path=teleop_config_path,
                            source_name=args.source,
                            leap_device=args.leap_device,
                            motor_model_numbers=driver.model_numbers,
                            goal_position_rad=last_goal,
                        )
                        print(f"recording grasp: {active.writer.episode_id}")

                if args.mock_auto_episodes > 0:
                    if active is None and auto_completed < args.mock_auto_episodes:
                        if tracking_ready and feedback_this_frame is not None:
                            active = _start_episode(
                                output,
                                feedback=feedback_this_frame,
                                run_start_s=run_start_s,
                                sample_period_s=sample_period_s,
                                settings=settings,
                                hardware_settings=hardware_settings,
                                collector_config_path=collector_config_path,
                                teleop_config_path=teleop_config_path,
                                source_name=args.source,
                                leap_device=args.leap_device,
                                motor_model_numbers=driver.model_numbers,
                                goal_position_rad=last_goal,
                            )
                    elif (
                        active is not None
                        and active.writer.step_count >= args.mock_frames_per_episode
                    ):
                        active, accepted_delta, rejected_delta = _finish_episode(active)
                        accepted_episodes += accepted_delta
                        rejected_episodes += rejected_delta
                        auto_completed += 1
                        if auto_completed >= args.mock_auto_episodes:
                            stop_reason = "mock_auto_complete"
                            break

                if (
                    active is not None
                    and iteration_start_s - active.started_monotonic_s
                    >= settings.maximum_episode_s
                ):
                    path = active.writer.reject("maximum_episode_duration_exceeded")
                    print(f"automatically rejected overlong episode: {path}")
                    rejected_episodes += 1
                    active = None

                control_frames += 1
                next_control_due_s += control_period_s
                remaining_s = next_control_due_s - time.monotonic()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
                elif -remaining_s > control_period_s:
                    if active is not None:
                        active.sticky_invalid_reasons.add(
                            "control_loop_deadline_missed"
                        )
                    next_control_due_s = time.monotonic()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except BaseException as error:
        fatal_error = error
        stop_reason = "runtime_error"
    finally:
        # Stop physical actuation before any potentially slow disk drain.
        try:
            driver.close()
        except Exception as error:
            if fatal_error is None:
                fatal_error = error
                stop_reason = "leap_shutdown_error"
        try:
            source.close()
        except Exception:
            pass
        if window_created:
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(WINDOW_NAME)
        if active is not None:
            try:
                path = active.writer.close_partial(reason=stop_reason)
                print(f"interrupted episode kept partial: {path}")
                interrupted_episodes += 1
            except Exception as error:
                if fatal_error is None:
                    fatal_error = error
                    stop_reason = "persistence_shutdown_error"

    summary = RuntimeSummary(
        stop_reason=stop_reason,
        accepted_episodes=accepted_episodes,
        rejected_episodes=rejected_episodes,
        interrupted_episodes=interrupted_episodes,
        control_frames=control_frames,
        submitted_steps=submitted_steps,
        feedback_failures=feedback_failures,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if fatal_error is not None:
        raise RuntimeError(
            f"LEAP proprio collector failed: {fatal_error}"
        ) from fatal_error
    return summary


def _record_resampled_samples(
    active: ActiveEpisode,
    *,
    feedback: LeapHandFeedback,
    run_start_s: float,
    sample_period_s: float,
    sample_period_tolerance_s: float,
    maximum_feedback_bracket_s: float,
    tracking_ready: bool,
    goal_position_rad: np.ndarray,
    vision_target_rad: np.ndarray,
) -> int:
    previous = active.previous_feedback
    previous_time_s = float(previous.monotonic_s)
    current_time_s = float(feedback.monotonic_s)
    if current_time_s <= previous_time_s:
        active.sticky_invalid_reasons.add("non_increasing_feedback_timestamp")
        return 0

    bracket_s = current_time_s - previous_time_s
    if not tracking_ready:
        active.sticky_invalid_reasons.add("hand_tracking_not_stable")
    if bracket_s > maximum_feedback_bracket_s:
        active.sticky_invalid_reasons.add("feedback_bracket_too_large")
    base_reasons = set(active.sticky_invalid_reasons)

    recorded = 0
    epsilon_s = 1e-9
    while active.next_sample_due_s <= current_time_s + epsilon_s:
        due_s = active.next_sample_due_s
        alpha = float(np.clip((due_s - previous_time_s) / bracket_s, 0.0, 1.0))
        actual_position = _interpolate(
            previous.actual_position_rad,
            feedback.actual_position_rad,
            alpha,
        )
        current_raw = np.rint(
            _interpolate(
                previous.present_current_raw,
                feedback.present_current_raw,
                alpha,
            )
        ).astype(np.int16)
        velocity_raw = np.rint(
            _interpolate(
                previous.present_velocity_raw,
                feedback.present_velocity_raw,
                alpha,
            )
        ).astype(np.int64)
        # The goal sent after the previous feedback remains the exact command
        # active on the motor until the current command is sent. Keep that
        # real post-slew command with zero-order hold; never invent an action
        # by interpolating two commands that were not actually transmitted.
        goal = active.previous_goal_position_rad.copy()
        timestamp_s = due_s - run_start_s
        reasons = set(base_reasons)
        sample_dt_s = timestamp_s - active.last_sample_timestamp_s
        if abs(sample_dt_s - sample_period_s) > sample_period_tolerance_s:
            reasons.add("sample_period_out_of_tolerance")
        active.writer.append(
            timestamp_s=timestamp_s,
            actual_position_rad=actual_position,
            present_current_raw=current_raw,
            goal_position_rad=goal,
            valid=not reasons,
            invalid_reasons=tuple(sorted(reasons)),
            extra={
                "resampled_to_fixed_clock": True,
                "feedback_bracket_start_s": previous_time_s - run_start_s,
                "feedback_bracket_end_s": current_time_s - run_start_s,
                "feedback_bracket_s": bracket_s,
                "interpolation_alpha": alpha,
                "action_zero_order_hold_from_s": previous_time_s - run_start_s,
                "present_velocity_raw": velocity_raw.tolist(),
                "right_bracket_vision_target_rad": np.asarray(
                    vision_target_rad, dtype=np.float32
                ).tolist(),
            },
        )
        if reasons:
            active.invalid_steps += 1
        active.last_sample_timestamp_s = timestamp_s
        active.next_sample_due_s += sample_period_s
        recorded += 1

    active.previous_feedback = feedback
    active.previous_goal_position_rad = np.asarray(
        goal_position_rad, dtype=np.float64
    ).copy()
    if recorded:
        active.sticky_invalid_reasons.clear()
    return recorded


def _interpolate(start: Any, end: Any, alpha: float) -> np.ndarray:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if start_array.shape != (16,) or end_array.shape != (16,):
        raise ValueError("LEAP interpolation inputs must contain 16 values")
    return start_array + float(alpha) * (end_array - start_array)


def _start_episode(
    output: Path,
    *,
    feedback: LeapHandFeedback,
    run_start_s: float,
    sample_period_s: float,
    settings: CollectorSettings,
    hardware_settings: HardwareSettings,
    collector_config_path: Path,
    teleop_config_path: Path,
    source_name: str,
    leap_device: str,
    motor_model_numbers: tuple[int, ...],
    goal_position_rad: np.ndarray,
) -> ActiveEpisode:
    initial_timestamp_s = float(feedback.monotonic_s - run_start_s)
    spec = ProprioEpisodeSpec(
        sample_period_s=sample_period_s,
        sample_period_tolerance_s=settings.sample_period_tolerance_s,
        joint_names=tuple(JOINT_NAMES),
        extra={
            "policy_observation": (
                "actual_position_rad[16] + finite_difference_velocity_rad_s[16] "
                "+ signed_present_current_raw[16]"
            ),
            "observation_action_alignment": (
                "pre_action_feedback_then_successful_post_slew_goal"
            ),
            "fixed_clock_resampling": (
                "linear_proprio_between_adjacent_feedback; "
                "zero_order_hold_of_actual_post_slew_goal"
            ),
            "maximum_feedback_bracket_s": settings.maximum_feedback_bracket_s,
            "excluded_from_policy": [
                "rgb",
                "depth",
                "franka_state",
                "wrist_pose",
            ],
            "source_used_for_human_mapping_only": source_name,
            "leap_device": leap_device,
            "operating_mode": POSITION_CURRENT_MODE,
            "operating_mode_name": "current_based_position_control",
            "outer_loop_force_control": False,
            "goal_current_raw": hardware_settings.current_limit,
            "present_current_unit": "signed_raw_register_count_model_dependent",
            "present_current_is_force_calibrated": False,
            "kp": hardware_settings.kp,
            "ki": hardware_settings.ki,
            "kd": hardware_settings.kd,
            "maximum_goal_step_rad": hardware_settings.maximum_step_rad,
            "motor_ids": list(hardware_settings.motor_ids),
            "motor_model_numbers": list(motor_model_numbers),
            "collector_config_sha256": _sha256(collector_config_path),
            "teleop_config_sha256": _sha256(teleop_config_path),
            "git_commit": _git_commit(),
        },
    )
    base = ProprioEpisodeWriter(
        output,
        spec,
        initial_timestamp_s=initial_timestamp_s,
        initial_actual_position_rad=feedback.actual_position_rad,
    )
    writer = QueuedProprioEpisodeWriter(
        base,
        max_pending_steps=settings.max_pending_steps,
    )
    return ActiveEpisode(
        writer=writer,
        started_monotonic_s=float(feedback.monotonic_s),
        next_sample_due_s=float(feedback.monotonic_s + sample_period_s),
        last_sample_timestamp_s=initial_timestamp_s,
        previous_feedback=feedback,
        previous_goal_position_rad=np.asarray(
            goal_position_rad, dtype=np.float64
        ).copy(),
    )


def _finish_episode(
    active: ActiveEpisode,
) -> tuple[None, int, int]:
    pending_invalid = sorted(active.sticky_invalid_reasons)
    if active.invalid_steps or pending_invalid:
        reason = f"contains_{active.invalid_steps}_invalid_steps"
        if pending_invalid:
            reason += "_and_pending_" + "-".join(pending_invalid)
        path = active.writer.reject(reason)
        print(f"cannot accept ({reason}); rejected: {path}")
        return None, 0, 1
    path = active.writer.accept(
        notes=("grasp-only LEAP proprio demonstration; arm motion and images excluded")
    )
    print(f"accepted: {path}")
    return None, 1, 0


def _show_preview(
    frame_bgr: np.ndarray,
    *,
    status: str,
    active: ActiveEpisode | None,
    feedback: LeapHandFeedback | None,
    goal_position_rad: np.ndarray,
    handedness: str,
    confidence: float,
    accepted: int,
    rejected: int,
) -> int:
    frame = np.asarray(frame_bgr).copy()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("camera preview must have shape HxWx3")
    lines = [
        f"{status} | hand={handedness} {confidence:.2f}",
        "G start grasp | SPACE accept | D reject | Q/E stop",
        "policy input: LEAP proprio 48D only (NO RGB-D / NO FR3)",
        f"accepted={accepted} rejected={rejected}",
    ]
    if feedback is not None:
        error = np.asarray(goal_position_rad) - feedback.actual_position_rad
        lines.append(
            "goal-actual mean/max="
            f"{np.mean(np.abs(error)):.3f}/{np.max(np.abs(error)):.3f} rad | "
            f"current |mean|max={np.mean(np.abs(feedback.present_current_raw)):.1f}/"
            f"{np.max(np.abs(feedback.present_current_raw))} raw"
        )
    if active is not None:
        lines.append(
            f"RECORDING grasp steps={active.writer.step_count} "
            f"pending={active.writer.pending_count} invalid={active.invalid_steps}"
        )
    overlay_height = min(frame.shape[0], 34 + 27 * len(lines))
    overlay = frame[:overlay_height].copy()
    overlay[:] = (18, 18, 18)
    cv2.addWeighted(overlay, 0.76, frame[:overlay_height], 0.24, 0.0, overlay)
    frame[:overlay_height] = overlay
    for index, line in enumerate(lines):
        color = (70, 220, 90)
        if "NO " in line or "invalid=" in line and active and active.invalid_steps:
            color = (80, 185, 255)
        cv2.putText(
            frame,
            line,
            (15, 27 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def _preview_window_open() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1.0
    except cv2.error:
        return False


def _decode_key(value: int) -> str | None:
    if value < 0 or value == 255:
        return None
    if value == 27:
        return "\x1b"
    try:
        return chr(value)
    except ValueError:
        return None


def _apply_setting_overrides(
    settings: CollectorSettings, args: argparse.Namespace
) -> CollectorSettings:
    updated = CollectorSettings(
        sample_hz=(settings.sample_hz if args.sample_hz is None else args.sample_hz),
        minimum_episode_steps=(
            settings.minimum_episode_steps
            if args.minimum_episode_steps is None
            else args.minimum_episode_steps
        ),
        maximum_episode_s=(
            settings.maximum_episode_s
            if args.maximum_episode_s is None
            else args.maximum_episode_s
        ),
        sample_period_tolerance_s=settings.sample_period_tolerance_s,
        maximum_feedback_bracket_s=settings.maximum_feedback_bracket_s,
        maximum_consecutive_feedback_failures=(
            settings.maximum_consecutive_feedback_failures
        ),
        max_pending_steps=settings.max_pending_steps,
    )
    updated.validate()
    return updated


def _validate_args(args: argparse.Namespace, settings: CollectorSettings) -> None:
    if args.leap_device == "real" and not args.enable_leap_torque:
        raise RuntimeError(
            "real LEAP Hand is disarmed; add --enable-leap-torque after checking COM"
        )
    if args.leap_device == "real" and not str(args.leap_port).strip():
        raise RuntimeError("--leap-port COMx is required for the real LEAP Hand")
    if args.enable_leap_torque and args.leap_device != "real":
        raise ValueError("--enable-leap-torque is only valid with --leap-device real")
    if args.leap_device == "real" and args.source == "mock":
        raise ValueError("refusing mock human input with a real LEAP Hand")
    if args.mock_auto_episodes < 0:
        raise ValueError("--mock-auto-episodes cannot be negative")
    if (
        args.mock_auto_episodes
        and args.mock_frames_per_episode < settings.minimum_episode_steps
    ):
        raise ValueError(
            "--mock-frames-per-episode cannot be below minimum_episode_steps"
        )
    if args.mock_auto_episodes and (
        args.source != "mock" or args.leap_device != "mock"
    ):
        raise ValueError("mock auto episodes require --source mock --leap-device mock")
    if not math.isfinite(args.duration_s) or args.duration_s < 0.0:
        raise ValueError("--duration-s must be finite and non-negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect grasp-only LEAP proprioception. Camera data drives the "
            "human-hand mapper but is never saved as a policy observation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teleop-config", type=Path, default=DEFAULT_TELEOP_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source", choices=("camera", "d455", "mock"), default="camera"
    )
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--d455-serial", default="")
    parser.add_argument("--leap-device", choices=("mock", "real"), default="mock")
    parser.add_argument("--leap-port", default="")
    parser.add_argument("--enable-leap-torque", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sample-hz", type=float)
    parser.add_argument("--minimum-episode-steps", type=int)
    parser.add_argument("--maximum-episode-s", type=float)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--mock-auto-episodes", type=int, default=0)
    parser.add_argument("--mock-frames-per-episode", type=int, default=30)
    return parser.parse_args(argv)


def main() -> None:
    run(_parse_args())
