"""Intel RealSense D455 RGB-D source for LEAP Hand tracking.

Depth is aligned to the color stream and used as a fail-closed palm tracking
gate. Finger angles come from MediaPipe world landmarks, so isolated fingertip
depth holes cannot directly close a robot joint. ``pyrealsense2`` is imported
lazily so the existing webcam and mock modes keep working without the add-on.
"""

from __future__ import annotations

import importlib
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

PALM_LANDMARK_INDICES = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class D455Settings:
    serial: str
    enforce_model: bool
    required_model: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    fps: int
    frame_timeout_ms: int
    warmup_frames: int
    depth_patch_radius_px: int
    minimum_palm_depth_samples: int
    minimum_depth_m: float
    maximum_depth_m: float
    emitter_enabled: bool

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        serial_override: str | None = None,
    ) -> D455Settings:
        raw = config.get("d455")
        if not isinstance(raw, dict):
            raise ValueError("missing config section 'd455'")
        serial = str(raw.get("serial", "")).strip()
        if serial_override is not None:
            serial = str(serial_override).strip()
        settings = cls(
            serial=serial,
            enforce_model=bool(raw.get("enforce_model", True)),
            required_model=str(raw.get("required_model", "D455")).strip(),
            color_width=int(raw["color_width"]),
            color_height=int(raw["color_height"]),
            depth_width=int(raw["depth_width"]),
            depth_height=int(raw["depth_height"]),
            fps=int(raw["fps"]),
            frame_timeout_ms=int(raw["frame_timeout_ms"]),
            warmup_frames=int(raw["warmup_frames"]),
            depth_patch_radius_px=int(raw["depth_patch_radius_px"]),
            minimum_palm_depth_samples=int(raw["minimum_palm_depth_samples"]),
            minimum_depth_m=float(raw["minimum_depth_m"]),
            maximum_depth_m=float(raw["maximum_depth_m"]),
            emitter_enabled=bool(raw.get("emitter_enabled", True)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        positive = (
            self.color_width,
            self.color_height,
            self.depth_width,
            self.depth_height,
            self.fps,
            self.frame_timeout_ms,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(
                "D455 stream dimensions, FPS, and timeout must be positive"
            )
        if self.warmup_frames < 0 or self.depth_patch_radius_px < 0:
            raise ValueError("D455 warmup frames and patch radius must be non-negative")
        if not 1 <= self.minimum_palm_depth_samples <= len(PALM_LANDMARK_INDICES):
            raise ValueError("D455 minimum palm depth samples must be in [1, 5]")
        if (
            not math.isfinite(self.minimum_depth_m)
            or not math.isfinite(self.maximum_depth_m)
            or self.minimum_depth_m <= 0.0
            or self.minimum_depth_m >= self.maximum_depth_m
        ):
            raise ValueError(
                "D455 depth range must be finite, positive, and increasing"
            )
        if self.enforce_model and not self.required_model:
            raise ValueError(
                "D455 required_model cannot be empty when enforcement is on"
            )


@dataclass(frozen=True)
class D455RGBDFrame:
    """One unannotated display-aligned RGB-D capture.

    ``color_bgr`` and ``depth_units`` are owned, contiguous copies.  Depth is
    stored in the camera's native units; multiply by ``depth_scale_m`` for
    metres.  ``camera_timestamp_ms`` is the RealSense color-frame timestamp
    when the SDK exposes one, while ``capture_monotonic_s`` is always the local
    monotonic clock sampled when the aligned frames were copied.
    """

    color_bgr: np.ndarray
    depth_units: np.ndarray
    capture_monotonic_s: float
    camera_timestamp_ms: float | None
    color_intrinsics: dict[str, Any] | None
    depth_scale_m: float


def median_depth_m(
    depth_units: np.ndarray,
    *,
    depth_scale_m: float,
    x_px: int,
    y_px: int,
    radius_px: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> float | None:
    """Return a robust metric depth around one aligned color pixel."""

    image = np.asarray(depth_units)
    if image.ndim != 2:
        raise ValueError("aligned D455 depth image must be two-dimensional")
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("D455 depth scale must be finite and positive")
    if radius_px < 0:
        raise ValueError("D455 depth patch radius must be non-negative")
    height, width = image.shape
    if width == 0 or height == 0:
        return None
    x = int(np.clip(x_px, 0, width - 1))
    y = int(np.clip(y_px, 0, height - 1))
    x0 = max(0, x - radius_px)
    x1 = min(width, x + radius_px + 1)
    y0 = max(0, y - radius_px)
    y1 = min(height, y + radius_px + 1)
    values_m = np.asarray(image[y0:y1, x0:x1], dtype=np.float64) * depth_scale_m
    valid = values_m[
        np.isfinite(values_m)
        & (values_m >= minimum_depth_m)
        & (values_m <= maximum_depth_m)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def palm_pixel_and_depth_m(
    normalized_landmarks: np.ndarray,
    depth_units: np.ndarray,
    *,
    depth_scale_m: float,
    radius_px: int,
    minimum_samples: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[tuple[int, int], float] | None:
    """Estimate palm pixel and depth from the wrist and four palm roots."""

    points = np.asarray(normalized_landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("MediaPipe landmarks must be a finite 21x3 array")
    depth = np.asarray(depth_units)
    if depth.ndim != 2:
        raise ValueError("aligned D455 depth image must be two-dimensional")
    height, width = depth.shape
    pixels: list[tuple[int, int]] = []
    depths_m: list[float] = []
    for index in PALM_LANDMARK_INDICES:
        x = int(round(float(points[index, 0]) * max(width - 1, 0)))
        y = int(round(float(points[index, 1]) * max(height - 1, 0)))
        sample = median_depth_m(
            depth,
            depth_scale_m=depth_scale_m,
            x_px=x,
            y_px=y,
            radius_px=radius_px,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
        )
        if sample is not None:
            pixels.append(
                (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))
            )
            depths_m.append(sample)
    if len(depths_m) < minimum_samples:
        return None
    pixel = tuple(
        int(round(float(np.median(axis)))) for axis in zip(*pixels, strict=True)
    )
    return (pixel[0], pixel[1]), float(np.median(depths_m))


class D455MediaPipeCamera:
    """MediaPipe hand pose backed by aligned D455 RGB-D frames."""

    def __init__(
        self,
        config: dict[str, Any],
        serial_override: str | None,
        *,
        disable_preview: bool,
        rs_module: Any | None = None,
    ) -> None:
        camera_cfg = config.get("camera")
        if not isinstance(camera_cfg, dict):
            raise ValueError("missing config section 'camera'")
        self.settings = D455Settings.from_config(
            config, serial_override=serial_override
        )
        self.flip_horizontal = bool(camera_cfg["flip_horizontal"])
        self.preview = bool(camera_cfg["preview"]) and not disable_preview
        self.required_handedness = str(camera_cfg["required_handedness"]).lower()
        if self.required_handedness not in {"any", "left", "right"}:
            raise ValueError("camera.required_handedness must be any, left, or right")

        if rs_module is None:
            try:
                rs_module = importlib.import_module("pyrealsense2")
            except ImportError as error:
                raise RuntimeError(
                    "D455 source requires pyrealsense2; install "
                    "requirements-d455.txt in the existing environment"
                ) from error
        self.rs = rs_module
        self.pipeline = self.rs.pipeline()
        stream_config = self.rs.config()
        if self.settings.serial:
            stream_config.enable_device(self.settings.serial)
        stream_config.enable_stream(
            self.rs.stream.depth,
            self.settings.depth_width,
            self.settings.depth_height,
            self.rs.format.z16,
            self.settings.fps,
        )
        stream_config.enable_stream(
            self.rs.stream.color,
            self.settings.color_width,
            self.settings.color_height,
            self.rs.format.bgr8,
            self.settings.fps,
        )

        self._started = False
        self.hands: Any | None = None
        try:
            profile = self.pipeline.start(stream_config)
            self._started = True
            device = profile.get_device()
            self.device_name = self._device_info(device, self.rs.camera_info.name)
            self.serial_number = self._device_info(
                device, self.rs.camera_info.serial_number
            )
            self.firmware_version = self._device_info(
                device, self.rs.camera_info.firmware_version
            )
            if (
                self.settings.enforce_model
                and self.settings.required_model.lower() not in self.device_name.lower()
            ):
                raise RuntimeError(
                    f"expected {self.settings.required_model}, got {self.device_name}"
                )
            depth_sensor = device.first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            if not math.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0.0:
                raise RuntimeError("D455 returned an invalid depth scale")
            if depth_sensor.supports(self.rs.option.emitter_enabled):
                depth_sensor.set_option(
                    self.rs.option.emitter_enabled,
                    1.0 if self.settings.emitter_enabled else 0.0,
                )
            self.align = self.rs.align(self.rs.stream.color)
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=float(camera_cfg["detection_confidence"]),
                min_tracking_confidence=float(camera_cfg["tracking_confidence"]),
            )
            self.drawer = mp.solutions.drawing_utils
            for _ in range(self.settings.warmup_frames):
                self.pipeline.wait_for_frames(self.settings.frame_timeout_ms)
        except BaseException:
            self.close()
            raise

        self.latest_palm_position_m: tuple[float, float, float] | None = None
        self.latest_palm_depth_m: float | None = None
        self.latest_hand_display_uv: np.ndarray | None = None
        self.depth_valid_frames = 0
        self.depth_invalid_frames = 0
        self._latest_color_bgr: np.ndarray | None = None
        self._latest_depth_units: np.ndarray | None = None
        self._latest_capture_monotonic_s: float | None = None
        self._latest_camera_timestamp_ms: float | None = None
        self._latest_color_intrinsics: dict[str, Any] | None = None

    def read(self) -> tuple[np.ndarray | None, np.ndarray, str, float]:
        frames = self.pipeline.wait_for_frames(self.settings.frame_timeout_ms)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("D455 returned an incomplete aligned RGB-D frameset")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        if color.ndim != 3 or color.shape[2] != 3:
            raise RuntimeError("D455 color frame must have shape HxWx3")
        if depth.ndim != 2:
            raise RuntimeError("D455 aligned depth frame must be two-dimensional")

        if self.flip_horizontal:
            display_color = cv2.flip(color, 1)
            display_depth = np.fliplr(depth)
        else:
            display_color = color
            display_depth = depth
        display_color = np.ascontiguousarray(display_color, dtype=np.uint8)
        display_depth = np.ascontiguousarray(display_depth, dtype=np.uint16)

        # Keep training observations independent from both the RealSense frame
        # buffers and the preview image that MediaPipe drawing mutates below.
        self._latest_color_bgr = display_color.copy(order="C")
        self._latest_depth_units = display_depth.copy(order="C")
        self._latest_capture_monotonic_s = time.monotonic()
        self._latest_camera_timestamp_ms = self._camera_timestamp_ms(color_frame)
        self._latest_color_intrinsics = self._color_intrinsics_metadata(color_frame)
        frame = display_color.copy(order="C")
        self.latest_hand_display_uv = None
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            self.latest_palm_position_m = None
            self.latest_palm_depth_m = None
            return None, frame, "unknown", 0.0

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = "unknown"
        confidence = 1.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label.lower()
            confidence = float(classification.score)
        self.drawer.draw_landmarks(
            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
        if self.required_handedness not in {"any", handedness}:
            self.latest_palm_position_m = None
            self.latest_palm_depth_m = None
            return None, frame, handedness, confidence

        normalized = np.asarray(
            [(point.x, point.y, point.z) for point in hand_landmarks.landmark],
            dtype=np.float64,
        )
        palm = palm_pixel_and_depth_m(
            normalized,
            display_depth,
            depth_scale_m=self.depth_scale_m,
            radius_px=self.settings.depth_patch_radius_px,
            minimum_samples=self.settings.minimum_palm_depth_samples,
            minimum_depth_m=self.settings.minimum_depth_m,
            maximum_depth_m=self.settings.maximum_depth_m,
        )
        if palm is None:
            self.depth_invalid_frames += 1
            self.latest_palm_position_m = None
            self.latest_palm_depth_m = None
            return None, frame, handedness, confidence

        (display_x, display_y), palm_depth_m = palm
        source_x = color.shape[1] - 1 - display_x if self.flip_horizontal else display_x
        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        palm_point = np.asarray(
            self.rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [float(source_x), float(display_y)],
                float(palm_depth_m),
            ),
            dtype=np.float64,
        )
        if palm_point.shape != (3,) or not np.isfinite(palm_point).all():
            self.depth_invalid_frames += 1
            self.latest_palm_position_m = None
            self.latest_palm_depth_m = None
            return None, frame, handedness, confidence
        if self.flip_horizontal:
            palm_point[0] *= -1.0
        self.latest_palm_position_m = tuple(float(value) for value in palm_point)
        self.latest_palm_depth_m = float(palm_depth_m)
        self.latest_hand_display_uv = normalized[:, :2].copy(order="C")
        self.depth_valid_frames += 1

        world_sets = getattr(result, "multi_hand_world_landmarks", None)
        if world_sets:
            points = np.asarray(
                [(point.x, point.y, point.z) for point in world_sets[0].landmark],
                dtype=np.float64,
            )
        else:
            points = normalized
        return points, frame, handedness, confidence

    def latest_rgbd(self) -> D455RGBDFrame:
        """Return safe copies of the most recently captured aligned RGB-D frame."""

        if (
            self._latest_color_bgr is None
            or self._latest_depth_units is None
            or self._latest_capture_monotonic_s is None
        ):
            raise RuntimeError("D455 has not captured an RGB-D frame yet; call read()")
        return D455RGBDFrame(
            color_bgr=self._latest_color_bgr.copy(order="C"),
            depth_units=self._latest_depth_units.copy(order="C"),
            capture_monotonic_s=float(self._latest_capture_monotonic_s),
            camera_timestamp_ms=self._latest_camera_timestamp_ms,
            color_intrinsics=deepcopy(self._latest_color_intrinsics),
            depth_scale_m=float(self.depth_scale_m),
        )

    def show(
        self,
        frame: np.ndarray,
        status: str,
        color: tuple[int, int, int],
        details: tuple[str, ...],
    ) -> int:
        if not self.preview:
            return -1
        cv2.putText(
            frame,
            status,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Q/E stop | SPACE pause/hold | L reload config.yml",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        depth_text = (
            "D455 palm depth=INVALID"
            if self.latest_palm_position_m is None
            else "D455 palm xyz="
            + "["
            + " ".join(
                f"{1000.0 * value:7.1f}" for value in self.latest_palm_position_m
            )
            + "] mm"
        )
        for line_index, line in enumerate((depth_text,) + details):
            cv2.putText(
                frame,
                line,
                (20, 110 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("D455: LEAP Hand real + MuJoCo", frame)
        return cv2.waitKey(1) & 0xFF

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "d455",
            "device_name": self.device_name,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "color_stream": [
                self.settings.color_width,
                self.settings.color_height,
                self.settings.fps,
            ],
            "depth_stream": [
                self.settings.depth_width,
                self.settings.depth_height,
                self.settings.fps,
            ],
            "depth_aligned_to_color": True,
            "flip_horizontal": self.flip_horizontal,
            "depth_scale_m": self.depth_scale_m,
            "depth_valid_frames": self.depth_valid_frames,
            "depth_invalid_frames": self.depth_invalid_frames,
            "final_palm_position_m": self.latest_palm_position_m,
            "latest_rgbd_available": self._latest_color_bgr is not None,
            "latest_capture_monotonic_s": self._latest_capture_monotonic_s,
            "latest_camera_timestamp_ms": self._latest_camera_timestamp_ms,
            "color_intrinsics": deepcopy(self._latest_color_intrinsics),
            "raw_rgbd_saved": False,
        }

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if self._started:
            self.pipeline.stop()
            self._started = False
        cv2.destroyAllWindows()

    @staticmethod
    def _device_info(device: Any, key: Any) -> str:
        try:
            return str(device.get_info(key))
        except Exception:
            return "unknown"

    @staticmethod
    def _camera_timestamp_ms(color_frame: Any) -> float | None:
        try:
            timestamp_ms = float(color_frame.get_timestamp())
        except Exception:
            return None
        return timestamp_ms if math.isfinite(timestamp_ms) else None

    @staticmethod
    def _color_intrinsics_metadata(color_frame: Any) -> dict[str, Any] | None:
        try:
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            return {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "model": str(intrinsics.model),
                "coeffs": [float(value) for value in intrinsics.coeffs],
            }
        except Exception:
            return None
