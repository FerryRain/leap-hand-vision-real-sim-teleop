"""OpenCV preview and mouse deadman state."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence

import cv2
import numpy as np

WINDOW_NAME = "FR3 + LEAP DP demonstration collector"


class PreviewUI:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._deadman_requested = False
        self._created = False

    @property
    def deadman_down(self) -> bool:
        if not self._deadman_requested:
            return False
        physical_state = _windows_left_button_in_foreground(WINDOW_NAME)
        if physical_state is None:
            return True
        if not physical_state:
            # Fail closed and require a new click inside the preview.  This
            # covers a missed LBUTTONUP callback and a window focus change.
            self._deadman_requested = False
            return False
        return True

    @deadman_down.setter
    def deadman_down(self, value: bool) -> None:
        self._deadman_requested = bool(value)

    def show(self, frame_bgr: np.ndarray, lines: Sequence[str]) -> int:
        if not self.enabled:
            return -1
        frame = np.asarray(frame_bgr).copy()
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("preview frame must have shape HxWx3")
        overlay_height = min(frame.shape[0], 34 + 25 * len(lines))
        overlay = frame[:overlay_height].copy()
        overlay[:] = (18, 18, 18)
        cv2.addWeighted(overlay, 0.72, frame[:overlay_height], 0.28, 0.0, overlay)
        frame[:overlay_height] = overlay
        for index, line in enumerate(lines):
            color = (80, 230, 90)
            if "INVALID" in line or "STOP" in line or "LOST" in line:
                color = (70, 90, 255)
            cv2.putText(
                frame,
                str(line),
                (16, 27 + 25 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(WINDOW_NAME, frame)
        if not self._created:
            cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
            self._created = True
        return cv2.waitKey(1) & 0xFF

    def window_open(self) -> bool:
        if not self.enabled or not self._created:
            return True
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1.0
        except cv2.error:
            return False

    def close(self) -> None:
        self.deadman_down = False
        if self.enabled:
            cv2.destroyWindow(WINDOW_NAME)

    def _mouse_callback(
        self,
        event: int,
        _x: int,
        _y: int,
        _flags: int,
        _parameter: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.deadman_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.deadman_down = False


def _windows_left_button_in_foreground(window_name: str) -> bool | None:
    """Poll the real Windows button/focus state; return None off Windows."""

    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        left_button_down = bool(user32.GetAsyncKeyState(0x01) & 0x8000)
        window = user32.FindWindowW(None, str(window_name))
        foreground = user32.GetForegroundWindow()
        return left_button_down and bool(window) and window == foreground
    except (AttributeError, OSError):
        return False
