from __future__ import annotations

from unittest import mock

from dp_collector.ui import PreviewUI


def test_windows_deadman_fails_closed_and_requires_a_new_click() -> None:
    ui = PreviewUI(enabled=True)
    ui.deadman_down = True

    with mock.patch(
        "dp_collector.ui._windows_left_button_in_foreground",
        return_value=False,
    ):
        assert ui.deadman_down is False

    # A stale callback state cannot reactivate when focus/button later returns.
    with mock.patch(
        "dp_collector.ui._windows_left_button_in_foreground",
        return_value=True,
    ):
        assert ui.deadman_down is False
        ui.deadman_down = True
        assert ui.deadman_down is True


def test_non_windows_deadman_uses_opencv_callback_state() -> None:
    ui = PreviewUI(enabled=True)
    ui.deadman_down = True
    with mock.patch(
        "dp_collector.ui._windows_left_button_in_foreground",
        return_value=None,
    ):
        assert ui.deadman_down is True
        ui.deadman_down = False
        assert ui.deadman_down is False
