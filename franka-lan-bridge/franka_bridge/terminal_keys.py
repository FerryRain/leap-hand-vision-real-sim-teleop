"""Dependency-free, non-blocking single-key input for Windows and Linux terminals."""

from __future__ import annotations

import asyncio
import os
import select
import sys
from types import ModuleType
from typing import TextIO


class TerminalKeys:
    """Put a terminal in single-key mode and restore it on exit."""

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
        """Return one key without blocking, or ``None`` when no key is ready."""

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

    async def wait(self, poll_period_s: float = 0.02) -> str:
        while True:
            key = self.poll()
            if key is not None:
                return key
            await asyncio.sleep(poll_period_s)
