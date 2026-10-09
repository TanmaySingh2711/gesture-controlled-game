"""Centring the command window run_game.bat opens, tested against stand-in Win32 calls."""

from __future__ import annotations

import ctypes
import sys
from typing import Any

import pytest

from src.console_window import EXIT_STATUS, Rect, center_console_window, centred


def rect(left: int, top: int, right: int, bottom: int) -> Rect:
    return Rect(left, top, right, bottom)


class FakeUser32:
    """Just enough of user32 to stand in for a console window on a 1536 x 816 work area."""

    def __init__(self, visible: bool = True, frame: Rect | None = None) -> None:
        self.visible = visible
        self.frame = frame or rect(40, 700, 1040, 1220)
        self.moved_to: tuple[int, int] | None = None

    def IsWindowVisible(self, window: int) -> bool:  # noqa: N802 - the Win32 name
        return self.visible

    def GetWindowRect(self, window: int, out: Any) -> bool:  # noqa: N802
        ctypes.memmove(out, ctypes.byref(self.frame), ctypes.sizeof(Rect))
        return True

    def SystemParametersInfoW(self, action: int, _: int, out: Any, __: int) -> bool:  # noqa: N802
        ctypes.memmove(out, ctypes.byref(rect(0, 0, 1536, 816)), ctypes.sizeof(Rect))
        return True

    def SetWindowPos(self, window: int, after: Any, x: int, y: int, *_: int) -> bool:  # noqa: N802
        self.moved_to = (x, y)
        return True


class FakeKernel32:
    def __init__(self, window: int = 42) -> None:
        self.window = window

    def GetConsoleWindow(self) -> int:  # noqa: N802
        return self.window


def test_a_window_is_centred_on_the_work_area() -> None:
    assert centred(rect(0, 0, 1000, 520), rect(0, 0, 1536, 816)) == (268, 148)


def test_a_window_bigger_than_the_screen_keeps_its_top_left_corner_visible() -> None:
    assert centred(rect(0, 0, 2000, 1000), rect(0, 0, 1536, 816)) == (0, 0)


def test_the_console_window_is_moved_to_the_middle() -> None:
    user32 = FakeUser32()
    assert center_console_window(user32, FakeKernel32()) == "moved"
    assert user32.moved_to == (268, 148)


def test_a_process_without_a_console_has_nothing_to_move() -> None:
    user32 = FakeUser32()
    assert center_console_window(user32, FakeKernel32(window=0)) == "no window"
    assert user32.moved_to is None


def test_a_console_hosted_by_windows_terminal_is_reported_not_moved() -> None:
    """run_game.bat reopens itself in a classic console window when it hears this."""
    user32 = FakeUser32(visible=False)
    assert center_console_window(user32, FakeKernel32()) == "hidden"
    assert user32.moved_to is None


def test_each_outcome_has_its_own_exit_status() -> None:
    assert EXIT_STATUS == {"moved": 0, "no window": 1, "hidden": 2}


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows this would move a real window")
def test_away_from_windows_there_is_nothing_to_do() -> None:
    assert center_console_window() == "no window"
