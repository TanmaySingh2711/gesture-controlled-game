"""Move the command window that run_game.bat opens to the middle of the screen (Windows only).

`run_game.bat` runs this before the game, so the window never lands somewhere random. It uses
nothing outside the standard library, so it runs in a few milliseconds, long before the game
itself has finished loading.

Exit status, which run_game.bat acts on:
    0  the window was moved to the middle of the screen
    1  nothing to move: not Windows, or no console window at all
    2  the console is hosted invisibly by a terminal such as Windows Terminal, which places its
       own window; run_game.bat then reopens itself in the classic console window, which can be
       moved
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any, Literal

SPI_GETWORKAREA = 0x0030
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004

Outcome = Literal["moved", "no window", "hidden"]
EXIT_STATUS: dict[Outcome, int] = {"moved": 0, "no window": 1, "hidden": 2}


class Rect(ctypes.Structure):
    """The Win32 RECT, defined here because ctypes.wintypes is Windows-only on some Pythons."""

    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def centred(frame: Rect, work: Rect) -> tuple[int, int]:
    """Top-left corner that puts `frame` in the middle of `work`, never off its top-left edge."""
    width, height = frame.right - frame.left, frame.bottom - frame.top
    x = work.left + max(0, (work.right - work.left - width) // 2)
    y = work.top + max(0, (work.bottom - work.top - height) // 2)
    return x, y


def center_console_window(user32: Any = None, kernel32: Any = None) -> Outcome:
    """Centre this process's console window on the work area (the screen minus the taskbar)."""
    if user32 is None or kernel32 is None:
        if sys.platform != "win32":
            return "no window"
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    window = kernel32.GetConsoleWindow()
    if not window:
        return "no window"
    if not user32.IsWindowVisible(window):
        return "hidden"
    frame, work = Rect(), Rect()
    if not user32.GetWindowRect(window, ctypes.byref(frame)):
        return "no window"
    if not user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(work), 0):
        return "no window"
    x, y = centred(frame, work)
    moved = user32.SetWindowPos(window, None, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER)
    return "moved" if moved else "no window"


if __name__ == "__main__":
    sys.exit(EXIT_STATUS[center_console_window()])
