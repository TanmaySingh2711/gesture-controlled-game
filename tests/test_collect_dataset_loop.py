"""The optional webcam capture tool, driven end to end with a scripted camera and keyboard.

Everything is written to a temporary dataset folder; the real `dataset/` is never touched.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src import collect_dataset as cd


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cd, "DATASET_DIR", str(tmp_path / "dataset"))
    return tmp_path / "dataset"


def frame(value: int = 100) -> np.ndarray:
    return np.full((cd.FRAME_HEIGHT, cd.FRAME_WIDTH, 3), value, np.uint8)


def roi() -> np.ndarray:
    return np.zeros((300, 300, 3), np.uint8)


class ScriptedCamera:
    def __init__(self, frames: list[np.ndarray | None], opened: bool = True) -> None:
        self.frames = deque(frames)
        self.opened = opened
        self.released = False
        self.settings: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self.opened

    def set(self, prop: int, value: float) -> None:
        self.settings[prop] = value

    def read(self) -> tuple[bool, np.ndarray | None]:
        item = self.frames.popleft() if self.frames else frame()
        return item is not None, item

    def release(self) -> None:
        self.released = True


# --- saving --------------------------------------------------------------------------------
def test_saves_never_overwrite_and_are_counted(dataset: Path) -> None:
    (dataset / "up").mkdir(parents=True)
    cv2.imwrite(str(dataset / "up" / "up_00000.jpg"), roi())
    assert cd.count_existing("up") == 1
    assert cd.count_existing("down") == 0  # no folder yet

    next_index = cd.save_roi("up", roi(), 0)

    assert next_index == 2  # up_00000 existed, so the image went to up_00001
    assert (dataset / "up" / "up_00001.jpg").exists()
    assert cd.count_existing("up") == 2


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (None, "empty ROI"),
        (np.zeros((0, 0, 3), np.uint8), "empty ROI"),
        (np.zeros((200, 300, 3), np.uint8), "unexpected ROI size"),
    ],
)
def test_bad_regions_are_skipped_not_saved(
    dataset: Path, image: Any, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cd.save_roi("left", image, 0) is None
    assert message in capsys.readouterr().out


def test_a_failed_write_is_reported(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cd.cv2, "imwrite", lambda *_args: False)
    assert cd.save_roi("left", roi(), 0) is None
    assert "failed to write" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("done", "capturing"), [(0, False), (5, True), (cd.TARGET_PER_CLASS, True)]
)
def test_the_overlay_marks_the_roi_in_the_state_colour(done: int, capturing: bool) -> None:
    display = frame(0)
    counts = dict.fromkeys(cd.CLASSES, 0) | {"right": done}
    cd.draw_overlay(display, "right", counts, capturing)
    expected = (0, 220, 0) if capturing else (0, 200, 255)
    assert tuple(display[cd.ROI_Y1 + 150, cd.ROI_X1]) == expected


# --- camera and self-test ------------------------------------------------------------------
def test_open_camera_sets_the_capture_size_or_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = ScriptedCamera([])
    monkeypatch.setattr(cd.cv2, "VideoCapture", lambda *_args: camera)
    assert cd.open_camera() is camera
    assert camera.settings[cv2.CAP_PROP_FRAME_WIDTH] == cd.FRAME_WIDTH

    monkeypatch.setattr(cd.cv2, "VideoCapture", lambda *_args: ScriptedCamera([], opened=False))
    with pytest.raises(RuntimeError, match="Could not open webcam"):
        cd.open_camera()


def test_the_selftest_checks_flip_roi_and_folders(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (dataset / "left").mkdir(parents=True)
    camera = ScriptedCamera([frame()])
    monkeypatch.setattr(cd, "open_camera", lambda: camera)
    assert cd.selftest() == 0
    output = capsys.readouterr().out
    assert "[PASS] ROI crop                300x300" in output
    assert "[PASS] dataset/left" in output
    assert "[FAIL] dataset/right    MISSING" in output
    assert camera.released


@pytest.mark.parametrize(
    ("image", "message"),
    [(None, "webcam returned no frame"), (np.zeros((200, 320, 3), np.uint8), "ROI crop is")],
)
def test_the_selftest_fails_on_a_bad_frame(
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image: Any,
    message: str,
) -> None:
    monkeypatch.setattr(cd, "open_camera", lambda: ScriptedCamera([image]))
    assert cd.selftest() == 1
    assert message in capsys.readouterr().out


# --- the capture loop ----------------------------------------------------------------------
def test_the_capture_loop_selects_captures_and_quits(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    camera = ScriptedCamera([None, frame(), frame(), frame(), frame(), frame()])
    keys = deque([ord("2"), ord(" "), -1, ord("c"), ord("q")])
    clock = iter([10.0, 10.1, 10.5, 11.0, 11.5, 12.0, 12.5])
    monkeypatch.setattr(cd, "open_camera", lambda: camera)
    monkeypatch.setattr(cd.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cd.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cd.cv2, "waitKey", lambda _delay: keys.popleft() if keys else ord("q"))
    monkeypatch.setattr(cd.cv2, "getWindowProperty", lambda *_args: 1.0)
    monkeypatch.setattr(cd.time, "time", lambda: next(clock))
    monkeypatch.setattr(sys, "argv", ["collect_dataset.py"])

    assert cd.main() == 0

    saved = sorted(p.name for p in (dataset / "right").glob("*.jpg"))
    # SPACE on frame 2 starts capture; frames 3, 4 and 5 each save automatically (their clock
    # readings are at least 0.25 s apart), and C on frame 4 saves one more: four images.
    assert len(saved) == 4
    assert not list((dataset / "left").glob("*.jpg"))
    output = capsys.readouterr().out
    assert "[warn] dropped frame from webcam" in output
    assert "Total: 4" in output
    assert camera.released


def test_closing_the_window_ends_the_loop(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    camera = ScriptedCamera([frame()])
    monkeypatch.setattr(cd, "open_camera", lambda: camera)
    monkeypatch.setattr(cd.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cd.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cd.cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cd.cv2, "getWindowProperty", lambda *_args: 0.0)
    monkeypatch.setattr(sys, "argv", ["collect_dataset.py"])
    assert cd.main() == 0
    assert camera.released


def test_main_runs_the_selftest_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "selftest", lambda: 7)
    monkeypatch.setattr(sys, "argv", ["collect_dataset.py", "--selftest"])
    assert cd.main() == 7
