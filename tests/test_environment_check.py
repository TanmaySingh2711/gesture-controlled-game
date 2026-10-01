"""The environment check's verdicts, in process - especially the ways it must FAIL.

tests/test_self_tests.py runs the script as a player would. These drive each check directly with
fakes, so every failure message a player could see is exercised on any machine.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from src import environment_check as env


@pytest.fixture(autouse=True)
def fresh_results(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, bool, str]]]:
    results: list[tuple[str, bool, str]] = []
    monkeypatch.setattr(env, "results", results)
    monkeypatch.setattr(env, "warnings", [])
    yield results


def verdicts(results: list[tuple[str, bool, str]]) -> dict[str, bool]:
    return {name: passed for name, passed, _ in results}


@pytest.mark.parametrize(("version", "ok"), [((3, 12, 10), True), ((3, 11, 9), False)])
def test_only_python_312_passes(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any], version: tuple[int, ...], ok: bool
) -> None:
    info = types.SimpleNamespace(major=version[0], minor=version[1], micro=version[2])
    monkeypatch.setattr(sys, "version_info", info)
    env.check_python()
    assert verdicts(fresh_results) == {"Python": ok}


def test_a_missing_library_fails_with_the_import_error(fresh_results: list[Any]) -> None:
    assert env.check_import("Nothing", "a_module_that_does_not_exist") is None
    (name, passed, detail) = fresh_results[0]
    assert (name, passed) == ("Nothing", False)
    assert "import failed" in detail


def test_no_pytorch_means_no_cuda_check(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    monkeypatch.setattr(env, "check_import", lambda *_args: None)
    env.check_torch_and_cuda()
    assert verdicts(fresh_results) == {"CUDA available": False}


def test_missing_cuda_is_information_not_failure(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    env.check_torch_and_cuda()
    assert all(verdicts(fresh_results).values())
    assert "gesture recognition will use the CPU" in capsys.readouterr().out


@pytest.mark.gpu
def test_a_working_gpu_passes_every_cuda_check(fresh_results: list[Any]) -> None:
    env.check_torch_and_cuda()
    checks = verdicts(fresh_results)
    assert checks["CUDA available"] and checks["GPU"] and checks["CUDA computation"]


@pytest.mark.gpu
def test_a_gpu_that_cannot_compute_fails(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    torch = pytest.importorskip("torch")

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(torch, "randn", broken)
    env.check_torch_and_cuda()
    assert verdicts(fresh_results)["CUDA computation"] is False


def test_the_gesture_model_check_loads_and_times_the_real_model(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    pytest.importorskip("torch")
    env.check_gesture_model()
    (name, passed, detail) = fresh_results[0]
    assert (name, passed) == ("Gesture model", True)
    assert "checksum verified" in detail


def test_a_model_slower_than_the_webcam_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """A slow CPU still plays: the game keeps 60 FPS, gestures are just recognised less often."""
    pytest.importorskip("torch")
    monkeypatch.setattr(env, "FRAME_BUDGET_MS", 0.0)
    env.check_gesture_model()
    assert verdicts(fresh_results) == {"Gesture model": True}
    assert [name for name, _ in env.warnings] == ["Gesture speed"]
    assert "gestures will be recognised a little later" in capsys.readouterr().out


def test_a_model_too_slow_to_steer_fails(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(env, "FRAME_BUDGET_MS", 0.0)
    monkeypatch.setattr(env, "USABLE_LIMIT_MS", 0.0)
    env.check_gesture_model()
    (_, passed, detail) = fresh_results[0]
    assert passed is False
    assert "too slow to steer the game (needs < 0 ms)" in detail


def test_main_passes_with_warnings_listed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("check_python", "check_torch_and_cuda", "check_pygame", "check_webcam"):
        monkeypatch.setattr(env, name, lambda: None)
    monkeypatch.setattr(env, "check_import", lambda *_args: None)

    def slow_but_working() -> None:
        env.record("Gesture model", True, "runs")
        env.warn("Gesture speed", "slow")

    monkeypatch.setattr(env, "check_gesture_model", slow_but_working)
    monkeypatch.setattr(sys, "argv", ["environment_check.py"])
    assert env.main() == 0
    assert "RESULT: PASS (all 1 checks passed, 1 warning: Gesture speed)" in capsys.readouterr().out


def test_a_model_that_cannot_load_fails(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    pytest.importorskip("torch")
    from src import gesture_recognizer

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("best_direction_model.pt is not the frozen checkpoint")

    monkeypatch.setattr(gesture_recognizer, "DirectionRecognizer", refuse)
    env.check_gesture_model()
    (_, passed, detail) = fresh_results[0]
    assert passed is False
    assert "not the frozen checkpoint" in detail


def test_pygame_that_cannot_start_fails(
    monkeypatch: pytest.MonkeyPatch, fresh_results: list[Any]
) -> None:
    def init() -> None:
        raise RuntimeError("no display")

    monkeypatch.setattr(env, "check_import", lambda *_args: types.SimpleNamespace(init=init))
    env.check_pygame()
    assert verdicts(fresh_results) == {"Pygame init": False}


class FakeCapture:
    def __init__(self, opened: bool, frame: Any) -> None:
        self.opened, self.frame, self.released = opened, frame, False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self.opened

    def read(self) -> tuple[bool, Any]:
        return self.frame is not None, self.frame

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize(
    ("opened", "frame", "passed", "detail"),
    [
        (True, np.full((480, 640, 3), 110, np.uint8), True, "captured a 640x480 frame"),
        (True, np.zeros((480, 640, 3), np.uint8), False, "its picture is black"),
        (
            True,
            np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8),
            False,
            "its picture is static",
        ),
        (True, None, False, "no frame was returned"),
        (False, None, False, "could not open the default webcam"),
    ],
)
def test_the_webcam_check(
    monkeypatch: pytest.MonkeyPatch,
    fresh_results: list[Any],
    opened: bool,
    frame: Any,
    passed: bool,
    detail: str,
) -> None:
    cv2 = pytest.importorskip("cv2")
    capture = FakeCapture(opened, frame)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    env.check_webcam()
    (_, ok, text) = fresh_results[0]
    assert ok is passed
    assert detail in text
    assert capture.released


def test_main_fails_when_any_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("check_python", "check_torch_and_cuda", "check_gesture_model", "check_pygame"):
        monkeypatch.setattr(env, name, lambda: None)
    monkeypatch.setattr(env, "check_import", lambda *_args: None)
    monkeypatch.setattr(env, "check_webcam", lambda: env.record("Webcam", False, "unplugged"))
    monkeypatch.setattr(sys, "argv", ["environment_check.py"])
    assert env.main() == 1
    assert "RESULT: FAIL (1 of 1 checks failed: Webcam)" in capsys.readouterr().out
