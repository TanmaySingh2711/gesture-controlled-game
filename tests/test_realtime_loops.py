"""The live tools' loops - preview, benchmark, trials, transitions, self-test - without a webcam.

Every loop is driven by a scripted camera, a scripted keyboard and a scripted recognizer, so the
exact frames, key presses and recognition results are known. OpenCV's window calls are replaced
with no-ops; everything else, including the CSV output and the printed summaries, is the real code.
"""

from __future__ import annotations

import csv
import sys
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from src import realtime_gesture
from src.gesture_recognizer import FRAME_HEIGHT, FRAME_WIDTH, ROI_X1, ROI_Y1, Prediction

KEY = {"space": ord(" "), "s": ord("s"), "r": ord("r"), "d": ord("d"), "q": ord("q"), "esc": 27}


def prediction(
    raw: str = "left",
    stable: str | None = None,
    *,
    confidence: float = 0.97,
    changed: bool = False,
) -> Prediction:
    return Prediction(
        raw_direction=raw,
        raw_confidence=confidence,
        thresholded_direction=raw if confidence >= 0.9 else None,
        stable_command=stable,
        stable_changed=changed,
        inference_ms=12.0,
    )


class ScriptedCamera:
    """Returns `frames` (None means a failed read), then keeps returning a blank frame."""

    def __init__(self, frames: list[np.ndarray | None] | None = None) -> None:
        self.frames = deque(frames or [])
        self.released = False
        self.reads = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        frame = self.frames.popleft() if self.frames else blank()
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


class ScriptedRecognizer:
    """Hands out `results` in order, repeating the last one; counts resets."""

    def __init__(self, results: list[Prediction]) -> None:
        self.results = deque(results)
        self.last = results[-1]
        self.threshold = 0.9
        self.window = 5
        self.min_agreement = 3
        self.history: deque[str | None] = deque(["left", None, "up"], maxlen=5)
        self.resets = 0

    def predict(self, _frame: np.ndarray) -> Prediction:
        if self.results:
            self.last = self.results.popleft()
        return self.last

    def reset(self) -> None:
        self.resets += 1


def blank() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def keyboard(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """A list the test fills with key codes; each frame consumes one (then 'no key')."""
    keys: list[int] = []
    monkeypatch.setattr(realtime_gesture.cv2, "waitKey", lambda _delay: keys.pop(0) if keys else -1)
    monkeypatch.setattr(realtime_gesture.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(realtime_gesture.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_gesture.cv2, "getWindowProperty", lambda *_args: 1.0)
    yield keys


def use_camera(monkeypatch: pytest.MonkeyPatch, camera: ScriptedCamera) -> None:
    monkeypatch.setattr(realtime_gesture, "open_camera", lambda: camera)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- overlay and helpers -------------------------------------------------------------------
@pytest.mark.parametrize("show_raw", [False, True])
def test_the_overlay_draws_the_roi_box_in_the_command_colour(show_raw: bool) -> None:
    recognizer: Any = ScriptedRecognizer([prediction()])
    frame = realtime_gesture.draw_overlay(
        blank(), prediction("up", "up"), 30.0, recognizer, show_raw, banner="TRIAL 1/2"
    )
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    # The ROI outline takes the stable command's colour, so the player sees what is in force.
    assert tuple(frame[ROI_Y1 + 150, ROI_X1]) == realtime_gesture.COLOR["up"]


def test_the_overlay_reports_a_rejected_frame() -> None:
    recognizer: Any = ScriptedRecognizer([prediction()])
    frame = realtime_gesture.draw_overlay(
        blank(), prediction("down", None, confidence=0.4), 0.0, recognizer, False
    )
    assert tuple(frame[ROI_Y1 + 150, ROI_X1]) == realtime_gesture.COLOR[None]


def test_closing_the_window_counts_as_quitting(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int]
) -> None:
    monkeypatch.setattr(realtime_gesture.cv2, "getWindowProperty", lambda *_args: 0.0)
    assert realtime_gesture.read_action("window") == "quit"
    monkeypatch.setattr(realtime_gesture.cv2, "getWindowProperty", lambda *_args: 1.0)
    keyboard.extend([KEY["esc"], KEY["space"]])
    assert realtime_gesture.read_action("window") == "quit"
    assert realtime_gesture.read_action("window") == "record"
    assert realtime_gesture.read_action("window") is None


def test_the_frame_rate_meter_averages_recent_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 0.1, 0.2, 0.3])
    monkeypatch.setattr(realtime_gesture.time, "perf_counter", lambda: next(clock))
    meter = realtime_gesture.FrameRate(window=5)
    assert meter.tick() == pytest.approx(10.0)
    assert meter.tick() == pytest.approx(10.0)


# --- preview and benchmark -----------------------------------------------------------------
def test_preview_toggles_debug_resets_and_quits(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    camera = ScriptedCamera([None, blank()])  # one failed read is skipped, not fatal
    use_camera(monkeypatch, camera)
    recognizer: Any = ScriptedRecognizer([prediction("right", "right")])
    keyboard.extend([KEY["d"], KEY["r"], -1, KEY["q"]])

    realtime_gesture.preview(recognizer)

    assert recognizer.resets == 1
    assert camera.released
    assert "frames 4 | mean CNN 12.00 ms" in capsys.readouterr().out


@pytest.mark.parametrize("show", [False, True])
def test_benchmark_measures_every_delivered_frame(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int], show: bool
) -> None:
    camera = ScriptedCamera([blank(), None, blank(), blank()])
    use_camera(monkeypatch, camera)
    recognizer: Any = ScriptedRecognizer([prediction()])

    report = realtime_gesture.benchmark(recognizer, 4, show)

    assert report is not None
    assert report["frames"] == 3  # the failed read is not a processed frame
    assert report["cnn_ms_mean"] == pytest.approx(12.0)
    assert camera.released


# --- guided trials -------------------------------------------------------------------------
def test_trials_record_skip_and_write_a_summary(
    monkeypatch: pytest.MonkeyPatch,
    keyboard: list[int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_camera(monkeypatch, ScriptedCamera([None]))
    recognizer: Any = ScriptedRecognizer(
        [
            prediction("up", "up"),  # trial 1: up -> correct
            prediction("down", "up"),  # trial 2: raw error rescued by smoothing
            prediction("left", "left", confidence=0.95),  # trial 3: no_hand -> false command
        ]
    )
    # The first frame is a failed read; then record, record, skip one, record, done.
    keyboard.extend([KEY["space"], KEY["space"], KEY["s"], KEY["space"]])
    out = tmp_path / "trials.csv"

    records = realtime_gesture.trials(recognizer, ["up", "no_hand"], 2, str(out))

    assert records is not None
    rows = read_csv(out)
    assert [row["expected_class"] for row in rows] == ["up", "up", "no_hand"]
    assert [row["correct"] for row in rows] == ["yes", "yes", "no"]
    assert recognizer.resets == 4  # after every trial, recorded or skipped
    output = capsys.readouterr().out
    assert "stable false commands: 1/1" in output
    assert "raw errors corrected by smoothing: 1" in output


def test_trials_quit_before_recording_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int], tmp_path: Path
) -> None:
    use_camera(monkeypatch, ScriptedCamera())
    keyboard.append(KEY["q"])
    out = tmp_path / "nothing.csv"
    recognizer: Any = ScriptedRecognizer([prediction()])
    assert realtime_gesture.trials(recognizer, ["left"], 1, str(out)) is None
    assert not out.exists()


# --- transitions ---------------------------------------------------------------------------
def test_transitions_time_each_move_and_count_spurious_commands(
    monkeypatch: pytest.MonkeyPatch,
    keyboard: list[int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(realtime_gesture, "TRANSITION_PAIRS", [("left", "right"), ("up", "down")])
    use_camera(monkeypatch, ScriptedCamera([None]))
    recognizer: Any = ScriptedRecognizer(
        [
            prediction("left", "left"),  # holding FROM; SPACE starts the timer
            prediction("up", "up", changed=True),  # a wrong command on the way: spurious
            prediction("right", "right", changed=True),  # target reached
            prediction("up", "up"),  # second pair: skipped
            prediction("up", "up"),
        ]
    )
    keyboard.extend([KEY["space"], -1, -1, KEY["s"]])
    out = tmp_path / "transitions.csv"

    records = realtime_gesture.transitions(recognizer, 1, str(out))

    assert records is not None
    (row,) = read_csv(out)
    assert row["transition"] == "left->right"
    assert row["spurious_commands"] == "up"
    output = capsys.readouterr().out
    assert "spurious stable cmds  : 1/1 transitions" in output
    assert "RECOGNIZER LAG" in output


def test_transitions_quit_without_records(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int], tmp_path: Path
) -> None:
    use_camera(monkeypatch, ScriptedCamera())
    keyboard.append(KEY["q"])
    recognizer: Any = ScriptedRecognizer([prediction()])
    assert realtime_gesture.transitions(recognizer, 1, str(tmp_path / "t.csv")) is None


# --- self-test against the real frozen model on the CPU ------------------------------------
@pytest.fixture(scope="module")
def cpu_recognizer() -> Any:
    from src.gesture_recognizer import DirectionRecognizer

    return DirectionRecognizer(device="cpu")


def sample_project(root: Path) -> None:
    """A project root holding the one image the parity check reads."""
    folder = root / "dataset" / "left"
    folder.mkdir(parents=True)
    image = np.random.default_rng(3).integers(0, 256, (180, 180, 3), np.uint8)
    cv2.imwrite(str(folder / "left_00003.jpg"), image)


def test_the_selftest_passes_on_the_cpu_with_a_scripted_camera(
    monkeypatch: pytest.MonkeyPatch,
    keyboard: list[int],
    tmp_path: Path,
    cpu_recognizer: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_project(tmp_path)
    monkeypatch.setattr(realtime_gesture, "PROJECT_ROOT", str(tmp_path))
    camera = ScriptedCamera([np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, np.uint8)])
    use_camera(monkeypatch, camera)

    assert realtime_gesture.selftest(cpu_recognizer)
    assert camera.released
    output = capsys.readouterr().out
    assert "RESULT: PASS" in output
    assert "[PASS] model device" in output
    assert "[PASS] preprocessing" in output


def test_the_selftest_fails_when_the_camera_returns_nothing(
    monkeypatch: pytest.MonkeyPatch,
    keyboard: list[int],
    tmp_path: Path,
    cpu_recognizer: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(realtime_gesture, "PROJECT_ROOT", str(tmp_path))  # no sample image either
    use_camera(monkeypatch, ScriptedCamera([None]))

    assert not realtime_gesture.selftest(cpu_recognizer)
    output = capsys.readouterr().out
    assert "[FAIL] preprocessing parity" in output
    assert "[FAIL] webcam" in output


# --- command line --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arguments", "called"),
    [
        (["--selftest"], "selftest"),
        (["--benchmark", "5"], "benchmark"),
        (["--transitions"], "transitions"),
        (["--idle"], "trials"),
        (["--trials", "--classes", "up,no_hand"], "trials"),
        ([], "preview"),
    ],
)
def test_main_dispatches_each_mode_on_the_requested_device(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], called: str
) -> None:
    seen: dict[str, Any] = {}

    class Recognizer(ScriptedRecognizer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__([prediction()])
            seen["kwargs"] = kwargs
            self.metadata = {"task": "pacman_direction_v1"}
            self.model = torch.nn.Linear(1, 1)  # main only reads its parameters' device

    monkeypatch.setattr(realtime_gesture, "DirectionRecognizer", Recognizer)
    for name in ("selftest", "benchmark", "transitions", "trials", "preview"):
        monkeypatch.setattr(
            realtime_gesture, name, lambda *args, _name=name: seen.setdefault("called", _name)
        )
    monkeypatch.setattr(sys, "argv", ["realtime_gesture.py", "--device", "cpu", *arguments])

    status = realtime_gesture.main()

    assert seen["called"] == called
    assert seen["kwargs"]["device"] == "cpu"
    assert status == 0


def test_main_rejects_unknown_trial_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    class Recognizer(ScriptedRecognizer):
        def __init__(self, **_kwargs: Any) -> None:
            super().__init__([prediction()])
            self.metadata = {"task": "pacman_direction_v1"}
            self.model = torch.nn.Linear(1, 1)

    monkeypatch.setattr(realtime_gesture, "DirectionRecognizer", Recognizer)
    monkeypatch.setattr(sys, "argv", ["realtime_gesture.py", "--trials", "--classes", "jump"])
    with pytest.raises(SystemExit):
        realtime_gesture.main()


# --- recording sessions with several people -------------------------------------------------
def test_session_files_are_new_named_and_never_the_p6_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(realtime_gesture, "SESSIONS_DIR", str(tmp_path))
    path = realtime_gesture.session_path("trials", "Riya S.", "dim room / evening")
    assert Path(path).parent == tmp_path
    assert Path(path).name.endswith("_riya-s_dim-room-evening_trials.csv")
    assert path not in (
        realtime_gesture.TRIALS_CSV,
        realtime_gesture.IDLE_CSV,
        realtime_gesture.TRANSITIONS_CSV,
    )
    assert realtime_gesture.slug("  ") == "unnamed"


def test_trial_rows_carry_the_participant_and_condition(
    monkeypatch: pytest.MonkeyPatch, keyboard: list[int], tmp_path: Path
) -> None:
    use_camera(monkeypatch, ScriptedCamera())
    recognizer: Any = ScriptedRecognizer([prediction("up", "up")])
    keyboard.append(KEY["space"])
    out = tmp_path / "session.csv"
    labels = {"participant": "riya", "condition": "dim-room"}
    realtime_gesture.trials(recognizer, ["up"], 1, str(out), labels)
    (row,) = read_csv(out)
    assert (row["participant"], row["condition"], row["correct"]) == ("riya", "dim-room", "yes")


def test_main_records_sessions_away_from_the_p6_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: dict[str, Any] = {}

    class Recognizer(ScriptedRecognizer):
        def __init__(self, **_kwargs: Any) -> None:
            super().__init__([prediction()])
            self.metadata = {"task": "pacman_direction_v1"}
            self.model = torch.nn.Linear(1, 1)

    def fake_trials(_recognizer: Any, _categories: Any, _count: int, out: str, labels: Any) -> None:
        written.update(out=out, labels=labels)

    monkeypatch.setattr(realtime_gesture, "DirectionRecognizer", Recognizer)
    monkeypatch.setattr(realtime_gesture, "trials", fake_trials)
    monkeypatch.setattr(realtime_gesture, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["realtime_gesture.py", "--idle", "--participant", "ravi", "--condition", "lamp"],
    )
    assert realtime_gesture.main() == 0
    assert written["labels"] == {"participant": "ravi", "condition": "lamp"}
    assert Path(written["out"]).parent == tmp_path
    assert written["out"].endswith("_ravi_lamp_idle.csv")
