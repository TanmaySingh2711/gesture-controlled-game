"""The recognizer's decision logic: threshold first, then a vote that can clear but never invent.

The vote, the reset and the ROI geometry are pure logic, so they are tested on an instance built
without its constructor (which insists on CUDA) and run everywhere, including CI. Loading, the
metadata guard and a full prediction need the GPU and are marked accordingly.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from torch import Tensor, nn

from src import gesture_recognizer
from src.gesture_recognizer import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_WINDOW,
    DIRECTIONS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INDEX_TO_DIRECTION,
    ROI_X1,
    ROI_Y1,
    DirectionRecognizer,
    open_camera,
)

gpu = pytest.mark.gpu


def voter(
    window: int = DEFAULT_WINDOW, min_agreement: int = DEFAULT_MIN_AGREEMENT
) -> DirectionRecognizer:
    """A recognizer with only its smoothing state - no model, no CUDA."""
    recognizer = object.__new__(DirectionRecognizer)
    recognizer.window = window
    recognizer.min_agreement = min_agreement
    recognizer.history = deque(maxlen=window)
    recognizer.stable_command = None
    recognizer.previous_stable = None
    return recognizer


# --- smoothing ---------------------------------------------------------------------------
def test_a_direction_needs_three_of_the_last_five_frames() -> None:
    recognizer = voter()
    outputs = [recognizer._smooth(d) for d in ("up", None, "up", "left", "up")]
    assert outputs == [None, None, None, None, "up"]


def test_a_single_confident_wrong_frame_cannot_turn_pacman() -> None:
    recognizer = voter()
    for _ in range(5):
        recognizer._smooth("left")
    assert recognizer._smooth("up") == "left"


def test_rejected_frames_clear_a_command_but_never_create_one() -> None:
    recognizer = voter()
    for _ in range(5):
        recognizer._smooth("down")
    outputs = [recognizer._smooth(None) for _ in range(5)]
    assert outputs == ["down", "down", None, None, None]


@given(st.lists(st.sampled_from([*DIRECTIONS, None]), max_size=60))
def test_a_command_always_holds_the_window_by_itself(frames: list[str | None]) -> None:
    recognizer = voter()
    for index, frame in enumerate(frames):
        command = recognizer._smooth(frame)
        recent = frames[max(0, index + 1 - DEFAULT_WINDOW) : index + 1]
        if command is None:
            assert all(recent.count(d) < DEFAULT_MIN_AGREEMENT for d in DIRECTIONS)
        else:
            assert recent.count(command) >= DEFAULT_MIN_AGREEMENT
        assert len(recognizer.history) <= DEFAULT_WINDOW


def test_reset_forgets_the_window() -> None:
    recognizer = voter()
    for _ in range(3):
        recognizer.stable_command = recognizer.previous_stable = recognizer._smooth("right")
    recognizer.reset()
    assert (len(recognizer.history), recognizer.stable_command, recognizer.previous_stable) == (
        0,
        None,
        None,
    )
    assert recognizer._smooth("right") is None


def test_roi_is_the_frozen_300_pixel_square() -> None:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[ROI_Y1, ROI_X1] = 255
    roi = DirectionRecognizer.crop_roi(frame)
    assert roi.shape == (300, 300, 3)
    assert roi[0, 0, 0] == 255


# --- camera ------------------------------------------------------------------------------
class FakeCapture:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.released = False
        self.settings: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self.opened

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> None:
        self.settings[prop] = value


@pytest.mark.parametrize("opened", [True, False])
def test_open_camera_sets_the_frozen_size_or_releases_on_failure(
    monkeypatch: pytest.MonkeyPatch, opened: bool
) -> None:
    capture = FakeCapture(opened)
    monkeypatch.setattr(gesture_recognizer.cv2, "VideoCapture", lambda *_args: capture)
    if not opened:
        with pytest.raises(RuntimeError, match="Could not open webcam device 3"):
            open_camera(3)
        assert capture.released
        return
    assert open_camera() is capture
    cv2 = gesture_recognizer.cv2
    assert capture.settings == {
        cv2.CAP_PROP_FRAME_WIDTH: FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT: FRAME_HEIGHT,
    }


# --- GPU: construction, loading and prediction -------------------------------------------
@gpu
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 0.0}, "threshold must be in"),
        ({"threshold": 1.5}, "threshold must be in"),
        ({"window": 5, "min_agreement": 6}, "min_agreement 6 must be between"),
        ({"window": 5, "min_agreement": 0}, "min_agreement 0 must be between"),
    ],
)
def test_constructor_rejects_settings_that_cannot_work(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectionRecognizer(**kwargs)


@gpu
def test_metadata_that_does_not_describe_the_direction_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "architecture": "resnet18",
        "task": "endless_runner",
        "num_classes": 4,
        "input_size": [3, 224, 224],
        "class_to_index": {"left": 0, "right": 1, "jump": 2, "neutral": 3},
    }
    monkeypatch.setattr(
        gesture_recognizer,
        "load_direction_checkpoint",
        lambda *_args, **_kwargs: (torch.nn.Identity(), payload),
    )
    with pytest.raises(RuntimeError, match="checkpoint metadata mismatch") as error:
        DirectionRecognizer()
    for problem in ("architecture=resnet18", "task=endless_runner", "input_size", "jump, neutral"):
        assert problem in str(error.value)


class FixedLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits: Tensor = torch.zeros(1, len(DIRECTIONS))

    def forward(self, batch: Tensor) -> Tensor:
        assert batch.shape == (1, 3, 160, 160)
        return self.logits.to(batch.device)


@gpu
def test_prediction_thresholds_then_smooths_and_flags_changes() -> None:
    recognizer = voter()
    recognizer.threshold = 0.9
    recognizer.device = torch.device("cuda")
    recognizer.transform = gesture_recognizer.inference_transform()
    recognizer.model = FixedLogits()
    roi = np.full((300, 300, 3), 128, dtype=np.uint8)

    up = DIRECTIONS.index("up")
    recognizer.model.logits[0, up] = 10.0  # confidence ~0.9999
    first = [recognizer.predict_roi(roi) for _ in range(3)]
    assert [p.raw_direction for p in first] == ["up"] * 3
    assert [p.thresholded_direction for p in first] == ["up"] * 3
    assert [p.stable_command for p in first] == [None, None, "up"]
    assert [p.stable_changed for p in first] == [False, False, True]
    assert all(p.inference_ms > 0.0 for p in first)

    recognizer.model.logits.zero_()
    recognizer.model.logits[0, up] = 0.5  # the most likely class, but far below 0.9
    weak = recognizer.predict_roi(roi)
    assert (weak.raw_direction, weak.thresholded_direction) == ("up", None)
    assert weak.raw_confidence < 0.9
    assert weak.stable_command == "up", "one rejected frame does not clear the vote"
    assert INDEX_TO_DIRECTION[up] == "up"


@gpu
def test_the_frozen_checkpoint_loads_and_predicts_a_full_frame() -> None:
    recognizer = DirectionRecognizer()
    frame = np.random.default_rng(0).integers(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    prediction = recognizer.predict(frame)
    assert prediction.raw_direction in DIRECTIONS
    assert 0.25 <= prediction.raw_confidence <= 1.0
    assert recognizer.metadata["class_to_index"] == {"left": 0, "right": 1, "up": 2, "down": 3}
