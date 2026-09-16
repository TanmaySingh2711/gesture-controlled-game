"""The live-trial measurement logic, tested without a webcam.

`TransitionTimer` produced the project's headline latency figure (143 ms mean recognizer lag), so
the separation it makes - the human's hand movement versus the recognizer's own decision time -
is checked here against hand-worked numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from src.realtime_gesture import (
    KEY_ACTIONS,
    NO_COMMAND_CATEGORIES,
    TransitionTimer,
    trial_record,
)


@dataclass
class Result:
    raw_direction: str
    stable_command: str | None
    stable_changed: bool = False
    raw_confidence: float = 0.95
    thresholded_direction: str | None = None
    inference_ms: float = 15.0


@dataclass
class Settings:
    threshold: float = 0.9
    window: int = 5
    min_agreement: int = 3


def test_timer_separates_hand_movement_from_recognizer_lag() -> None:
    timer = TransitionTimer("left", "right", started=10.0)
    timer.observe(Result("left", "left"), now=10.10)  # the hand still shows the old gesture
    timer.observe(Result("right", "left"), now=10.30)  # the CNN first sees the target
    timer.observe(Result("right", None, stable_changed=True), now=10.35)
    timer.observe(Result("right", "right", stable_changed=True), now=10.45)

    assert timer.done
    record = timer.record(10.45, Settings())
    assert record["ms_to_stable"] == 450.0  # includes the 300 ms of hand movement
    assert record["recognizer_lag_ms"] == 150.0  # only the recognizer's own decision
    assert record["recognizer_lag_frames"] == 3
    assert record["spurious_count"] == 0
    assert record["spurious_commands"] == "none"


def test_a_wrong_stable_command_on_the_way_is_recorded_as_spurious() -> None:
    timer = TransitionTimer("up", "down", started=0.0)
    timer.observe(Result("left", "left", stable_changed=True), now=0.1)
    timer.observe(Result("down", "down", stable_changed=True), now=0.3)
    record = timer.record(0.3, Settings())
    assert record["spurious_commands"] == "left"
    assert record["spurious_count"] == 1


def test_holding_the_source_gesture_is_not_spurious() -> None:
    timer = TransitionTimer("up", "down", started=0.0)
    timer.observe(Result("up", "up", stable_changed=True), now=0.1)
    assert timer.spurious == []
    assert not timer.done


def test_lag_is_unknown_when_the_raw_class_never_matched_the_target() -> None:
    timer = TransitionTimer("up", "down", started=0.0)
    timer.done = True
    record = timer.record(0.2, Settings())
    assert record["recognizer_lag_frames"] == 0
    assert math.isnan(record["recognizer_lag_ms"])


@pytest.mark.parametrize("category", NO_COMMAND_CATEGORIES)
def test_no_command_trials_are_correct_only_when_nothing_is_issued(category: str) -> None:
    quiet = trial_record(0, category, Result("left", None), Settings())
    noisy = trial_record(1, category, Result("left", "left"), Settings())
    assert quiet["correct"] == "yes"
    assert quiet["expected_command"] == "NO COMMAND"
    assert noisy["correct"] == "no"
    assert noisy["stable_command"] == "LEFT"


def test_direction_trials_score_the_stable_command_not_the_raw_frame() -> None:
    rescued = trial_record(0, "up", Result("down", "up"), Settings())
    assert rescued["correct"] == "yes", "smoothing rescued a bad raw frame"
    assert rescued["raw_prediction"] == "down"
    assert rescued["smoothing"] == "3-of-5"
    assert rescued["trial_number"] == 1


def test_key_actions_cover_the_documented_controls() -> None:
    assert set(KEY_ACTIONS.values()) == {"skip", "record", "reset", "debug"}
    assert KEY_ACTIONS[ord(" ")] == "record"
