"""The webcam capture tool's decisions, tested without a webcam or a window."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("cv2")

from src.collect_dataset import (
    CAPTURE_INTERVAL,
    CLASS_KEYS,
    TARGET_PER_CLASS,
    CaptureSession,
    auto_capture,
    handle_key,
)


class FakeSaver:
    """Stands in for `save_roi`: records calls, and can be told to fail."""

    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed
        self.calls: list[tuple[str, int]] = []

    def __call__(self, label: str, roi: Any, index: int) -> int | None:
        self.calls.append((label, index))
        return index + 1 if self.succeed else None


def session(**counts: int) -> CaptureSession:
    filled = {label: counts.get(label, 0) for label in ("left", "right", "up", "down")}
    return CaptureSession(counts=filled, next_index=dict(filled))


def test_number_keys_select_the_four_direction_classes() -> None:
    assert sorted(CLASS_KEYS.values()) == ["down", "left", "right", "up"]
    state = session()
    state.capturing = True
    assert handle_key(state, ord("3"), roi=None)
    assert state.label == "up"
    assert not state.capturing, "switching class must pause automatic capture"


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_quit_keys_end_the_loop(key: int) -> None:
    assert handle_key(session(), key, roi=None) is False


def test_space_toggles_capture_unless_the_class_is_full() -> None:
    state = session()
    handle_key(state, ord(" "), roi=None)
    assert state.capturing
    handle_key(state, ord(" "), roi=None)
    assert not state.capturing

    full = session(left=TARGET_PER_CLASS)
    handle_key(full, ord(" "), roi=None)
    assert not full.capturing


def test_single_capture_saves_and_counts() -> None:
    saver = FakeSaver()
    state = session(left=5)
    handle_key(state, ord("c"), roi=object(), save=saver)
    assert saver.calls == [("left", 5)]
    assert state.counts["left"] == 6 and state.next_index["left"] == 6


def test_a_failed_save_changes_nothing() -> None:
    state = session(left=5)
    handle_key(state, ord("c"), roi=object(), save=FakeSaver(succeed=False))
    assert state.counts["left"] == 5 and state.next_index["left"] == 5


def test_auto_capture_respects_the_interval() -> None:
    saver = FakeSaver()
    state = session()
    state.capturing = True
    auto_capture(state, object(), now=100.0, save=saver)
    auto_capture(state, object(), now=100.0 + CAPTURE_INTERVAL / 2, save=saver)
    auto_capture(state, object(), now=100.0 + CAPTURE_INTERVAL, save=saver)
    assert len(saver.calls) == 2


def test_auto_capture_does_nothing_when_paused() -> None:
    saver = FakeSaver()
    auto_capture(session(), object(), now=100.0, save=saver)
    assert saver.calls == []


def test_auto_capture_stops_itself_at_the_target() -> None:
    state = session(left=TARGET_PER_CLASS - 1)
    state.capturing = True
    auto_capture(state, object(), now=100.0, save=FakeSaver())
    assert state.counts["left"] == TARGET_PER_CLASS
    assert not state.capturing
