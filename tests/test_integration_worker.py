"""The real recognition worker loop, driven by scripted fake cameras - no webcam, no GPU.

The worker's camera, recognizer and mirror are injectable, so these tests run the production
`RecognitionWorker.run()` itself: the capture loop, reconnection, reset requests, shutdown and
error reporting. Nothing here imports torch or OpenCV.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from game.engine import PLAYING, Game
from src.game_integration import (
    CAMERA_FAILURE_LIMIT,
    LATENCY_HISTORY,
    RECONNECT_ATTEMPTS,
    GestureController,
    RecognitionWorker,
    SharedState,
    Snapshot,
    percentile,
)


@dataclass
class FakePrediction:
    raw_direction: str
    raw_confidence: float
    thresholded_direction: str | None
    stable_command: str | None
    stable_changed: bool
    inference_ms: float


class ScriptedCamera:
    """Returns frames or empty reads from a script, then a default once the script runs out."""

    def __init__(self, script: tuple[bool, ...] = (), *, default: bool = True) -> None:
        self.script = list(script)
        self.default = default
        self.reads = 0
        self.released = 0

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        ok = self.script.pop(0) if self.script else self.default
        time.sleep(0.0005)
        return (True, np.zeros((480, 640, 3), dtype=np.uint8)) if ok else (False, None)

    def release(self) -> None:
        self.released += 1


class FakeRecognizer:
    def __init__(self, command: str | None = "up") -> None:
        self.command = command
        self.resets = 0

    def predict(self, mirrored_frame: Any) -> FakePrediction:
        return FakePrediction("up", 0.99, self.command, self.command, False, 4.0)

    def crop_roi(self, frame: Any) -> Any:
        return frame[90:390, 300:600]

    def reset(self) -> None:
        self.resets += 1


def cameras(*items: ScriptedCamera | Exception) -> Callable[[], ScriptedCamera]:
    """A camera factory handing out the given cameras (or raising) in order."""
    queue = list(items)

    def factory() -> ScriptedCamera:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    return factory


def make_worker(
    state: SharedState,
    factory: Callable[[], ScriptedCamera],
    recognizer: FakeRecognizer | None = None,
    **kwargs: Any,
) -> RecognitionWorker:
    fake = recognizer or FakeRecognizer()
    return RecognitionWorker(
        state,
        camera_factory=factory,
        recognizer_factory=lambda: fake,
        mirror=lambda frame: frame[:, ::-1],
        **kwargs,
    )


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# --- the normal loop -------------------------------------------------------------------------
def test_worker_publishes_the_latest_result_and_stops_cleanly() -> None:
    state = SharedState()
    camera = ScriptedCamera()
    worker = make_worker(state, cameras(camera), FakeRecognizer("left"))
    controller = GestureController(state, worker)
    controller.start()
    assert wait_for(lambda: state.read().sequence >= 5)
    live = state.read()
    assert controller.stop()

    assert live.stable_command == "left"
    assert live.status == "ready" and live.camera_ok
    assert 0 < live.captured_at <= live.timestamp
    assert live.preview.shape == (300, 300, 3)
    assert camera.released == 1
    assert state.read().status == "stopped"
    assert not worker.failed


def test_reset_request_reaches_the_recognizer() -> None:
    state = SharedState()
    recognizer = FakeRecognizer()
    worker = make_worker(state, cameras(ScriptedCamera()), recognizer)
    worker.start()
    assert wait_for(lambda: state.read().sequence >= 1)
    worker.request_reset()
    assert wait_for(lambda: recognizer.resets >= 1)
    worker.stop()
    worker.join(2.0)


def test_worker_output_steers_the_game_through_the_seam() -> None:
    state = SharedState()
    worker = make_worker(state, cameras(ScriptedCamera()), FakeRecognizer("down"))
    controller = GestureController(state, worker)
    game = Game(headless=True)
    game.state = PLAYING
    controller.start()
    assert wait_for(lambda: state.read().sequence >= 2)
    assert controller.apply_to(game) == "down"
    assert game.controls.pending == "down"
    assert controller.stop()


# --- faults ----------------------------------------------------------------------------------
def test_worker_reconnects_after_the_camera_stops_delivering() -> None:
    state = SharedState()
    dropping = ScriptedCamera((True,) * 3 + (False,) * CAMERA_FAILURE_LIMIT, default=False)
    healthy = ScriptedCamera()
    recognizer = FakeRecognizer()
    worker = make_worker(state, cameras(dropping, healthy), recognizer, reconnect_delay=0.01)
    worker.start()
    assert wait_for(lambda: healthy.reads > 5 and state.read().sequence > 3)
    worker.stop()
    worker.join(2.0)

    assert worker.reconnects == 1
    assert dropping.released == 1 and healthy.released == 1
    assert recognizer.resets >= 1, "frames from before the outage must not keep voting"
    assert not worker.failed
    assert state.read().status == "stopped"


def test_worker_gives_up_after_its_reconnect_budget() -> None:
    state = SharedState()
    dead = ScriptedCamera(default=False)
    worker = make_worker(state, cameras(dead, RuntimeError("no device")), reconnect_delay=0.005)
    worker.start()
    worker.join(5.0)

    assert not worker.is_alive(), "a worker that gives up must exit on its own"
    assert worker.failed
    assert worker.reconnects == RECONNECT_ATTEMPTS
    final = state.read()
    assert final.status == "camera error"
    assert "could not be reopened" in (final.error or "")
    assert final.stable_command is None
    assert dead.released == 1


def test_stopping_during_a_reconnect_is_a_clean_stop_not_an_error() -> None:
    state = SharedState()
    worker = make_worker(
        state,
        cameras(ScriptedCamera(default=False), RuntimeError("no device")),
        reconnect_delay=10.0,
    )
    worker.start()
    assert wait_for(lambda: state.read().status == "reconnecting")
    started = time.perf_counter()
    worker.stop()
    worker.join(2.0)
    assert time.perf_counter() - started < 1.0, "stop must interrupt the reconnect wait"
    assert not worker.failed
    assert state.read().status == "stopped"


def test_model_load_failure_is_reported_once_and_never_opens_the_camera() -> None:
    state = SharedState()
    opened: list[bool] = []

    def camera() -> ScriptedCamera:
        opened.append(True)
        return ScriptedCamera()

    def broken() -> FakeRecognizer:
        raise RuntimeError("CUDA GPU is required for CNN inference.")

    worker = RecognitionWorker(state, camera_factory=camera, recognizer_factory=broken)
    worker.start()
    worker.join(2.0)
    assert worker.failed and not opened
    assert state.read().status == "camera error"
    assert "CUDA GPU is required" in (state.read().error or "")


def test_start_is_safe_after_the_worker_has_already_finished() -> None:
    state = SharedState()

    def broken() -> FakeRecognizer:
        raise RuntimeError("no model")

    worker = RecognitionWorker(state, recognizer_factory=broken)
    controller = GestureController(state, worker)
    controller.start()
    worker.join(2.0)
    controller.start()  # must not raise "threads can only be started once"
    assert controller.stop()


# --- latency and freshness --------------------------------------------------------------------
def test_controller_measures_capture_to_request_latency_once_per_frame() -> None:
    state = SharedState()
    controller = GestureController(state)
    game = Game(headless=True)
    now = time.perf_counter()
    for sequence in range(1, 11):
        state.publish(
            sequence=sequence,
            timestamp=now,
            captured_at=now - 0.050,
            stable_command="left",
            camera_ok=True,
            status="ready",
        )
        controller.apply_to(game, now)
        controller.apply_to(game, now)  # a second game frame on the same snapshot
    summary = controller.latency_summary()
    assert summary["samples"] == 10
    assert summary["median_ms"] == pytest.approx(50.0, abs=0.01)


def test_latency_history_is_bounded() -> None:
    worker = RecognitionWorker(SharedState())
    assert worker.latencies.maxlen == LATENCY_HISTORY
    assert GestureController().capture_to_request_ms.maxlen == LATENCY_HISTORY


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([7.0], 0.5) == 7.0
    assert percentile([float(v) for v in range(1, 101)], 0.95) == 95.0


def test_freshness_accepts_a_timestamp_of_zero() -> None:
    """Regression: `now or perf_counter()` treated now=0.0 as missing."""
    snapshot = Snapshot(sequence=1, timestamp=0.0, camera_ok=True)
    assert snapshot.is_fresh(now=0.0)


def test_reconnecting_has_its_own_status_line() -> None:
    state = SharedState()
    state.publish(status="reconnecting", camera_ok=False)
    assert GestureController(state).status_line() == "GESTURE: RECONNECTING CAMERA"
