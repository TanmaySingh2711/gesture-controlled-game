"""Bridge between the gesture recognizer and the Pac-Man game.

Deliberately lives in `src/`, not in `game/`. The game's own tests assert that no module under
`game/` imports torch, OpenCV or anything in `src`, and that neither reaches `sys.modules` while
the game runs. Keeping the bridge out here preserves that guarantee: the maze game stays fully
testable, and fully playable, with no CNN present at all.

Architecture
------------
Two loops that never wait for each other:

    main thread              RecognitionWorker thread
    ------------             ------------------------
    pygame events            camera.read()
    game.update(dt)          mirror  ->  ROI  ->  CUDA inference
    game.draw()              publish a snapshot into SharedState
    ~60 FPS                  ~28-31 FPS, camera limited

The game never calls `camera.read()` and never runs inference, so a slow frame or a camera
stall cannot drop the game below 60 FPS. Communication is one small immutable snapshot behind a
lock - **latest state only, never a queue**. Old camera frames are worthless here, so there is
nothing to back up: a new snapshot simply replaces the previous one.

How a gesture becomes a turn
----------------------------
`GestureController.apply_to(game)` runs once per game frame and, while the newest snapshot is
fresh and carries a direction, calls `game.request_direction(...)` - the same seam the keyboard
uses. That call is idempotent (it sets the pending request and resets its age), so calling it
repeatedly is not a duplicated event but a *refreshed intent*: holding thumbs-up keeps UP
requested until an UP turn becomes legal.

When the recognizer reports `None`, the controller simply stops refreshing, and the game's own
0.35 s request grace expires the stale request on its own. `None` therefore means "no new
command" - Pac-Man carries on in its current direction. It never means "stop".

Fault tolerance
---------------
A webcam that stops delivering frames is reopened up to RECONNECT_ATTEMPTS times, with a pause
between attempts, before gesture control is declared failed. A USB hiccup or a driver reset
costs a couple of seconds of RECONNECTING rather than the rest of the session. Whatever finally
ends the worker, it is logged once, shown as CAMERA ERROR, and the keyboard keeps working.

Testability
-----------
The worker takes optional `camera_factory`, `recognizer_factory` and `mirror` callables. The
application passes none and gets the real webcam, the CUDA model and OpenCV's flip; the test
suite passes fakes, so the real capture loop - reconnection, reset requests, shutdown and error
reporting - runs in CI with no camera and no GPU.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from game.engine import Game

log = logging.getLogger(__name__)

# How long a snapshot stays usable. Past this the recognizer is treated as unavailable and no
# direction is issued, so a worker that dies cannot leave an ancient LEFT stuck on the game.
# 0.75 s is a little over twenty camera frames at 28 FPS: long enough that an ordinary hiccup
# does not flicker the HUD, short enough that a dead thread is noticed almost immediately.
STALE_SECONDS: Final = 0.75

CAMERA_FAILURE_LIMIT: Final = 30  # consecutive empty reads before the camera counts as lost
RECONNECT_ATTEMPTS: Final = 3
RECONNECT_DELAY: Final = 1.0  # seconds between reconnection attempts
PREVIEW_EVERY: Final = 1  # publish a preview frame every N recognitions

# Latency samples are kept in bounded windows, so a long session cannot grow memory. Ten
# minutes of frames at ~30 FPS is far more than any summary needs.
LATENCY_HISTORY: Final = 30 * 60 * 10

# HUD wording for worker states that override whatever the last recognition said.
STATUS_LINES: Final = {
    "camera error": "CAMERA ERROR - keyboard still works",
    "starting": "INITIALIZING CAMERA / GESTURE CONTROL",
    "loading model": "INITIALIZING CAMERA / GESTURE CONTROL",
    "opening camera": "INITIALIZING CAMERA / GESTURE CONTROL",
    "reconnecting": "GESTURE: RECONNECTING CAMERA",
    "stopped": "GESTURE OFF - keyboard only",
}


class Capture(Protocol):
    """The part of `cv2.VideoCapture` the worker uses."""

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


class PredictionLike(Protocol):
    """The fields of a recognition result the worker publishes.

    Structural on purpose: `gesture_recognizer.Prediction` satisfies it, and so does any test
    fake with the same fields, without either importing the other.
    """

    @property
    def raw_direction(self) -> str: ...

    @property
    def raw_confidence(self) -> float: ...

    @property
    def thresholded_direction(self) -> str | None: ...

    @property
    def stable_command(self) -> str | None: ...

    @property
    def inference_ms(self) -> float: ...


class Recognizer(Protocol):
    """The part of `DirectionRecognizer` the worker uses."""

    def predict(self, mirrored_frame: Any) -> PredictionLike: ...

    def crop_roi(self, frame: Any) -> Any: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class Snapshot:
    """One recognition result. Immutable, so a reader can never see a half-written state."""

    sequence: int = 0
    timestamp: float = 0.0  # when the result was published
    captured_at: float = 0.0  # when its camera frame was read, for end-to-end latency
    stable_command: str | None = None
    raw_direction: str | None = None
    raw_confidence: float = 0.0
    thresholded_direction: str | None = None
    inference_ms: float = 0.0
    camera_ok: bool = False
    status: str = "starting"
    error: str | None = None
    preview: Any = None  # small BGR ndarray, or None

    def is_fresh(self, now: float | None = None, stale_seconds: float = STALE_SECONDS) -> bool:
        if not self.camera_ok or self.sequence == 0:
            return False
        current = now if now is not None else time.perf_counter()
        return current - self.timestamp <= stale_seconds


class SharedState:
    """The single slot the worker writes and the game reads. Latest value wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._writes = 0

    def publish(self, **fields: Any) -> Snapshot:
        with self._lock:
            self._snapshot = replace(self._snapshot, **fields)
            self._writes += 1
            return self._snapshot

    def read(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    @property
    def writes(self) -> int:
        with self._lock:
            return self._writes


def percentile(samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[rank]


def _real_camera() -> Capture:
    from src.gesture_recognizer import open_camera

    capture: Capture = open_camera()
    return capture


def _cv2_mirror(frame: Any) -> Any:
    import cv2  # imported lazily so `game/` and the tests never need OpenCV

    return cv2.flip(frame, 1)


class RecognitionWorker(threading.Thread):
    """Owns the webcam and the model. The only thread that touches either."""

    def __init__(  # noqa: PLR0913 - the keyword-only factories exist for dependency injection
        self,
        state: SharedState,
        threshold: float | None = None,
        window: int | None = None,
        agreement: int | None = None,
        name: str = "recognition",
        *,
        camera_factory: Callable[[], Capture] | None = None,
        recognizer_factory: Callable[[], Recognizer] | None = None,
        mirror: Callable[[Any], Any] | None = None,
        reconnect_delay: float = RECONNECT_DELAY,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.state = state
        self.threshold = threshold
        self.window = window
        self.agreement = agreement
        self._camera_factory = camera_factory or _real_camera
        self._recognizer_factory = recognizer_factory or self._real_recognizer
        self._mirror = mirror or _cv2_mirror
        self.reconnect_delay = reconnect_delay
        # NOT `self._stop`: threading.Thread already has a private _stop() method and
        # shadowing it breaks Thread.join().
        self._stop_event = threading.Event()
        self._reset_request = threading.Event()
        self.recognizer: Recognizer | None = None
        self.frames = 0
        self.reconnects = 0
        self.started_at: float | None = None
        self.latencies: deque[float] = deque(maxlen=LATENCY_HISTORY)
        self.failed = False

    def _real_recognizer(self) -> Recognizer:
        from src.gesture_recognizer import (
            DEFAULT_MIN_AGREEMENT,
            DEFAULT_THRESHOLD,
            DEFAULT_WINDOW,
            DirectionRecognizer,
        )

        return DirectionRecognizer(
            threshold=self.threshold if self.threshold is not None else DEFAULT_THRESHOLD,
            window=self.window if self.window is not None else DEFAULT_WINDOW,
            min_agreement=self.agreement if self.agreement is not None else DEFAULT_MIN_AGREEMENT,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def request_reset(self) -> None:
        """Ask the worker to clear its smoothing window at the next safe moment."""
        self._reset_request.set()

    def _fail(self, message: str) -> None:
        """Record a camera failure once, as a state rather than a repeated exception.

        Set before the loop exits and deliberately *not* overwritten by the shutdown publish,
        so CAMERA ERROR stays on screen instead of being replaced by "stopped" a moment later.
        """
        self.failed = True
        log.error("gesture control unavailable: %s", message)
        self.state.publish(
            camera_ok=False, status="camera error", error=message, stable_command=None
        )

    def _publish_exit(self) -> None:
        """The final publish when the loop ends, whatever ended it.

        An ordinary stop reads as "gesture off"; a failure must keep reading as CAMERA ERROR.
        Publishing "stopped" over a failure would hide the reason the worker died behind a
        message that looks deliberate.
        """
        if self.failed:
            self.state.publish(camera_ok=False, stable_command=None)
        else:
            self.state.publish(status="stopped", camera_ok=False, stable_command=None)

    def _reconnect(self, lost: Capture) -> Capture | None:
        """Reopen a webcam that stopped delivering frames. None if it cannot be recovered."""
        lost.release()
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            self.reconnects += 1
            self.state.publish(status="reconnecting", camera_ok=False, stable_command=None)
            log.warning(
                "webcam stopped delivering frames; reconnect attempt %d of %d",
                attempt,
                RECONNECT_ATTEMPTS,
            )
            if self._stop_event.wait(self.reconnect_delay):
                return None
            try:
                capture = self._camera_factory()
            except (RuntimeError, OSError) as error:
                log.warning("reconnect attempt %d failed: %s", attempt, error)
                continue
            if self.recognizer is not None:
                self.recognizer.reset()  # frames from before the outage must not vote
            self.state.publish(status="ready", camera_ok=True, error=None)
            log.info("webcam reconnected on attempt %d", attempt)
            return capture
        return None

    def run(self) -> None:
        capture: Capture | None = None
        try:
            self.state.publish(status="loading model", camera_ok=False)
            recognizer = self._recognizer_factory()
            self.recognizer = recognizer

            self.state.publish(status="opening camera")
            capture = self._camera_factory()
            self.state.publish(status="ready", camera_ok=True, error=None)
            self.started_at = time.perf_counter()

            failures = 0
            while not self._stop_event.is_set():
                if self._reset_request.is_set():
                    recognizer.reset()
                    self._reset_request.clear()

                ok, frame = capture.read()
                captured_at = time.perf_counter()
                if not ok or frame is None:
                    failures += 1
                    if failures < CAMERA_FAILURE_LIMIT:
                        continue
                    lost, capture = capture, None
                    capture = self._reconnect(lost)
                    if capture is None:
                        if not self._stop_event.is_set():
                            self._fail(
                                "the webcam stopped returning frames and could not be reopened"
                            )
                        break
                    failures = 0
                    continue
                failures = 0

                frame = self._mirror(frame)  # mirror convention, applied first
                result = recognizer.predict(frame)
                self.frames += 1
                self.latencies.append(result.inference_ms)

                preview = None
                if self.frames % PREVIEW_EVERY == 0:
                    # A copy, because the worker overwrites `frame` on the next pass.
                    preview = recognizer.crop_roi(frame).copy()

                self.state.publish(
                    sequence=self.frames,
                    timestamp=time.perf_counter(),
                    captured_at=captured_at,
                    stable_command=result.stable_command,
                    raw_direction=result.raw_direction,
                    raw_confidence=result.raw_confidence,
                    thresholded_direction=result.thresholded_direction,
                    inference_ms=result.inference_ms,
                    camera_ok=True,
                    status="ready",
                    error=None,
                    preview=preview,
                )
        except Exception as error:  # reported once as CAMERA ERROR, never swallowed
            log.debug("recognition worker traceback", exc_info=True)
            self._fail(f"{type(error).__name__}: {error}")
        finally:
            if capture is not None:
                capture.release()
            self._publish_exit()

    # --- statistics, for the performance report ----------------------------------------
    def summary(self) -> dict[str, float]:
        if not self.started_at or not self.frames:
            return {}
        elapsed = time.perf_counter() - self.started_at
        latencies = list(self.latencies)
        return {
            "frames": float(self.frames),
            "seconds": elapsed,
            "camera_fps": self.frames / elapsed if elapsed else 0.0,
            "cnn_mean_ms": statistics.fmean(latencies),
            "cnn_median_ms": statistics.median(latencies),
            "cnn_p95_ms": percentile(latencies, 0.95),
            "reconnects": float(self.reconnects),
        }


class GestureController:
    """Turns recognition snapshots into direction requests. Pure logic, no threads."""

    def __init__(
        self,
        state: SharedState | None = None,
        worker: RecognitionWorker | None = None,
        stale_seconds: float = STALE_SECONDS,
    ) -> None:
        self.state = state if state is not None else SharedState()
        self.worker = worker
        self.stale_seconds = stale_seconds
        self.last_applied: str | None = None
        self.last_sequence = 0
        self.applied_count = 0
        self.snapshot = Snapshot()
        # Camera frame read -> direction request submitted, one sample per new snapshot.
        self.capture_to_request_ms: deque[float] = deque(maxlen=LATENCY_HISTORY)

    # --- lifecycle -----------------------------------------------------------------------
    @classmethod
    def with_worker(
        cls, threshold: float | None = None, window: int | None = None, agreement: int | None = None
    ) -> GestureController:
        state = SharedState()
        worker = RecognitionWorker(state, threshold, window, agreement)
        return cls(state, worker)

    def start(self) -> None:
        if self.worker is not None and not self.worker.is_alive() and self.worker.ident is None:
            self.worker.start()

    def stop(self, timeout: float = 3.0) -> bool:
        """Signal the worker, wait for it, and confirm it has exited."""
        if self.worker is None:
            return True
        self.worker.stop()
        if self.worker.ident is not None:  # never started: there is nothing to join
            self.worker.join(timeout=timeout)
        return not self.worker.is_alive()

    def reset(self) -> None:
        """Forget the current command and clear the recognizer's smoothing window.

        Called on death, round change and restart so a direction held before the pause cannot
        immediately steer the respawned player.
        """
        self.last_applied = None
        self.last_sequence = 0
        if self.worker is not None:
            self.worker.request_reset()

    # --- per-frame -----------------------------------------------------------------------
    def poll(self, now: float | None = None) -> tuple[Snapshot, bool]:
        """The newest snapshot, and whether it is fresh enough to act on."""
        current = now if now is not None else time.perf_counter()
        self.snapshot = self.state.read()
        return self.snapshot, self.snapshot.is_fresh(current, self.stale_seconds)

    def apply_to(self, game: Game, now: float | None = None) -> str | None:
        """Refresh the game's direction request from the newest recognition.

        Returns the direction submitted, or None. Called once per game frame: a held gesture
        therefore keeps its request alive, and releasing the gesture lets the game's own 0.35 s
        grace period expire it.
        """
        current = now if now is not None else time.perf_counter()
        snapshot, fresh = self.poll(current)
        if not fresh or snapshot.stable_command is None:
            self.last_applied = None
            return None

        game.request_direction(snapshot.stable_command)
        if snapshot.sequence != self.last_sequence and snapshot.captured_at:
            self.capture_to_request_ms.append((current - snapshot.captured_at) * 1000.0)
        self.applied_count += 1
        self.last_applied = snapshot.stable_command
        self.last_sequence = snapshot.sequence
        return snapshot.stable_command

    def latency_summary(self) -> dict[str, float]:
        samples = list(self.capture_to_request_ms)
        if not samples:
            return {}
        return {
            "samples": float(len(samples)),
            "median_ms": statistics.median(samples),
            "p95_ms": percentile(samples, 0.95),
        }

    # --- HUD -----------------------------------------------------------------------------
    def status_line(self, now: float | None = None) -> str:
        """The one-line gesture state the game HUD shows beside the maze."""
        snapshot, fresh = self.poll(now)
        if snapshot.status in STATUS_LINES:
            return STATUS_LINES[snapshot.status]
        if not fresh:
            return "GESTURE: WAITING"
        if snapshot.stable_command is None:
            return "GESTURE: NO COMMAND"
        return f"GESTURE: {snapshot.stable_command.upper()}"
