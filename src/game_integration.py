"""Bridge between the frozen P6 gesture recognizer and the frozen P7 Pac-Man game.

Deliberately lives in `src/`, not in `game/`. The P7 test suite asserts that no module under
`game/` imports torch, OpenCV or the recognizer, and that neither ever reaches `sys.modules`
while the game runs. Keeping the bridge outside that package preserves the guarantee: the
maze game stays fully testable, and fully playable, with no CNN present at all.

Architecture
------------
Two loops that never wait for each other:

    main thread              RecognitionWorker thread
    ------------             ------------------------
    pygame events            camera.read()
    game.update(dt)          cv2.flip  ->  ROI  ->  CUDA inference
    game.draw()              publish a snapshot into SharedState
    ~60 FPS                  ~28-31 FPS, camera limited

The game never calls `camera.read()` and never runs inference, so a slow frame or a camera
stall cannot drop the game below 60 FPS. Communication is one small immutable snapshot
behind a lock - **latest state only, never a queue**. Old camera frames are worthless here,
so there is nothing to back up: a new snapshot simply replaces the previous one.

How a gesture becomes a turn
----------------------------
`GestureController.apply_to(game)` runs once per game frame and, while the newest snapshot is
fresh and carries a direction, calls `game.request_direction(...)` - the same seam the
keyboard uses. Because that call is idempotent (it sets the pending request and resets its
age), calling it repeatedly is not a duplicated event but a *refreshed intent*: holding
thumbs-up keeps UP requested until an UP turn becomes legal.

When the recognizer reports `None`, the controller simply stops refreshing, and the game's own
0.35 s request grace expires the stale request on its own. `None` therefore means "no new
command" - Pac-Man carries on in its current direction. It never means "stop".
"""

import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# How long a snapshot stays usable. Past this the recognizer is treated as unavailable and no
# direction is issued, so a worker that dies cannot leave an ancient LEFT stuck on the game.
# 0.75 s is a little over twenty camera frames at 28 FPS: long enough that an ordinary hiccup
# does not flicker the HUD, short enough that a dead thread is noticed almost immediately.
STALE_SECONDS = 0.75

CAMERA_FAILURE_LIMIT = 30       # consecutive empty reads before declaring the camera lost
PREVIEW_EVERY = 1               # publish a preview frame every N recognitions


@dataclass(frozen=True)
class Snapshot:
    """One recognition result. Immutable, so a reader can never see a half-written state."""

    sequence: int = 0
    timestamp: float = 0.0
    stable_command: Optional[str] = None
    raw_direction: Optional[str] = None
    raw_confidence: float = 0.0
    thresholded_direction: Optional[str] = None
    inference_ms: float = 0.0
    camera_ok: bool = False
    status: str = "starting"
    error: Optional[str] = None
    preview: object = None          # small BGR ndarray, or None

    def is_fresh(self, now=None, stale_seconds=STALE_SECONDS):
        if not self.camera_ok or self.sequence == 0:
            return False
        return (now or time.perf_counter()) - self.timestamp <= stale_seconds


class SharedState:
    """The single slot the worker writes and the game reads. Latest value wins."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = Snapshot()
        self._writes = 0

    def publish(self, **fields):
        with self._lock:
            self._snapshot = replace(self._snapshot, **fields)
            self._writes += 1
            return self._snapshot

    def read(self):
        with self._lock:
            return self._snapshot

    @property
    def writes(self):
        with self._lock:
            return self._writes


class RecognitionWorker(threading.Thread):
    """Owns the webcam and the model. The only thread that touches either."""

    def __init__(self, state, threshold=None, window=None, agreement=None, name="recognition"):
        super().__init__(name=name, daemon=True)
        self.state = state
        self.threshold = threshold
        self.window = window
        self.agreement = agreement
        # NOT `self._stop`: threading.Thread already has a private _stop() method and
        # shadowing it breaks Thread.join().
        self._stop_event = threading.Event()
        self._reset_request = threading.Event()
        self.recognizer = None
        self.frames = 0
        self.started_at = None
        self.latencies = []
        self.failed = False

    def stop(self):
        self._stop_event.set()

    def _fail(self, message):
        """Record a camera failure once, as a state rather than a repeated exception.

        Set before the loop exits and deliberately *not* overwritten by the shutdown
        publish, so CAMERA ERROR stays on screen instead of being replaced by the ordinary
        "stopped" message a moment later.
        """
        self.failed = True
        self.state.publish(camera_ok=False, status="camera error", error=message,
                           stable_command=None)

    def _publish_exit(self):
        """The final publish when the loop ends, whatever ended it.

        An ordinary stop reads as "gesture off"; a failure must keep reading as CAMERA
        ERROR. Publishing "stopped" over a failure would hide the reason the worker died
        behind a message that looks deliberate. Kept as a method so the shutdown tests can
        drive the real path without opening a camera.
        """
        if self.failed:
            self.state.publish(camera_ok=False, stable_command=None)
        else:
            self.state.publish(status="stopped", camera_ok=False, stable_command=None)

    def request_reset(self):
        """Ask the worker to clear its smoothing window at the next safe moment."""
        self._reset_request.set()

    def run(self):
        import cv2                      # imported here so `game/` never pulls OpenCV in

        capture = None
        try:
            self.state.publish(status="loading model", camera_ok=False)
            from gesture_recognizer import (DEFAULT_MIN_AGREEMENT, DEFAULT_THRESHOLD,
                                            DEFAULT_WINDOW, DirectionRecognizer, open_camera)
            self.recognizer = DirectionRecognizer(
                threshold=self.threshold if self.threshold is not None else DEFAULT_THRESHOLD,
                window=self.window if self.window is not None else DEFAULT_WINDOW,
                min_agreement=(self.agreement if self.agreement is not None
                               else DEFAULT_MIN_AGREEMENT))

            self.state.publish(status="opening camera")
            capture = open_camera()
            self.state.publish(status="ready", camera_ok=True, error=None)
            self.started_at = time.perf_counter()

            failures = 0
            while not self._stop_event.is_set():
                if self._reset_request.is_set():
                    self.recognizer.reset()
                    self._reset_request.clear()

                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= CAMERA_FAILURE_LIMIT:
                        self._fail("the webcam stopped returning frames")
                        break
                    continue
                failures = 0

                frame = cv2.flip(frame, 1)          # mirror convention, applied first
                result = self.recognizer.predict(frame)
                self.frames += 1
                self.latencies.append(result.inference_ms)

                preview = None
                if self.frames % PREVIEW_EVERY == 0:
                    # A copy, because the worker will overwrite `frame` on the next pass.
                    preview = self.recognizer.crop_roi(frame).copy()

                self.state.publish(
                    sequence=self.frames,
                    timestamp=time.perf_counter(),
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
        except Exception as error:                  # noqa: BLE001 - reported, not swallowed
            # Recorded once and turned into a HUD state. The loop has already exited, so
            # there is no way for this to spam every frame.
            self._fail(f"{type(error).__name__}: {error}")
        finally:
            if capture is not None:
                capture.release()
            self._publish_exit()

    # --- statistics, for the final performance report ------------------------------------
    def summary(self):
        if not self.started_at or not self.frames:
            return {}
        import statistics
        elapsed = time.perf_counter() - self.started_at
        return {
            "frames": self.frames,
            "seconds": elapsed,
            "camera_fps": self.frames / elapsed if elapsed else 0.0,
            "cnn_mean_ms": statistics.fmean(self.latencies),
            "cnn_median_ms": statistics.median(self.latencies),
            "cnn_p95_ms": sorted(self.latencies)[int(len(self.latencies) * 0.95) - 1],
        }


class GestureController:
    """Turns recognition snapshots into direction requests. Pure logic, no threads."""

    def __init__(self, state=None, worker=None, stale_seconds=STALE_SECONDS):
        self.state = state if state is not None else SharedState()
        self.worker = worker
        self.stale_seconds = stale_seconds
        self.last_applied = None
        self.last_sequence = 0
        self.applied_count = 0
        self.snapshot = Snapshot()

    # --- lifecycle -----------------------------------------------------------------------
    @classmethod
    def with_worker(cls, threshold=None, window=None, agreement=None):
        state = SharedState()
        worker = RecognitionWorker(state, threshold, window, agreement)
        return cls(state, worker)

    def start(self):
        if self.worker is not None and not self.worker.is_alive():
            self.worker.start()

    def stop(self, timeout=3.0):
        """Signal the worker, wait for it, and confirm the camera was released."""
        if self.worker is None:
            return True
        self.worker.stop()
        if self.worker.ident is not None:       # never started: there is nothing to join
            self.worker.join(timeout=timeout)
        return not self.worker.is_alive()

    def reset(self):
        """Forget the current command and clear the recognizer's smoothing window.

        Called on death, round change and restart so a direction held before the pause
        cannot immediately steer the respawned player.
        """
        self.last_applied = None
        self.last_sequence = 0
        if self.worker is not None:
            self.worker.request_reset()

    # --- per-frame -----------------------------------------------------------------------
    def poll(self, now=None):
        """The newest snapshot, and whether it is fresh enough to act on."""
        now = now if now is not None else time.perf_counter()
        self.snapshot = self.state.read()
        return self.snapshot, self.snapshot.is_fresh(now, self.stale_seconds)

    def apply_to(self, game, now=None):
        """Refresh the game's direction request from the newest recognition.

        Returns the direction submitted, or None. Called once per game frame: a held gesture
        therefore keeps its request alive, and releasing the gesture lets the game's own
        0.35 s grace period expire it.
        """
        snapshot, fresh = self.poll(now)
        if not fresh or snapshot.stable_command is None:
            self.last_applied = None
            return None

        game.request_direction(snapshot.stable_command)
        self.applied_count += 1
        self.last_applied = snapshot.stable_command
        self.last_sequence = snapshot.sequence
        return snapshot.stable_command

    # --- HUD -----------------------------------------------------------------------------
    def status_line(self, now=None):
        snapshot, fresh = self.poll(now)
        if snapshot.status == "camera error":
            return "CAMERA ERROR - keyboard still works"
        if snapshot.status in ("starting", "loading model", "opening camera"):
            return "INITIALIZING CAMERA / GESTURE CONTROL"
        if snapshot.status == "stopped":
            return "GESTURE OFF - keyboard only"
        if not fresh:
            return "GESTURE: WAITING"
        if snapshot.stable_command is None:
            return "GESTURE: NO COMMAND"
        return f"GESTURE: {snapshot.stable_command.upper()}"
