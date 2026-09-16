"""Reusable real-time direction recognition for the frozen four-direction MobileNetV2.

Deliberately knows nothing about Pygame, OpenCV windows, or any particular UI: it takes a
mirrored webcam frame (or a bare ROI) and returns raw, thresholded and smoothed results. The
game drives itself from this without either side knowing the other's internals.

    recognizer = DirectionRecognizer()
    result = recognizer.predict(mirrored_frame)
    result.stable_command      # "left" / "right" / "up" / "down" / None  <- act on this
    result.stable_changed      # True on the frame the stable command just changed
    result.raw_direction       # single-frame CNN output, for debugging and overlays
    result.raw_confidence      # softmax probability of the raw class

There is no NEUTRAL class
-------------------------
The CNN has exactly four outputs and is mathematically forced to pick one of them for every
frame, including frames containing no hand at all. "No reliable command" is therefore not a
class - it is the absence of one, represented as `None`. It never appears in CLASS_TO_INDEX and
is never a model prediction.

For Pac-Man this is the correct semantics: `None` means *do not issue a new direction request*,
so the character keeps travelling the way it already was. It does not mean "stop".

Why smoothing is not optional
-----------------------------
Offline evaluation produced a `down` misread as `up` at 0.9031 confidence. A confidence
threshold alone cannot catch an error like that, so a direction must also hold a majority of a
short rolling window before it becomes a command. A single bad frame - motion blur during a
transition, a hand halfway between gestures - cannot by itself issue a turn.

Preprocessing is not reimplemented here: the ROI is handed to
`data_pipeline.inference_transform()`, the exact transform used for the validation split and
the test evaluation, so webcam tensors are built identically to the ones the model was scored on.
"""

from __future__ import annotations

import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import cv2
import torch
from PIL import Image

from src.data_pipeline import CLASS_TO_INDEX, IMAGE_SIZE, inference_transform
from src.train_model import CHECKPOINT_PATH, load_direction_checkpoint

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    Frame = npt.NDArray[np.uint8]

DIRECTIONS: Final[list[str]] = list(CLASS_TO_INDEX)  # left, right, up, down
INDEX_TO_DIRECTION: Final[dict[int, str]] = {index: name for name, index in CLASS_TO_INDEX.items()}

# The absence of a command. Internally `None`; this string is for display only and must never be
# treated as a class, stored in CLASS_TO_INDEX, or returned as a prediction.
NO_COMMAND_LABEL: Final = "NO COMMAND"

EXPECTED_TASK: Final = "pacman_direction_v1"

# Fixed capture geometry, frozen in PROJECT_SPEC.md section 7 and unchanged since the dataset was
# built. The ROI is what the CNN sees; everything outside it is ignored.
FRAME_WIDTH: Final = 640
FRAME_HEIGHT: Final = 480
ROI_X1: Final = 300
ROI_Y1: Final = 90
ROI_X2: Final = 600
ROI_Y2: Final = 390

# Frozen after the P6 live trials, on live evidence rather than inherited from the retired
# project: intentional gestures never fell below 0.961 confidence while idle and absent hands
# never exceeded 0.617, and 3-of-5 smoothing produced no spurious turns at natural speed. They
# are never tuned against the offline test set.
DEFAULT_THRESHOLD: Final = 0.90
DEFAULT_WINDOW: Final = 5
DEFAULT_MIN_AGREEMENT: Final = 3

DIRECTION_HELP: Final[dict[str, str]] = {
    "left": "Fist",
    "right": "Open palm",
    "up": "Thumbs up",
    "down": "Thumbs down",
}


def label_of(command: str | None) -> str:
    """Display text for a command that may be None."""
    return NO_COMMAND_LABEL if command is None else command.upper()


@dataclass
class Prediction:
    """One frame of recognition state. Raw, thresholded and smoothed are kept separate."""

    raw_direction: str  # straight from the CNN, no threshold applied
    raw_confidence: float  # softmax probability of raw_direction
    thresholded_direction: str | None  # raw_direction, or None if below threshold
    stable_command: str | None  # smoothed - act on this one; None = no command
    stable_changed: bool  # True on the frame stable_command just changed
    inference_ms: float  # CNN forward pass only


class DirectionRecognizer:
    """Frame in, direction request out. No UI, no game, no Pygame."""

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        window: int = DEFAULT_WINDOW,
        min_agreement: int = DEFAULT_MIN_AGREEMENT,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for CNN inference.")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not 1 <= min_agreement <= window:
            raise ValueError(f"min_agreement {min_agreement} must be between 1 and window {window}")
        self.device = torch.device("cuda")
        self.threshold = threshold
        self.window = window
        self.min_agreement = min_agreement

        # Bounded history, so a long session cannot grow memory. Holds directions and None.
        self.history: deque[str | None] = deque(maxlen=window)
        self.stable_command: str | None = None
        self.previous_stable: str | None = None

        self.model, self.metadata = self._load(checkpoint_path)
        self.transform = inference_transform()
        self._last_inference_ms = 0.0

    # --- setup ---------------------------------------------------------------------------
    def _load(self, checkpoint_path: str) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Load once, at construction. Never per frame.

        Goes through the guarded loader, which verifies the file's SHA-256 against the frozen
        checkpoint, loads it without executing pickled code, and refuses any class mapping that
        differs from the active one. The mapping check is what stops the retired endless-runner
        model: it has four outputs too, so shape alone would not catch it, and index 2 would
        silently mean `jump` instead of `up`.
        """
        model, payload = load_direction_checkpoint(checkpoint_path, device=self.device)
        model.eval()

        problems = []
        if payload.get("architecture") != "mobilenet_v2":
            problems.append(f"architecture={payload.get('architecture')}")
        if payload.get("task") != EXPECTED_TASK:
            problems.append(f"task={payload.get('task')}")
        if payload.get("num_classes") != len(DIRECTIONS):
            problems.append(f"num_classes={payload.get('num_classes')}")
        if list(payload.get("input_size", [])) != [3, IMAGE_SIZE, IMAGE_SIZE]:
            problems.append(f"input_size={payload.get('input_size')}")
        if payload.get("class_to_index") != CLASS_TO_INDEX:
            problems.append(f"class_to_index={payload.get('class_to_index')}")
        retired = sorted({"jump", "neutral"} & set(payload.get("class_to_index", {})))
        if retired:
            problems.append(f"retired class names: {', '.join(retired)}")
        if problems:
            raise RuntimeError("checkpoint metadata mismatch: " + ", ".join(problems))

        return model, payload

    # --- geometry ------------------------------------------------------------------------
    @staticmethod
    def crop_roi(frame: Frame) -> Frame:
        """The fixed 300x300 ROI of an already-mirrored 640x480 frame."""
        return frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    # --- inference -----------------------------------------------------------------------
    def preprocess(self, roi_bgr: Frame) -> torch.Tensor:
        """BGR ROI -> 1x3x160x160 CUDA tensor, via the frozen evaluation transform.

        The frame is already mirrored by the caller before the ROI is cut; nothing here flips,
        rotates or jitters it. Augmentation belongs to training only.
        """
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = self.transform(Image.fromarray(rgb))
        return tensor.unsqueeze(0).to(self.device, non_blocking=True)

    def predict_roi(self, roi_bgr: Frame) -> Prediction:
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(self.preprocess(roi_bgr))  # raw logits, no softmax inside
            probabilities = torch.softmax(logits, dim=1)  # softmax only for confidence
            confidence, index = probabilities.max(dim=1)
            torch.cuda.synchronize()
        self._last_inference_ms = (time.perf_counter() - started) * 1000.0

        raw_direction = INDEX_TO_DIRECTION[int(index.item())]
        raw_confidence = float(confidence.item())

        # Threshold first, smoothing second. The raw prediction is never overwritten.
        thresholded = raw_direction if raw_confidence >= self.threshold else None
        stable = self._smooth(thresholded)

        changed = stable != self.previous_stable
        self.previous_stable = stable
        self.stable_command = stable

        return Prediction(
            raw_direction=raw_direction,
            raw_confidence=raw_confidence,
            thresholded_direction=thresholded,
            stable_command=stable,
            stable_changed=changed,
            inference_ms=self._last_inference_ms,
        )

    def predict(self, mirrored_frame: Frame) -> Prediction:
        """Convenience entry point: takes the full mirrored frame, crops the ROI itself."""
        return self.predict_roi(self.crop_roi(mirrored_frame))

    # --- smoothing -----------------------------------------------------------------------
    def _smooth(self, thresholded_direction: str | None) -> str | None:
        """A direction becomes a command only once it holds the window by itself.

        Only real directions are counted. Rejected frames enter the history as None and dilute
        the count without ever being able to win it, so a burst of low-confidence frames can
        clear a command but can never fabricate one.
        """
        self.history.append(thresholded_direction)
        votes = Counter(item for item in self.history if item is not None)
        if not votes:
            return None
        candidate, count = votes.most_common(1)[0]
        return candidate if count >= self.min_agreement else None

    def reset(self) -> None:
        """Forget the rolling window - use between trials, after a pause, or on reconnection."""
        self.history.clear()
        self.stable_command = None
        self.previous_stable = None


# Backwards-compatible alias. The class was called GestureRecognizer while the project was a
# gesture-named endless runner; the concept is now a direction request.
GestureRecognizer = DirectionRecognizer


def open_camera(index: int = 0) -> Any:
    """Open the webcam at the frozen capture size. Returns a `cv2.VideoCapture`."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam device {index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return capture
