"""The webcam: the frozen capture geometry and opening the device, using OpenCV alone.

Kept apart from `gesture_recognizer`, which imports PyTorch, so the recognition worker can open
the camera on a helper thread while PyTorch is still loading. Importing PyTorch takes a few
seconds; the camera is usually open long before it finishes.
"""

from __future__ import annotations

import os
from typing import Any, Final

import cv2

# Fixed capture geometry, frozen in docs/PROJECT_SPEC.md section 7 and unchanged since the dataset
# was built. Every frame is mirrored first; the ROI is then the 300x300 square on the right of
# the mirrored frame, where a right hand naturally sits, and is all the CNN ever sees.
FRAME_WIDTH: Final = 640
FRAME_HEIGHT: Final = 480
ROI_X1: Final = 300
ROI_Y1: Final = 90
ROI_X2: Final = 600
ROI_Y2: Final = 390


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
