"""OPTIONAL webcam capture tool - NOT part of the primary dataset workflow.

The project dataset comes from HaGRID hand-region crops, built by
`src/crop_hagrid_hands.py`. This script is kept for two secondary uses:

  * capturing a small set of custom webcam images to sanity-check the trained model on the
    machine it will actually run on, and
  * as the reference implementation of the capture geometry (flip-then-crop and the fixed
    ROI constants below) that real-time gameplay inference must reuse.

Do not use it to build the training set. Its on-screen gesture hints follow the P1 mapping
frozen for the Pac-Man direction: fist / palm / like / dislike.

Collects raw ROI images for the four gesture classes: left, right, up, down.

Mirror convention (frozen in PROJECT_SPEC.md section 7):
    Every frame is horizontally flipped immediately after capture, and every label refers to
    what the user sees in the flipped preview. The gesture the user sees is the label saved.
    The same flip and the same ROI are used later during real-time gameplay, so the model
    never sees a differently-framed image at inference time.

Controls:
    1 / 2 / 3 / 4   select class LEFT / RIGHT / JUMP / NEUTRAL
    SPACE           start or pause automatic capture
    C               capture a single image
    Q or ESC        quit

Usage:
    python src/collect_dataset.py
    python src/collect_dataset.py --selftest    # verify webcam, flip and ROI without saving
"""

import os
import sys
import time

import cv2

# --- Fixed capture geometry -------------------------------------------------------------
# Webcam feed is 640x480. The ROI is a 300x300 square on the right-hand side of the FLIPPED
# preview, which is where a right hand naturally sits in a mirror view, vertically centred.
# These four numbers are the contract between dataset collection and gameplay inference:
# any later script that feeds the CNN must crop exactly this region from a flipped frame.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_X1, ROI_Y1 = 300, 90
ROI_X2, ROI_Y2 = 600, 390
ROI_SIZE = (ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1)  # (300, 300)

TARGET_PER_CLASS = 400
CAPTURES_PER_SECOND = 4          # 4 fps -> ~0.25 s between saved images
CAPTURE_INTERVAL = 1.0 / CAPTURES_PER_SECOND

CLASSES = ["left", "right", "up", "down"]
CLASS_KEYS = {ord("1"): "left", ord("2"): "right", ord("3"): "up", ord("4"): "down"}

GESTURE_HINTS = {
    "left": "CLOSED FIST facing the camera (HaGRID 'fist')",
    "right": "OPEN PALM, five fingers spread (HaGRID 'palm')",
    "up": "THUMBS UP, thumb pointing upward (HaGRID 'like')",
    "down": "THUMBS DOWN, thumb pointing downward (HaGRID 'dislike')",
}

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def class_dir(label):
    return os.path.join(DATASET_DIR, label)


def count_existing(label):
    """Number of images already collected for a class."""
    folder = class_dir(label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


def next_free_path(label, index):
    """Build a unique path, skipping any filename that already exists (never overwrite)."""
    folder = class_dir(label)
    while True:
        path = os.path.join(folder, f"{label}_{index:05d}.jpg")
        if not os.path.exists(path):
            return path, index
        index += 1


def save_roi(label, roi, index):
    """Validate and write one ROI image. Returns the next index, or None if the save failed."""
    if roi is None or roi.size == 0:
        print("[warn] empty ROI, frame skipped")
        return None
    if (roi.shape[1], roi.shape[0]) != ROI_SIZE:
        print(f"[warn] unexpected ROI size {roi.shape[:2]}, frame skipped")
        return None

    path, index = next_free_path(label, index)
    if not cv2.imwrite(path, roi):
        print(f"[warn] failed to write {path}")
        return None
    return index + 1


def draw_overlay(display, label, counts, capturing):
    """Draw the ROI box, the current state and the on-screen instructions."""
    in_roi_color = (0, 220, 0) if capturing else (0, 200, 255)
    cv2.rectangle(display, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), in_roi_color, 2)
    cv2.putText(display, "ROI", (ROI_X1 + 6, ROI_Y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, in_roi_color, 2)

    # Dark band behind the text so it stays readable on any background.
    cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 74), (0, 0, 0), -1)
    cv2.rectangle(display, (0, FRAME_HEIGHT - 54), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)

    done = counts[label]
    state = "CAPTURING" if capturing else "PAUSED"
    state_color = (0, 220, 0) if capturing else (0, 200, 255)
    if done >= TARGET_PER_CLASS:
        state, state_color = "TARGET REACHED", (0, 220, 220)

    cv2.putText(display, f"CLASS: {label.upper()}   {done}/{TARGET_PER_CLASS}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, state, (430, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    cv2.putText(display, GESTURE_HINTS[label], (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    totals = "  ".join(f"{c[:1].upper()}:{counts[c]}" for c in CLASSES)
    cv2.putText(display, f"totals  {totals}   (target {TARGET_PER_CLASS} each)",
                (10, FRAME_HEIGHT - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(display, "1 left  2 right  3 up  4 down  |  SPACE capture  C single  Q quit",
                (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def open_camera():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam device 0.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return capture


def selftest():
    """Non-interactive check: capture one frame, flip it, crop the ROI, save nothing."""
    print("Self-test: webcam -> horizontal flip -> ROI crop (no images are saved)")
    capture = open_camera()
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            print("[FAIL] webcam returned no frame")
            return 1
        print(f"[PASS] frame captured          {frame.shape[1]}x{frame.shape[0]}")

        flipped = cv2.flip(frame, 1)
        print("[PASS] horizontal flip applied cv2.flip(frame, 1)")

        roi = flipped[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        if (roi.shape[1], roi.shape[0]) != ROI_SIZE:
            print(f"[FAIL] ROI crop is {roi.shape[1]}x{roi.shape[0]}, expected {ROI_SIZE[0]}x{ROI_SIZE[1]}")
            return 1
        print(f"[PASS] ROI crop                {roi.shape[1]}x{roi.shape[0]} "
              f"at x[{ROI_X1}:{ROI_X2}] y[{ROI_Y1}:{ROI_Y2}]")

        for label in CLASSES:
            folder = class_dir(label)
            status = "ok" if os.path.isdir(folder) else "MISSING"
            print(f"[{'PASS' if status == 'ok' else 'FAIL'}] dataset/{label:<8} {status}, "
                  f"{count_existing(label)} images present")
        return 0
    finally:
        capture.release()
        print("[PASS] webcam released cleanly")


def main():
    if "--selftest" in sys.argv:
        return selftest()

    for label in CLASSES:
        os.makedirs(class_dir(label), exist_ok=True)

    counts = {label: count_existing(label) for label in CLASSES}
    next_index = {label: counts[label] for label in CLASSES}
    label = "left"
    capturing = False
    last_capture = 0.0

    print(__doc__.split("Controls:")[1].split("Usage:")[0])
    print("Existing images:", ", ".join(f"{c}={counts[c]}" for c in CLASSES))

    capture = open_camera()
    window = "Gesture dataset collection"
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[warn] dropped frame from webcam")
                continue

            # Mirror convention: flip first, then everything downstream sees the mirror view.
            frame = cv2.flip(frame, 1)
            roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2].copy()

            now = time.time()
            if capturing and counts[label] < TARGET_PER_CLASS and now - last_capture >= CAPTURE_INTERVAL:
                new_index = save_roi(label, roi, next_index[label])
                if new_index is not None:
                    next_index[label] = new_index
                    counts[label] += 1
                    last_capture = now
                    if counts[label] >= TARGET_PER_CLASS:
                        capturing = False
                        print(f"[info] target reached for {label} ({counts[label]} images)")

            display = frame.copy()  # overlay is drawn on a copy so it never lands in saved data
            draw_overlay(display, label, counts, capturing)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in CLASS_KEYS:
                label = CLASS_KEYS[key]
                capturing = False
                print(f"[info] class -> {label} ({counts[label]}/{TARGET_PER_CLASS})")
            elif key == ord(" "):
                if counts[label] >= TARGET_PER_CLASS:
                    print(f"[info] {label} already has {counts[label]} images")
                else:
                    capturing = not capturing
                    last_capture = 0.0
                    print(f"[info] capture {'started' if capturing else 'paused'} for {label}")
            elif key == ord("c"):
                new_index = save_roi(label, roi, next_index[label])
                if new_index is not None:
                    next_index[label] = new_index
                    counts[label] += 1
                    print(f"[info] saved 1 image  {label} {counts[label]}/{TARGET_PER_CLASS}")

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("\nFinal counts:", ", ".join(f"{c}={counts[c]}" for c in CLASSES))
    print("Total:", sum(counts.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
