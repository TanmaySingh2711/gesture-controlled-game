"""Standalone webcam demo and live-trial recorder for four-direction recognition.

Recognition itself lives in `gesture_recognizer.py`; this file is only the camera loop, the
overlay, and the measurement tools. No game logic here.

Modes:
    python src/realtime_gesture.py                      live preview with overlay
    python src/realtime_gesture.py --trials             guided held-gesture recorder
    python src/realtime_gesture.py --trials --classes up,down --count 20 --out model/x.csv
    python src/realtime_gesture.py --idle               no-hand / idle-hand recorder
    python src/realtime_gesture.py --transitions        transition stress test + latency
    python src/realtime_gesture.py --benchmark 300      measure the live pipeline, then exit
    python src/realtime_gesture.py --selftest           headless verification, no window

Trial categories include two that are not CNN classes:

    no_hand     empty ROI            - the correct result is NO COMMAND
    idle_hand   hand, but no gesture - the correct result is NO COMMAND

They exist because the CNN has no reject class and must output one of four directions for
every frame. What matters is not what it predicts, but whether the threshold and smoothing
stop that prediction from becoming a command.

Controls in preview mode:
    Q or ESC   quit
    R          reset the smoothing history
    D          toggle the raw-prediction debug line

Controls in trial mode:
    SPACE      record the current stable command for the prompted category
    S          skip the current trial
    Q or ESC   quit early (records what has been collected so far)
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data_pipeline import CLASS_TO_INDEX, PROJECT_ROOT, eval_transform
from src.gesture_recognizer import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    DIRECTION_HELP,
    DIRECTIONS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NO_COMMAND_LABEL,
    ROI_X1,
    ROI_X2,
    ROI_Y1,
    ROI_Y2,
    DirectionRecognizer,
    label_of,
    open_camera,
)

TRIALS_PER_CLASS = 20
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
TRIALS_CSV = os.path.join(MODEL_DIR, "live_direction_test_baseline.csv")
IDLE_CSV = os.path.join(MODEL_DIR, "live_direction_idle_baseline.csv")
TRANSITIONS_CSV = os.path.join(MODEL_DIR, "live_direction_transitions.csv")

# Categories whose correct outcome is "no command issued" rather than a direction.
NO_COMMAND_CATEGORIES = ("no_hand", "idle_hand")
CATEGORIES = list(DIRECTIONS) + list(NO_COMMAND_CATEGORIES)

CATEGORY_HELP = dict(DIRECTION_HELP)
CATEGORY_HELP["no_hand"] = "EMPTY box - take your hand out"
CATEGORY_HELP["idle_hand"] = "Relaxed hand, no deliberate gesture"

# Natural pairs for the transition stress test: every one crosses a real decision boundary.
TRANSITION_PAIRS = [
    ("left", "right"),
    ("right", "up"),
    ("up", "down"),
    ("down", "left"),
    ("left", "up"),
    ("right", "down"),
]

COLOR = {
    "left": (80, 200, 255),
    "right": (120, 220, 120),
    "up": (255, 190, 90),
    "down": (200, 140, 255),
    None: (170, 170, 170),
}


def draw_overlay(frame, result, fps, recognizer, show_raw, banner=None):
    stable = result.stable_command
    colour = COLOR.get(stable, (220, 220, 220))

    cv2.rectangle(frame, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), colour, 2)
    cv2.putText(
        frame,
        "keep the whole hand inside this box",
        (ROI_X1 - 4, ROI_Y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        colour,
        1,
    )

    cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 100), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"RAW: {result.raw_direction.upper()} {result.raw_confidence * 100:.1f}%",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    thresholded = (
        label_of(result.thresholded_direction) if result.thresholded_direction else "REJECTED"
    )
    cv2.putText(
        frame,
        f"THRESHOLDED: {thresholded}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame, f"STABLE: {label_of(stable)}", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2
    )

    cv2.putText(
        frame,
        f"Threshold: {recognizer.threshold:.2f}",
        (400, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 170, 170),
        1,
    )
    cv2.putText(
        frame,
        f"Smoothing: {recognizer.min_agreement}-of-{recognizer.window}",
        (400, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 170, 170),
        1,
    )
    cv2.putText(
        frame,
        f"CNN: {result.inference_ms:4.1f} ms",
        (400, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 170, 170),
        1,
    )
    cv2.putText(
        frame, f"FPS: {fps:5.1f}", (400, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1
    )

    if show_raw:
        window_text = " ".join(
            "-" if item is None else item[0].upper() for item in recognizer.history
        )
        cv2.putText(
            frame,
            f"window [{window_text}]",
            (10, FRAME_HEIGHT - 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 220, 150),
            1,
        )

    cv2.rectangle(frame, (0, FRAME_HEIGHT - 78), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
    legend = "  |  ".join(f"{DIRECTION_HELP[d]} = {d.upper()}" for d in DIRECTIONS)
    cv2.putText(
        frame, legend, (10, FRAME_HEIGHT - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
    )
    cv2.putText(
        frame,
        "hand should fill most of the box  |  Q quit  R reset  D debug",
        (10, FRAME_HEIGHT - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
    )

    if banner:
        cv2.rectangle(
            frame, (0, FRAME_HEIGHT - 112), (FRAME_WIDTH, FRAME_HEIGHT - 80), (30, 30, 30), -1
        )
        cv2.putText(
            frame,
            banner,
            (10, FRAME_HEIGHT - 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (90, 230, 255),
            1,
        )
    return frame


class FrameRate:
    """Rolling frame rate over the last `window` frames."""

    def __init__(self, window=30):
        self.times: deque[float] = deque(maxlen=window)
        self.previous = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.times.append(now - self.previous)
        self.previous = now
        average = sum(self.times) / len(self.times)
        return 1.0 / average if average > 0 else 0.0


KEY_ACTIONS = {ord("s"): "skip", ord(" "): "record", ord("r"): "reset", ord("d"): "debug"}


def read_action(window):
    """This frame's action: "quit", "skip", "record", "reset", "debug" or None.

    Closing the window counts as quitting, exactly like Q or ESC.
    """
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        return "quit"
    return KEY_ACTIONS.get(key)


def write_records(out_path, records):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def preview(recognizer):
    capture = open_camera()
    meter = FrameRate()
    show_raw = False
    window = "Real-time direction recognition"
    latencies = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)  # mirror convention, applied first
            result = recognizer.predict(frame)
            latencies.append(result.inference_ms)

            draw_overlay(frame, result, meter.tick(), recognizer, show_raw)
            cv2.imshow(window, frame)

            action = read_action(window)
            if action == "quit":
                break
            if action == "reset":
                recognizer.reset()
            elif action == "debug":
                show_raw = not show_raw
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if latencies:
        print(
            f"frames {len(latencies)} | mean CNN {np.mean(latencies):.2f} ms | "
            f"p95 {np.percentile(latencies, 95):.2f} ms"
        )


def benchmark(recognizer, frames, show):
    """Measure the whole live pipeline: capture, mirror, crop, preprocess, CNN, smooth, draw."""
    capture = open_camera()
    meter = FrameRate(window=frames)
    latencies, loop_times = [], []
    window = "benchmark"
    started = time.perf_counter()
    try:
        for _ in range(frames):
            loop_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            result = recognizer.predict(frame)
            latencies.append(result.inference_ms)
            draw_overlay(frame, result, meter.tick(), recognizer, False)
            if show:
                cv2.imshow(window, frame)
                cv2.waitKey(1)
            loop_times.append((time.perf_counter() - loop_started) * 1000.0)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    total = time.perf_counter() - started
    fps = len(latencies) / total
    frame_ms = 1000.0 / fps if fps else 0.0
    print(f"frames processed      : {len(latencies)}")
    print(f"wall time             : {total:.2f} s")
    print(f"end-to-end frame rate : {fps:.1f} FPS ({frame_ms:.1f} ms per frame)")
    print(
        f"loop time  mean/p95   : {np.mean(loop_times):.2f} / "
        f"{np.percentile(loop_times, 95):.2f} ms"
    )
    print(f"CNN latency mean      : {np.mean(latencies):.2f} ms")
    print(f"CNN latency median    : {np.median(latencies):.2f} ms")
    print(f"CNN latency p95       : {np.percentile(latencies, 95):.2f} ms")
    print(f"CNN share of loop     : {np.mean(latencies) / np.mean(loop_times) * 100:.1f}%")
    print(
        f"smoothing cost        : {recognizer.min_agreement} frames = "
        f"{recognizer.min_agreement * frame_ms:.0f} ms at this frame rate"
    )
    return {
        "frames": len(latencies),
        "fps": fps,
        "frame_ms": frame_ms,
        "loop_ms_mean": float(np.mean(loop_times)),
        "cnn_ms_mean": float(np.mean(latencies)),
        "cnn_ms_median": float(np.median(latencies)),
        "cnn_ms_p95": float(np.percentile(latencies, 95)),
    }


def trial_record(position, expected, result, recognizer):
    """One recorded trial. For the no-command categories the correct answer is NO COMMAND."""
    expected_command = None if expected in NO_COMMAND_CATEGORIES else expected
    correct = result.stable_command == expected_command
    return {
        "trial_number": position + 1,
        "expected_class": expected,
        "expected_command": label_of(expected_command),
        "raw_prediction": result.raw_direction,
        "raw_confidence": round(result.raw_confidence, 6),
        "thresholded_prediction": label_of(result.thresholded_direction),
        "stable_command": label_of(result.stable_command),
        "correct": "yes" if correct else "no",
        "inference_ms": round(result.inference_ms, 3),
        "threshold": recognizer.threshold,
        "smoothing": f"{recognizer.min_agreement}-of-{recognizer.window}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def trials(recognizer, category_list, count, out_path):
    """Guided recorder. SPACE records the current stable command for the prompt.

    For the four directions the correct answer is that direction. For no_hand and idle_hand the
    correct answer is NO COMMAND - those categories measure whether the recognizer stays quiet
    when it should, which matters more than raw accuracy for a model with no reject class.
    """
    capture = open_camera()
    meter = FrameRate()
    window = "Live direction trials"
    records = []
    plan = [(name, index + 1) for name in category_list for index in range(count)]
    position = 0

    print(f"Recording {len(plan)} trials ({count} per category: {', '.join(category_list)}).")
    print("Hold the prompted state in the box, let the reading settle, then press SPACE.")
    try:
        while position < len(plan):
            expected, trial_number = plan[position]
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            result = recognizer.predict(frame)

            wanted = NO_COMMAND_LABEL if expected in NO_COMMAND_CATEGORIES else expected.upper()
            banner = (
                f"TRIAL {position + 1}/{len(plan)}  -  {CATEGORY_HELP[expected]}"
                f"  (want {wanted})  #{trial_number}/{count}"
                f"   SPACE=record  S=skip"
            )
            draw_overlay(frame, result, meter.tick(), recognizer, True, banner)
            cv2.imshow(window, frame)

            action = read_action(window)
            if action == "quit":
                break
            if action not in ("skip", "record"):
                continue
            if action == "record":
                record = trial_record(position, expected, result, recognizer)
                records.append(record)
                mark = "OK " if record["correct"] == "yes" else "MISS"
                print(
                    f"  [{mark}] {position + 1:>3}/{len(plan)}  {expected:<10} "
                    f"raw {result.raw_direction:<6} {result.raw_confidence:.3f}  "
                    f"stable {record['stable_command']}"
                )
            position += 1
            recognizer.reset()
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if not records:
        print("no trials recorded")
        return None

    write_records(out_path, records)
    summarise(records, category_list)
    print(f"saved {os.path.relpath(out_path, PROJECT_ROOT)}")
    return records


def summarise(records, category_list):
    correct = sum(1 for r in records if r["correct"] == "yes")
    print("-" * 72)
    print(f"overall: {correct}/{len(records)} ({correct / len(records) * 100:.1f}%)")
    for name in category_list:
        subset = [r for r in records if r["expected_class"] == name]
        if not subset:
            continue
        hits = sum(1 for r in subset if r["correct"] == "yes")
        confidences = [r["raw_confidence"] for r in subset]
        wrong = Counter(r["stable_command"] for r in subset if r["correct"] == "no")
        detail = f"  got: {dict(wrong)}" if wrong else ""
        print(
            f"  {name:<10} {hits}/{len(subset)} ({hits / len(subset) * 100:5.1f}%)  "
            f"raw conf mean {np.mean(confidences):.3f} min {min(confidences):.3f}"
            f"{detail}"
        )

    # For the no-command categories, the headline number is how often a false command actually
    # survived smoothing - that is what would turn Pac-Man the wrong way.
    quiet = [r for r in records if r["expected_class"] in NO_COMMAND_CATEGORIES]
    if quiet:
        false_commands = [r for r in quiet if r["stable_command"] != NO_COMMAND_LABEL]
        print(
            f"  stable false commands: {len(false_commands)}/{len(quiet)} "
            f"({len(false_commands) / len(quiet) * 100:.1f}%)"
        )
        if false_commands:
            print(
                "    "
                + ", ".join(
                    f"{r['expected_class']}->{r['stable_command']} ({r['raw_confidence']:.3f})"
                    for r in false_commands
                )
            )

    # Raw mistakes that smoothing caught are the direct evidence that smoothing earns its place.
    rescued = [
        r
        for r in records
        if r["expected_class"] in DIRECTIONS
        and r["raw_prediction"] != r["expected_class"]
        and r["correct"] == "yes"
    ]
    if rescued:
        print(f"  raw errors corrected by smoothing: {len(rescued)}")
        for r in rescued:
            print(
                f"    trial {r['trial_number']}: raw {r['raw_prediction']} "
                f"({r['raw_confidence']:.3f}) -> stable {r['stable_command']}"
            )


@dataclass
class TransitionTimer:
    """Times one transition, from the SPACE press until the target command stabilises."""

    source: str
    target: str
    started: float
    frames: int = 0
    raw_seen: float | None = None
    raw_frames: int = 0
    spurious: list[str] = field(default_factory=list)
    done: bool = False

    def observe(self, result: Any, now: float) -> None:
        self.frames += 1
        # The moment the hand first *looks* like the target to the CNN. Everything before this
        # is the human moving; everything after is the recognizer deciding. Only the second
        # part is the recognizer's own latency.
        if self.raw_seen is None and result.raw_direction == self.target:
            self.raw_seen = now
            self.raw_frames = self.frames
        if result.stable_changed and result.stable_command is not None:
            if result.stable_command == self.target:
                self.done = True
            elif result.stable_command != self.source:
                self.spurious.append(result.stable_command)

    def record(self, now: float, recognizer: Any) -> dict[str, Any]:
        elapsed = (now - self.started) * 1000.0
        if self.raw_seen is not None:
            lag_ms = (now - self.raw_seen) * 1000.0
            lag_frames = self.frames - self.raw_frames + 1
        else:
            lag_ms, lag_frames = float("nan"), 0
        return {
            "transition": f"{self.source}->{self.target}",
            "from_direction": self.source,
            "to_direction": self.target,
            "frames_to_stable": self.frames,
            "ms_to_stable": round(elapsed, 1),
            "recognizer_lag_frames": lag_frames,
            "recognizer_lag_ms": round(lag_ms, 1),
            "spurious_commands": "|".join(self.spurious) or "none",
            "spurious_count": len(self.spurious),
            "threshold": recognizer.threshold,
            "smoothing": f"{recognizer.min_agreement}-of-{recognizer.window}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def print_transition(record, position, total):
    mark = "OK " if not record["spurious_count"] else "SPUR"
    spurious = (
        f"  spurious: {record['spurious_commands'].split('|')}" if record["spurious_count"] else ""
    )
    print(
        f"  [{mark}] {position + 1:>3}/{total}  {record['transition']}  "
        f"total {record['ms_to_stable']:.0f} ms  |  recognizer lag {record['recognizer_lag_ms']:.0f} ms "
        f"({record['recognizer_lag_frames']} frames){spurious}"
    )


def summarise_transitions(records):
    times = [r["ms_to_stable"] for r in records]
    lags = [r["recognizer_lag_ms"] for r in records if r["recognizer_lag_frames"]]
    spurious = [r for r in records if r["spurious_count"]]
    print("-" * 72)
    print(f"transitions recorded  : {len(records)}")
    print(
        f"total move+settle     : mean {np.mean(times):.0f} ms | "
        f"median {np.median(times):.0f} ms   (includes your hand movement)"
    )
    if lags:
        print(
            f"RECOGNIZER LAG        : mean {np.mean(lags):.0f} ms | "
            f"median {np.median(lags):.0f} ms | p95 {np.percentile(lags, 95):.0f} ms"
        )
        print(
            "  (raw prediction first matches the target -> stable command issued; "
            "this is the part the game feels)"
        )
    print(f"spurious stable cmds  : {len(spurious)}/{len(records)} transitions")
    if spurious:
        tally = Counter(c for r in spurious for c in r["spurious_commands"].split("|"))
        print(f"  which: {dict(tally)}")


def transitions(recognizer, repeats, out_path):
    """Transition stress test: press SPACE as you START moving to the target gesture.

    Measures how long the intended command takes to stabilise, and whether any *other* stable
    command appears in between. A brief NO COMMAND during the movement is fine and expected; a
    confident wrong direction is not.
    """
    capture = open_camera()
    meter = FrameRate()
    window = "Transition stress test"
    records = []
    plan = [pair for _ in range(repeats) for pair in TRANSITION_PAIRS]
    position = 0
    timer = None

    print(
        f"{len(plan)} transitions. Hold the FROM gesture, then press SPACE at the moment "
        "you start moving to the TO gesture."
    )
    try:
        while position < len(plan):
            source, target = plan[position]
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            result = recognizer.predict(frame)

            if timer is not None:
                timer.observe(result, time.perf_counter())
                if timer.done:
                    record = timer.record(time.perf_counter(), recognizer)
                    records.append(record)
                    print_transition(record, position, len(plan))
                    timer = None
                    position += 1
                    recognizer.reset()
                    continue

            state = "MOVING - waiting for stable" if timer else "hold FROM, SPACE to start"
            banner = (
                f"{position + 1}/{len(plan)}  {source.upper()} -> {target.upper()}"
                f"   [{state}]   S=skip"
            )
            draw_overlay(frame, result, meter.tick(), recognizer, True, banner)
            cv2.imshow(window, frame)

            action = read_action(window)
            if action == "quit":
                break
            if action == "skip":
                timer = None
                position += 1
                recognizer.reset()
            elif action == "record" and timer is None:
                timer = TransitionTimer(source, target, time.perf_counter())
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if not records:
        print("no transitions recorded")
        return None

    write_records(out_path, records)
    summarise_transitions(records)
    print(f"saved {os.path.relpath(out_path, PROJECT_ROOT)}")
    return records


def _selftest_model(recognizer, check):
    """Checkpoint identity, device, the obsolete-checkpoint guard and preprocessing parity."""
    meta = recognizer.metadata
    check(
        "checkpoint",
        meta.get("task") == "pacman_direction_v1",
        f"{meta['architecture']}, task {meta.get('task')}, input {meta['input_size']}",
    )
    check("class mapping", meta["class_to_index"] == CLASS_TO_INDEX, f"{meta['class_to_index']}")
    check(
        "no retired classes",
        not ({"jump", "neutral"} & set(meta["class_to_index"])),
        "no jump/neutral in the checkpoint mapping",
    )

    device = next(recognizer.model.parameters()).device
    check(
        "model device",
        device.type == "cuda",
        f"{device}, eval mode {not recognizer.model.training}",
    )

    # The obsolete checkpoint must be refused, not merely different.
    obsolete = os.path.join(MODEL_DIR, "archive_endless_runner", "best_gesture_model_OBSOLETE.pt")
    if os.path.exists(obsolete):
        try:
            DirectionRecognizer(checkpoint_path=obsolete)
            check("obsolete rejected", False, "the obsolete checkpoint loaded")
        except RuntimeError as error:
            check(
                "obsolete rejected",
                "Refusing to load" in str(error),
                "guard refused the endless-runner checkpoint",
            )
    else:
        check("obsolete rejected", True, "obsolete checkpoint not present")

    # Preprocessing parity: the webcam path must build the same tensor as the evaluation path.
    sample = os.path.join(PROJECT_ROOT, "dataset", "left", "left_00003.jpg")
    bgr = cv2.imread(sample)
    webcam_tensor = recognizer.preprocess(bgr).cpu()
    with Image.open(sample) as image:
        dataset_tensor = eval_transform()(image.convert("RGB")).unsqueeze(0)
    identical = torch.equal(webcam_tensor, dataset_tensor)
    delta = (webcam_tensor - dataset_tensor).abs().max().item()
    check(
        "preprocessing", identical, f"identical to the P3/P5 eval transform (max delta {delta:.2e})"
    )


def _selftest_smoothing(recognizer, check):
    """The rolling-window semantics that stand between one bad frame and a wrong turn."""
    recognizer.reset()
    stable = [recognizer._smooth(d) for d in ["up", "up", None, "up", "up"]]
    check(
        "smoothing majority",
        stable[-1] == "up",
        f"['up','up',None,'up','up'] -> {label_of(stable[-1])}",
    )

    recognizer.reset()
    flicker = [recognizer._smooth(d) for d in ["up", "down", "left", "down", "up"]]
    check("no majority", flicker[-1] is None, f"flickering directions -> {label_of(flicker[-1])}")

    recognizer.reset()
    rejected = [recognizer._smooth(None) for _ in range(5)]
    check(
        "rejections stay quiet",
        all(s is None for s in rejected),
        f"five rejected frames -> {label_of(rejected[-1])} (never fabricates a direction)",
    )

    recognizer.reset()
    for direction in ["down", "down", "down"]:
        recognizer._smooth(direction)
    cleared = [recognizer._smooth(None) for _ in range(3)]
    check(
        "command can clear",
        cleared[-1] is None,
        f"down held, then three rejected frames -> {label_of(cleared[-1])}",
    )

    recognizer.reset()
    needed = 0
    for _ in range(recognizer.window):
        needed += 1
        if recognizer._smooth("left") == "left":
            break
    check(
        "stabilisation cost",
        needed == recognizer.min_agreement,
        f"a clean new direction stabilises after {needed} frames "
        f"({recognizer.min_agreement}-of-{recognizer.window})",
    )
    recognizer.reset()


def _selftest_webcam(recognizer, check):
    """A real frame through the real pipeline: capture, mirror, ROI, inference."""
    capture = open_camera()
    try:
        ok_frame, frame = capture.read()
        if not ok_frame or frame is None:
            check("webcam", False, "no frame returned")
            return
        frame = cv2.flip(frame, 1)
        check(
            "webcam",
            frame.shape[1] == FRAME_WIDTH and frame.shape[0] == FRAME_HEIGHT,
            f"{frame.shape[1]}x{frame.shape[0]} frame, mirrored",
        )
        roi = recognizer.crop_roi(frame)
        check(
            "ROI",
            roi.shape[:2] == (300, 300),
            f"{roi.shape[1]}x{roi.shape[0]} at x[{ROI_X1}:{ROI_X2}] y[{ROI_Y1}:{ROI_Y2}]",
        )
        result = recognizer.predict(frame)
        check(
            "live inference",
            np.isfinite(result.raw_confidence),
            f"raw {result.raw_direction} {result.raw_confidence:.3f} -> "
            f"stable {label_of(result.stable_command)} in {result.inference_ms:.2f} ms",
        )
        check(
            "no-command type",
            result.stable_command is None or result.stable_command in DIRECTIONS,
            f"stable_command is {result.stable_command!r} "
            "(a direction string or None, never 'neutral')",
        )
    finally:
        capture.release()
        cv2.destroyAllWindows()
        recognizer.reset()


def selftest(recognizer):
    """Headless checks: metadata, device, preprocessing parity, smoothing, webcam."""
    results = []

    def check(name, passed, detail):
        results.append((name, passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<20} {detail}")

    print("Self-test - real-time four-direction recognition")
    print("-" * 72)
    _selftest_model(recognizer, check)
    _selftest_smoothing(recognizer, check)
    _selftest_webcam(recognizer, check)

    print("-" * 72)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)}: {', '.join(failed)})")
        return False
    print(f"RESULT: PASS (all {len(results)} checks passed)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Real-time direction recognition demo.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--agreement", type=int, default=DEFAULT_MIN_AGREEMENT)
    parser.add_argument("--trials", action="store_true", help="guided held-gesture recorder")
    parser.add_argument(
        "--idle", action="store_true", help="no-hand and idle-hand recorder (expects NO COMMAND)"
    )
    parser.add_argument(
        "--transitions",
        action="store_true",
        help="transition stress test and stabilisation latency",
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="repeats of the 6 transition pairs for --transitions"
    )
    parser.add_argument("--classes", default=None, help="comma-separated categories for --trials")
    parser.add_argument("--count", type=int, default=TRIALS_PER_CLASS, help="trials per category")
    parser.add_argument("--out", default=None, help="CSV path for recorded results")
    parser.add_argument(
        "--benchmark",
        type=int,
        metavar="FRAMES",
        help="measure the live pipeline over N frames, then exit",
    )
    parser.add_argument("--show", action="store_true", help="display during --benchmark")
    parser.add_argument("--selftest", action="store_true", help="headless verification")
    args = parser.parse_args()

    recognizer = DirectionRecognizer(
        threshold=args.threshold, window=args.window, min_agreement=args.agreement
    )
    print(
        f"checkpoint {os.path.basename(recognizer.metadata.get('task', 'unknown'))} | "
        f"model on {next(recognizer.model.parameters()).device} | "
        f"threshold {args.threshold} | smoothing {args.agreement}-of-{args.window}"
    )

    if args.selftest:
        return 0 if selftest(recognizer) else 1
    if args.benchmark:
        benchmark(recognizer, args.benchmark, args.show)
        return 0
    if args.transitions:
        transitions(recognizer, args.repeats, args.out or TRANSITIONS_CSV)
        return 0
    if args.idle:
        trials(recognizer, list(NO_COMMAND_CATEGORIES), args.count, args.out or IDLE_CSV)
        return 0
    if args.trials:
        selected = (
            [c.strip() for c in args.classes.split(",")] if args.classes else list(DIRECTIONS)
        )
        unknown = [c for c in selected if c not in CATEGORIES]
        if unknown:
            parser.error(f"unknown category/categories: {unknown}; valid: {CATEGORIES}")
        trials(recognizer, selected, args.count, args.out or TRIALS_CSV)
        return 0
    preview(recognizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
