"""Dataset integrity check for the CNN-Based Gesture Controlled Gaming Application.

Inspects the four-class gesture dataset built from HaGRID hand-region crops and reports
per-class counts, image dimensions, unreadable files, class balance, duplicate filenames
and byte-identical duplicates, with a PASS/FAIL verdict.

The active classes are the four Pac-Man directions frozen in P1:

    left   <- HaGRID fist      (closed fist)
    right  <- HaGRID palm      (open palm)
    up     <- HaGRID like      (thumbs up)
    down   <- HaGRID dislike   (thumbs down)

There is no neutral class. The dataset folder is also checked for stray class folders, so
that the retired endless-runner classes (jump, neutral) cannot silently come back.

Crops are squared but not resized, so they do NOT share one resolution; dimensions are
reported as statistics rather than required to be identical. Resizing belongs to P3.

This is a file-level integrity check only - no CNN, no preprocessing, no training analysis.

Usage:
    python src/check_dataset.py
    python src/check_dataset.py --grid                   # also write a sample contact sheet
    python src/check_dataset.py --updown-grid            # thumbs-up vs thumbs-down QA sheet
    python src/check_dataset.py --dir dataset_cropped    # check a candidate dataset first
"""

import argparse
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any

import cv2
import numpy as np

CLASSES = ["left", "right", "up", "down"]
SOURCE_CLASS = {"left": "fist", "right": "palm", "up": "like", "down": "dislike"}
GESTURE_NAME = {
    "left": "closed fist",
    "right": "open palm",
    "up": "thumbs up",
    "down": "thumbs down",
}
TARGET_PER_CLASS = 500

BALANCE_TOLERANCE = 0.10  # largest class may exceed the smallest by at most 10%
MIN_FRACTION_OF_TARGET = 0.90  # a class below 90% of target counts as under-collected
MIN_SIDE = 64  # anything smaller than this is not a usable training image
MAX_ASPECT = 1.15  # hand crops are squared off, so w/h must stay close to 1

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
GRID_PATH = os.path.join(PROJECT_ROOT, "dataset_sample_grid.jpg")

GRID_SAMPLES = 10
GRID_THUMB = 128
GRID_SEED = 7

UPDOWN_PATH = os.path.join(PROJECT_ROOT, "updown_orientation_grid.jpg")
UPDOWN_COLUMNS = 8  # per class, per row block
UPDOWN_ROWS = 3  # so 24 samples of up and 24 of down

results: list[tuple[str, bool, str]] = []


def class_dir(label):
    return os.path.join(DATASET_DIR, label)


def record(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<22} {detail}")


def scan():
    """Read every image once, returning per-class stats."""
    stats = {}
    for label in CLASSES:
        folder = class_dir(label)
        entry: dict[str, Any] = {
            "exists": os.path.isdir(folder),
            "files": [],
            "unreadable": [],
            "sizes": Counter(),
            "tiny": [],
            "skewed": [],
            "not_rgb": [],
            "hashes": defaultdict(list),
        }
        if entry["exists"]:
            entry["files"] = sorted(
                f
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
            )
            for name in entry["files"]:
                path = os.path.join(folder, name)
                if os.path.getsize(path) == 0:
                    entry["unreadable"].append(name)
                    continue
                image = cv2.imread(path)
                if image is None:
                    entry["unreadable"].append(name)
                    continue
                height, width = image.shape[:2]
                if image.ndim != 3 or image.shape[2] != 3:
                    entry["not_rgb"].append(name)
                entry["sizes"][(width, height)] += 1
                if min(width, height) < MIN_SIDE:
                    entry["tiny"].append(name)
                if max(width, height) / max(1, min(width, height)) > MAX_ASPECT:
                    entry["skewed"].append(f"{name} ({width}x{height})")
                with open(path, "rb") as handle:
                    entry["hashes"][
                        hashlib.md5(handle.read(), usedforsecurity=False).hexdigest()
                    ].append(name)
        stats[label] = entry
    return stats


def write_sample_grid(stats):
    """Contact sheet: one row per class, GRID_SAMPLES random thumbnails each."""
    rng = random.Random(GRID_SEED)
    label_width = 110
    rows = []
    for label in CLASSES:
        files = stats[label]["files"]
        picks = rng.sample(files, min(GRID_SAMPLES, len(files)))
        cells = []
        for name in picks:
            image = cv2.imread(os.path.join(class_dir(label), name))
            if image is None:
                image = np.zeros((GRID_THUMB, GRID_THUMB, 3), np.uint8)
            cells.append(cv2.resize(image, (GRID_THUMB, GRID_THUMB)))
        while len(cells) < GRID_SAMPLES:
            cells.append(np.zeros((GRID_THUMB, GRID_THUMB, 3), np.uint8))

        strip = np.hstack(cells)
        caption = np.zeros((GRID_THUMB, label_width, 3), np.uint8)
        cv2.putText(
            caption,
            label.upper(),
            (8, GRID_THUMB // 2 - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            caption,
            SOURCE_CLASS[label],
            (8, GRID_THUMB // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (120, 200, 255),
            1,
        )
        cv2.putText(
            caption,
            GESTURE_NAME[label],
            (8, GRID_THUMB // 2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (160, 160, 160),
            1,
        )
        rows.append(np.hstack([caption, strip]))

    grid = np.vstack(rows)
    cv2.imwrite(GRID_PATH, grid)
    print(
        f"[NOTE] sample grid written to {os.path.relpath(GRID_PATH, PROJECT_ROOT)} "
        f"({grid.shape[1]}x{grid.shape[0]})"
    )


def write_updown_grid(stats):
    """A larger, side-by-side sheet for the highest-risk pair: thumbs up vs thumbs down.

    `up` and `down` are the same hand rotated about 180 degrees, so a mis-cropped or
    mislabelled sample is far harder to spot than for fist vs palm. This sheet puts many
    samples of each class together at a readable size so the thumb direction can be
    checked by eye, which is the only reliable way to catch an orientation problem.
    """
    rng = random.Random(GRID_SEED + 1)
    thumb, pad = 148, 6
    blocks: list[np.ndarray] = []
    for label in ("up", "down"):
        files = stats[label]["files"]
        wanted = UPDOWN_COLUMNS * UPDOWN_ROWS
        picks = rng.sample(files, min(wanted, len(files)))
        header = np.zeros((42, UPDOWN_COLUMNS * (thumb + pad) + pad, 3), np.uint8)
        header[:] = (40, 40, 40)
        cv2.putText(
            header,
            f"{label.upper()}  <-  HaGRID '{SOURCE_CLASS[label]}'  "
            f"({GESTURE_NAME[label]})  -  {len(picks)} random samples",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        blocks.append(header)
        for row in range(UPDOWN_ROWS):
            cells: list[np.ndarray] = []
            for column in range(UPDOWN_COLUMNS):
                index = row * UPDOWN_COLUMNS + column
                cell: np.ndarray = np.zeros((thumb, thumb, 3), np.uint8)
                if index < len(picks):
                    image = cv2.imread(os.path.join(class_dir(label), picks[index]))
                    if image is not None:
                        cell = cv2.resize(image, (thumb, thumb))
                cells.append(
                    cv2.copyMakeBorder(
                        cell, 0, pad, pad, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)
                    )
                )
            strip = np.hstack(cells)
            strip = cv2.copyMakeBorder(strip, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            blocks.append(strip)

    sheet = np.vstack(blocks)
    cv2.imwrite(UPDOWN_PATH, sheet)
    print(
        f"[NOTE] up/down orientation sheet written to "
        f"{os.path.relpath(UPDOWN_PATH, PROJECT_ROOT)} ({sheet.shape[1]}x{sheet.shape[0]})"
    )


def main():
    parser = argparse.ArgumentParser(description="Check gesture dataset integrity.")
    parser.add_argument(
        "--dir", default="dataset", help="dataset folder to check, relative to the project root"
    )
    parser.add_argument(
        "--grid", action="store_true", help="also write a random-sample contact sheet"
    )
    parser.add_argument(
        "--updown-grid",
        action="store_true",
        help="also write the thumbs-up vs thumbs-down orientation sheet",
    )
    args = parser.parse_args()

    # The --dir option rebinds the module paths exactly once, before anything reads them.
    global DATASET_DIR, GRID_PATH  # noqa: PLW0603
    DATASET_DIR = os.path.join(PROJECT_ROOT, args.dir)
    GRID_PATH = os.path.join(PROJECT_ROOT, f"{os.path.basename(args.dir)}_sample_grid.jpg")

    print("Dataset check - CNN-Based Gesture Controlled Gaming Application")
    print(f"dataset root: {DATASET_DIR}")
    print("-" * 78)

    stats = scan()
    counts = {label: len(stats[label]["files"]) for label in CLASSES}
    total = sum(counts.values())

    missing = [label for label in CLASSES if not stats[label]["exists"]]
    record(
        "Class folders",
        not missing,
        f"all four present: {', '.join(CLASSES)}"
        if not missing
        else f"missing: {', '.join(missing)}",
    )

    # Nothing but the four active classes may live here, or a later script that enumerates
    # subfolders would silently pick up a retired class such as jump or neutral.
    present = (
        sorted(
            name
            for name in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, name)) and not name.startswith(".")
        )
        if os.path.isdir(DATASET_DIR)
        else []
    )
    stray = [name for name in present if name not in CLASSES]
    record(
        "No stray classes",
        not stray,
        "dataset holds the four active classes and nothing else"
        if not stray
        else f"unexpected folder(s) in dataset/: {', '.join(stray)}",
    )

    # --- per-class table ------------------------------------------------------------------
    print()
    print(f"{'class':<10}{'source':<12}{'images':>8}{'target':>8}   dimensions (w x h)")
    for label in CLASSES:
        sizes = stats[label]["sizes"]
        if sizes:
            widths = [w for (w, _), n in sizes.items() for _ in range(n)]
            heights = [h for (_, h), n in sizes.items() for _ in range(n)]
            dims = (
                f"{len(sizes)} distinct | w {min(widths)}-{max(widths)} "
                f"(avg {sum(widths) // len(widths)}) | h {min(heights)}-{max(heights)} "
                f"(avg {sum(heights) // len(heights)})"
            )
        else:
            dims = "-"
        print(
            f"{label:<10}{SOURCE_CLASS[label]:<12}{counts[label]:>8}{TARGET_PER_CLASS:>8}   {dims}"
        )
    print(f"{'TOTAL':<30}{total:>8}{TARGET_PER_CLASS * len(CLASSES):>8}")
    print()

    under = [
        f"{label}={counts[label]}"
        for label in CLASSES
        if counts[label] < TARGET_PER_CLASS * MIN_FRACTION_OF_TARGET
    ]
    record(
        "Image counts",
        not under and total > 0,
        f"{total} images, every class at or near the {TARGET_PER_CLASS} target"
        if not under and total > 0
        else f"under target: {', '.join(under) if under else 'dataset is empty'}",
    )

    wrong = [f"{label}={counts[label]}" for label in CLASSES if counts[label] != TARGET_PER_CLASS]
    record(
        "Exact counts",
        not wrong,
        f"every class holds exactly {TARGET_PER_CLASS} images "
        f"({TARGET_PER_CLASS * len(CLASSES)} total)"
        if not wrong
        else f"not exactly {TARGET_PER_CLASS}: {', '.join(wrong)}",
    )

    smallest, largest = min(counts.values()), max(counts.values())
    if largest == 0:
        record("Class balance", False, "no images to balance")
    else:
        spread = (largest - smallest) / largest
        record(
            "Class balance",
            spread <= BALANCE_TOLERANCE,
            f"min {smallest}, max {largest}, spread {spread * 100:.1f}% "
            f"(limit {BALANCE_TOLERANCE * 100:.0f}%)",
        )

    unreadable = [(label, name) for label in CLASSES for name in stats[label]["unreadable"]]
    record(
        "Readable images",
        not unreadable,
        "every image decoded successfully"
        if not unreadable
        else f"{len(unreadable)} unreadable: "
        + ", ".join(f"{label}/{name}" for label, name in unreadable[:5]),
    )

    not_rgb = [(label, name) for label in CLASSES for name in stats[label]["not_rgb"]]
    record(
        "Three-channel RGB",
        not not_rgb and total > 0,
        "every image decodes to 3 colour channels"
        if not not_rgb and total
        else f"{len(not_rgb)} image(s) are not 3-channel: "
        + ", ".join(f"{label}/{name}" for label, name in not_rgb[:5]),
    )

    tiny = [(label, name) for label in CLASSES for name in stats[label]["tiny"]]
    record(
        "Usable dimensions",
        not tiny and total > 0,
        f"every image is at least {MIN_SIDE}px on its shorter side"
        if not tiny and total
        else f"{len(tiny)} image(s) smaller than {MIN_SIDE}px",
    )

    skewed = [(label, name) for label in CLASSES for name in stats[label]["skewed"]]
    record(
        "Square crops",
        not skewed and total > 0,
        f"every crop is square within {MAX_ASPECT:.2f}:1"
        if not skewed and total
        else f"{len(skewed)} crop(s) too far from square: "
        + ", ".join(f"{label}/{name}" for label, name in skewed[:4]),
    )

    name_owners = defaultdict(list)
    for label in CLASSES:
        for name in stats[label]["files"]:
            name_owners[name].append(label)
    dup_names = {n: owners for n, owners in name_owners.items() if len(owners) > 1}
    record(
        "Unique filenames",
        not dup_names,
        "no filename appears in more than one class"
        if not dup_names
        else f"{len(dup_names)} duplicated: " + ", ".join(list(dup_names)[:5]),
    )

    # Byte-identical duplicates within a class, and the same file landing in two classes.
    global_hashes = defaultdict(list)
    for label in CLASSES:
        for digest, names in stats[label]["hashes"].items():
            for name in names:
                global_hashes[digest].append(f"{label}/{name}")
    dup_content = {d: paths for d, paths in global_hashes.items() if len(paths) > 1}
    extra = sum(len(paths) - 1 for paths in dup_content.values())
    record(
        "No duplicate images",
        not dup_content,
        "no byte-identical duplicates"
        if not dup_content
        else f"{extra} duplicate file(s) across {len(dup_content)} image(s): "
        + "; ".join(" == ".join(p) for p in list(dup_content.values())[:3]),
    )

    if args.grid and total:
        print()
        write_sample_grid(stats)

    if args.updown_grid and total:
        if not args.grid:
            print()
        write_updown_grid(stats)

    print("-" * 78)
    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)} checks failed: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} checks passed, {total} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
