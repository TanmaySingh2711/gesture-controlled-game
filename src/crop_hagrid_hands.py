"""Build a hand-region gesture dataset from HaGRID using the official bounding boxes.

Why this exists
---------------
HaGRID's published images are full scenes, while gameplay feeds the CNN a tight webcam hand
ROI. This script closes that domain gap by cropping each sample to its annotated hand region.

Source (verified when the dataset was first built, and reused unchanged for P2)
------------------------------------------------------------------------------
`cj-mills/hagrid-sample-500k-384p` — 509,323 HaGRID images downscaled to 384p, CC-BY-SA-4.0.
Crucially this archive is a pure downscale, so HaGRID's normalized bounding boxes still line
up with the pixels, and it bundles the official `ann_train_val/*.json` annotations. (The 150k
classification repack tried first was cropped rather than resized, so official boxes do NOT
align with it — that was verified by overlaying boxes on those images, and is the reason the
image source changed to this one.)

The 13.4 GB archive is never downloaded in full: it is read over HTTP range requests, so only
the annotation files and the ~2000 images actually used are transferred.

Class mapping (project class <- HaGRID box label), frozen in P1 for the Pac-Man direction:
    left  <- fist       (closed fist)
    right <- palm       (open palm)
    up    <- like       (thumbs up)
    down  <- dislike    (thumbs down)

Every class draws only from its own gesture's annotation file, so the four classes are
disjoint by construction: a HaGRID image belongs to exactly one gesture folder.

`up` and `down` are the same hand rotated roughly 180 degrees, so they are orientation
sensitive. Each is taken from its own genuine HaGRID class; neither is ever produced by
flipping or rotating the other, and no augmentation of any kind happens in this script.

Crop geometry:
    1. take the largest box carrying the required label,
    2. pad it by PADDING on every side (hand + wrist + a little background),
    3. expand the shorter side to make the region square, never shrinking the longer side,
    4. clamp to the image, shifting rather than distorting.

No resizing, normalization, augmentation or splitting happens here - P3 owns those.

Usage:
    python src/crop_hagrid_hands.py                        # all four classes -> dataset_cropped/
    python src/crop_hagrid_hands.py --classes up down      # only the named classes
    python src/crop_hagrid_hands.py --promote              # ...then move them into dataset/
    python src/crop_hagrid_hands.py --per-class 500
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np

ARCHIVE_URL = ("https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p/"
               "resolve/main/hagrid-sample-500k-384p.zip")
ROOT = "hagrid-sample-500k-384p"

GESTURE_FOR = {"left": "fist", "right": "palm", "up": "like", "down": "dislike"}
CLASSES = list(GESTURE_FOR)

DEFAULT_PER_CLASS = 500
SEED = 42
PADDING = 0.25          # 25% of the box size added on every side
MIN_CROP = 96           # px; smaller crops are too coarse to be useful, so they are replaced
MIN_BOX_FRACTION = 0.10  # skip boxes smaller than this fraction of the image's larger side

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CROPPED_DIR = os.path.join(PROJECT_ROOT, "dataset_cropped")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".hagrid_cache")


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with retries."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.fetched, self.requests = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    chunk = response.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"range request failed after retries: {last}")
        self.pos += len(chunk)
        self.fetched += len(chunk)
        self.requests += 1
        return chunk


def open_archive():
    head = urllib.request.Request(ARCHIVE_URL, method="HEAD")
    size = int(urllib.request.urlopen(head, timeout=120).headers["Content-Length"])
    handle = RangeFile(ARCHIVE_URL, size)
    return handle, zipfile.ZipFile(handle)


def load_annotations(zf, gestures):
    """Official HaGRID train/val annotations, cached locally after the first run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    annotations = {}
    for gesture in gestures:
        cached = os.path.join(CACHE_DIR, f"ann_{gesture}.json")
        if os.path.exists(cached):
            with open(cached, encoding="utf-8") as handle:
                annotations[gesture] = json.load(handle)
            print(f"  cached     {gesture}.json ({len(annotations[gesture])} images)")
            continue
        member = f"{ROOT}/ann_train_val/{gesture}.json"
        print(f"  fetching   {gesture}.json ...", end="", flush=True)
        payload = zf.read(member)
        annotations[gesture] = json.loads(payload)
        with open(cached, "wb") as handle:
            handle.write(payload)
        print(f" {len(annotations[gesture])} images")
    return annotations


def largest_box(record, wanted_label):
    """The biggest box carrying `wanted_label`, or None. Deterministic tie-break by index."""
    best, best_area = None, -1.0
    for index, (box, label) in enumerate(zip(record["bboxes"], record["labels"])):
        if label != wanted_label:
            continue
        area = box[2] * box[3]
        if area > best_area:
            best, best_area = box, area
    return best


def build_candidates(annotations, selected):
    """project class -> deterministic list of (gesture_folder, uuid, box)."""
    candidates = {label: [] for label in selected}
    for project_class in selected:
        gesture = GESTURE_FOR[project_class]
        for uuid in sorted(annotations[gesture]):
            record = annotations[gesture][uuid]
            box = largest_box(record, gesture)
            if box is None:
                continue
            # Implausibly small boxes are too coarse to crop from; skip and replace them.
            if max(box[2], box[3]) < MIN_BOX_FRACTION:
                continue
            candidates[project_class].append((gesture, uuid, box))
    return candidates


def existing_hashes(selected):
    """Hashes of images already in dataset/, so a new crop can never duplicate one.

    Classes being rebuilt are excluded, since their own folders are about to be replaced.
    """
    digests = set()
    if not os.path.isdir(DATASET_DIR):
        return digests
    for label in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, label)
        if label in selected or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.startswith("."):
                continue
            with open(os.path.join(folder, name), "rb") as handle:
                digests.add(hashlib.md5(handle.read()).hexdigest())
    return digests


def square_crop(image, box):
    """Pad the box, square it off, clamp to the image. Returns the crop or None."""
    height, width = image.shape[:2]
    x, y, bw, bh = box
    x1, y1 = x * width, y * height
    x2, y2 = (x + bw) * width, (y + bh) * height

    pad_x, pad_y = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    # Square by expanding the shorter side around the centre; never shrink the longer one.
    side = max(x2 - x1, y2 - y1)
    side = min(side, width, height)           # cannot exceed the image
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    left, top = cx - half, cy - half

    # Shift (not shrink) the window so it stays inside the image.
    left = max(0.0, min(left, width - side))
    top = max(0.0, min(top, height - side))

    left, top, side = int(round(left)), int(round(top)), int(round(side))
    crop = image[top:top + side, left:left + side]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop


def main():
    parser = argparse.ArgumentParser(description="Crop HaGRID samples to annotated hand regions.")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--classes", nargs="+", choices=CLASSES, default=CLASSES,
                        metavar="CLASS",
                        help="build only these project classes (default: all four)")
    parser.add_argument("--promote", action="store_true",
                        help="replace dataset/ with the cropped dataset once it is built")
    parser.add_argument("--keep-cache", action="store_true",
                        help="keep the cached annotation JSONs")
    args = parser.parse_args()
    target = args.per_class
    selected = [label for label in CLASSES if label in args.classes]

    print("HaGRID hand-region cropping")
    print(f"source  : {ARCHIVE_URL.rsplit('/', 1)[-1]} (read over HTTP range requests)")
    print(f"mapping : " + ", ".join(f"{GESTURE_FOR[c]} -> {c}" for c in selected))
    print(f"padding : {PADDING:.0%} per side, squared, seed {SEED}, target {target}/class")
    if selected != CLASSES:
        untouched = ", ".join(c for c in CLASSES if c not in selected)
        print(f"building: {', '.join(selected)} only (not rebuilt: {untouched})")
    print("-" * 78)

    handle, zf = open_archive()
    print(f"archive  : {handle.size / 1e9:.2f} GB, {len(zf.namelist())} members")
    print("annotations:")
    annotations = load_annotations(zf, [GESTURE_FOR[label] for label in selected])

    candidates = build_candidates(annotations, selected)
    print("\ncandidate boxes per class:")
    for label in selected:
        print(f"  {label:<8} ({GESTURE_FOR[label]:<10}) {len(candidates[label])}")
    for label in selected:
        if len(candidates[label]) < target:
            print(f"ERROR: only {len(candidates[label])} candidates for {label}, need {target}")
            return 1

    # Deterministic order; classes are kept disjoint so one frame never feeds two classes.
    rng = random.Random(SEED)
    order = {}
    for label in selected:
        shuffled = list(candidates[label])
        rng.shuffle(shuffled)
        order[label] = shuffled

    for label in selected:
        os.makedirs(os.path.join(CROPPED_DIR, label), exist_ok=True)
        for name in os.listdir(os.path.join(CROPPED_DIR, label)):
            if not name.startswith("."):
                os.remove(os.path.join(CROPPED_DIR, label, name))

    members = set(zf.namelist())
    used_uuids = set()
    # Seeded with the classes we are keeping, so a new crop can never byte-match one of them.
    seen_hashes = existing_hashes(selected)
    if seen_hashes:
        print(f"\nguarding against {len(seen_hashes)} images already in dataset/")
    counts = {label: 0 for label in selected}
    skipped = {label: {"missing": 0, "unreadable": 0, "too_small": 0,
                       "duplicate": 0, "reused": 0} for label in selected}
    sizes = {label: [] for label in selected}

    print("\ncropping:")
    for label in selected:
        gesture_label = GESTURE_FOR[label]
        for gesture, uuid, box in order[label]:
            if counts[label] >= target:
                break
            if uuid in used_uuids:
                skipped[label]["reused"] += 1
                continue
            member = f"{ROOT}/hagrid_500k/train_val_{gesture}/{uuid}.jpg"
            if member not in members:
                skipped[label]["missing"] += 1
                continue
            try:
                payload = zf.read(member)
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                image = None
            if image is None:
                skipped[label]["unreadable"] += 1
                continue

            crop = square_crop(image, box)
            if crop is None or crop.shape[0] < MIN_CROP:
                skipped[label]["too_small"] += 1
                continue

            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                skipped[label]["unreadable"] += 1
                continue
            digest = hashlib.md5(encoded.tobytes()).hexdigest()
            if digest in seen_hashes:
                skipped[label]["duplicate"] += 1
                continue

            path = os.path.join(CROPPED_DIR, label, f"{label}_{counts[label]:05d}.jpg")
            with open(path, "wb") as target_file:
                target_file.write(encoded.tobytes())

            seen_hashes.add(digest)
            used_uuids.add(uuid)
            sizes[label].append(crop.shape[0])
            counts[label] += 1
            if counts[label] % 100 == 0:
                print(f"  {label:<8} {counts[label]}/{target} "
                      f"({handle.fetched / 1e6:.0f} MB fetched)")
        print(f"  {label:<8} done: {counts[label]}/{target}, "
              f"crop side min {min(sizes[label]) if sizes[label] else 0} "
              f"max {max(sizes[label]) if sizes[label] else 0} "
              f"avg {sum(sizes[label]) // len(sizes[label]) if sizes[label] else 0}, "
              f"label='{gesture_label}'")

    print("-" * 78)
    print(f"{'class':<10}{'source':<12}{'images':>8}   skipped")
    for label in selected:
        detail = ", ".join(f"{k}={v}" for k, v in skipped[label].items() if v)
        print(f"{label:<10}{GESTURE_FOR[label]:<12}{counts[label]:>8}   {detail or '-'}")
    print(f"{'TOTAL':<22}{sum(counts.values()):>8}")
    print(f"transferred {handle.fetched / 1e6:.0f} MB in {handle.requests} range requests")

    if not args.keep_cache:
        for gesture in (GESTURE_FOR[label] for label in selected):
            cached = os.path.join(CACHE_DIR, f"ann_{gesture}.json")
            if os.path.exists(cached):
                os.remove(cached)
        if os.path.isdir(CACHE_DIR) and not os.listdir(CACHE_DIR):
            os.rmdir(CACHE_DIR)

    complete = all(counts[label] == target for label in selected)
    if not complete:
        print("RESULT: INCOMPLETE - some classes did not reach the target")
        return 1

    if args.promote:
        print("\npromoting dataset_cropped/ to dataset/ ...")
        for label in selected:
            destination = os.path.join(DATASET_DIR, label)
            os.makedirs(destination, exist_ok=True)
            for name in os.listdir(destination):
                if not name.startswith("."):
                    os.remove(os.path.join(destination, name))
            for name in os.listdir(os.path.join(CROPPED_DIR, label)):
                shutil.move(os.path.join(CROPPED_DIR, label, name),
                            os.path.join(destination, name))
            os.rmdir(os.path.join(CROPPED_DIR, label))
        if os.path.isdir(CROPPED_DIR) and not os.listdir(CROPPED_DIR):
            os.rmdir(CROPPED_DIR)
        print("dataset/ now holds the hand-region crops; dataset_cropped/ removed")

    print("RESULT: PASS - all classes complete and balanced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
