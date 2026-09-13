"""Verification for the P3 four-direction data pipeline.

Checks the frozen class mapping, split integrity and determinism, tensor shapes and values,
validation/test determinism, DataLoader configuration, CUDA transfer, and that no transform
can invert an image vertically. Also writes augmentation contact sheets for visual inspection.

The vertical-flip check is the important one. `up` (thumbs up) and `down` (thumbs down) are
the same hand rotated about 180 degrees, so a vertical flip anywhere in the training
transform would silently relabel half the data. The check walks the composed transforms
rather than trusting the source to stay correct.

No model is built, no weights are downloaded, nothing is trained.

Usage:
    python src/check_data_pipeline.py
    python src/check_data_pipeline.py --workers 0
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from data_pipeline import (BATCH_SIZE, CLASSES, CLASS_TO_INDEX, IMAGE_SIZE, IMAGENET_MEAN,
                           IMAGENET_STD, MAPPING_PATH, MAX_SAFE_ROTATION, NUM_WORKERS,
                           PIN_MEMORY, PROJECT_ROOT, SPLIT_PATH, GestureDataset, build_split,
                           denormalize, eval_transform, get_dataloaders, train_transform)

TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS = 400, 50, 50
EXPECTED = {"train": 1600, "val": 200, "test": 200}
REQUIRED_MAPPING = {"left": 0, "right": 1, "up": 2, "down": 3}
GRID_PATH = os.path.join(PROJECT_ROOT, "augmentation_sample_grid.jpg")
UPDOWN_GRID_PATH = os.path.join(PROJECT_ROOT, "augmentation_updown_grid.jpg")

# Any transform whose name contains one of these could invert or heavily rotate an image.
BANNED_TRANSFORMS = ("verticalflip", "randomrotation", "randomperspective")

results = []


def record(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<26} {detail}")


def flatten(transform):
    """Every transform in a (possibly nested) Compose, depth first."""
    children = getattr(transform, "transforms", None)
    if children is None:
        return [transform]
    found = []
    for child in children:
        found.extend(flatten(child))
    return found


def check_mapping(split):
    """The frozen mapping must agree in three places: the code, and both JSON files."""
    record("Mapping constant", CLASS_TO_INDEX == REQUIRED_MAPPING,
           f"{CLASS_TO_INDEX}" if CLASS_TO_INDEX == REQUIRED_MAPPING
           else f"{CLASS_TO_INDEX} != required {REQUIRED_MAPPING}")

    import json
    with open(MAPPING_PATH, encoding="utf-8") as handle:
        mapping_file = json.load(handle)
    from_file = mapping_file["class_to_index"]
    reverse = {int(k): v for k, v in mapping_file["index_to_class"].items()}
    consistent = (from_file == REQUIRED_MAPPING
                  and reverse == {v: k for k, v in REQUIRED_MAPPING.items()}
                  and sorted(from_file.values()) == [0, 1, 2, 3]
                  and len(from_file) == 4)
    record("class_mapping.json", consistent,
           "four classes, indices 0-3, forward and reverse maps agree" if consistent
           else f"file disagrees with the code: {from_file}")

    retired = sorted({"jump", "neutral", "peace", "no_gesture"}
                     & (set(from_file) | set(reverse.values()) | set(CLASS_TO_INDEX)))
    record("No retired classes", not retired,
           "no jump/neutral/peace/no_gesture anywhere in the mapping" if not retired
           else f"retired class names still present: {', '.join(retired)}")

    record("Split file mapping", split["class_to_index"] == REQUIRED_MAPPING,
           "data_splits.json carries the same mapping" if
           split["class_to_index"] == REQUIRED_MAPPING
           else f"split file says {split['class_to_index']}")


def check_regeneration():
    """Seed 42 must reproduce the split exactly - verified, not assumed."""
    with open(SPLIT_PATH, "rb") as handle:
        before = handle.read()
    build_split(force=True)
    with open(SPLIT_PATH, "rb") as handle:
        after = handle.read()
    identical = before == after
    record("Split regeneration", identical,
           f"rebuilding with seed 42 reproduced the file byte-for-byte "
           f"({len(after)} bytes)" if identical
           else "regenerating the split produced a different file")


def check_transform_safety():
    """No transform may be able to turn a thumbs-up into a thumbs-down."""
    train = flatten(train_transform())
    names = [type(t).__name__ for t in train]

    offenders = [n for n in names
                 if n.lower() in BANNED_TRANSFORMS or "vertical" in n.lower()]
    record("No vertical flip", not offenders,
           "no vertical flip or orientation-reversing transform in the training pipeline"
           if not offenders else f"forbidden transform(s): {', '.join(sorted(set(offenders)))}")

    affine = [t for t in train if type(t).__name__ == "RandomAffine"]
    degrees = [max(abs(d) for d in t.degrees) for t in affine]
    bounded = bool(affine) and all(d <= MAX_SAFE_ROTATION for d in degrees)
    record("Rotation bounded", bounded,
           f"RandomAffine rotates at most +/-{max(degrees):.0f} degrees "
           f"(limit {MAX_SAFE_ROTATION})" if affine
           else "no RandomAffine found in the training transform")

    has_hflip = any(type(t).__name__ == "RandomHorizontalFlip" for t in train)
    record("Horizontal flip present", has_hflip,
           "RandomHorizontalFlip(p=0.5) is present and is safe for all four classes"
           if has_hflip else "horizontal flip is missing")

    jitter = any(type(t).__name__ == "ColorJitter" for t in train)
    record("Augmentation present", has_hflip and bool(affine) and jitter,
           "h-flip + affine + colour jitter, applied to the training split only")

    evaluation = [type(t).__name__ for t in flatten(eval_transform())]
    stochastic = [n for n in evaluation if n.startswith("Random") or n == "ColorJitter"]
    record("Eval transform clean", not stochastic,
           f"deterministic only: {' -> '.join(evaluation)}" if not stochastic
           else f"stochastic transform(s) in eval: {', '.join(stochastic)}")


def check_every_class(split):
    """One image of every class must survive both transforms with the right label."""
    detail, ok = [], True
    for name in CLASSES:
        entry = next(e for e in split["splits"]["train"] if e["class"] == name)
        train_tensor, label = GestureDataset([entry], train_transform())[0]
        eval_tensor, _ = GestureDataset([entry], eval_transform())[0]
        good = (tuple(train_tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
                and tuple(eval_tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
                and label == CLASS_TO_INDEX[name]
                and bool(torch.isfinite(train_tensor).all())
                and bool(torch.isfinite(eval_tensor).all()))
        ok = ok and good
        detail.append(f"{name}->{label}")
    record("All four classes", ok,
           " ".join(detail) + f", each 3x{IMAGE_SIZE}x{IMAGE_SIZE} through both transforms")


def check_loader_config(loaders, batch_size, workers):
    train, val, test = loaders["train"], loaders["val"], loaders["test"]
    shuffles = (not isinstance(train.sampler, torch.utils.data.SequentialSampler),
                isinstance(val.sampler, torch.utils.data.SequentialSampler),
                isinstance(test.sampler, torch.utils.data.SequentialSampler))
    record("Loader shuffling", all(shuffles),
           "train shuffled, validation and test in fixed order")
    sizes_ok = all(loader.batch_size == batch_size for loader in (train, val, test))
    record("Loader settings", sizes_ok,
           f"batch {batch_size}, workers {workers}, pin_memory {PIN_MEMORY}, "
           f"persistent_workers {workers > 0}")


def check_split(split):
    counts = {name: len(items) for name, items in split["splits"].items()}
    record("Split sizes", counts == EXPECTED, f"{counts} (expected {EXPECTED})")

    per_class_ok = True
    detail = []
    for name, expected_count in (("train", TRAIN_PER_CLASS), ("val", VAL_PER_CLASS),
                                 ("test", TEST_PER_CLASS)):
        tally = Counter(entry["class"] for entry in split["splits"][name])
        detail.append(f"{name}=" + "/".join(str(tally[c]) for c in CLASSES))
        if any(tally[c] != expected_count for c in CLASSES):
            per_class_ok = False
    record("Per-class split", per_class_ok,
           " ".join(detail) + f"  (order {'/'.join(CLASSES)})")

    paths = {name: {entry["path"] for entry in items}
             for name, items in split["splits"].items()}
    overlaps = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = paths[a] & paths[b]
        if shared:
            overlaps.append(f"{a}&{b}={len(shared)}")
    record("Split overlap", not overlaps, "no image appears in two splits" if not overlaps
           else ", ".join(overlaps))

    all_assigned = [entry["path"] for items in split["splits"].values() for entry in items]
    unique = set(all_assigned)
    on_disk = set()
    for name in CLASSES:
        folder = os.path.join(PROJECT_ROOT, "dataset", name)
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(".jpg"):
                on_disk.add(f"dataset/{name}/{file_name}")
    complete = len(all_assigned) == len(unique) == len(on_disk) == 2000 and unique == on_disk
    record("Full coverage", complete,
           f"{len(all_assigned)} assignments, {len(unique)} unique, {len(on_disk)} on disk")

    labels_ok = all(entry["label"] == CLASS_TO_INDEX[entry["class"]]
                    for items in split["splits"].values() for entry in items)
    record("Label mapping", labels_ok,
           "left=0, right=1, up=2, down=3 applied consistently" if labels_ok
           else "label does not match the frozen mapping")


def check_batches(loaders, train_batches=3):
    """Shapes, finiteness and label range across batches of each loader.

    Validation and test are swept in full: they are unshuffled and the split file is
    ordered by class, so the first few batches would only ever contain `left` and `right`
    and the check would never see `up` or `down`. Train is shuffled, so a few batches
    already mix all four classes.
    """
    for name, loader in loaders.items():
        limit = train_batches if name == "train" else None
        seen = []
        for index, batch in enumerate(loader):
            seen.append(batch)
            if limit is not None and index + 1 >= limit:
                break

        images, labels = seen[0]
        shape_ok = all(b[0].ndim == 4 and b[0].shape[1] == 3
                       and b[0].shape[2] == IMAGE_SIZE and b[0].shape[3] == IMAGE_SIZE
                       and b[1].shape[0] == b[0].shape[0] for b in seen)
        record(f"{name} batch shape", shape_ok,
               f"images {tuple(images.shape)}, labels {tuple(labels.shape)}, "
               f"dtype {images.dtype} over {len(seen)} batches")

        finite = all(bool(torch.isfinite(b[0]).all()) for b in seen)
        low = min(float(b[0].min()) for b in seen)
        high = max(float(b[0].max()) for b in seen)
        record(f"{name} values finite", finite,
               f"no NaN/Inf across {len(seen)} batches, range [{low:.3f}, {high:.3f}], "
               f"mean {float(images.mean()):.3f}")

        labels_ok = all(bool(b[1].dtype in (torch.int64, torch.int32)
                             and b[1].min() >= 0 and b[1].max() <= 3) for b in seen)
        from collections import Counter as _Counter
        tally = _Counter(int(v) for b in seen for v in b[1].tolist())
        observed = sorted(tally)
        record(f"{name} labels valid", labels_ok,
               f"dtype {labels.dtype}, values {observed} within 0-3, counts "
               + "/".join(f"{CLASSES[i]}:{tally[i]}" for i in observed))


def check_determinism(split):
    entries = split["splits"]["val"][:8]
    dataset = GestureDataset(entries, eval_transform())
    first = torch.stack([dataset[i][0] for i in range(len(entries))])
    second = torch.stack([dataset[i][0] for i in range(len(entries))])
    identical = torch.equal(first, second)
    record("Eval determinism", identical,
           "validation preprocessing is bit-identical across loads" if identical
           else "validation preprocessing differs between loads")

    train_entries = split["splits"]["train"][:8]
    train_ds = GestureDataset(train_entries, train_transform())
    a = torch.stack([train_ds[i][0] for i in range(len(train_entries))])
    b = torch.stack([train_ds[i][0] for i in range(len(train_entries))])
    varies = not torch.equal(a, b)
    record("Train augmentation live", varies,
           "training samples differ between loads, as expected" if varies
           else "training augmentation produced identical tensors")


def check_cuda(loader):
    if not torch.cuda.is_available():
        record("CUDA transfer", False, "CUDA is not available")
        return
    device = torch.device("cuda")
    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    torch.cuda.synchronize()
    ok = images.device.type == "cuda" and labels.device.type == "cuda"
    record("CUDA transfer", ok,
           f"images {images.device}, labels {labels.device}, "
           f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated")
    del images, labels
    torch.cuda.empty_cache()


def write_augmentation_grid(split, variants=6):
    """One row per class: the original crop, then several augmented versions of it."""
    transform = train_transform()
    rows = []
    torch.manual_seed(SEED_FOR_GRID)
    for name in CLASSES:
        entry = next(e for e in split["splits"]["train"] if e["class"] == name)
        path = os.path.join(PROJECT_ROOT, entry["path"].replace("/", os.sep))

        original = cv2.imread(path)
        original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.putText(original, "original", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cells = [original]

        from PIL import Image
        with Image.open(path) as image:
            image = image.convert("RGB")
            for _ in range(variants):
                tensor = denormalize(transform(image))
                array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cells.append(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

        strip = np.zeros((IMAGE_SIZE, 110, 3), np.uint8)
        cv2.putText(strip, name.upper(), (6, IMAGE_SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        rows.append(np.hstack([strip] + cells))

    grid = np.vstack(rows)
    cv2.imwrite(GRID_PATH, grid)
    print(f"[NOTE] augmentation grid written to "
          f"{os.path.relpath(GRID_PATH, PROJECT_ROOT)} ({grid.shape[1]}x{grid.shape[0]})")


def write_updown_grid(split, rows_per_class=3, variants=7):
    """A focused sheet: many augmented thumbs-up and thumbs-down, for orientation QA.

    This is the pair the augmentation could plausibly break, so it gets more samples and
    more variants than the overview grid. Training-split images only - the test split is
    reserved for P5 and must not influence any augmentation decision.
    """
    from PIL import Image
    transform = train_transform()
    torch.manual_seed(SEED_FOR_GRID + 1)
    blocks = []
    for name in ("up", "down"):
        entries = [e for e in split["splits"]["train"] if e["class"] == name][:rows_per_class]
        header = np.zeros((34, (variants + 1) * IMAGE_SIZE, 3), np.uint8)
        header[:] = (40, 40, 40)
        cv2.putText(header, f"{name.upper()}  original + {variants} augmented   "
                            f"(h-flip, +/-{MAX_SAFE_ROTATION} deg, 5% shift, 0.9-1.1 scale, "
                            f"brightness/contrast 0.2)",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        blocks.append(header)
        for entry in entries:
            path = os.path.join(PROJECT_ROOT, entry["path"].replace("/", os.sep))
            original = cv2.resize(cv2.imread(path), (IMAGE_SIZE, IMAGE_SIZE))
            cv2.putText(original, "original", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            cells = [original]
            with Image.open(path) as image:
                image = image.convert("RGB")
                for _ in range(variants):
                    tensor = denormalize(transform(image))
                    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    cells.append(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            blocks.append(np.hstack(cells))

    sheet = np.vstack(blocks)
    cv2.imwrite(UPDOWN_GRID_PATH, sheet)
    print(f"[NOTE] up/down augmentation sheet written to "
          f"{os.path.relpath(UPDOWN_GRID_PATH, PROJECT_ROOT)} "
          f"({sheet.shape[1]}x{sheet.shape[0]})")


SEED_FOR_GRID = 7


def main():
    parser = argparse.ArgumentParser(description="Verify the gesture data pipeline.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("Data pipeline check - CNN-Based Gesture Controlled Gaming Application")
    print(f"input {IMAGE_SIZE}x{IMAGE_SIZE} | batch {args.batch_size} | "
          f"workers {args.workers} | split file {os.path.basename(SPLIT_PATH)}")
    print("-" * 82)

    split = build_split()
    check_mapping(split)

    print()
    check_split(split)

    print()
    check_regeneration()

    print()
    check_transform_safety()

    print()
    check_every_class(split)

    loaders = {}
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    print()
    check_loader_config(loaders, args.batch_size, args.workers)

    print()
    check_batches(loaders)

    print()
    check_determinism(split)

    print()
    check_cuda(train_loader)

    print()
    write_augmentation_grid(split)
    write_updown_grid(split)

    print("-" * 82)
    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)} checks failed: "
              f"{', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
