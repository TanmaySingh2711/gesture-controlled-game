"""Recover the HaGRID origin of every dataset crop, audit subject leakage, build an unseen-subject set.

Why this exists
---------------
P2 wrote its crops without recording where they came from, which left two questions open:

1. **Subject leakage.** Does one person's hand appear in more than one of train, validation
   and test? If so, part of the 99% test accuracy measures recognising *people* rather than
   gestures.
2. **Generalisation.** How does the frozen model do on people it has never seen?

How lineage is recovered
------------------------
The crop pipeline is deterministic: sorted annotation ids, a seeded shuffle, fixed crop
geometry and JPEG quality 95, hashed with MD5. Replaying that order and hashing each crop
reproduces the bytes on disk exactly, so a hash match identifies the HaGRID image - and its
annotated ``user_id`` - behind every file. Nothing is guessed: a file either matches byte for
byte or is reported as unmatched.

The shuffle draws from one random stream across the classes built in a run, so a class's order
depends on which classes were built alongside it. Each plausible run configuration is probed
and the one whose crops actually match is used.

Outputs
-------
    reports/dataset_lineage.json       dataset file -> HaGRID uuid, user_id, gesture
    reports/subject_leakage.json       people shared between train / val / test
    dataset_external/<class>/*.jpg     unseen-subject evaluation images (gitignored)
    dataset_external/manifest.json     their HaGRID uuids and user_ids

Usage::

    python -m src.audit_hagrid_lineage                  # lineage and leakage audit
    python -m src.audit_hagrid_lineage --external 500   # ...plus an unseen-subject test set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.crop_hagrid_hands import (
    CLASSES,
    GESTURE_FOR,
    MIN_CROP,
    ROOT,
    SEED,
    build_candidates,
    load_annotations,
    open_archive,
    square_crop,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SPLITS_FILE = PROJECT_ROOT / "data_splits.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXTERNAL_DIR = PROJECT_ROOT / "dataset_external"

# Every run configuration that could have produced a class's shuffle order.
RUN_CONFIGS: tuple[tuple[str, ...], ...] = (
    tuple(CLASSES),
    ("up", "down"),
    ("left", "right"),
    ("left",),
    ("right",),
    ("up",),
    ("down",),
)
PROBE = 15  # crops tried per configuration before choosing one
MAX_WALK = 4000  # upper bound on crops walked while matching one class
EXTERNAL_SEED = 2026  # independent of the P2 seed, so the external draw is a fresh sample

# Crop hashes are persisted as they are computed, so a dropped connection part-way through
# never throws away finished work: a rerun picks up exactly where the last one stopped.
DIGEST_CACHE = PROJECT_ROOT / ".hagrid_cache" / "lineage_digests.json"
DIGEST_SAVE_EVERY = 25

Candidate = tuple[str, str, list[float]]


def load_digest_cache() -> dict[str, str | None]:
    if not DIGEST_CACHE.exists():
        return {}
    cache: dict[str, str | None] = json.loads(DIGEST_CACHE.read_text(encoding="utf-8"))
    return cache


def save_digest_cache(cache: dict[str, str | None]) -> None:
    DIGEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DIGEST_CACHE.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(DIGEST_CACHE)


def md5(data: bytes) -> str:
    """Content identity, matching the hash the original crop pipeline used (not security)."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def crop_bytes(
    zf: Any, members: set[str], gesture: str, uuid: str, box: list[float]
) -> bytes | None:
    """Reproduce the P2 crop of one HaGRID image, byte for byte, or None if it was skipped."""
    member = f"{ROOT}/hagrid_500k/train_val_{gesture}/{uuid}.jpg"
    if member not in members:
        return None
    image = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    crop = square_crop(image, box)
    if crop is None or crop.shape[0] < MIN_CROP:
        return None
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return encoded.tobytes() if ok else None


def shuffled_order(
    candidates: dict[str, list[Candidate]], config: tuple[str, ...], label: str
) -> list[Candidate]:
    """The order P2 walked `label` in, had it been built in a run of `config`."""
    rng = random.Random(SEED)
    for name in config:
        order = list(candidates[name])
        rng.shuffle(order)
        if name == label:
            return order
    raise ValueError(f"{label} is not part of {config}")


def dataset_hashes(label: str) -> dict[str, str]:
    return {
        md5(path.read_bytes()): path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted((DATASET_DIR / label).glob("*.jpg"))
    }


def recover_lineage(
    zf: Any,
    members: set[str],
    annotations: dict[str, Any],
    candidates: dict[str, list[Candidate]],
    digest_of: dict[str, str | None],
) -> tuple[dict[str, Any], ...]:
    files: dict[str, dict[str, Any]] = {}
    configs: dict[str, list[str]] = {}
    unmatched: dict[str, list[str]] = {}

    def digest(gesture: str, uuid: str, box: list[float]) -> str | None:
        if uuid not in digest_of:
            data = crop_bytes(zf, members, gesture, uuid, box)
            digest_of[uuid] = md5(data) if data is not None else None
            if len(digest_of) % DIGEST_SAVE_EVERY == 0:
                save_digest_cache(digest_of)
        return digest_of[uuid]

    for label in CLASSES:
        wanted = dataset_hashes(label)
        gesture = GESTURE_FOR[label]
        best: tuple[str, ...] | None = None
        best_hits = 0
        for config in (c for c in RUN_CONFIGS if label in c):
            order = shuffled_order(candidates, config, label)
            hits = sum(1 for _, uuid, box in order[:PROBE] if digest(gesture, uuid, box) in wanted)
            print(f"  {label:<6} probe {'+'.join(config):<22} {hits:>2}/{PROBE} match", flush=True)
            if hits > best_hits:
                best, best_hits = config, hits
            if hits >= PROBE - 3:
                break

        if best is None:
            unmatched[label] = sorted(wanted.values())
            print(f"  {label:<6} no configuration reproduces these crops", flush=True)
            continue

        configs[label] = list(best)
        remaining = dict(wanted)
        for step, (_, uuid, box) in enumerate(
            shuffled_order(candidates, best, label)[:MAX_WALK], 1
        ):
            if not remaining:
                break
            path = remaining.pop(digest(gesture, uuid, box) or "", None)
            if path is not None:
                record = annotations[gesture][uuid]
                files[path] = {"uuid": uuid, "user_id": record.get("user_id"), "gesture": gesture}
            if step % 100 == 0:
                print(
                    f"  {label:<6} walked {step}, matched {len(wanted) - len(remaining)}/"
                    f"{len(wanted)}",
                    flush=True,
                )
        unmatched[label] = sorted(remaining.values())
        print(
            f"  {label:<6} matched {len(wanted) - len(remaining)}/{len(wanted)} "
            f"using run {'+'.join(best)}",
            flush=True,
        )
    return files, configs, unmatched


def load_splits() -> dict[str, list[str]]:
    """Every split's files, normalised to `dataset/<class>/<file>` whatever the JSON layout."""
    data = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))

    def find(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            # The split file also has a `counts` block keyed train/val/test with plain numbers;
            # only a node whose splits are lists of entries is the real one.
            if {"train", "val", "test"} <= set(node) and all(
                isinstance(node[name], list) for name in ("train", "val", "test")
            ):
                return node
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    def paths(node: Any) -> list[str]:
        if isinstance(node, str):
            return [node] if node.lower().endswith((".jpg", ".jpeg", ".png")) else []
        if isinstance(node, dict):
            return [p for value in node.values() for p in paths(value)]
        if isinstance(node, list):
            return [p for value in node for p in paths(value)]
        return []

    splits = find(data)
    if splits is None:
        raise RuntimeError(f"no train/val/test section found in {SPLITS_FILE.name}")
    return {
        name: ["dataset/" + "/".join(Path(p).parts[-2:]) for p in paths(splits[name])]
        for name in ("train", "val", "test")
    }


def leakage_report(files: dict[str, Any], splits: dict[str, list[str]]) -> dict[str, Any]:
    people = {name: Counter[str]() for name in splits}
    without_lineage = 0
    for name, entries in splits.items():
        for entry in entries:
            user = files.get(entry, {}).get("user_id")
            if user is None:
                without_lineage += 1
            else:
                people[name][user] += 1

    pairs: dict[str, Any] = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(people[first]) & set(people[second])
        pairs[f"{first}/{second}"] = {
            "shared_people": len(shared),
            f"{second}_images_from_shared_people": sum(people[second][u] for u in shared),
        }
    return {
        "images_per_split": {n: len(e) for n, e in splits.items()},
        "people_per_split": {n: len(c) for n, c in people.items()},
        "max_images_from_one_person": {n: max(c.values(), default=0) for n, c in people.items()},
        "shared_between_splits": pairs,
        "split_images_without_lineage": without_lineage,
    }


def build_external(
    zf: Any,
    members: set[str],
    annotations: dict[str, Any],
    candidates: dict[str, list[Candidate]],
    files: dict[str, Any],
    per_class: int,
    lineage_complete: bool,
) -> dict[str, Any]:
    """Draw `per_class` crops per class from people who contributed nothing to the dataset."""
    seen_people = {info["user_id"] for info in files.values() if info["user_id"]}
    seen_images = {info["uuid"] for info in files.values()}
    rng = random.Random(EXTERNAL_SEED)
    manifest: dict[str, Any] = {
        "source": "HaGRID sample 500k (384p), train_val annotations",
        "seed": EXTERNAL_SEED,
        "per_class": per_class,
        "rule": "one image per person per class; nobody who appears anywhere in dataset/",
        "excluded_people": len(seen_people),
        "lineage_complete": lineage_complete,
        "files": {},
    }
    for label in CLASSES:
        gesture = GESTURE_FOR[label]
        folder = EXTERNAL_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.jpg"):
            old.unlink()
        order = list(candidates[label])
        rng.shuffle(order)
        taken: set[str] = set()
        count = 0
        for _, uuid, box in order:
            if count >= per_class:
                break
            user = annotations[gesture][uuid].get("user_id")
            if not user or user in seen_people or user in taken or uuid in seen_images:
                continue
            data = crop_bytes(zf, members, gesture, uuid, box)
            if data is None:
                continue
            name = f"{label}_{count:05d}.jpg"
            (folder / name).write_bytes(data)
            manifest["files"][f"{label}/{name}"] = {
                "uuid": uuid,
                "user_id": user,
                "gesture": gesture,
            }
            taken.add(user)
            count += 1
            if count % 100 == 0:
                print(f"  external {label:<6} {count}/{per_class}", flush=True)
        print(
            f"  external {label:<6} done {count}/{per_class} from {len(taken)} people", flush=True
        )
    (EXTERNAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--external",
        type=int,
        default=0,
        metavar="PER_CLASS",
        help="also build an unseen-subject test set with this many per class",
    )
    args = parser.parse_args()

    handle, zf = open_archive()
    members = set(zf.namelist())
    top = Counter("/".join(m.split("/")[:3]) for m in members if m.count("/") >= 2)
    print(f"archive {handle.size / 1e9:.2f} GB, {len(members)} members")
    print("  folders: " + ", ".join(sorted(k for k in top if "ann" in k or "test" in k))[:400])

    annotations = load_annotations(zf, [GESTURE_FOR[c] for c in CLASSES])
    sample = next(iter(annotations[GESTURE_FOR["up"]].values()))
    print(f"  annotation fields: {sorted(sample)}")
    if "user_id" not in sample:
        print("ERROR: these annotations carry no user_id, so subject leakage cannot be audited")
        return 1
    candidates = build_candidates(annotations, CLASSES)

    print("\nrecovering lineage")
    digest_cache = load_digest_cache()
    print(f"  {len(digest_cache)} crop hashes already cached from earlier runs")
    try:
        files, configs, unmatched = recover_lineage(
            zf, members, annotations, candidates, digest_cache
        )
    finally:
        save_digest_cache(digest_cache)
    total = sum(len(dataset_hashes(label)) for label in CLASSES)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_lineage.json").write_text(
        json.dumps(
            {
                "method": "byte-exact replay of the deterministic P2 crop pipeline",
                "matched": len(files),
                "total": total,
                "run_configuration": configs,
                "unmatched": unmatched,
                "files": dict(sorted(files.items())),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report = leakage_report(files, load_splits())
    report["lineage_matched"] = f"{len(files)}/{total}"
    (REPORTS_DIR / "subject_leakage.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("\nsubject leakage")
    print(json.dumps(report, indent=1))

    if args.external:
        print("\nbuilding unseen-subject test set")
        build_external(
            zf,
            members,
            annotations,
            candidates,
            files,
            args.external,
            lineage_complete=len(files) == total,
        )

    print(f"\ntransferred {handle.fetched / 1e6:.0f} MB in {handle.requests} range requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
