"""RETIRED - superseded by `src/crop_hagrid_hands.py`. This script no longer runs.

It built the dataset from the HaGRID *classification repack*, whose images were
geometrically cropped rather than downscaled. HaGRID's official bounding boxes therefore do
not line up with those pixels, which was confirmed by overlaying the boxes on the images.
Because the hand region cannot be located reliably in that source, the project moved to
`cj-mills/hagrid-sample-500k-384p` (a pure downscale) and to annotation-aligned cropping in
`src/crop_hagrid_hands.py`, which is the only supported way to build the dataset.

The file is kept as a record of the earlier approach and of the class mapping it used, which
belonged to the retired endless-runner direction (fist/palm/peace/no_gesture ->
left/right/jump/neutral). Running it would produce full-scene images under the wrong class
names, so main() refuses to execute.

Source it used: https://huggingface.co/datasets/cj-mills/hagrid-classification-512p-no-gesture-150k
    153,735 images from HaGRID (Kapitanov et al.), downscaled to 512p and re-packaged for
    image classification. Licensed CC-BY-SA-4.0, same as the original HaGRID dataset.
    Stored as 8 parquet shards of ~480 MB; this script downloads only as many shards as it
    needs to fill the per-class target, instead of the 716 GB original dataset.

Class mapping it used (retired with the endless-runner direction):
    fist        -> left
    palm        -> right
    peace       -> jump
    no_gesture  -> neutral

The current mapping is fist/palm/like/dislike -> left/right/up/down; see
`src/crop_hagrid_hands.py`.

Usage:
    python src/crop_hagrid_hands.py --promote             # use this instead
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request

import pyarrow.parquet as pq

SOURCE_REPO = "cj-mills/hagrid-classification-512p-no-gesture-150k"
API_TREE = f"https://huggingface.co/api/datasets/{SOURCE_REPO}/tree/main/data"
RESOLVE = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/main/"
CARD_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}/raw/main/README.md"

# project class -> HaGRID source class
MAPPING = {
    "left": "fist",
    "right": "palm",
    "jump": "peace",
    "neutral": "no_gesture",
}
CLASSES = list(MAPPING)

DEFAULT_PER_CLASS = 500
SEED = 42
# Gather more candidates than needed so the random sample is drawn from a real pool
# rather than simply taking the first N rows of the first shard.
POOL_FACTOR = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".hagrid_cache")


def class_dir(label):
    return os.path.join(DATASET_DIR, label)


def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def list_shards():
    """Parquet shard paths in the source repo, in a stable order."""
    entries = http_json(API_TREE)
    shards = sorted(e["path"] for e in entries if e["path"].endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"No parquet shards found in {SOURCE_REPO}")
    return shards


def download(path):
    """Download one shard into the local cache, skipping it if already complete."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, os.path.basename(path))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"  cached  {os.path.basename(path)} ({os.path.getsize(local) / 1e6:.0f} MB)")
        return local

    url = RESOLVE + path
    print(f"  downloading {os.path.basename(path)} ...", end="", flush=True)
    partial = local + ".part"
    with urllib.request.urlopen(url, timeout=600) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, local)
    print(f" done ({os.path.getsize(local) / 1e6:.0f} MB)")
    return local


def source_class_names():
    """Read the label id -> gesture name mapping from the dataset card.

    The parquet files store `label` as a bare int64 with no schema metadata, so the
    authoritative mapping is the class_label block in the dataset card's YAML header.
    Parsing it at run time means the mapping is verified against the source rather than
    hardcoded here.
    """
    with urllib.request.urlopen(CARD_URL, timeout=60) as response:
        card = response.read().decode("utf-8", "replace")

    block = re.search(r"class_label:\s*\n\s*names:\s*\n((?:\s*'?\d+'?:\s*\S+\s*\n)+)", card)
    if not block:
        return None
    pairs = re.findall(r"'?(\d+)'?:\s*(\S+)", block.group(1))
    if not pairs:
        return None
    return [name for _, name in sorted(pairs, key=lambda p: int(p[0]))]


def extension_for(data):
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return None


def main():
    print(__doc__.split("Source it used:")[0].strip())
    print("-" * 78)
    print("REFUSING TO RUN. Build the dataset with:")
    print("    python src/crop_hagrid_hands.py --promote")
    return 2


def _retired_main():
    parser = argparse.ArgumentParser(description="Prepare the HaGRID-based gesture dataset.")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete existing images in the four class folders first")
    parser.add_argument("--keep-cache", action="store_true",
                        help="keep the downloaded parquet shards after preparing the dataset")
    args = parser.parse_args()
    target = args.per_class

    print(f"HaGRID dataset preparation")
    print(f"source : {SOURCE_REPO}")
    print(f"mapping: " + ", ".join(f"{src} -> {dst}" for dst, src in MAPPING.items()))
    print(f"target : {target} images per class, seed {SEED}")
    print("-" * 74)

    # --- guard against silently mixing a new dataset into an old one --------------------
    for label in CLASSES:
        os.makedirs(class_dir(label), exist_ok=True)
        existing = [f for f in os.listdir(class_dir(label)) if not f.startswith(".")]
        if existing and not args.overwrite:
            print(f"ERROR: dataset/{label}/ already contains {len(existing)} files.")
            print("Re-run with --overwrite to rebuild the dataset from scratch.")
            return 1
    if args.overwrite:
        removed = 0
        for label in CLASSES:
            for name in os.listdir(class_dir(label)):
                if not name.startswith("."):
                    os.remove(os.path.join(class_dir(label), name))
                    removed += 1
        if removed:
            print(f"removed {removed} existing image(s) before rebuilding")

    shards = list_shards()
    print(f"source has {len(shards)} parquet shard(s)")

    # --- pass 1: collect candidate row references until every class has a deep pool -----
    wanted = {src: dst for dst, src in MAPPING.items()}
    pool = {label: [] for label in CLASSES}          # label -> [(shard_index, row_index)]
    label_names = None
    used_shards = []

    for shard_index, shard in enumerate(shards):
        if all(len(pool[c]) >= target * POOL_FACTOR for c in CLASSES):
            break
        print(f"shard {shard_index}:")
        local = download(shard)
        table = pq.read_table(local, columns=["label"])

        if label_names is None:
            label_names = source_class_names()
            if not label_names:
                print("ERROR: could not read class names from the source schema.")
                return 1
            print(f"  source classes ({len(label_names)}): {', '.join(label_names)}")
            missing = [s for s in MAPPING.values() if s not in label_names]
            if missing:
                print(f"ERROR: required source class(es) missing from HaGRID subset: {missing}")
                return 1

        labels = table.column("label").to_pylist()
        found = {label: 0 for label in CLASSES}
        for row_index, value in enumerate(labels):
            name = label_names[value]
            if name in wanted:
                project_class = wanted[name]
                pool[project_class].append((shard_index, row_index))
                found[project_class] += 1
        used_shards.append(shard)
        print("  rows found: " + ", ".join(f"{c}={found[c]}" for c in CLASSES)
              + " | pool: " + ", ".join(f"{c}={len(pool[c])}" for c in CLASSES))
        del table

    short = [c for c in CLASSES if len(pool[c]) < target]
    if short:
        print(f"ERROR: not enough source images for {short} "
              f"(have {[len(pool[c]) for c in short]}, need {target})")
        return 1

    # --- reproducible sampling ----------------------------------------------------------
    rng = random.Random(SEED)
    chosen = {label: sorted(rng.sample(pool[label], target)) for label in CLASSES}

    # --- pass 2: write the selected images out ------------------------------------------
    print("-" * 74)
    by_shard = {}
    for label in CLASSES:
        for shard_index, row_index in chosen[label]:
            by_shard.setdefault(shard_index, []).append((row_index, label))

    counters = {label: 0 for label in CLASSES}
    seen_hashes = set()
    duplicates = 0
    written = 0

    for shard_index in sorted(by_shard):
        local = os.path.join(CACHE_DIR, os.path.basename(shards[shard_index]))
        print(f"extracting {len(by_shard[shard_index])} image(s) from shard {shard_index}")
        table = pq.read_table(local, columns=["image"])
        images = table.column("image").to_pylist()
        for row_index, label in sorted(by_shard[shard_index]):
            data = images[row_index]["bytes"]
            if not data:
                print(f"  [warn] empty image at row {row_index}, skipped")
                continue
            digest = hashlib.md5(data).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(digest)
            suffix = extension_for(data)
            if suffix is None:
                print(f"  [warn] unrecognised image format at row {row_index}, skipped")
                continue

            name = f"{label}_{counters[label]:05d}{suffix}"
            path = os.path.join(class_dir(label), name)
            while os.path.exists(path):  # never overwrite
                counters[label] += 1
                name = f"{label}_{counters[label]:05d}{suffix}"
                path = os.path.join(class_dir(label), name)
            with open(path, "wb") as handle:
                handle.write(data)
            counters[label] += 1
            written += 1
        del table, images

    # --- top up anything lost to duplicates or bad rows ----------------------------------
    if any(counters[c] < target for c in CLASSES):
        print("topping up classes that lost images to duplicates or bad rows ...")
        for label in CLASSES:
            if counters[label] >= target:
                continue
            spare = [ref for ref in pool[label] if ref not in set(chosen[label])]
            rng.shuffle(spare)
            for shard_index, row_index in spare:
                if counters[label] >= target:
                    break
                local = os.path.join(CACHE_DIR, os.path.basename(shards[shard_index]))
                table = pq.read_table(local, columns=["image"])
                data = table.column("image")[row_index].as_py()["bytes"]
                del table
                digest = hashlib.md5(data).hexdigest()
                suffix = extension_for(data) if data else None
                if not data or digest in seen_hashes or suffix is None:
                    continue
                seen_hashes.add(digest)
                name = f"{label}_{counters[label]:05d}{suffix}"
                with open(os.path.join(class_dir(label), name), "wb") as handle:
                    handle.write(data)
                counters[label] += 1
                written += 1

    # --- summary --------------------------------------------------------------------------
    print("-" * 74)
    print(f"{'project class':<16}{'source class':<14}{'images':>8}")
    for label in CLASSES:
        print(f"{label:<16}{MAPPING[label]:<14}{counters[label]:>8}")
    print(f"{'TOTAL':<30}{sum(counters.values()):>8}")
    print(f"written: {written} | byte-identical duplicates skipped: {duplicates}")

    if not args.keep_cache:
        for shard in used_shards:
            local = os.path.join(CACHE_DIR, os.path.basename(shard))
            if os.path.exists(local):
                os.remove(local)
        if os.path.isdir(CACHE_DIR) and not os.listdir(CACHE_DIR):
            os.rmdir(CACHE_DIR)
        print("parquet cache removed (use --keep-cache to keep it)")

    balanced = len(set(counters.values())) == 1 and all(c == target for c in counters.values())
    print("RESULT:", "PASS" if balanced else "CHECK",
          "- dataset is balanced" if balanced else "- counts differ from target, inspect above")
    return 0 if balanced else 1


if __name__ == "__main__":
    sys.exit(main())
