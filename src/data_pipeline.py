"""Reusable PyTorch data pipeline for the CNN-Based Gesture Controlled Gaming Application.

Owns everything between the raw images in `dataset/` and the batches a model consumes:
the frozen class mapping, the deterministic split, the transforms, the Dataset and the
DataLoaders. No model code lives here.

Key decisions (frozen in P3, carried over from the earlier pipeline except for the classes)
------------------------------------------------------------------------------------------
* Input tensors are 3 x 160 x 160. 160px suits hand crops that average 137-152px, is
  cheaper than 224px on a 4 GB GPU, and is a valid MobileNetV2 input size.
* The images in `dataset/` are never modified or duplicated. Resizing and augmentation
  happen at run time, per batch.
* Split is 80/10/10, stratified per class, seed 42, persisted to `data_splits.json`
  so every future run sees exactly the same train/val/test membership.
* Class indices are pinned explicitly (left=0, right=1, up=2, down=3) rather than
  inherited from alphabetical folder order.

ORIENTATION CONSTRAINT - the one rule that must never be relaxed
----------------------------------------------------------------
`up` is thumbs-up and `down` is thumbs-down: the same hand rotated about 180 degrees. The
only thing separating the two classes is vertical orientation, so any transform able to
invert an image vertically would relabel the data rather than augment it.

    FORBIDDEN: RandomVerticalFlip, 90/180 degree rotation, any large rotation.
    ALLOWED  : horizontal flip (a mirrored thumbs-up is still thumbs-up), rotation up to
               about 10 degrees, mild translation, mild scale, mild brightness/contrast.

`check_data_pipeline.py` walks the composed transforms and fails if a vertical flip appears
or if the affine rotation exceeds the safe limit.

INFERENCE CONTRACT - read before writing webcam code in a later objective
-------------------------------------------------------------------------
Real-time webcam inference MUST use exactly the evaluation preprocessing below, via
`inference_transform()`. Any divergence silently degrades accuracy:

    webcam frame
      -> cv2.flip(frame, 1)                  (mirror convention, PROJECT_SPEC.md section 7)
      -> crop the fixed 300x300 ROI          (x 300-600, y 90-390 of the 640x480 frame)
      -> BGR to RGB                          (OpenCV gives BGR, the model was trained on RGB)
      -> resize to 160x160
      -> to float tensor scaled to [0, 1]
      -> normalize with the ImageNet mean/std below
      -> add a batch dimension -> 1 x 3 x 160 x 160
      -> move to CUDA

Never apply training augmentation at inference time.
"""

import json
import os
import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# --- frozen constants -------------------------------------------------------------------
CLASS_TO_INDEX = {"left": 0, "right": 1, "up": 2, "down": 3}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
CLASSES = list(CLASS_TO_INDEX)

IMAGE_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
MAX_SAFE_ROTATION = 10  # degrees; beyond this an augmented thumbs-up drifts toward sideways
TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 50
TEST_PER_CLASS = 50

BATCH_SIZE = 32
NUM_WORKERS = 2  # conservative and verified stable on this Windows machine
PIN_MEMORY = True  # batches are copied to CUDA during training

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "data_splits.json")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "class_mapping.json")


# --- split ------------------------------------------------------------------------------
def build_split(force=False):
    """Create the deterministic stratified split and persist it. Returns the split dict."""
    if os.path.exists(SPLIT_PATH) and not force:
        return load_split()

    rng = random.Random(SEED)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for name in CLASSES:  # fixed class order, not os.listdir order
        folder = os.path.join(DATASET_DIR, name)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        expected = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
        if len(files) != expected:
            raise RuntimeError(f"{name}: expected {expected} images, found {len(files)}")

        shuffled = list(files)
        rng.shuffle(shuffled)
        chunks = {
            "train": shuffled[:TRAIN_PER_CLASS],
            "val": shuffled[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS],
            "test": shuffled[TRAIN_PER_CLASS + VAL_PER_CLASS :],
        }
        for split, names in chunks.items():
            for file_name in sorted(names):
                # Relative POSIX-style path so the file is portable across machines.
                splits[split].append(
                    {
                        "path": f"dataset/{name}/{file_name}",
                        "label": CLASS_TO_INDEX[name],
                        "class": name,
                    }
                )

    payload = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "class_to_index": CLASS_TO_INDEX,
        "counts": {split: len(items) for split, items in splits.items()},
        "splits": splits,
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    with open(MAPPING_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_to_index": CLASS_TO_INDEX,
                "index_to_class": {str(i): n for i, n in INDEX_TO_CLASS.items()},
                "classes": CLASSES,
                "gesture": {"left": "fist", "right": "palm", "up": "like", "down": "dislike"},
                "note": "Frozen in P3 for the Pac-Man direction. Training and real-time "
                "inference must both use this mapping; never rely on alphabetical "
                "folder ordering. There is no neutral class: below-threshold "
                "predictions mean 'no new command', not a fifth class.",
            },
            handle,
            indent=1,
            sort_keys=True,
        )

    return payload


def load_split():
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"{SPLIT_PATH} is missing - run build_split() first.")
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# --- transforms -------------------------------------------------------------------------
def train_transform():
    """Mild, realistic augmentation. Nothing here may change what a gesture means."""
    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Horizontal mirroring changes handedness and viewpoint but never the gesture: a
            # mirrored fist is a fist, and a mirrored thumbs-up still points up. There is
            # deliberately no vertical counterpart - see the orientation constraint above.
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=MAX_SAFE_ROTATION, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform():
    """Deterministic preprocessing for validation, test and real-time inference."""
    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def inference_transform():
    """Alias used by webcam code, so the contract is impossible to miss."""
    return eval_transform()


def denormalize(tensor):
    """Undo ImageNet normalization so a tensor can be viewed as an image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.detach().cpu() * std + mean).clamp(0, 1)


# --- dataset ----------------------------------------------------------------------------
class GestureDataset(Dataset):
    """Reads the images named by one split entry list, applying `transform` on the fly."""

    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        path = os.path.join(PROJECT_ROOT, entry["path"].replace("/", os.sep))
        with Image.open(path) as source:
            image = source.convert("RGB")
        return self.transform(image), entry["label"]


def get_datasets(split=None):
    split = split or build_split()
    return (
        GestureDataset(split["splits"]["train"], train_transform()),
        GestureDataset(split["splits"]["val"], eval_transform()),
        GestureDataset(split["splits"]["test"], eval_transform()),
    )


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    """Train loader shuffles; validation and test stay in a fixed order."""
    train_set, val_set, test_set = get_datasets()
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **common),
        DataLoader(val_set, shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


if __name__ == "__main__":
    payload = build_split(force=True)
    print(f"split written to {os.path.relpath(SPLIT_PATH, PROJECT_ROOT)}")
    print(f"mapping written to {os.path.relpath(MAPPING_PATH, PROJECT_ROOT)}")
    print("counts:", payload["counts"])
