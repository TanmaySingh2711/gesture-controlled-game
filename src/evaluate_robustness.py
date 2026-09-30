"""How the frozen model holds up under the camera conditions a real player's room can produce.

The unseen-subject set (`dataset_external/`, 2,000 HaGRID crops from 1,727 people the model never
saw) is re-scored under simulated webcam problems - dim or harsh light, blur, sensor noise, heavy
JPEG compression, low resolution, a colour cast, a tilted hand - each at fixed strengths.

This is measurement only:
- the frozen checkpoint is loaded read-only, through the guarded loader,
- nothing is trained, and no setting (threshold, smoothing) is tuned from these numbers,
- the P3 test split is never touched.

Every corruption is deterministic (noise is seeded per image), so a rerun reproduces the report.
Simulated conditions are not a substitute for live trials with real people and rooms; they show
which conditions are worth testing live, and how gracefully accuracy degrades.

Usage:
    python -m src.evaluate_robustness
    python -m src.evaluate_robustness --device cpu
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from src.data_pipeline import CLASS_TO_INDEX, eval_transform
from src.evaluate_model import BATCH_SIZE, load_model

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
MANIFEST_PATH: Final = PROJECT_ROOT / "dataset_external" / "manifest.json"
REPORT_PATH: Final = PROJECT_ROOT / "reports" / "robustness.json"
LIVE_THRESHOLD: Final = 0.90

Corruption = Callable[[Image.Image, int], Image.Image]


def brightness(factor: float) -> Corruption:
    return lambda image, _seed: ImageEnhance.Brightness(image).enhance(factor)


def contrast(factor: float) -> Corruption:
    return lambda image, _seed: ImageEnhance.Contrast(image).enhance(factor)


def blur(radius: float) -> Corruption:
    return lambda image, _seed: image.filter(ImageFilter.GaussianBlur(radius))


def noise(sigma: float) -> Corruption:
    """Gaussian sensor noise, seeded per image so every run adds the same noise."""

    def apply(image: Image.Image, seed: int) -> Image.Image:
        pixels = np.asarray(image, dtype=np.float32)
        grain = np.random.default_rng(seed).normal(0.0, sigma, pixels.shape)
        return Image.fromarray(np.clip(pixels + grain, 0, 255).astype(np.uint8))

    return apply


def jpeg(quality: int) -> Corruption:
    def apply(image: Image.Image, _seed: int) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    return apply


def low_resolution(side: int) -> Corruption:
    """Down to `side` pixels and back up: a distant hand or a low-resolution webcam."""

    def apply(image: Image.Image, _seed: int) -> Image.Image:
        small = image.resize((side, side), Image.Resampling.BILINEAR)
        return small.resize(image.size, Image.Resampling.BILINEAR)

    return apply


def colour_cast(red: float, blue: float) -> Corruption:
    """Warm (tungsten) or cool (daylight, screen glow) light: scale the red and blue channels."""

    def apply(image: Image.Image, _seed: int) -> Image.Image:
        pixels = np.asarray(image, dtype=np.float32) * np.array([red, 1.0, blue], np.float32)
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    return apply


def rotate(degrees: float) -> Corruption:
    """A tilted hand. Training only ever saw up to 10 degrees, so 20 is outside it."""
    return lambda image, _seed: image.rotate(degrees, resample=Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class Condition:
    name: str
    group: str
    corrupt: Corruption | None  # None is the unmodified image


CONDITIONS: Final[tuple[Condition, ...]] = (
    Condition("clean", "reference", None),
    Condition("dim light (brightness x0.4)", "lighting", brightness(0.4)),
    Condition("dark room (brightness x0.25)", "lighting", brightness(0.25)),
    Condition("harsh light (brightness x1.6)", "lighting", brightness(1.6)),
    Condition("flat light (contrast x0.5)", "lighting", contrast(0.5)),
    Condition("warm light (tungsten cast)", "lighting", colour_cast(1.25, 0.75)),
    Condition("cool light (blue cast)", "lighting", colour_cast(0.8, 1.25)),
    Condition("slight blur (radius 1.5)", "camera", blur(1.5)),
    Condition("strong blur (radius 3)", "camera", blur(3.0)),
    Condition("sensor noise (sigma 15)", "camera", noise(15.0)),
    Condition("heavy sensor noise (sigma 30)", "camera", noise(30.0)),
    Condition("cheap webcam JPEG (quality 15)", "camera", jpeg(15)),
    Condition("low resolution (48 px)", "camera", low_resolution(48)),
    Condition("tilted hand (+20 degrees)", "pose", rotate(20.0)),
    Condition("tilted hand (-20 degrees)", "pose", rotate(-20.0)),
)


class CorruptedImages(Dataset[tuple[torch.Tensor, int]]):
    """The external images, each passed through `corrupt` before the evaluation transform."""

    def __init__(
        self, entries: list[dict[str, Any]], root: Path, corrupt: Corruption | None
    ) -> None:
        self.entries, self.root, self.corrupt = entries, root, corrupt
        self.transform = eval_transform()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[index]
        with Image.open(self.root / entry["path"]) as source:
            image = source.convert("RGB")
        if self.corrupt is not None:
            image = self.corrupt(image, index)
        return self.transform(image), entry["label"]


def load_entries(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = []
    for relative in sorted(manifest["files"]):
        label = relative.split("/", 1)[0]
        if label not in CLASS_TO_INDEX:
            raise ValueError(f"{relative}: unknown class {label!r}")
        entries.append({"path": f"dataset_external/{relative}", "label": CLASS_TO_INDEX[label]})
    return entries


@torch.inference_mode()
def score(
    model: torch.nn.Module, loader: DataLoader[Any], device: torch.device
) -> dict[str, float | int]:
    """Accuracy, and what the live 0.90 threshold would let through."""
    correct, accepted, accepted_wrong, total = 0, 0, 0, 0
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
        confidence, predicted = probabilities.max(dim=1)
        hit = predicted == labels
        passes = confidence >= LIVE_THRESHOLD
        correct += int(hit.sum())
        accepted += int(passes.sum())
        accepted_wrong += int((passes & ~hit).sum())
        total += len(labels)
    return {
        "images": total,
        "accuracy": correct / total,
        "accepted_at_0.90": accepted / total,
        "wrong_and_accepted_at_0.90": accepted_wrong / total,
    }


def evaluate(
    entries: list[dict[str, Any]],
    root: Path,
    device: torch.device,
    conditions: tuple[Condition, ...] = CONDITIONS,
) -> list[dict[str, Any]]:
    model, _payload = load_model(device)
    rows = []
    for condition in conditions:
        loader = DataLoader(
            CorruptedImages(entries, root, condition.corrupt), batch_size=BATCH_SIZE, num_workers=0
        )
        rows.append(
            {"condition": condition.name, "group": condition.group, **score(model, loader, device)}
        )
        print(
            f"  {condition.name:<34} accuracy {rows[-1]['accuracy']:.4f}  "
            f"accepted {rows[-1]['accepted_at_0.90']:.1%}  "
            f"wrong+accepted {rows[-1]['wrong_and_accepted_at_0.90']:.2%}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args(argv)

    if not MANIFEST_PATH.exists():
        print(
            "ERROR: dataset_external/manifest.json not found - build it with "
            "`python -m src.audit_hagrid_lineage --external 500`"
        )
        return 1
    entries = load_entries(MANIFEST_PATH)
    missing = [e["path"] for e in entries if not (PROJECT_ROOT / e["path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} image(s) missing, e.g. {missing[0]}")
        return 1

    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if wants_cuda else "cpu")
    print(f"{len(entries)} unseen-subject images, {len(CONDITIONS)} conditions, on {device}")
    rows = evaluate(entries, PROJECT_ROOT, device)

    clean = rows[0]["accuracy"]
    worst = min(rows, key=lambda row: row["accuracy"])
    report = {
        "source": "dataset_external/ - 2,000 unseen-subject HaGRID crops, frozen model",
        "rules": [
            "measurement only: nothing trained, no setting tuned from these numbers",
            "the P3 test split is never touched",
            "corruptions are deterministic; noise is seeded per image",
        ],
        "live_threshold": LIVE_THRESHOLD,
        "clean_accuracy": clean,
        "worst_condition": worst["condition"],
        "worst_accuracy": worst["accuracy"],
        "conditions": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"worst: {worst['condition']} at {worst['accuracy']:.4f} (clean {clean:.4f})")
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
