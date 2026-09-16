"""Architecture and hyperparameter study for the gesture CNN - train and validation splits only.

Why this exists
---------------
MobileNetV2 and the P4 recipe (stage-2 learning rate 1e-4, unfreezing from block 14) were
chosen once and never compared against anything. This study asks two questions without
touching anything that is frozen:

1. Would another small ImageNet backbone have been a better fit for this task and this laptop
   GPU, once accuracy, calibration, size and latency are weighed together?
2. Was the P4 recipe a lucky point, or does performance hold across nearby settings and seeds?

Rules
-----
* **Train and validation only.** The test split is never iterated; P5 used it exactly once.
* **The frozen checkpoint is never written.** Study weights go to `model/studies/` (gitignored).
* **Three seeds per configuration.** On a 200-image validation set a single image is 0.5%, so a
  single run cannot separate configurations; the report gives mean and spread.
* **Resumable.** Results are written after every run, and a rerun skips finished ones.

Usage::

    python -m src.model_study            # the full study, about 42 runs
    python -m src.model_study --quick    # one short run per architecture, as a smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from torch import nn
from torchvision import models

from src.analyze_evaluation import calibration
from src.data_pipeline import BATCH_SIZE, IMAGE_SIZE, PROJECT_ROOT, get_dataloaders
from src.train_model import (
    NUM_CLASSES,
    PATIENCE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_EPOCHS,
    STAGE2_LR,
    UNFREEZE_FROM,
    WEIGHT_DECAY,
    require_cuda,
    run_epoch,
    set_seed,
)

REPORT_PATH: Final = Path(PROJECT_ROOT) / "reports" / "model_study.json"
QUICK_REPORT_PATH: Final = Path(PROJECT_ROOT) / "reports" / "model_study_quick.json"
STUDY_DIR: Final = Path(PROJECT_ROOT) / "model" / "studies"

SEEDS: Final = (42, 7, 1234)
HP_LEARNING_RATES: Final = (3e-5, 1e-4, 3e-4)
HP_UNFREEZE_FROM: Final = (10, 14, 17)
LATENCY_WARMUP: Final = 30
LATENCY_RUNS: Final = 200

# The frozen model's own P4 numbers, recorded for comparison only.
P4_REFERENCE: Final = {
    "architecture": "mobilenet_v2",
    "seed": 42,
    "val_loss": 0.0362,
    "val_accuracy": 0.995,
    "note": "single P4 run that produced the frozen checkpoint",
}


# --- architectures --------------------------------------------------------------------------
def _new_head(container: nn.Sequential, index: int) -> None:
    layer = container[index]
    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"expected a Linear classifier at index {index}, found {type(layer).__name__}"
        )
    container[index] = nn.Linear(layer.in_features, NUM_CLASSES)


def _mobilenet_v2() -> nn.Module:
    model: Any = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    _new_head(model.classifier, 1)
    return cast("nn.Module", model)


def _mobilenet_v3_small() -> nn.Module:
    model: Any = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    _new_head(model.classifier, 3)
    return cast("nn.Module", model)


def _mobilenet_v3_large() -> nn.Module:
    model: Any = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    _new_head(model.classifier, 3)
    return cast("nn.Module", model)


def _efficientnet_b0() -> nn.Module:
    model: Any = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    _new_head(model.classifier, 1)
    return cast("nn.Module", model)


def _resnet18() -> nn.Module:
    model: Any = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return cast("nn.Module", model)


def _shufflenet_v2() -> nn.Module:
    model: Any = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return cast("nn.Module", model)


@dataclass(frozen=True)
class Architecture:
    """How to build a backbone, which parameters are its head, and what stage 2 unfreezes.

    Stage 2 unfreezes roughly the last quarter of each backbone, mirroring P4's `features[14:]`
    of MobileNetV2's 19 blocks, so every architecture gets a comparable fine-tuning budget.
    """

    name: str
    build: Callable[[], nn.Module]
    head: Callable[[Any], Iterable[nn.Parameter]]
    finetune: Callable[[Any, int | None], list[nn.Module]]


ARCHITECTURES: Final[dict[str, Architecture]] = {
    arch.name: arch
    for arch in (
        Architecture(
            "mobilenet_v2",
            _mobilenet_v2,
            lambda m: m.classifier.parameters(),
            lambda m, start: list(m.features[UNFREEZE_FROM if start is None else start :]),
        ),
        Architecture(
            "mobilenet_v3_small",
            _mobilenet_v3_small,
            lambda m: m.classifier.parameters(),
            lambda m, _start: list(m.features[-4:]),
        ),
        Architecture(
            "mobilenet_v3_large",
            _mobilenet_v3_large,
            lambda m: m.classifier.parameters(),
            lambda m, _start: list(m.features[-5:]),
        ),
        Architecture(
            "efficientnet_b0",
            _efficientnet_b0,
            lambda m: m.classifier.parameters(),
            lambda m, _start: list(m.features[-3:]),
        ),
        Architecture(
            "resnet18",
            _resnet18,
            lambda m: m.fc.parameters(),
            lambda m, _start: [m.layer3, m.layer4],
        ),
        Architecture(
            "shufflenet_v2_x1_0",
            _shufflenet_v2,
            lambda m: m.fc.parameters(),
            lambda m, _start: [m.stage4, m.conv5],
        ),
    )
}


@dataclass(frozen=True)
class RunConfig:
    architecture: str
    seed: int
    stage2_lr: float = STAGE2_LR
    unfreeze_from: int | None = None  # MobileNetV2 only; None means the architecture default

    @property
    def key(self) -> str:
        return f"{self.architecture}|lr={self.stage2_lr:g}|unfreeze={self.unfreeze_from}|seed={self.seed}"


def study_plan(quick: bool) -> list[RunConfig]:
    seeds = SEEDS[:1] if quick else SEEDS
    plan = [RunConfig(name, seed) for name in ARCHITECTURES for seed in seeds]
    if quick:
        return plan
    for learning_rate in HP_LEARNING_RATES:
        for start in HP_UNFREEZE_FROM:
            if (learning_rate, start) == (STAGE2_LR, UNFREEZE_FROM):
                continue  # identical to the MobileNetV2 runs of the architecture study
            plan.extend(RunConfig("mobilenet_v2", seed, learning_rate, start) for seed in seeds)
    return plan


# --- one training run ------------------------------------------------------------------------
@torch.inference_mode()
def validation_outputs(
    model: nn.Module, loader: Any, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    """Float32 validation confidences, correctness and NLL - the inference path the app uses."""
    model.eval()
    confidences, corrects = [], []
    nll_total, seen = 0.0, 0
    for batch_images, batch_labels in loader:
        images = batch_images.to(device, non_blocking=True)
        labels = batch_labels.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images).float(), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        confidences.append(confidence.cpu())
        corrects.append((predicted == labels).cpu())
        chosen = probabilities.gather(1, labels[:, None]).clamp_min(1e-12)
        nll_total += float(-torch.log(chosen).sum())
        seen += labels.numel()
    return (
        torch.cat(confidences).numpy().astype(np.float64),
        torch.cat(corrects).numpy(),
        nll_total / seen,
    )


@torch.inference_mode()
def batch1_latency(model: nn.Module, device: torch.device) -> tuple[float, float]:
    """Median and p95 of single-image float32 forward passes, in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    for _ in range(LATENCY_WARMUP):
        model(dummy)
    torch.cuda.synchronize()
    samples = []
    for _ in range(LATENCY_RUNS):
        started = time.perf_counter()
        model(dummy)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[int(0.95 * len(samples)) - 1]


def train_one(
    config: RunConfig,
    loaders: tuple[Any, Any],
    device: torch.device,
    *,
    stage1_epochs: int,
    stage2_epochs: int,
    save_to: Path | None,
) -> dict[str, Any]:
    """The P4 two-stage recipe on one architecture and seed, selected on validation loss."""
    architecture = ARCHITECTURES[config.architecture]
    set_seed(config.seed)
    model = architecture.build().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    train_loader, val_loader = loaders

    best_loss, best_accuracy, best_epoch, best_stage = math.inf, -1.0, 0, ""
    best_state: dict[str, torch.Tensor] | None = None
    epochs_run = 0
    started = time.perf_counter()
    for stage, epochs, learning_rate in (
        ("stage1_head", stage1_epochs, STAGE1_LR),
        ("stage2_finetune", stage2_epochs, config.stage2_lr),
    ):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in architecture.head(model):
            parameter.requires_grad = True
        if stage == "stage2_finetune":
            for module in architecture.finetune(model, config.unfreeze_from):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        stale = 0
        for epoch in range(1, epochs + 1):
            run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            epochs_run += 1
            if val_loss < best_loss or (val_loss == best_loss and val_accuracy > best_accuracy):
                best_loss, best_accuracy, best_epoch, best_stage = (
                    val_loss,
                    val_accuracy,
                    epoch,
                    stage,
                )
                best_state = {
                    k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stage == "stage2_finetune" and stale >= PATIENCE:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError(f"{config.key}: no epoch completed")

    model.load_state_dict(best_state)
    confidence, correct, nll = validation_outputs(model, val_loader, device)
    calibration_report = calibration(confidence, correct)
    latency_median, latency_p95 = batch1_latency(model, device)
    result = {
        **asdict(config),
        "key": config.key,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "val_ece": calibration_report["ece"],
        "val_nll": nll,
        "best_stage": best_stage,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "train_seconds": train_seconds,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latency_batch1_median_ms": latency_median,
        "latency_batch1_p95_ms": latency_p95,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "result": result}, save_to)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


# --- reporting -------------------------------------------------------------------------------------
def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "runs": float(len(values)),
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_architecture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_recipe: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for run in results:
        default_recipe = run["stage2_lr"] == STAGE2_LR and run["unfreeze_from"] in (
            None,
            UNFREEZE_FROM,
        )
        if default_recipe:
            by_architecture[run["architecture"]].append(run)
        if run["architecture"] == "mobilenet_v2":
            start = UNFREEZE_FROM if run["unfreeze_from"] is None else run["unfreeze_from"]
            by_recipe[(run["stage2_lr"], start)].append(run)

    metrics = ("val_loss", "val_accuracy", "val_ece", "latency_batch1_median_ms", "train_seconds")
    architectures = {
        name: {
            "parameters": runs[0]["parameters"],
            **{metric: _spread([r[metric] for r in runs]) for metric in metrics},
        }
        for name, runs in sorted(by_architecture.items())
    }
    recipes = [
        {
            "stage2_lr": learning_rate,
            "unfreeze_from": start,
            **{
                metric: _spread([r[metric] for r in runs])
                for metric in ("val_loss", "val_accuracy", "val_ece")
            },
        }
        for (learning_rate, start), runs in sorted(by_recipe.items())
    ]
    return {"architectures": architectures, "mobilenet_v2_recipes": recipes}


def write_report(
    path: Path, results: list[dict[str, Any]], quick: bool, notes: list[str] | None = None
) -> None:
    """Rewrite the whole report. `notes` carry provenance - such as why runs were discarded -
    across resumed runs, which would otherwise silently drop anything not rebuilt here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": quick,
        "notes": list(notes or []),
        "rules": [
            "train and validation splits only - the test split is never iterated",
            "the frozen checkpoint is never written",
            f"seeds {list(SEEDS[:1] if quick else SEEDS)} per configuration",
            "selection on validation loss, accuracy as tie-break, exactly as in P4",
        ],
        "test_split_used": False,
        "p4_reference": P4_REFERENCE,
        "summary": summarise(results),
        "runs": results,
    }
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Architecture and hyperparameter study (train/val only)."
    )
    parser.add_argument("--quick", action="store_true", help="one short run per architecture")
    args = parser.parse_args(argv)

    require_cuda()
    device = torch.device("cuda")
    report_path = QUICK_REPORT_PATH if args.quick else REPORT_PATH
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        results = existing.get("runs", [])
        notes = existing.get("notes", [])
    done = {run["key"] for run in results}

    train_loader, val_loader, _test_loader_never_iterated = get_dataloaders(batch_size=BATCH_SIZE)
    plan = study_plan(args.quick)
    print(f"{len(plan)} runs planned, {len(done)} already finished", flush=True)
    for position, config in enumerate(plan, start=1):
        if config.key in done:
            continue
        keep_weights = (
            not args.quick
            and config.seed == SEEDS[0]
            and config.stage2_lr == STAGE2_LR
            and config.unfreeze_from is None
        )
        save_to = (
            STUDY_DIR / f"{config.architecture}_seed{config.seed}.pt" if keep_weights else None
        )
        print(f"[{position}/{len(plan)}] {config.key}", flush=True)
        result = train_one(
            config,
            (train_loader, val_loader),
            device,
            stage1_epochs=1 if args.quick else STAGE1_EPOCHS,
            stage2_epochs=1 if args.quick else STAGE2_EPOCHS,
            save_to=save_to,
        )
        results.append(result)
        write_report(report_path, results, args.quick, notes)
        print(
            f"    val loss {result['val_loss']:.4f}  acc {result['val_accuracy']:.4f}  "
            f"ECE {result['val_ece']:.4f}  {result['parameters'] / 1e6:.2f}M params  "
            f"{result['latency_batch1_median_ms']:.2f} ms  {result['train_seconds']:.0f}s",
            flush=True,
        )
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
