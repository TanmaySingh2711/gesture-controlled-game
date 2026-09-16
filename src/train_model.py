"""Train the direction CNN: MobileNetV2 transfer learning on CUDA.

Four outputs, in the mapping frozen in P3: 0=left (fist), 1=right (palm), 2=up (thumbs up),
3=down (thumbs down). There is no neutral class.

Two-stage strategy:
  Stage 1 - freeze the whole feature extractor, train only the 4-class classifier head.
  Stage 2 - unfreeze the last few feature blocks and fine-tune them at a lower learning rate.

Selection is done purely on validation performance, with **validation loss** as the primary
criterion and validation accuracy as the tie-break. The test split is never loaded here: it
stays untouched until P5.

Why the checkpoint carries its own class mapping
------------------------------------------------
The retired endless-runner model also had four outputs (left/right/jump/neutral), so it
loads into this architecture without any shape error and then means something entirely
different at indices 2 and 3. Every checkpoint written here therefore records its mapping,
and `load_direction_checkpoint()` refuses to return a model whose mapping disagrees with the
active one. That guard is what P5 and P6 should use rather than torch.load directly.

Usage:
    python src/train_model.py
    python src/train_model.py --batch-size 16
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Sized
from dataclasses import dataclass, field
from typing import Any, cast

import matplotlib
from torch.utils.data import DataLoader

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.data_pipeline import (
    BATCH_SIZE,
    CLASS_TO_INDEX,
    CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INDEX_TO_CLASS,
    PROJECT_ROOT,
    SEED,
    get_dataloaders,
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_direction_model.pt")
HISTORY_PATH = os.path.join(MODEL_DIR, "direction_training_history.json")
CURVES_PATH = os.path.join(MODEL_DIR, "direction_training_curves.png")

# SHA-256 of the frozen checkpoint - the exact file trained in P4, evaluated once in P5 and
# live-tested since. `load_direction_checkpoint` refuses any other bytes by default, so a
# swapped, truncated or tampered model file cannot be loaded silently. After a deliberate,
# approved retrain this constant must be updated in the same change as the new checkpoint.
FROZEN_CHECKPOINT_SHA256 = "e57cab3b2fc5ddeba362f7e41586663b02feeed7ca2af49420f7bc7bc255cd3c"

ARCHITECTURE = "mobilenet_v2"
NUM_CLASSES = 4

STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 8
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3  # early stopping during stage 2
UNFREEZE_FROM = 14  # features[14:] -> the last inverted-residual blocks + conv head


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for CNN training.")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark keeps convolution autotuning on; this is a small fixed-size input,
    # so performance matters more than bit-exact determinism across machines.
    torch.backends.cudnn.benchmark = True


def build_model() -> nn.Module:
    """MobileNetV2 with ImageNet weights and a fresh 4-class head."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    # Output order is the frozen project mapping: 0=left, 1=right, 2=up, 3=down.
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return cast(nn.Module, model)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total - trainable, total


def freeze_features(model: nn.Module) -> None:
    for parameter in cast(nn.Sequential, model.features).parameters():
        parameter.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, from_index: int = UNFREEZE_FROM) -> None:
    for index, block in enumerate(cast(nn.Sequential, model.features)):
        if index >= from_index:
            for parameter in block.parameters():
                parameter.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """One pass over a loader. Training when an optimizer is given, else evaluation."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_images, batch_labels in loader:
            images = batch_images.to(device, non_blocking=True)
            labels = batch_labels.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if optimizer is not None:
                if scaler is None:
                    raise ValueError("training needs a GradScaler for float16 autocast")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += batch

    return total_loss / seen, correct / seen


def save_checkpoint(
    model: nn.Module, stage: str, epoch: int, val_acc: float, val_loss: float, batch_size: int
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": ARCHITECTURE,
            "num_classes": NUM_CLASSES,
            "class_to_index": CLASS_TO_INDEX,
            "index_to_class": {str(i): n for i, n in INDEX_TO_CLASS.items()},
            "classes": list(CLASSES),
            "gesture": {"left": "fist", "right": "palm", "up": "like", "down": "dislike"},
            "task": "pacman_direction_v1",
            "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "best_val_accuracy": val_acc,
            "best_val_loss": val_loss,
            "epoch": epoch,
            "stage": stage,
            "seed": SEED,
            "batch_size": batch_size,
            "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
            "selection_metric": "validation loss (accuracy as tie-break)",
            "note": "Four directional outputs for the Pac-Man project. NOT interchangeable with "
            "the retired left/right/jump/neutral checkpoint, which has the same shape.",
        },
        CHECKPOINT_PATH,
    )


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_direction_checkpoint(
    path: str = CHECKPOINT_PATH,
    device: torch.device | None = None,
    *,
    expected_sha256: str | None = FROZEN_CHECKPOINT_SHA256,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the directional model through three independent guards.

    1. **Integrity.** The file's SHA-256 must equal `expected_sha256` (the frozen checkpoint by
       default). Pass `expected_sha256=None` only for a checkpoint that was just trained and
       therefore has no pinned digest yet.
    2. **No code execution.** `torch.load(weights_only=True)` restricts unpickling to tensors
       and plain containers, so a malicious file cannot run code while loading.
    3. **Meaning.** The stored class mapping must match the active one. A shape check is not
       enough: the retired endless-runner checkpoint has four outputs too, so it would load
       cleanly and then report `jump` as `up`.
    """
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"{os.path.basename(path)} is not the frozen checkpoint (sha256 {actual[:16]}..., "
                f"expected {expected_sha256[:16]}...). Refusing to load an unverified model file."
            )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    stored = payload.get("class_to_index")
    if stored != CLASS_TO_INDEX:
        raise RuntimeError(
            f"{os.path.basename(path)} was trained for {stored}, but the active mapping is "
            f"{CLASS_TO_INDEX}. Refusing to load - output indices would mean the wrong "
            "gestures."
        )
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(payload["state_dict"], strict=True)
    if device is not None:
        model = model.to(device)
    return model, payload


@dataclass
class TrainingRun:
    """Everything a training run accumulates across its two stages."""

    history: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] = field(
        default_factory=lambda: {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0, "stage": ""}
    )
    stage_epochs: dict[str, int] = field(
        default_factory=lambda: {"stage1_head": 0, "stage2_finetune": 0}
    )
    early_stopped: bool = False


@dataclass
class StageContext:
    """What every epoch of either stage needs."""

    model: nn.Module
    train_loader: Any
    val_loader: Any
    criterion: nn.Module
    scaler: Any
    device: torch.device
    batch_size: int


def _print_environment(device: torch.device, batch_size: int) -> None:
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA runtime   : {torch.version.cuda}")
    print(f"Device         : {torch.cuda.get_device_name(0)} ({device})")
    print(f"VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"VRAM allocated : {torch.cuda.memory_allocated() / 1024**2:.1f} MB (before model)")
    print(f"Seed           : {SEED} | batch size {batch_size} | input {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("-" * 78)


def _run_stage(context: StageContext, run: TrainingRun, stage: str, epochs: int, lr: float) -> None:
    """One stage of the two-stage recipe, selecting on validation loss as it goes."""
    model = context.model
    if stage == "stage1_head":
        freeze_features(model)
    else:
        unfreeze_last_blocks(model)

    trainable, frozen, total = count_parameters(model)
    print("-" * 78)
    print(f"{stage}: lr {lr}, weight decay {WEIGHT_DECAY}, up to {epochs} epochs")
    print(
        f"  trainable {trainable:,} | frozen {frozen:,} | total {total:,} "
        f"({trainable / total:.1%} trainable)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY
    )
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        train_loss, train_acc = run_epoch(
            model,
            context.train_loader,
            context.criterion,
            context.device,
            optimizer,
            context.scaler,
        )
        val_loss, val_acc = run_epoch(model, context.val_loader, context.criterion, context.device)
        elapsed = time.time() - epoch_started
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < run.best["val_loss"] or (
            val_loss == run.best["val_loss"] and val_acc > run.best["val_acc"]
        )
        if improved:
            run.best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch, "stage": stage}
            save_checkpoint(model, stage, epoch, val_acc, val_loss, context.batch_size)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        run.stage_epochs[stage] = epoch
        run.history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "learning_rate": current_lr,
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"  Epoch {epoch}/{epochs} [{stage}]"
            f"  Train Loss {train_loss:.4f}  Train Acc {train_acc:.4f}"
            f"  |  Val Loss {val_loss:.4f}  Val Acc {val_acc:.4f}"
            f"  |  lr {current_lr:.1e}  {elapsed:.1f}s"
            f"  peak {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB"
            f"{'  *best' if improved else ''}"
        )

        # Early stopping only applies to fine-tuning, as specified.
        if stage == "stage2_finetune" and epochs_without_improvement >= PATIENCE:
            print(f"  early stopping: validation loss did not improve for {PATIENCE} epochs")
            run.early_stopped = True
            break


def _write_history(
    run: TrainingRun,
    batch_size: int,
    breakdown: dict[str, Any],
    total_time: float,
    peak_mb: float,
) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "architecture": ARCHITECTURE,
                "pretrained_weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
                "task": "pacman_direction_v1",
                "class_to_index": CLASS_TO_INDEX,
                "seed": SEED,
                "batch_size": batch_size,
                "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
                "mixed_precision": True,
                "selection_metric": "validation loss (accuracy as tie-break)",
                "stage_config": {
                    "stage1_head": {
                        "max_epochs": STAGE1_EPOCHS,
                        "lr": STAGE1_LR,
                        "weight_decay": WEIGHT_DECAY,
                        "trainable": "classifier only",
                    },
                    "stage2_finetune": {
                        "max_epochs": STAGE2_EPOCHS,
                        "lr": STAGE2_LR,
                        "weight_decay": WEIGHT_DECAY,
                        "patience": PATIENCE,
                        "trainable": f"features[{UNFREEZE_FROM}:] + classifier",
                    },
                },
                "stage_epochs": run.stage_epochs,
                "early_stopped": run.early_stopped,
                "best": run.best,
                "validation_breakdown": breakdown,
                "test_split_used": False,
                "total_seconds": round(total_time, 2),
                "peak_vram_mb": round(peak_mb, 1),
                "epochs": run.history,
            },
            handle,
            indent=1,
        )


def train(batch_size: int) -> tuple[dict[str, Any], int]:
    """The P4 two-stage transfer-learning run. The test split is never loaded."""
    device = torch.device("cuda")
    set_seed()
    _print_environment(device, batch_size)

    # Only train and validation loaders are bound; the test loader is deliberately dropped so it
    # cannot influence anything.
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    print(
        f"train batches {len(train_loader)} ({len(cast(Sized, train_loader.dataset))} images) | "
        f"val batches {len(val_loader)} ({len(cast(Sized, val_loader.dataset))} images)"
    )

    scaler = torch.amp.GradScaler("cuda")
    print(f"AMP enabled    : {scaler.is_enabled()} (float16 autocast on CUDA)")
    context = StageContext(
        build_model().to(device),
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        scaler,
        device,
        batch_size,
    )

    # Validation LOSS is the primary selection criterion; accuracy breaks ties. Loss moves before
    # accuracy does on a 200-image validation set, where one image is worth 0.5%.
    run = TrainingRun()
    started = time.time()
    for stage, epochs, lr in (
        ("stage1_head", STAGE1_EPOCHS, STAGE1_LR),
        ("stage2_finetune", STAGE2_EPOCHS, STAGE2_LR),
    ):
        _run_stage(context, run, stage, epochs, lr)
    total_time = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # The in-memory model is whatever the last epoch produced, which after early stopping is not
    # the selected one. Reload the best checkpoint - written by this very run, so it has no
    # pinned digest yet - before summarising it.
    best_model, _ = load_direction_checkpoint(CHECKPOINT_PATH, device=device, expected_sha256=None)
    breakdown = print_validation_breakdown(validation_breakdown(best_model, val_loader, device))
    del best_model

    _write_history(run, batch_size, breakdown, total_time, peak_mb)
    plot_curves(run.history, run.best)

    print("-" * 78)
    print(
        f"best val accuracy {run.best['val_acc']:.4f} (loss {run.best['val_loss']:.4f}) "
        f"at {run.best['stage']} epoch {run.best['epoch']}"
    )
    print(
        f"epochs run: stage1={run.stage_epochs['stage1_head']}, "
        f"stage2={run.stage_epochs['stage2_finetune']}"
    )
    print(f"peak VRAM {peak_mb:.0f} MB | total time {total_time / 60:.1f} min")
    print(f"checkpoint {os.path.relpath(CHECKPOINT_PATH, PROJECT_ROOT)}")
    print(f"history    {os.path.relpath(HISTORY_PATH, PROJECT_ROOT)}")
    return run.best, batch_size


def validation_breakdown(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> Any:
    """Per-class validation accuracy and the full 4x4 validation confusion counts.

    Validation only - the test split is not touched in P4. The point is to catch a class
    collapse that a high overall number would hide, particularly up <-> down.
    """
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for batch_images, labels in loader:
            images = batch_images.to(device, non_blocking=True)
            predicted = model(images).argmax(dim=1).cpu()
            for true, guess in zip(labels.tolist(), predicted.tolist(), strict=True):
                confusion[true][guess] += 1
    return confusion


def print_validation_breakdown(confusion: Any) -> dict[str, Any]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    print("-" * 78)
    print(f"validation breakdown (validation split only, {total} images)")
    header = "  ".join(f"{name:>6}" for name in CLASSES)
    print(f"  {'true':<8}{header}   correct")
    for index, name in enumerate(CLASSES):
        row = "  ".join(f"{confusion[index][j]:>6}" for j in range(NUM_CLASSES))
        per_class = confusion[index][index] / max(1, confusion[index].sum())
        print(
            f"  {name:<8}{row}   {confusion[index][index]}/"
            f"{confusion[index].sum()} ({per_class:.1%})"
        )
    print(f"  overall  {correct}/{total} ({correct / total:.4f})")

    up, down = CLASS_TO_INDEX["up"], CLASS_TO_INDEX["down"]
    up_as_down = int(confusion[up][down])
    down_as_up = int(confusion[down][up])
    print(f"  up->down {up_as_down}   down->up {down_as_up}   (the orientation-sensitive pair)")
    return {
        "confusion": confusion.tolist(),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "per_class": {
            name: {"correct": int(confusion[i][i]), "total": int(confusion[i].sum())}
            for i, name in enumerate(CLASSES)
        },
        "up_predicted_as_down": up_as_down,
        "down_predicted_as_up": down_as_up,
    }


def plot_curves(history: list[dict[str, Any]], best: dict[str, Any]) -> None:
    """Loss and accuracy over epochs, train and validation only - no test metrics."""
    steps = list(range(1, len(history) + 1))
    boundary = sum(1 for row in history if row["stage"] == "stage1_head")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for axis, (key_train, key_val, title, label) in zip(
        axes,
        (
            ("train_loss", "val_loss", "Loss", "loss"),
            ("train_accuracy", "val_accuracy", "Accuracy", "accuracy"),
        ),
        strict=True,
    ):
        axis.plot(steps, [row[key_train] for row in history], "o-", label=f"train {label}")
        axis.plot(steps, [row[key_val] for row in history], "s-", label=f"validation {label}")
        if 0 < boundary < len(history):
            axis.axvline(boundary + 0.5, color="grey", linestyle="--", linewidth=1)
            axis.text(
                boundary + 0.6, axis.get_ylim()[1], " stage 2", fontsize=8, va="top", color="grey"
            )
        axis.set_title(f"{title} (MobileNetV2, four directions)")
        axis.set_xlabel("epoch (stage 1 then stage 2)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

    best_step = next(
        (
            i + 1
            for i, row in enumerate(history)
            if row["stage"] == best["stage"] and row["epoch"] == best["epoch"]
        ),
        None,
    )
    if best_step:
        for axis in axes:
            axis.axvline(best_step, color="green", alpha=0.35, linewidth=6)

    figure.suptitle(
        f"P4 training - best validation loss {best['val_loss']:.4f}, "
        f"accuracy {best['val_acc']:.4f} "
        f"({best['stage']} epoch {best['epoch']})",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(CURVES_PATH, dpi=130)
    plt.close(figure)
    print(f"curves     {os.path.relpath(CURVES_PATH, PROJECT_ROOT)}")


def verify_checkpoint(batch_size: int) -> bool:
    """Rebuild the architecture from scratch, load the checkpoint, run one val batch."""
    print("-" * 78)
    print("checkpoint verification")
    device = torch.device("cuda")

    # Goes through the guard, so a mapping mismatch fails here rather than silently.
    # A just-trained checkpoint has no pinned digest; the mapping guard still applies.
    fresh, payload = load_direction_checkpoint(CHECKPOINT_PATH, device=device, expected_sha256=None)
    fresh.eval()
    print(f"  architecture {payload['architecture']} | task {payload.get('task')}")
    print(f"  classes {payload['class_to_index']}")
    print(
        f"  input {payload['input_size']} | stage {payload['stage']} "
        f"epoch {payload['epoch']} | val acc {payload['best_val_accuracy']:.4f} "
        f"| val loss {payload['best_val_loss']:.4f}"
    )

    mapping_ok = payload["class_to_index"] == CLASS_TO_INDEX
    retired = sorted({"jump", "neutral"} & set(payload["class_to_index"]))
    print(
        f"  [{'PASS' if mapping_ok else 'FAIL'}] checkpoint mapping equals the active "
        f"mapping (strict state_dict load succeeded)"
    )
    print(
        f"  [{'PASS' if not retired else 'FAIL'}] no retired class names in metadata"
        f"{'' if not retired else ': ' + ', '.join(retired)}"
    )

    _, val_loader, _ = get_dataloaders(batch_size=batch_size)
    images, _labels = next(iter(val_loader))
    images = images.to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = fresh(images)

    shape_ok = outputs.shape == (images.shape[0], NUM_CLASSES)
    finite = bool(torch.isfinite(outputs).all())
    on_cuda = outputs.device.type == "cuda"
    ok = mapping_ok and not retired
    print(
        f"  [{'PASS' if shape_ok else 'FAIL'}] output shape {tuple(outputs.shape)} "
        f"(expected ({images.shape[0]}, {NUM_CLASSES}))"
    )
    print(
        f"  [{'PASS' if finite else 'FAIL'}] finite outputs, range "
        f"[{outputs.min().item():.3f}, {outputs.max().item():.3f}]"
    )
    print(f"  [{'PASS' if on_cuda else 'FAIL'}] model output on {outputs.device}")

    return shape_ok and finite and on_cuda and ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the gesture CNN on CUDA.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    require_cuda()
    batch_size = args.batch_size
    try:
        _best, batch_size = train(batch_size)
    except torch.cuda.OutOfMemoryError:
        if batch_size <= 16:
            raise
        print(f"\nCUDA out of memory at batch size {batch_size}; clearing cache and retrying at 16")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _best, batch_size = train(16)

    ok = verify_checkpoint(batch_size)
    print("-" * 78)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
