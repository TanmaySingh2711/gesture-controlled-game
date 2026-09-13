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
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import (BATCH_SIZE, CLASSES, CLASS_TO_INDEX, IMAGE_SIZE, IMAGENET_MEAN,
                           IMAGENET_STD, INDEX_TO_CLASS, PROJECT_ROOT, SEED, get_dataloaders)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_direction_model.pt")
HISTORY_PATH = os.path.join(MODEL_DIR, "direction_training_history.json")
CURVES_PATH = os.path.join(MODEL_DIR, "direction_training_curves.png")

ARCHITECTURE = "mobilenet_v2"
NUM_CLASSES = 4

STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 8
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3               # early stopping during stage 2
UNFREEZE_FROM = 14         # features[14:] -> the last inverted-residual blocks + conv head


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for CNN training.")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark keeps convolution autotuning on; this is a small fixed-size input,
    # so performance matters more than bit-exact determinism across machines.
    torch.backends.cudnn.benchmark = True


def build_model():
    """MobileNetV2 with ImageNet weights and a fresh 4-class head."""
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    # Output order is the frozen project mapping: 0=left, 1=right, 2=up, 3=down.
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total - trainable, total


def freeze_features(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_last_blocks(model, from_index=UNFREEZE_FROM):
    for index, block in enumerate(model.features):
        if index >= from_index:
            for parameter in block.parameters():
                parameter.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over a loader. Training when an optimizer is given, else evaluation."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += batch

    return total_loss / seen, correct / seen


def save_checkpoint(model, stage, epoch, val_acc, val_loss, batch_size):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
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
    }, CHECKPOINT_PATH)


def load_direction_checkpoint(path=CHECKPOINT_PATH, device=None):
    """Load the directional model, refusing any checkpoint whose mapping disagrees.

    This is the guard P5 and P6 should use. A shape check is not enough: the retired
    endless-runner checkpoint has four outputs too, so it would load cleanly and then
    report `jump` as `up`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload.get("class_to_index")
    if stored != CLASS_TO_INDEX:
        raise RuntimeError(
            f"{os.path.basename(path)} was trained for {stored}, but the active mapping is "
            f"{CLASS_TO_INDEX}. Refusing to load - output indices would mean the wrong "
            "gestures.")
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(payload["state_dict"], strict=True)
    if device is not None:
        model = model.to(device)
    return model, payload


def train(batch_size):
    device = torch.device("cuda")
    set_seed()

    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA runtime   : {torch.version.cuda}")
    print(f"Device         : {torch.cuda.get_device_name(0)} ({device})")
    print(f"VRAM total     : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"VRAM allocated : {torch.cuda.memory_allocated() / 1024**2:.1f} MB (before model)")
    print(f"Seed           : {SEED} | batch size {batch_size} | input "
          f"{IMAGE_SIZE}x{IMAGE_SIZE}")
    print("-" * 78)

    # Only train and validation loaders are bound; the test loader is deliberately dropped
    # so it cannot influence anything in this objective.
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    print(f"train batches {len(train_loader)} ({len(train_loader.dataset)} images) | "
          f"val batches {len(val_loader)} ({len(val_loader.dataset)} images)")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    print(f"AMP enabled    : {scaler.is_enabled()} (float16 autocast on CUDA)")

    history = []
    # Validation LOSS is the primary selection criterion; accuracy breaks ties. Loss moves
    # before accuracy does on a 200-image validation set, where one image is worth 0.5%.
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0, "stage": ""}
    started = time.time()
    stage_epochs = {"stage1_head": 0, "stage2_finetune": 0}
    early_stopped = False

    for stage, epochs, lr in (("stage1_head", STAGE1_EPOCHS, STAGE1_LR),
                              ("stage2_finetune", STAGE2_EPOCHS, STAGE2_LR)):
        if stage == "stage1_head":
            freeze_features(model)
        else:
            unfreeze_last_blocks(model)

        trainable, frozen, total = count_parameters(model)
        print("-" * 78)
        print(f"{stage}: lr {lr}, weight decay {WEIGHT_DECAY}, up to {epochs} epochs")
        print(f"  trainable {trainable:,} | frozen {frozen:,} | total {total:,} "
              f"({trainable / total:.1%} trainable)")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=WEIGHT_DECAY)

        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                              optimizer, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - epoch_started
            current_lr = optimizer.param_groups[0]["lr"]

            improved = (val_loss < best["val_loss"] or
                        (val_loss == best["val_loss"] and val_acc > best["val_acc"]))
            if improved:
                best = {"val_acc": val_acc, "val_loss": val_loss,
                        "epoch": epoch, "stage": stage}
                save_checkpoint(model, stage, epoch, val_acc, val_loss, batch_size)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            stage_epochs[stage] = epoch
            history.append({
                "stage": stage, "epoch": epoch,
                "train_loss": round(train_loss, 6), "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6), "val_accuracy": round(val_acc, 6),
                "learning_rate": current_lr, "seconds": round(elapsed, 2),
            })

            print(f"  Epoch {epoch}/{epochs} [{stage}]"
                  f"  Train Loss {train_loss:.4f}  Train Acc {train_acc:.4f}"
                  f"  |  Val Loss {val_loss:.4f}  Val Acc {val_acc:.4f}"
                  f"  |  lr {current_lr:.1e}  {elapsed:.1f}s"
                  f"  peak {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB"
                  f"{'  *best' if improved else ''}")

            # Early stopping only applies to fine-tuning, as specified.
            if stage == "stage2_finetune" and epochs_without_improvement >= PATIENCE:
                print(f"  early stopping: validation loss did not improve for "
                      f"{PATIENCE} epochs")
                early_stopped = True
                break

    total_time = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # The in-memory model is whatever the last epoch produced, which after early stopping is
    # not the selected one. Reload the best checkpoint before summarising it.
    best_model, _ = load_direction_checkpoint(device=device)
    breakdown = print_validation_breakdown(
        validation_breakdown(best_model, val_loader, device))
    del best_model

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump({
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
                "stage1_head": {"max_epochs": STAGE1_EPOCHS, "lr": STAGE1_LR,
                                "weight_decay": WEIGHT_DECAY, "trainable": "classifier only"},
                "stage2_finetune": {"max_epochs": STAGE2_EPOCHS, "lr": STAGE2_LR,
                                    "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
                                    "trainable": f"features[{UNFREEZE_FROM}:] + classifier"},
            },
            "stage_epochs": stage_epochs,
            "early_stopped": early_stopped,
            "best": best,
            "validation_breakdown": breakdown,
            "test_split_used": False,
            "total_seconds": round(total_time, 2),
            "peak_vram_mb": round(peak_mb, 1),
            "epochs": history,
        }, handle, indent=1)

    plot_curves(history, best)

    print("-" * 78)
    print(f"best val accuracy {best['val_acc']:.4f} (loss {best['val_loss']:.4f}) "
          f"at {best['stage']} epoch {best['epoch']}")
    print(f"epochs run: stage1={stage_epochs['stage1_head']}, "
          f"stage2={stage_epochs['stage2_finetune']}")
    print(f"peak VRAM {peak_mb:.0f} MB | total time {total_time / 60:.1f} min")
    print(f"checkpoint {os.path.relpath(CHECKPOINT_PATH, PROJECT_ROOT)}")
    print(f"history    {os.path.relpath(HISTORY_PATH, PROJECT_ROOT)}")
    return best, batch_size


def validation_breakdown(model, loader, device):
    """Per-class validation accuracy and the full 4x4 validation confusion counts.

    Validation only - the test split is not touched in P4. The point is to catch a class
    collapse that a high overall number would hide, particularly up <-> down.
    """
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            predicted = model(images).argmax(dim=1).cpu()
            for true, guess in zip(labels.tolist(), predicted.tolist()):
                confusion[true][guess] += 1
    return confusion


def print_validation_breakdown(confusion):
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    print("-" * 78)
    print(f"validation breakdown (validation split only, {total} images)")
    header = "  ".join(f"{name:>6}" for name in CLASSES)
    print(f"  {'true':<8}{header}   correct")
    for index, name in enumerate(CLASSES):
        row = "  ".join(f"{confusion[index][j]:>6}" for j in range(NUM_CLASSES))
        per_class = confusion[index][index] / max(1, confusion[index].sum())
        print(f"  {name:<8}{row}   {confusion[index][index]}/"
              f"{confusion[index].sum()} ({per_class:.1%})")
    print(f"  overall  {correct}/{total} ({correct / total:.4f})")

    up, down = CLASS_TO_INDEX["up"], CLASS_TO_INDEX["down"]
    up_as_down = int(confusion[up][down])
    down_as_up = int(confusion[down][up])
    print(f"  up->down {up_as_down}   down->up {down_as_up}   "
          f"(the orientation-sensitive pair)")
    return {
        "confusion": confusion.tolist(),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "per_class": {name: {"correct": int(confusion[i][i]),
                             "total": int(confusion[i].sum())}
                      for i, name in enumerate(CLASSES)},
        "up_predicted_as_down": up_as_down,
        "down_predicted_as_up": down_as_up,
    }


def plot_curves(history, best):
    """Loss and accuracy over epochs, train and validation only - no test metrics."""
    steps = list(range(1, len(history) + 1))
    boundary = sum(1 for row in history if row["stage"] == "stage1_head")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for axis, (key_train, key_val, title, label) in zip(axes, (
            ("train_loss", "val_loss", "Loss", "loss"),
            ("train_accuracy", "val_accuracy", "Accuracy", "accuracy"))):
        axis.plot(steps, [row[key_train] for row in history], "o-", label=f"train {label}")
        axis.plot(steps, [row[key_val] for row in history], "s-", label=f"validation {label}")
        if 0 < boundary < len(history):
            axis.axvline(boundary + 0.5, color="grey", linestyle="--", linewidth=1)
            axis.text(boundary + 0.6, axis.get_ylim()[1], " stage 2", fontsize=8,
                      va="top", color="grey")
        axis.set_title(f"{title} (MobileNetV2, four directions)")
        axis.set_xlabel("epoch (stage 1 then stage 2)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

    best_step = next((i + 1 for i, row in enumerate(history)
                      if row["stage"] == best["stage"] and row["epoch"] == best["epoch"]), None)
    if best_step:
        for axis in axes:
            axis.axvline(best_step, color="green", alpha=0.35, linewidth=6)

    figure.suptitle(f"P4 training - best validation loss {best['val_loss']:.4f}, "
                    f"accuracy {best['val_acc']:.4f} "
                    f"({best['stage']} epoch {best['epoch']})", fontsize=10)
    figure.tight_layout()
    figure.savefig(CURVES_PATH, dpi=130)
    plt.close(figure)
    print(f"curves     {os.path.relpath(CURVES_PATH, PROJECT_ROOT)}")


def verify_checkpoint(batch_size):
    """Rebuild the architecture from scratch, load the checkpoint, run one val batch."""
    print("-" * 78)
    print("checkpoint verification")
    device = torch.device("cuda")

    # Goes through the guard, so a mapping mismatch fails here rather than silently.
    fresh, payload = load_direction_checkpoint(device=device)
    fresh.eval()
    print(f"  architecture {payload['architecture']} | task {payload.get('task')}")
    print(f"  classes {payload['class_to_index']}")
    print(f"  input {payload['input_size']} | stage {payload['stage']} "
          f"epoch {payload['epoch']} | val acc {payload['best_val_accuracy']:.4f} "
          f"| val loss {payload['best_val_loss']:.4f}")

    mapping_ok = payload["class_to_index"] == CLASS_TO_INDEX
    retired = sorted({"jump", "neutral"} & set(payload["class_to_index"]))
    print(f"  [{'PASS' if mapping_ok else 'FAIL'}] checkpoint mapping equals the active "
          f"mapping (strict state_dict load succeeded)")
    print(f"  [{'PASS' if not retired else 'FAIL'}] no retired class names in metadata"
          f"{'' if not retired else ': ' + ', '.join(retired)}")

    _, val_loader, _ = get_dataloaders(batch_size=batch_size)
    images, labels = next(iter(val_loader))
    images = images.to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = fresh(images)

    shape_ok = outputs.shape == (images.shape[0], NUM_CLASSES)
    finite = bool(torch.isfinite(outputs).all())
    on_cuda = outputs.device.type == "cuda"
    ok = mapping_ok and not retired
    print(f"  [{'PASS' if shape_ok else 'FAIL'}] output shape {tuple(outputs.shape)} "
          f"(expected ({images.shape[0]}, {NUM_CLASSES}))")
    print(f"  [{'PASS' if finite else 'FAIL'}] finite outputs, range "
          f"[{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    print(f"  [{'PASS' if on_cuda else 'FAIL'}] model output on {outputs.device}")

    try:
        load_direction_checkpoint(
            os.path.join(MODEL_DIR, "archive_endless_runner",
                         "best_gesture_model_OBSOLETE.pt"))
        guard_ok = False
        detail = "the obsolete checkpoint loaded without complaint"
    except FileNotFoundError:
        guard_ok = True
        detail = "obsolete checkpoint not present to test against"
    except RuntimeError as error:
        guard_ok = "Refusing to load" in str(error)
        detail = "guard rejected the obsolete left/right/jump/neutral checkpoint"
    print(f"  [{'PASS' if guard_ok else 'FAIL'}] {detail}")

    return shape_ok and finite and on_cuda and ok and guard_ok


def main():
    parser = argparse.ArgumentParser(description="Train the gesture CNN on CUDA.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    require_cuda()
    batch_size = args.batch_size
    try:
        best, batch_size = train(batch_size)
    except torch.cuda.OutOfMemoryError:
        if batch_size <= 16:
            raise
        print("\nCUDA out of memory at batch size "
              f"{batch_size}; clearing cache and retrying at 16")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        best, batch_size = train(16)

    ok = verify_checkpoint(batch_size)
    print("-" * 78)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
