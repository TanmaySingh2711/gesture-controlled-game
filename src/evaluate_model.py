"""Final offline evaluation of the frozen direction CNN on the untouched test split.

Loads model/best_direction_model.pt through the P4 guard, runs the 200-image frozen P3 test
split through it on CUDA under inference mode, and writes metrics, a confusion matrix,
per-image predictions, a misclassification sheet and a CNN-only latency benchmark.

Classes are the four Pac-Man directions: 0=left (fist), 1=right (palm), 2=up (thumbs up),
3=down (thumbs down).

Nothing here trains, fine-tunes, or modifies the checkpoint, and no test result is used to
tune anything - not the learning rate, not the augmentation, and not the live confidence
threshold, which P6 must derive from webcam trials instead.

The loader is `train_model.load_direction_checkpoint()`, which refuses any checkpoint whose
class mapping disagrees with the active one. This matters because the retired endless-runner
checkpoint also has four outputs and would load without a shape error while meaning `jump`
at index 2.

Training curves are not produced here: P4 owns `model/direction_training_curves.png`.

Usage:
    python src/evaluate_model.py
"""

import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import (CLASS_TO_INDEX, IMAGE_SIZE, PROJECT_ROOT, GestureDataset,
                           eval_transform, load_split)
from train_model import CHECKPOINT_PATH, load_direction_checkpoint

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
METRICS_PATH = os.path.join(MODEL_DIR, "direction_evaluation_metrics.json")
CONFUSION_PATH = os.path.join(MODEL_DIR, "direction_confusion_matrix.png")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "direction_test_predictions.csv")
MISCLASSIFIED_PATH = os.path.join(MODEL_DIR, "direction_misclassified_samples.png")
LOW_CONFIDENCE_PATH = os.path.join(MODEL_DIR, "direction_low_confidence_correct.png")
OBSOLETE_PATH = os.path.join(MODEL_DIR, "archive_endless_runner",
                             "best_gesture_model_OBSOLETE.pt")

CLASSES = list(CLASS_TO_INDEX)                 # left, right, up, down - fixed order
NUM_CLASSES = len(CLASSES)
EXPECTED_PER_CLASS = 50
BATCH_SIZE = 32
BENCHMARK_WARMUP = 50
BENCHMARK_RUNS = 300
LOW_CONFIDENCE_SHOWN = 6


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for CNN evaluation.")


def load_model(device):
    """Fresh architecture + saved weights. The checkpoint file is never modified.

    Identity is checked before anything is evaluated: a checkpoint that is the right shape
    but the wrong experiment would produce a confident, entirely meaningless report.
    """
    model, payload = load_direction_checkpoint(device=device)
    model.eval()

    problems = []
    if payload.get("architecture") != "mobilenet_v2":
        problems.append(f"architecture={payload.get('architecture')}")
    if list(payload.get("input_size", [])) != [3, IMAGE_SIZE, IMAGE_SIZE]:
        problems.append(f"input_size={payload.get('input_size')}")
    if payload.get("class_to_index") != CLASS_TO_INDEX:
        problems.append(f"class_to_index={payload.get('class_to_index')}")
    if payload.get("num_classes") != NUM_CLASSES:
        problems.append(f"num_classes={payload.get('num_classes')}")
    if payload.get("task") != "pacman_direction_v1":
        problems.append(f"task={payload.get('task')}")
    retired = sorted({"jump", "neutral"} & set(payload.get("class_to_index", {})))
    if retired:
        problems.append(f"retired class names present: {', '.join(retired)}")
    if problems:
        raise RuntimeError("checkpoint metadata mismatch: " + ", ".join(problems))

    return model, payload


def verify_obsolete_rejected():
    """The guard must refuse the retired four-output checkpoint, not merely differ from it."""
    if not os.path.exists(OBSOLETE_PATH):
        return True, "obsolete checkpoint not present to test against"
    try:
        load_direction_checkpoint(OBSOLETE_PATH)
    except RuntimeError as error:
        return "Refusing to load" in str(error), "guard rejected the obsolete checkpoint"
    return False, "the obsolete checkpoint loaded without complaint"


def verify_test_split(split):
    """The test split must be the frozen P3 one: 200 images, 50 per class, disjoint."""
    entries = split["splits"]["test"]
    tally = {name: sum(1 for e in entries if e["class"] == name) for name in CLASSES}
    paths = {name: {e["path"] for e in items} for name, items in split["splits"].items()}
    overlap = len(paths["test"] & (paths["train"] | paths["val"]))
    problems = []
    if len(entries) != NUM_CLASSES * EXPECTED_PER_CLASS:
        problems.append(f"{len(entries)} test images")
    if any(count != EXPECTED_PER_CLASS for count in tally.values()):
        problems.append(f"per-class {tally}")
    if overlap:
        problems.append(f"{overlap} image(s) shared with train/val")
    if split.get("class_to_index") != CLASS_TO_INDEX:
        problems.append("split file mapping disagrees with the active mapping")
    if problems:
        raise RuntimeError("test split is not the frozen P3 split: " + "; ".join(problems))
    return tally, overlap


@torch.inference_mode()
def evaluate(model, loader, device):
    """One pass over the test split. No gradients, no optimizer, no weight update."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, seen = 0.0, 0, 0
    all_true, all_pred, all_conf, all_probs = [], [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)                       # loss takes raw logits, never softmax
        total_loss += criterion(logits, labels).item()

        probabilities = torch.softmax(logits, dim=1)  # softmax only for confidence
        confidence, predicted = probabilities.max(dim=1)

        correct += (predicted == labels).sum().item()
        seen += labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predicted.cpu().tolist())
        all_conf.extend(confidence.cpu().tolist())
        all_probs.append(probabilities.cpu())

    probs = torch.cat(all_probs)
    sums = probs.sum(dim=1)
    return {
        "loss": total_loss / seen,
        "correct": correct,
        "total": seen,
        "accuracy": correct / seen,
        "true": all_true,
        "pred": all_pred,
        "confidence": all_conf,
        "probabilities": probs.float().numpy(),
        "finite": bool(torch.isfinite(probs).all()),
        "softmax_sum_ok": bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-4)),
        "softmax_sum_range": [float(sums.min()), float(sums.max())],
    }


def plot_confusion(matrix):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="images")

    ax.set_xticks(range(NUM_CLASSES), [c.upper() for c in CLASSES])
    ax.set_yticks(range(NUM_CLASSES), [c.upper() for c in CLASSES])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix - test set (200 images, four directions)")

    peak = matrix.max()
    for row in range(NUM_CLASSES):
        support = matrix[row].sum()
        for column in range(NUM_CLASSES):
            count = matrix[row, column]
            percent = 100.0 * count / support if support else 0.0
            ax.text(column, row, f"{count}\n{percent:.0f}%", ha="center", va="center",
                    fontsize=11,
                    color="white" if count > peak * 0.5 else "black")

    fig.tight_layout()
    fig.subplots_adjust(left=0.16)     # keep the "True class" label from being clipped
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)


def plot_misclassified(entries, results):
    wrong = [i for i, (t, p) in enumerate(zip(results["true"], results["pred"])) if t != p]
    if not wrong:
        return 0

    shown = wrong[:12]
    columns = min(4, len(shown))
    rows = (len(shown) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis in axes:
        axis.axis("off")
    for axis, index in zip(axes, shown):
        entry = entries[index]
        path = os.path.join(PROJECT_ROOT, entry["path"].replace("/", os.sep))
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"true: {CLASSES[results['true'][index]].upper()}\n"
            f"pred: {CLASSES[results['pred'][index]].upper()} "
            f"({results['confidence'][index]:.2f})",
            fontsize=10)
        axis.axis("off")

    fig.suptitle(f"Misclassified test images ({len(wrong)} total, showing {len(shown)})")
    fig.tight_layout()
    fig.savefig(MISCLASSIFIED_PATH, dpi=150)
    plt.close(fig)
    return len(wrong)


def plot_low_confidence_correct(entries, results, indices):
    """The correct predictions the model was least sure about.

    These are the offline samples closest to the decision boundary, so they are the ones
    most likely to become unstable under live webcam conditions. They are NOT errors.
    """
    if not indices:
        return
    columns = min(3, len(indices))
    rows = (len(indices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, index in zip(axes, indices):
        entry = entries[index]
        path = os.path.join(PROJECT_ROOT, entry["path"].replace("/", os.sep))
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{CLASSES[results['true'][index]].upper()} (correct)\n"
                       f"confidence {results['confidence'][index]:.3f}", fontsize=10)
        axis.axis("off")
    fig.suptitle("Lowest-confidence correct test predictions (not errors)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.32)
    fig.savefig(LOW_CONFIDENCE_PATH, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def benchmark(model, device, batch=1, runs=BENCHMARK_RUNS):
    """CNN-only forward-pass latency: no webcam, no OpenCV, no Pygame, no disk I/O.

    Timed with CUDA events around each individual pass, so the result is a distribution
    rather than one averaged number. This measures the model alone - the end-to-end webcam
    frame rate in P6 will be lower, because capture, ROI cropping and drawing all add time.
    """
    dummy = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    for _ in range(BENCHMARK_WARMUP):
        model(dummy)
    torch.cuda.synchronize()

    samples = []
    for _ in range(runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        model(dummy)
        end_event.record()
        torch.cuda.synchronize()
        samples.append(start_event.elapsed_time(end_event))

    samples = np.array(samples)
    return {
        "batch_size": batch,
        "runs": runs,
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(samples.min()),
        "max_ms": float(samples.max()),
        "predictions_per_second": float(batch * 1000.0 / samples.mean()),
    }


def main():
    require_cuda()
    device = torch.device("cuda")

    print("P5 final offline evaluation - four-direction gesture CNN")
    print(f"PyTorch        : {torch.__version__} | CUDA runtime {torch.version.cuda}")
    print(f"device         : {torch.cuda.get_device_name(0)} ({device})")

    model, payload = load_model(device)
    parameter_device = next(model.parameters()).device
    print(f"checkpoint     : {os.path.relpath(CHECKPOINT_PATH, PROJECT_ROOT)} "
          f"({os.path.getsize(CHECKPOINT_PATH) / 1e6:.2f} MB)")
    print(f"task           : {payload.get('task')} | architecture "
          f"{payload['architecture']} | input {payload['input_size']}")
    print(f"class mapping  : {payload['class_to_index']}")
    print(f"val accuracy   : {payload['best_val_accuracy']:.4f} "
          f"(loss {payload['best_val_loss']:.4f}; model-selection metric, "
          f"stage {payload['stage']} epoch {payload['epoch']})")
    print(f"model device   : {parameter_device} | eval mode: {not model.training} | "
          f"autocast: not used (full float32 inference)")

    guard_ok, guard_detail = verify_obsolete_rejected()
    print(f"guard          : [{'PASS' if guard_ok else 'FAIL'}] {guard_detail}")

    split = load_split()
    tally, overlap = verify_test_split(split)
    entries = split["splits"]["test"]
    dataset = GestureDataset(entries, eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=True)
    transform_names = [type(t).__name__ for t in eval_transform().transforms]
    print(f"test split     : {len(entries)} images "
          + ", ".join(f"{name} {count}" for name, count in tally.items())
          + f" | overlap with train/val: {overlap}")
    print(f"test transform : {' -> '.join(transform_names)} (deterministic, unshuffled)")
    print("-" * 78)

    results = evaluate(model, loader, device)
    true = np.array(results["true"])
    pred = np.array(results["pred"])
    confidence = np.array(results["confidence"])

    print(f"Test Loss    : {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}% "
          f"({results['correct']}/{results['total']})")
    print(f"Finite outputs: {results['finite']}")
    print()

    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    macro = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(true, pred, average="weighted", zero_division=0)

    print(f"{'class':<10}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for index, name in enumerate(CLASSES):
        print(f"{name:<10}{precision[index]:>10.4f}{recall[index]:>9.4f}"
              f"{f1[index]:>8.4f}{support[index]:>9}")
    print(f"{'macro avg':<10}{macro[0]:>10.4f}{macro[1]:>9.4f}{macro[2]:>8.4f}"
          f"{results['total']:>9}")
    print(f"{'weighted':<10}{weighted[0]:>10.4f}{weighted[1]:>9.4f}{weighted[2]:>8.4f}"
          f"{results['total']:>9}")
    print()

    matrix = confusion_matrix(true, pred, labels=list(range(NUM_CLASSES)))
    print("confusion matrix (rows = true, columns = predicted, order "
          f"{'/'.join(CLASSES)}):")
    for index, row in enumerate(matrix):
        print(f"  {CLASSES[index]:<8} {row}")
    plot_confusion(matrix)
    print(f"  saved {os.path.relpath(CONFUSION_PATH, PROJECT_ROOT)}")

    # The pairs worth naming explicitly: up/down are the orientation-sensitive pair, and
    # left->down was the single validation error in P4.
    index_of = {name: i for i, name in enumerate(CLASSES)}
    pairs = {
        "up_as_down": int(matrix[index_of["up"]][index_of["down"]]),
        "down_as_up": int(matrix[index_of["down"]][index_of["up"]]),
        "left_as_down": int(matrix[index_of["left"]][index_of["down"]]),
        "down_as_left": int(matrix[index_of["down"]][index_of["left"]]),
    }
    print()
    print("targeted confusions:")
    print(f"  up   -> down : {pairs['up_as_down']}/50")
    print(f"  down -> up   : {pairs['down_as_up']}/50")
    print(f"  left -> down : {pairs['left_as_down']}/50")
    print(f"  down -> left : {pairs['down_as_left']}/50")
    print()

    # --- per-image predictions -----------------------------------------------------------
    probabilities = results["probabilities"]
    with open(PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true_class", "true_index", "predicted_class",
                         "predicted_index", "correct", "confidence"]
                        + [f"prob_{name}" for name in CLASSES])
        for index, entry in enumerate(entries):
            writer.writerow([entry["path"], CLASSES[true[index]], int(true[index]),
                             CLASSES[pred[index]], int(pred[index]),
                             "yes" if true[index] == pred[index] else "no",
                             f"{confidence[index]:.6f}"]
                            + [f"{probabilities[index][c]:.6f}"
                               for c in range(NUM_CLASSES)])
    print(f"  saved {os.path.relpath(PREDICTIONS_PATH, PROJECT_ROOT)} "
          f"({len(entries)} rows)")

    # --- misclassifications ---------------------------------------------------------------
    wrong_count = plot_misclassified(entries, results)
    if wrong_count:
        print(f"  saved {os.path.relpath(MISCLASSIFIED_PATH, PROJECT_ROOT)} "
              f"({wrong_count} misclassified)")
    else:
        print("  no misclassified test images - misclassification sheet not created")

    # --- confidence (descriptive only, never used to tune a threshold) --------------------
    correct_mask = true == pred

    def describe(values):
        if not len(values):
            return None
        return {"count": int(len(values)), "mean": float(values.mean()),
                "median": float(np.median(values)), "min": float(values.min()),
                "max": float(values.max())}

    stats = {
        "correct": describe(confidence[correct_mask]),
        "incorrect": describe(confidence[~correct_mask]),
        "overall_mean": float(confidence.mean()),
        "overall_median": float(np.median(confidence)),
        "incorrect_values": [round(float(v), 6) for v in confidence[~correct_mask]],
    }
    print("confidence (maximum softmax probability of the predicted class):")
    for label in ("correct", "incorrect"):
        block = stats[label]
        if block is None:
            print(f"  {label:<10} none")
            continue
        print(f"  {label:<10} n={block['count']:<4} mean {block['mean']:.4f}  "
              f"median {block['median']:.4f}  min {block['min']:.4f}  "
              f"max {block['max']:.4f}")
    if stats["incorrect_values"]:
        print(f"  incorrect confidences: "
              + ", ".join(f"{v:.4f}" for v in stats["incorrect_values"]))
    print("  (diagnostic only - the live threshold is tuned from webcam trials in P6)")

    # --- lowest-confidence correct predictions ------------------------------------------------
    correct_indices = [i for i in range(len(entries)) if correct_mask[i]]
    lowest = sorted(correct_indices, key=lambda i: confidence[i])[:LOW_CONFIDENCE_SHOWN]
    print()
    print(f"lowest-confidence CORRECT predictions (borderline, not errors):")
    low_records = []
    for index in lowest:
        runner_up = float(np.sort(probabilities[index])[-2])
        runner_class = CLASSES[int(np.argsort(probabilities[index])[-2])]
        low_records.append({
            "file": entries[index]["path"], "true_class": CLASSES[true[index]],
            "confidence": round(float(confidence[index]), 6),
            "runner_up_class": runner_class, "runner_up_probability": round(runner_up, 6),
        })
        print(f"  {CLASSES[true[index]]:<6} {confidence[index]:.4f}  "
              f"(runner-up {runner_class} {runner_up:.4f})  "
              f"{os.path.basename(entries[index]['path'])}")
    plot_low_confidence_correct(entries, results, lowest)
    print(f"  saved {os.path.relpath(LOW_CONFIDENCE_PATH, PROJECT_ROOT)}")

    latency = benchmark(model, device, batch=1)
    throughput = benchmark(model, device, batch=32, runs=100)
    print()
    print(f"CNN-only GPU latency, batch 1: mean {latency['mean_ms']:.3f} ms | "
          f"median {latency['median_ms']:.3f} | p95 {latency['p95_ms']:.3f} | "
          f"~{latency['predictions_per_second']:.0f} predictions/sec")
    print(f"CNN-only GPU throughput, batch 32: mean {throughput['mean_ms']:.3f} ms/batch | "
          f"~{throughput['predictions_per_second']:.0f} images/sec")
    print("  (model forward pass only - webcam capture, ROI cropping and drawing are extra)")

    payload_metrics = {
        "evaluated_split": "test",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "P5",
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "inference": "torch.inference_mode, float32, no autocast, model.eval()",
        "class_to_index": CLASS_TO_INDEX,
        "test_per_class": tally,
        "test_images": results["total"],
        "test_loss": round(results["loss"], 6),
        "test_accuracy": round(results["accuracy"], 6),
        "correct": results["correct"],
        "class_order": CLASSES,
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            } for i, name in enumerate(CLASSES)
        },
        "macro_avg": {"precision": round(float(macro[0]), 6),
                      "recall": round(float(macro[1]), 6),
                      "f1": round(float(macro[2]), 6)},
        "weighted_avg": {"precision": round(float(weighted[0]), 6),
                         "recall": round(float(weighted[1]), 6),
                         "f1": round(float(weighted[2]), 6)},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": CLASSES,
        "targeted_confusions": pairs,
        "misclassified": int(wrong_count),
        "misclassified_detail": [
            {"file": entries[i]["path"], "true_class": CLASSES[true[i]],
             "predicted_class": CLASSES[pred[i]],
             "confidence": round(float(confidence[i]), 6)}
            for i in range(len(entries)) if true[i] != pred[i]
        ],
        "confidence": stats,
        "lowest_confidence_correct": low_records,
        "softmax_sums_to_one": results["softmax_sum_ok"],
        "softmax_sum_range": results["softmax_sum_range"],
        "outputs_finite": results["finite"],
        "latency_batch1": latency,
        "throughput_batch32": throughput,
        "validation_reference": {
            "val_accuracy": payload["best_val_accuracy"],
            "val_loss": payload["best_val_loss"],
            "note": "from P4; recorded here only as a generalization comparison",
        },
        "checkpoint": {
            "path": "model/best_direction_model.pt",
            "size_bytes": os.path.getsize(CHECKPOINT_PATH),
            "architecture": payload["architecture"],
            "input_size": payload["input_size"],
            "class_to_index": payload["class_to_index"],
            "task": payload.get("task"),
            "best_val_accuracy": payload["best_val_accuracy"],
            "best_val_loss": payload["best_val_loss"],
            "epoch": payload["epoch"],
            "stage": payload["stage"],
        },
        "note": "Validation loss was the model-selection metric in P4; this test result is "
                "the final offline evaluation and was not used to tune anything. It measures "
                "held-out HaGRID hand crops, NOT live webcam reliability - P6 must establish "
                "that separately from webcam trials.",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload_metrics, handle, indent=1)
    print(f"  saved {os.path.relpath(METRICS_PATH, PROJECT_ROOT)}")

    # --- safety summary ---------------------------------------------------------------------
    checks = [
        ("checkpoint mapping matches active", payload["class_to_index"] == CLASS_TO_INDEX),
        ("obsolete checkpoint rejected", guard_ok),
        ("CUDA inference", parameter_device.type == "cuda"),
        ("model in eval mode", not model.training),
        ("frozen test split, 200 images", results["total"] == 200),
        ("50 images per class", all(c == EXPECTED_PER_CLASS for c in tally.values())),
        ("no overlap with train/val", overlap == 0),
        ("deterministic test transform",
         not any(n.startswith("Random") or n == "ColorJitter" for n in transform_names)),
        ("predictions within 0-3",
         bool(pred.min() >= 0 and pred.max() < NUM_CLASSES)),
        ("probabilities finite", results["finite"]),
        ("softmax sums to 1", results["softmax_sum_ok"]),
        ("test loss finite", bool(np.isfinite(results["loss"]))),
        ("confusion matrix totals 200", int(matrix.sum()) == 200),
        ("support totals 200", int(sum(support)) == 200),
        ("prediction rows written", len(entries) == 200),
    ]
    print("-" * 78)
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    failed = [name for name, ok in checks if not ok]
    print("-" * 78)
    if failed:
        print(f"RESULT: FAIL ({len(failed)} check(s): {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(checks)} safety checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
