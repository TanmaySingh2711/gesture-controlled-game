"""Statistical rigour for the frozen P5 test evaluation: confidence intervals and calibration.

Reads the per-image predictions that P5 already saved (`model/direction_test_predictions.csv`).
The model is not run and the test split is not evaluated again, so P5's "test set used exactly
once" rule still holds - this analyses numbers that were recorded at the time.

What it adds
------------
* **Wilson 95% intervals** for overall and per-class accuracy. With 200 test images, "99.0%"
  on its own overstates what is known; the interval says where the true accuracy plausibly sits.
* **Bootstrap 95% interval for macro-F1** (10,000 resamples, fixed seed).
* **Calibration**: expected and maximum calibration error, Brier score and negative
  log-likelihood, with a reliability diagram. The frozen 0.90 live threshold only means what
  it appears to mean if 0.90 confidence really corresponds to high accuracy.
* **Selective prediction**: how much of the data is accepted, and how accurately, at each
  confidence threshold - including the live 0.90.

Usage::

    python -m src.analyze_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH: Final = PROJECT_ROOT / "model" / "direction_test_predictions.csv"
REPORT_PATH: Final = PROJECT_ROOT / "reports" / "evaluation_rigor.json"
RELIABILITY_PATH: Final = PROJECT_ROOT / "reports" / "reliability_diagram.png"

CLASSES: Final = ("left", "right", "up", "down")
Z_95: Final = 1.959963984540054
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 2026
CALIBRATION_BINS: Final = 10
LIVE_THRESHOLD: Final = 0.90
THRESHOLDS: Final = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Predictions:
    true: IntArray
    predicted: IntArray
    confidence: FloatArray
    probabilities: FloatArray  # shape (n, 4), columns in CLASSES order

    @property
    def correct(self) -> BoolArray:
        return np.asarray(self.true == self.predicted)


def load_predictions(path: Path = PREDICTIONS_PATH) -> Predictions:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    return Predictions(
        true=np.array([int(row["true_index"]) for row in rows], dtype=np.int64),
        predicted=np.array([int(row["predicted_index"]) for row in rows], dtype=np.int64),
        confidence=np.array([float(row["confidence"]) for row in rows], dtype=np.float64),
        probabilities=np.array(
            [[float(row[f"prob_{name}"]) for name in CLASSES] for row in rows], dtype=np.float64
        ),
    )


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves well
    near 100%, which is exactly where this model's accuracy sits.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
    ) / denominator
    # At 0 or all successes the matching bound is exactly 0 or 1 by definition; computing it in
    # floating point would give 0.9999999999999999 and look like a real, if tiny, gap.
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == trials else min(1.0, centre + margin)
    return low, high


def macro_f1(true: IntArray, predicted: IntArray, classes: int = len(CLASSES)) -> float:
    """Unweighted mean F1 over classes; a class with no support or predictions scores 0."""
    confusion = np.bincount(true * classes + predicted, minlength=classes * classes).reshape(
        classes, classes
    )
    hits = np.diag(confusion).astype(np.float64)
    predicted_counts = confusion.sum(axis=0)
    true_counts = confusion.sum(axis=1)
    precision = np.divide(hits, predicted_counts, out=np.zeros(classes), where=predicted_counts > 0)
    recall = np.divide(hits, true_counts, out=np.zeros(classes), where=true_counts > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros(classes), where=total > 0)
    return float(f1.mean())


def bootstrap_interval(
    true: IntArray,
    predicted: IntArray,
    statistic: Callable[[IntArray, IntArray], float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of `statistic`, resampling images with replacement."""
    rng = np.random.default_rng(seed)
    count = len(true)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, count, count)
        values[index] = statistic(true[sample], predicted[sample])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def calibration(
    confidence: FloatArray, correct: BoolArray, bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    """Expected and maximum calibration error over equal-width confidence bins.

    Bins are closed on the right, so a confidence of exactly 1.0 lands in the top bin.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = len(confidence)
    ece, mce = 0.0, 0.0
    table: list[dict[str, Any]] = []
    for index in range(bins):
        members = assigned == index
        count = int(members.sum())
        row: dict[str, Any] = {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "accuracy": None,
            "confidence": None,
            "gap": None,
        }
        if count:
            accuracy = float(correct[members].mean())
            mean_confidence = float(confidence[members].mean())
            gap = abs(accuracy - mean_confidence)
            ece += gap * count / total
            mce = max(mce, gap)
            row.update(accuracy=accuracy, confidence=mean_confidence, gap=gap)
        table.append(row)
    return {"ece": ece, "mce": mce, "bins": table}


def brier_score(probabilities: FloatArray, true: IntArray) -> float:
    """Mean squared distance between predicted distributions and one-hot truth (0 is perfect)."""
    one_hot = np.eye(probabilities.shape[1])[true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: FloatArray, true: IntArray) -> float:
    chosen = probabilities[np.arange(len(true)), true]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def selective_prediction(
    confidence: FloatArray, correct: BoolArray, thresholds: tuple[float, ...] = THRESHOLDS
) -> list[dict[str, Any]]:
    """Coverage and accuracy when only predictions at or above each threshold are accepted."""
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        rows.append(
            {
                "threshold": threshold,
                "accepted": count,
                "coverage": count / len(confidence),
                "accuracy": float(correct[accepted].mean()) if count else None,
                "errors_accepted": int((~correct[accepted]).sum()),
            }
        )
    return rows


def plot_reliability(bins: list[dict[str, Any]], ece: float, path: Path) -> None:
    centres = [(row["lower"] + row["upper"]) / 2 for row in bins]
    accuracy = [row["accuracy"] if row["count"] else 0.0 for row in bins]
    counts = [row["count"] for row in bins]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 6.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    top.bar(centres, accuracy, width=0.09, color="#4c78a8", edgecolor="white", label="accuracy")
    top.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect calibration")
    top.set_ylabel("accuracy in bin")
    top.set_ylim(0, 1.02)
    top.set_title(f"Reliability diagram - frozen model, test split (ECE {ece:.4f})")
    top.legend(loc="upper left")
    bottom.bar(centres, counts, width=0.09, color="#9d9da1")
    bottom.set_yscale("symlog")
    bottom.set_xlabel("predicted confidence")
    bottom.set_ylabel("images")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def analyse(predictions: Predictions) -> dict[str, Any]:
    correct = predictions.correct
    total = len(correct)
    hits = int(correct.sum())
    overall_low, overall_high = wilson_interval(hits, total)

    per_class = {}
    for index, name in enumerate(CLASSES):
        members = predictions.true == index
        support = int(members.sum())
        class_hits = int(correct[members].sum())
        low, high = wilson_interval(class_hits, support) if support else (0.0, 0.0)
        per_class[name] = {
            "correct": class_hits,
            "support": support,
            "recall": class_hits / support if support else None,
            "wilson_95": [low, high],
        }

    f1_low, f1_high = bootstrap_interval(predictions.true, predictions.predicted, macro_f1)
    calibration_report = calibration(predictions.confidence, correct)
    selective = selective_prediction(predictions.confidence, correct)
    live = next(row for row in selective if row["threshold"] == LIVE_THRESHOLD)
    return {
        "source": "model/direction_test_predictions.csv (saved by P5; the model is not rerun)",
        "images": total,
        "accuracy": hits / total,
        "accuracy_wilson_95": [overall_low, overall_high],
        "per_class": per_class,
        "macro_f1": macro_f1(predictions.true, predictions.predicted),
        "macro_f1_bootstrap_95": [f1_low, f1_high],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "calibration": calibration_report,
        "brier_score": brier_score(predictions.probabilities, predictions.true),
        "negative_log_likelihood": negative_log_likelihood(
            predictions.probabilities, predictions.true
        ),
        "selective_prediction": selective,
        "at_live_threshold": live,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=RELIABILITY_PATH)
    args = parser.parse_args(argv)

    report = analyse(load_predictions(args.predictions))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    plot_reliability(report["calibration"]["bins"], report["calibration"]["ece"], args.figure)

    low, high = report["accuracy_wilson_95"]
    f1_low, f1_high = report["macro_f1_bootstrap_95"]
    live = report["at_live_threshold"]
    print(f"accuracy        {report['accuracy']:.4f}  Wilson 95% [{low:.4f}, {high:.4f}]")
    for name, block in report["per_class"].items():
        c_low, c_high = block["wilson_95"]
        print(
            f"  {name:<6} {block['correct']}/{block['support']}  Wilson 95% [{c_low:.3f}, {c_high:.3f}]"
        )
    print(f"macro-F1        {report['macro_f1']:.4f}  bootstrap 95% [{f1_low:.4f}, {f1_high:.4f}]")
    print(
        f"calibration     ECE {report['calibration']['ece']:.4f}  MCE {report['calibration']['mce']:.4f}"
        f"  Brier {report['brier_score']:.4f}  NLL {report['negative_log_likelihood']:.4f}"
    )
    print(
        f"at threshold {LIVE_THRESHOLD}: {live['accepted']}/{report['images']} accepted "
        f"({live['coverage']:.1%}), accuracy {live['accuracy']:.4f}, "
        f"{live['errors_accepted']} errors accepted"
    )
    print(
        f"saved {args.report.relative_to(PROJECT_ROOT)} and {args.figure.relative_to(PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
