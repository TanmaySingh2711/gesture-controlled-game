"""Evaluate the frozen model on people it has never seen: the unseen-subject HaGRID set.

HaGRID was not split by person, so the P5 test split could share people with training, and its
99% may partly reflect familiar hands. `dataset_external/`, built by
`python -m src.audit_hagrid_lineage --external 500`, holds crops from people with no image
anywhere in `dataset/`, one image per person per class. Accuracy here is the better estimate of
how the model does for a new player.

Rules:
- The frozen checkpoint is loaded read-only, through the guarded loader.
- The P3 test split is never touched; P5's saved predictions are only read, for comparison.
- Nothing measured here is used to tune any setting.
- The run is refused if any external image is byte-identical to a dataset image.

Usage:
    python -m src.evaluate_external
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import analyze_evaluation as rigor
from src.data_pipeline import CLASS_TO_INDEX, GestureDataset, eval_transform
from src.evaluate_model import BATCH_SIZE, evaluate, load_model, require_cuda, write_predictions

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
EXTERNAL_DIR: Final = PROJECT_ROOT / "dataset_external"
MANIFEST_PATH: Final = EXTERNAL_DIR / "manifest.json"
DATASET_DIR: Final = PROJECT_ROOT / "dataset"
REPORT_PATH: Final = PROJECT_ROOT / "reports" / "external_evaluation.json"
PREDICTIONS_PATH: Final = PROJECT_ROOT / "reports" / "external_predictions.csv"


def load_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Manifest files -> dataset entries in the pipeline's format, in a fixed order."""
    entries = []
    for relative in sorted(manifest["files"]):
        label = relative.split("/", 1)[0]
        if label not in CLASS_TO_INDEX:
            raise ValueError(f"{relative}: unknown class {label!r}")
        entries.append(
            {"path": f"dataset_external/{relative}", "label": CLASS_TO_INDEX[label], "class": label}
        )
    return entries


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def duplicates_of_dataset(
    entries: list[dict[str, Any]], root: Path = PROJECT_ROOT, dataset_dir: Path = DATASET_DIR
) -> list[str]:
    """External images whose bytes also appear in `dataset/` - there must be none."""
    dataset = {md5_of(path) for path in dataset_dir.rglob("*.jpg")}
    return [entry["path"] for entry in entries if md5_of(root / entry["path"]) in dataset]


def difference_interval(
    hits_a: int, total_a: int, hits_b: int, total_b: int
) -> tuple[float, float]:
    """95% interval for the difference of two independent proportions, a minus b.

    Newcombe's hybrid score method (method 10 of Newcombe, 1998), built from the two Wilson
    intervals. It stays well behaved near 100%, where a simple normal interval would not.
    """
    p_a, p_b = hits_a / total_a, hits_b / total_b
    low_a, high_a = rigor.wilson_interval(hits_a, total_a)
    low_b, high_b = rigor.wilson_interval(hits_b, total_b)
    difference = p_a - p_b
    low = difference - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    high = difference + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return max(-1.0, low), min(1.0, high)


def compare_with_p5(hits: int, total: int, p5: rigor.Predictions) -> dict[str, Any]:
    p5_hits, p5_total = int(p5.correct.sum()), len(p5.correct)
    low, high = difference_interval(hits, total, p5_hits, p5_total)
    return {
        "p5_test_accuracy": p5_hits / p5_total,
        "p5_test_wilson_95": list(rigor.wilson_interval(p5_hits, p5_total)),
        "difference_external_minus_p5": hits / total - p5_hits / p5_total,
        "difference_newcombe_95": [low, high],
        "difference_is_significant": not low <= 0.0 <= high,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.parse_args(argv)

    if not MANIFEST_PATH.exists():
        print(
            "ERROR: dataset_external/manifest.json not found - build it with "
            "`python -m src.audit_hagrid_lineage --external 500`"
        )
        return 1
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = load_entries(manifest)
    missing = [entry["path"] for entry in entries if not (PROJECT_ROOT / entry["path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} manifest image(s) missing, e.g. {missing[0]}")
        return 1
    duplicates = duplicates_of_dataset(entries)
    if duplicates:
        print(
            f"ERROR: {len(duplicates)} external image(s) duplicate dataset images: {duplicates[:3]}"
        )
        return 1

    require_cuda()
    device = torch.device("cuda")
    model, _payload = load_model(device)
    loader = DataLoader(
        GestureDataset(entries, eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    results = evaluate(model, loader, device)
    predictions = rigor.Predictions(
        true=np.asarray(results["true"], dtype=np.int64),
        predicted=np.asarray(results["pred"], dtype=np.int64),
        confidence=np.asarray(results["confidence"], dtype=np.float64),
        probabilities=np.asarray(results["probabilities"], dtype=np.float64),
    )
    hits, total = int(predictions.correct.sum()), len(entries)

    report = rigor.analyse(predictions)
    report["source"] = "dataset_external/ - unseen-subject HaGRID crops, frozen model, one pass"
    report["provisional"] = not manifest.get("lineage_complete", False)
    report["provenance"] = {
        key: manifest.get(key)
        for key in ("source", "seed", "per_class", "rule", "excluded_people", "lineage_complete")
    }
    report["distinct_people"] = len({info["user_id"] for info in manifest["files"].values()})
    report["duplicates_of_dataset_images"] = 0
    report["loss"] = results["loss"]
    report["outputs_finite"] = results["finite"]
    report["softmax_sums_to_one"] = results["softmax_sum_ok"]
    report["comparison_with_p5_test"] = compare_with_p5(hits, total, rigor.load_predictions())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_predictions(
        entries,
        results["true"],
        results["pred"],
        results["confidence"],
        results["probabilities"],
        str(PREDICTIONS_PATH),
    )

    low, high = report["accuracy_wilson_95"]
    comparison = report["comparison_with_p5_test"]
    diff_low, diff_high = comparison["difference_newcombe_95"]
    print(f"unseen-subject images {total} from {report['distinct_people']} people")
    print(f"accuracy {hits}/{total} = {hits / total:.4f}  Wilson 95% [{low:.4f}, {high:.4f}]")
    for name, row in report["per_class"].items():
        print(f"  {name:<6} recall {row['recall']:.4f}  ({row['correct']}/{row['support']})")
    print(f"macro-F1 {report['macro_f1']:.4f}  ECE {report['calibration']['ece']:.4f}")
    at_live = report["at_live_threshold"]
    print(
        f"at the live 0.90 threshold: {at_live['coverage']:.1%} accepted, "
        f"accuracy {at_live['accuracy']:.4f}, {at_live['errors_accepted']} errors accepted"
    )
    print(
        f"vs P5 test {comparison['p5_test_accuracy']:.4f}: difference "
        f"{comparison['difference_external_minus_p5']:+.4f}, 95% [{diff_low:+.4f}, {diff_high:+.4f}]"
    )
    if report["provisional"]:
        print("WARNING: lineage was incomplete, so some dataset people may not have been excluded")
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
