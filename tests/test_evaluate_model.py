"""The evaluation pipeline end to end on the GPU - on the validation split, into a temp folder.

`evaluate_model.py` produced the project's headline 99% test accuracy, and P5 used the test split
exactly once. So this never touches the test split: it runs the identical pipeline on the
validation split, writes every artefact to a temporary folder, and then proves the frozen P5
artefacts on disk are byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src import evaluate_model as ev
from src.data_pipeline import CLASSES, load_split

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

P5_ARTEFACTS = (ev.METRICS_PATH, ev.PREDICTIONS_PATH, ev.CONFUSION_PATH)
DATASET = Path(__file__).resolve().parent.parent / "dataset"


def digests(paths: tuple[str, ...]) -> dict[str, str]:
    return {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
        if Path(path).exists()
    }


def test_pipeline_runs_on_validation_without_touching_the_p5_artefacts(tmp_path: Path) -> None:
    before = digests(P5_ARTEFACTS)
    # Guard added after a real incident: a shadowed `path` variable made the plot functions
    # save their figures over dataset images. Every image must be byte-identical afterwards.
    images = tuple(str(p) for p in sorted(DATASET.rglob("*.jpg")))
    images_before = digests(images)
    device = torch.device("cuda")
    model, payload = ev.load_model(device)
    entries = load_split()["splits"]["val"]
    tally = {name: sum(1 for entry in entries if entry["class"] == name) for name in CLASSES}
    paths = ev.EvaluationPaths(
        metrics=str(tmp_path / "metrics.json"),
        predictions=str(tmp_path / "predictions.csv"),
        confusion=str(tmp_path / "confusion.png"),
        misclassified=str(tmp_path / "misclassified.png"),
        low_confidence=str(tmp_path / "low_confidence.png"),
    )

    record = ev.run_evaluation(
        model,
        payload,
        entries,
        device,
        paths,
        ev.SplitInfo("validation", tally, overlap=0, guard_ok=True),
    )

    assert record.results["total"] == 200
    assert record.results["accuracy"] >= 0.97, "P4 measured 0.995 on this split"
    assert ev.safety_checks(record) == 0

    metrics = json.loads(Path(paths.metrics).read_text(encoding="utf-8"))
    assert metrics["evaluated_split"] == "validation"
    assert metrics["confusion_matrix_order"] == list(CLASSES)
    rows = Path(paths.predictions).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 201, "a header plus one row per image"
    assert Path(paths.confusion).is_file()
    assert Path(paths.low_confidence).is_file()

    assert digests(P5_ARTEFACTS) == before, "a frozen P5 artefact changed"
    assert digests(images) == images_before, "a dataset image was modified"
