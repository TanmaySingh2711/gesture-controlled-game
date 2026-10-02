"""The unseen-subject evaluation's pure parts: entries, the duplicate guard and the comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src import data_pipeline
from src import evaluate_external as ext
from src.analyze_evaluation import Predictions
from src.evaluate_external import (
    compare_with_p5,
    difference_interval,
    duplicates_of_dataset,
    load_entries,
)


def test_entries_follow_the_pipeline_format_in_a_fixed_order() -> None:
    manifest = {
        "files": {
            "up/up_00000.jpg": {"user_id": "a"},
            "left/left_00001.jpg": {"user_id": "b"},
            "left/left_00000.jpg": {"user_id": "c"},
        }
    }
    assert load_entries(manifest) == [
        {"path": "dataset/external/left/left_00000.jpg", "label": 0, "class": "left"},
        {"path": "dataset/external/left/left_00001.jpg", "label": 0, "class": "left"},
        {"path": "dataset/external/up/up_00000.jpg", "label": 2, "class": "up"},
    ]


def test_unknown_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown class 'jump'"):
        load_entries({"files": {"jump/jump_00000.jpg": {}}})


def test_duplicate_guard_finds_byte_identical_images(tmp_path: Path) -> None:
    (tmp_path / "dataset" / "left").mkdir(parents=True)
    (tmp_path / "dataset" / "external" / "left").mkdir(parents=True)
    (tmp_path / "dataset" / "left" / "left_00000.jpg").write_bytes(b"same bytes")
    (tmp_path / "dataset" / "external" / "left" / "left_00000.jpg").write_bytes(b"same bytes")
    (tmp_path / "dataset" / "external" / "left" / "left_00001.jpg").write_bytes(b"new person")
    entries = load_entries({"files": {"left/left_00000.jpg": {}, "left/left_00001.jpg": {}}})
    assert duplicates_of_dataset(entries, tmp_path, tmp_path / "dataset") == [
        "dataset/external/left/left_00000.jpg"
    ]


def test_difference_interval_matches_newcombes_worked_example() -> None:
    """Newcombe (1998), Statistics in Medicine 17:873, method 10: 56/70 - 48/80."""
    low, high = difference_interval(56, 70, 48, 80)
    assert low == pytest.approx(0.0524, abs=5e-4)
    assert high == pytest.approx(0.3339, abs=5e-4)


def test_equal_proportions_give_an_interval_around_zero() -> None:
    low, high = difference_interval(198, 200, 1980, 2000)
    assert low < 0.0 < high


def test_difference_interval_stays_inside_minus_one_and_one() -> None:
    low, high = difference_interval(0, 5, 5, 5)
    assert -1.0 <= low <= high <= 1.0


def test_comparison_reports_the_p5_side_and_significance() -> None:
    p5 = Predictions(
        true=np.array([0] * 200, dtype=np.int64),
        predicted=np.array([0] * 198 + [1, 1], dtype=np.int64),
        confidence=np.ones(200),
        probabilities=np.tile([1.0, 0.0, 0.0, 0.0], (200, 1)),
    )
    close = compare_with_p5(1970, 2000, p5)
    assert close["p5_test_accuracy"] == pytest.approx(0.99)
    assert close["difference_external_minus_p5"] == pytest.approx(-0.005)
    assert close["difference_is_significant"] is False

    far = compare_with_p5(1700, 2000, p5)
    assert far["difference_is_significant"] is True
    assert far["difference_newcombe_95"][1] < 0.0


# --- the whole run, on the tiny stand-in set -----------------------------------------------
@pytest.fixture
def external_project(
    tiny_dataset: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The tiny dataset's validation images, laid out and manifested as an external set."""
    root: Path = tiny_dataset["root"]
    files = {}
    for number, entry in enumerate(tiny_dataset["splits"]["val"]):
        relative = f"{entry['class']}/{entry['class']}_{number:05d}.jpg"
        target = root / "dataset" / "external" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(root / entry["path"], target)
        files[relative] = {"uuid": f"u{number}", "user_id": f"person-{number}"}
    manifest = {"seed": 2026, "per_class": 2, "lineage_complete": False, "files": files}
    (root / "dataset" / "external" / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    monkeypatch.setattr(ext, "PROJECT_ROOT", root)
    monkeypatch.setattr(ext, "EXTERNAL_DIR", root / "dataset" / "external")
    monkeypatch.setattr(ext, "MANIFEST_PATH", root / "dataset" / "external" / "manifest.json")
    monkeypatch.setattr(ext, "DATASET_DIR", root / "dataset")
    monkeypatch.setattr(ext, "REPORT_PATH", tmp_path / "out" / "external.json")
    monkeypatch.setattr(ext, "PREDICTIONS_PATH", tmp_path / "out" / "external.csv")
    assert str(root) == data_pipeline.PROJECT_ROOT
    return root


@pytest.mark.slow
def test_the_frozen_model_is_scored_on_unseen_people(
    external_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ext.main([]) == 0

    report = json.loads((tmp_path / "out" / "external.json").read_text("utf-8"))
    assert report["distinct_people"] == 8
    assert report["provisional"] is True  # the manifest says lineage was incomplete
    assert report["outputs_finite"] and report["softmax_sums_to_one"]
    assert set(report["comparison_with_p5_test"]) >= {"difference_newcombe_95"}
    rows = (tmp_path / "out" / "external.csv").read_text("utf-8").splitlines()
    assert len(rows) == 9
    output = capsys.readouterr().out
    assert "unseen-subject images 8 from 8 people" in output
    assert "WARNING: lineage was incomplete" in output


def test_a_missing_manifest_is_explained(
    external_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (external_project / "dataset" / "external" / "manifest.json").unlink()
    assert ext.main([]) == 1
    assert "manifest.json not found" in capsys.readouterr().out


def test_a_missing_image_stops_the_run(
    external_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    next((external_project / "dataset" / "external").rglob("*.jpg")).unlink()
    assert ext.main([]) == 1
    assert "manifest image(s) missing" in capsys.readouterr().out


def test_an_image_shared_with_the_dataset_stops_the_run(
    external_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = next((external_project / "dataset" / "external").rglob("*.jpg"))
    shutil.copy(shared, external_project / "dataset" / "left" / "copy.jpg")
    assert ext.main([]) == 1
    assert "duplicate dataset images" in capsys.readouterr().out
