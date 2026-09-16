"""The unseen-subject evaluation's pure parts: entries, the duplicate guard and the comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

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
        {"path": "dataset_external/left/left_00000.jpg", "label": 0, "class": "left"},
        {"path": "dataset_external/left/left_00001.jpg", "label": 0, "class": "left"},
        {"path": "dataset_external/up/up_00000.jpg", "label": 2, "class": "up"},
    ]


def test_unknown_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown class 'jump'"):
        load_entries({"files": {"jump/jump_00000.jpg": {}}})


def test_duplicate_guard_finds_byte_identical_images(tmp_path: Path) -> None:
    (tmp_path / "dataset" / "left").mkdir(parents=True)
    (tmp_path / "dataset_external" / "left").mkdir(parents=True)
    (tmp_path / "dataset" / "left" / "left_00000.jpg").write_bytes(b"same bytes")
    (tmp_path / "dataset_external" / "left" / "left_00000.jpg").write_bytes(b"same bytes")
    (tmp_path / "dataset_external" / "left" / "left_00001.jpg").write_bytes(b"new person")
    entries = load_entries({"files": {"left/left_00000.jpg": {}, "left/left_00001.jpg": {}}})
    assert duplicates_of_dataset(entries, tmp_path, tmp_path / "dataset") == [
        "dataset_external/left/left_00000.jpg"
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
