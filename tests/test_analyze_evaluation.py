"""The evaluation statistics, checked against values that can be worked out by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analyze_evaluation import (
    CLASSES,
    Predictions,
    analyse,
    bootstrap_interval,
    brier_score,
    calibration,
    load_predictions,
    macro_f1,
    negative_log_likelihood,
    selective_prediction,
    wilson_interval,
)


def test_wilson_interval_matches_the_textbook_value_for_198_of_200() -> None:
    low, high = wilson_interval(198, 200)
    assert low == pytest.approx(0.9643, abs=5e-4)
    assert high == pytest.approx(0.9973, abs=5e-4)


def test_wilson_interval_stays_inside_zero_and_one_at_the_extremes() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(10, 10)[0] < 1.0, "a perfect score on 10 images is not certainty"


@pytest.mark.parametrize(("successes", "trials"), [(-1, 10), (11, 10), (1, 0)])
def test_wilson_interval_rejects_impossible_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_macro_f1_is_one_when_every_prediction_is_right() -> None:
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    assert macro_f1(labels, labels) == 1.0


def test_macro_f1_matches_a_hand_worked_example() -> None:
    true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    predicted = np.array([0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    # class 0: P=1, R=0.5, F1=2/3 ; class 1: P=2/3, R=1, F1=0.8 ; classes 2 and 3: F1=1
    assert macro_f1(true, predicted) == pytest.approx((2 / 3 + 0.8 + 1 + 1) / 4)


def test_bootstrap_interval_collapses_for_perfect_predictions() -> None:
    labels = np.repeat(np.arange(4, dtype=np.int64), 25)
    assert bootstrap_interval(labels, labels, macro_f1, resamples=200) == (1.0, 1.0)


def test_bootstrap_is_reproducible_with_its_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    true = rng.integers(0, 4, 120).astype(np.int64)
    predicted = np.where(rng.random(120) < 0.8, true, rng.integers(0, 4, 120)).astype(np.int64)
    first = bootstrap_interval(true, predicted, macro_f1, resamples=300)
    assert first == bootstrap_interval(true, predicted, macro_f1, resamples=300)
    assert first[0] < macro_f1(true, predicted) < first[1]


def test_calibration_is_zero_for_a_perfectly_calibrated_model() -> None:
    confidence = np.array([1.0] * 10 + [0.5] * 10)
    correct = np.array([True] * 10 + [True] * 5 + [False] * 5)
    report = calibration(confidence, correct)
    assert report["ece"] == pytest.approx(0.0)
    assert report["mce"] == pytest.approx(0.0)
    assert sum(row["count"] for row in report["bins"]) == 20


def test_calibration_measures_overconfidence() -> None:
    confidence = np.full(10, 0.95)
    correct = np.array([True] * 5 + [False] * 5)
    report = calibration(confidence, correct)
    assert report["ece"] == pytest.approx(0.45)
    top = report["bins"][-1]
    assert top["count"] == 10 and top["accuracy"] == 0.5


def test_brier_and_nll_are_zero_for_certain_correct_predictions() -> None:
    probabilities = np.eye(4)[[0, 1, 2, 3]].astype(np.float64)
    labels = np.arange(4, dtype=np.int64)
    assert brier_score(probabilities, labels) == 0.0
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(0.0)


def test_selective_prediction_trades_coverage_for_accuracy() -> None:
    confidence = np.array([0.99, 0.95, 0.6, 0.55])
    correct = np.array([True, True, False, True])
    rows = {row["threshold"]: row for row in selective_prediction(confidence, correct, (0.0, 0.9))}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["errors_accepted"] == 1
    assert rows[0.9]["coverage"] == 0.5 and rows[0.9]["accuracy"] == 1.0


def test_the_saved_p5_predictions_reproduce_the_reported_accuracy() -> None:
    path = Path(__file__).resolve().parent.parent / "model" / "direction_test_predictions.csv"
    if not path.exists():
        pytest.skip("P5 predictions file not present")
    predictions = load_predictions(path)
    assert len(predictions.true) == 200
    report = analyse(predictions)
    assert report["accuracy"] == pytest.approx(0.99)
    assert set(report["per_class"]) == set(CLASSES)
    low, high = report["accuracy_wilson_95"]
    assert low < 0.99 < high


def test_load_rejects_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("file,true_index,predicted_index,confidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no predictions"):
        load_predictions(empty)


def test_predictions_correct_mask() -> None:
    predictions = Predictions(
        true=np.array([0, 1], dtype=np.int64),
        predicted=np.array([0, 2], dtype=np.int64),
        confidence=np.array([0.9, 0.8]),
        probabilities=np.zeros((2, 4)),
    )
    assert predictions.correct.tolist() == [True, False]
