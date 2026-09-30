"""The training and evaluation code on the CPU, over a tiny stand-in dataset - so CI runs it too.

The GPU tests in test_train_model.py and test_evaluate_model.py exercise the same code on the real
dataset, but CI has neither a GPU nor the dataset images. These run the identical functions on a
handful of generated images, write everything to a temporary folder, and prove the frozen
checkpoint and the frozen P5 artefacts are byte-for-byte unchanged afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from torch.utils.data import DataLoader

from src import data_pipeline, train_model
from src import evaluate_model as ev

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaders(
    split: dict[str, Any], batch_size: int
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """The real datasets and transforms over the tiny split, without worker processes."""
    train_set, val_set, test_set = data_pipeline.get_datasets(split)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )


# --- training ------------------------------------------------------------------------------
def test_both_training_stages_run_on_the_cpu_and_save_a_loadable_model(
    tiny_dataset: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_path = train_model.CHECKPOINT_PATH  # remembered before it is redirected below
    frozen_digest = sha256(frozen_path)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(train_model, "MODEL_DIR", str(out))
    monkeypatch.setattr(train_model, "CHECKPOINT_PATH", str(out / "trained.pt"))
    monkeypatch.setattr(train_model, "HISTORY_PATH", str(out / "history.json"))
    monkeypatch.setattr(train_model, "CURVES_PATH", str(out / "curves.png"))
    monkeypatch.setattr(train_model, "STAGE1_EPOCHS", 1)
    monkeypatch.setattr(train_model, "STAGE2_EPOCHS", 2)
    monkeypatch.setattr(train_model, "PATIENCE", 1)
    monkeypatch.setattr(
        train_model, "get_dataloaders", lambda batch_size: loaders(tiny_dataset, batch_size)
    )

    best, batch_size = train_model.train(8, CPU)

    assert batch_size == 8
    assert best["stage"] in ("stage1_head", "stage2_finetune")
    for name in ("trained.pt", "history.json", "curves.png"):
        assert (out / name).is_file(), name
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert history["test_split_used"] is False
    assert history["stage_epochs"]["stage1_head"] == 1
    assert history["validation_breakdown"]["total"] == 8  # two validation images per class
    assert train_model.verify_checkpoint(8, CPU)

    _model, payload = train_model.load_direction_checkpoint(
        str(out / "trained.pt"), expected_sha256=None
    )
    assert payload["class_to_index"] == data_pipeline.CLASS_TO_INDEX
    assert sha256(frozen_path) == frozen_digest, "the frozen checkpoint changed"


def test_the_validation_breakdown_counts_the_up_down_confusions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import numpy as np

    confusion = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 2, 0]])
    summary = train_model.print_validation_breakdown(confusion)
    assert summary["up_predicted_as_down"] == 1
    assert summary["down_predicted_as_up"] == 2
    assert summary["correct"] == 5
    assert "up->down 1   down->up 2" in capsys.readouterr().out


def test_training_retries_at_batch_16_after_running_out_of_gpu_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes: list[int] = []

    def fake_train(batch_size: int) -> tuple[dict[str, Any], int]:
        sizes.append(batch_size)
        if batch_size > 16:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return {}, batch_size

    monkeypatch.setattr(train_model, "require_cuda", lambda: None)
    monkeypatch.setattr(train_model, "train", fake_train)
    monkeypatch.setattr(train_model, "verify_checkpoint", lambda batch_size: batch_size == 16)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr("sys.argv", ["train_model.py", "--batch-size", "32"])

    assert train_model.main() == 0
    assert sizes == [32, 16]


def test_training_insists_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required for CNN training"):
        train_model.require_cuda()


# --- evaluation ----------------------------------------------------------------------------
def test_the_evaluation_pipeline_runs_on_the_cpu_and_leaves_p5_untouched(
    tiny_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frozen = {path: sha256(path) for path in (ev.METRICS_PATH, ev.PREDICTIONS_PATH)}
    monkeypatch.setattr(ev, "PROJECT_ROOT", str(tiny_dataset["root"]))
    monkeypatch.setattr(ev, "BENCHMARK_WARMUP", 1)
    real_benchmark = ev.benchmark
    monkeypatch.setattr(
        ev,
        "benchmark",
        lambda model, device, batch=1, runs=0: real_benchmark(model, device, batch=batch, runs=2),
    )
    model, payload = ev.load_model(CPU)
    entries = tiny_dataset["splits"]["val"]
    out = tmp_path / "eval"
    out.mkdir()
    paths = ev.EvaluationPaths(
        metrics=str(out / "metrics.json"),
        predictions=str(out / "predictions.csv"),
        confusion=str(out / "confusion.png"),
        misclassified=str(out / "misclassified.png"),
        low_confidence=str(out / "low.png"),
    )
    tally = dict.fromkeys(data_pipeline.CLASSES, 2)

    record = ev.run_evaluation(
        model, payload, entries, CPU, paths, ev.SplitInfo("validation", tally, overlap=0)
    )

    assert record.results["total"] == 8
    assert record.results["finite"] and record.results["softmax_sum_ok"]
    metrics = json.loads(Path(paths.metrics).read_text(encoding="utf-8"))
    assert metrics["device"] == "cpu"
    assert metrics["latency_batch1"]["runs"] == 2
    assert len(Path(paths.predictions).read_text(encoding="utf-8").splitlines()) == 9
    assert Path(paths.confusion).is_file()
    assert "CNN-only CPU latency" in capsys.readouterr().out

    # The safety checks exist for the real P5 run: on 8 CPU images they must refuse to pass.
    assert ev.safety_checks(record) == 1
    output = capsys.readouterr().out
    assert "[FAIL] CUDA inference" in output
    assert "[FAIL] frozen validation split, 200 images" in output
    assert "[PASS] probabilities finite" in output

    assert {path: sha256(path) for path in frozen} == frozen, "a frozen P5 artefact changed"


def test_the_frozen_test_split_passes_verification() -> None:
    tally, overlap = ev.verify_test_split(data_pipeline.load_split())
    assert tally == dict.fromkeys(data_pipeline.CLASSES, 50)
    assert overlap == 0


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda split: split["splits"]["test"].pop(), "199 test images"),
        (
            lambda split: split["splits"]["test"].append(split["splits"]["train"][0]),
            "shared with train/val",
        ),
        (lambda split: split.update(class_to_index={"up": 0}), "mapping disagrees"),
    ],
)
def test_a_tampered_test_split_is_refused(tamper: Any, message: str) -> None:
    split = data_pipeline.load_split()
    tamper(split)
    with pytest.raises(RuntimeError, match=message):
        ev.verify_test_split(split)


def test_evaluation_insists_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required for CNN evaluation"):
        ev.require_cuda()
