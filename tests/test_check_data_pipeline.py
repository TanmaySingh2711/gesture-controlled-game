"""The P3 data-pipeline checks, run against the tiny stand-in dataset so CI exercises them too.

The full script (`python src/check_data_pipeline.py`) needs all 2,000 dataset images and runs in
tests/test_self_tests.py when they are present. Here each check runs on generated images: the
ones that must pass on any correct pipeline are asserted to pass, and the ones that check the
frozen 2,000-image split are asserted to *fail* on the tiny one - proving they actually look.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src import check_data_pipeline as cdp
from src import data_pipeline


@pytest.fixture
def results(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, str]]:
    """A fresh result list for each test, instead of the module's global one."""
    fresh: list[tuple[str, bool, str]] = []
    monkeypatch.setattr(cdp, "results", fresh)
    return fresh


def outcome(results: list[tuple[str, bool, str]]) -> dict[str, bool]:
    return {name: passed for name, passed, _ in results}


def test_the_frozen_mapping_agrees_everywhere(results: list[Any]) -> None:
    cdp.check_mapping(data_pipeline.load_split())
    assert outcome(results) == {
        "Mapping constant": True,
        "class_mapping.json": True,
        "No retired classes": True,
        "Split file mapping": True,
    }


def test_a_split_with_another_mapping_is_caught(results: list[Any]) -> None:
    split = data_pipeline.load_split()
    split["class_to_index"] = {"left": 0, "right": 1, "jump": 2, "neutral": 3}
    cdp.check_mapping(split)
    assert outcome(results)["Split file mapping"] is False


def test_the_training_transform_can_never_flip_a_thumb(results: list[Any]) -> None:
    cdp.check_transform_safety()
    assert all(outcome(results).values()), results


def test_a_vertical_flip_or_large_rotation_would_be_caught(
    results: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    dangerous = v2.Compose(
        [v2.RandomVerticalFlip(), v2.Compose([v2.RandomAffine(degrees=45)]), v2.ToImage()]
    )
    monkeypatch.setattr(cdp, "train_transform", lambda: dangerous)
    monkeypatch.setattr(cdp, "eval_transform", lambda: v2.Compose([v2.RandomHorizontalFlip()]))
    cdp.check_transform_safety()
    checks = outcome(results)
    assert checks["No vertical flip"] is False
    assert checks["Rotation bounded"] is False
    assert checks["Horizontal flip present"] is False
    assert checks["Eval transform clean"] is False


def test_every_class_survives_both_transforms(
    tiny_dataset: dict[str, Any], results: list[Any]
) -> None:
    cdp.check_every_class(tiny_dataset)
    assert outcome(results) == {"All four classes": True}


def test_loaders_batches_and_determinism(tiny_dataset: dict[str, Any], results: list[Any]) -> None:
    train_set, val_set, test_set = data_pipeline.get_datasets(tiny_dataset)
    loaders: dict[str, DataLoader[Any]] = {
        "train": DataLoader(train_set, batch_size=4, shuffle=True),
        "val": DataLoader(val_set, batch_size=4),
        "test": DataLoader(test_set, batch_size=4),
    }
    cdp.check_loader_config(loaders, 4, 0)
    cdp.check_batches(loaders, train_batches=2)
    cdp.check_determinism(tiny_dataset)
    assert all(outcome(results).values()), results
    assert "val labels valid" in outcome(results)


def test_the_split_checks_notice_a_split_that_is_not_the_frozen_one(
    tiny_dataset: dict[str, Any], results: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cdp, "PROJECT_ROOT", str(tiny_dataset["root"]))
    split = {**tiny_dataset, "splits": dict(tiny_dataset["splits"])}
    split["splits"]["test"] = [*split["splits"]["test"], split["splits"]["train"][0]]
    cdp.check_split(split)
    checks = outcome(results)
    assert checks["Split sizes"] is False  # 16/8/9 is not 1600/200/200
    assert checks["Per-class split"] is False
    assert checks["Split overlap"] is False  # one image put in both train and test
    assert checks["Full coverage"] is False
    assert checks["Label mapping"] is True


def test_the_qa_sheets_are_written_where_they_are_ignored(
    tiny_dataset: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa = tmp_path / "qa"
    monkeypatch.setattr(cdp, "PROJECT_ROOT", str(tiny_dataset["root"]))
    monkeypatch.setattr(cdp, "QA_DIR", str(qa))
    monkeypatch.setattr(cdp, "GRID_PATH", str(qa / "grid.jpg"))
    monkeypatch.setattr(cdp, "UPDOWN_GRID_PATH", str(qa / "updown.jpg"))

    cdp.write_augmentation_grid(tiny_dataset, variants=2)
    cdp.write_updown_grid(tiny_dataset, rows_per_class=1, variants=2)

    grid = cv2.imread(str(qa / "grid.jpg"))
    assert grid.shape[0] == 4 * data_pipeline.IMAGE_SIZE  # one row per class
    assert cv2.imread(str(qa / "updown.jpg")) is not None


def test_an_unreadable_image_stops_the_sheet(
    tiny_dataset: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cdp, "PROJECT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="cannot read"):
        cdp.write_augmentation_grid(tiny_dataset)
    with pytest.raises(RuntimeError, match="cannot read"):
        cdp.write_updown_grid(tiny_dataset)


def test_cuda_transfer_is_skipped_rather_than_failed_without_a_gpu(
    tiny_dataset: dict[str, Any],
    results: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train_set, _, _ = data_pipeline.get_datasets(tiny_dataset)
    cdp.check_cuda(DataLoader(train_set, batch_size=4))
    assert results == []
    assert "[SKIP] CUDA transfer" in capsys.readouterr().out


@pytest.mark.gpu
def test_batches_reach_the_gpu(tiny_dataset: dict[str, Any], results: list[Any]) -> None:
    train_set, _, _ = data_pipeline.get_datasets(tiny_dataset)
    cdp.check_cuda(DataLoader(train_set, batch_size=4))
    assert outcome(results) == {"CUDA transfer": True}
