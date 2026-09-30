"""The dataset integrity check against a healthy dataset and a deliberately broken one.

The script's value is in what it refuses, so the broken dataset below plants one example of
every problem it is meant to catch, and each must turn its check to FAIL.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src import check_dataset as cd


def write(path: Path, width: int = 96, height: int = 96, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 256, (height, width, 3), np.uint8)
    cv2.imwrite(str(path), image)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root whose `dataset/` holds five good square crops per class."""
    seed = 0
    for label in cd.CLASSES:
        for index in range(5):
            seed += 1
            write(tmp_path / "dataset" / label / f"{label}_{index:05d}.jpg", seed=seed)
    monkeypatch.setattr(cd, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(cd, "QA_DIR", str(tmp_path / "qa"))
    monkeypatch.setattr(cd, "UPDOWN_PATH", str(tmp_path / "qa" / "updown.jpg"))
    # main() rebinds these two globals; registering them here makes sure they are restored.
    monkeypatch.setattr(cd, "DATASET_DIR", cd.DATASET_DIR)
    monkeypatch.setattr(cd, "GRID_PATH", cd.GRID_PATH)
    monkeypatch.setattr(cd, "TARGET_PER_CLASS", 5)
    monkeypatch.setattr(cd, "results", [])
    return tmp_path


def run(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["check_dataset.py", *arguments])
    return cd.main()


def verdicts() -> dict[str, bool]:
    return {name: passed for name, passed, _ in cd.results}


def test_a_healthy_dataset_passes_and_writes_both_qa_sheets(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "--grid", "--updown-grid") == 0
    assert all(verdicts().values()), cd.results
    assert (project / "qa" / "dataset_sample_grid.jpg").is_file()
    assert (project / "qa" / "updown.jpg").is_file()
    assert "RESULT: PASS (all 11 checks passed, 20 images)" in capsys.readouterr().out


def test_a_candidate_folder_can_be_checked_before_promotion(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(project / "dataset", project / "dataset_cropped")
    assert run(monkeypatch, "--dir", "dataset_cropped", "--updown-grid") == 0
    assert str(project / "dataset_cropped") == cd.DATASET_DIR


def test_every_planted_problem_is_caught(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = project / "dataset"
    shutil.rmtree(dataset / "down")  # a missing class folder
    (dataset / "jump").mkdir()  # a retired class creeping back
    (dataset / "up" / "up_00005.jpg").write_bytes(b"")  # an empty file
    (dataset / "up" / "up_00006.jpg").write_bytes(b"not an image")  # a corrupt file
    write(dataset / "up" / "up_00007.jpg", 32, 32)  # far too small to train on
    write(dataset / "up" / "up_00008.jpg", 120, 80)  # not square
    shutil.copy(dataset / "left" / "left_00000.jpg", dataset / "right" / "left_00000.jpg")

    assert run(monkeypatch) == 1

    checks = verdicts()
    for name in (
        "Class folders",
        "No stray classes",
        "Image counts",
        "Exact counts",
        "Class balance",
        "Readable images",
        "Usable dimensions",
        "Square crops",
        "Unique filenames",
        "No duplicate images",
    ):
        assert checks[name] is False, name
    assert checks["Three-channel RGB"] is True
    assert "RESULT: FAIL" in capsys.readouterr().out


def test_an_empty_dataset_fails_without_crashing(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for label in cd.CLASSES:
        shutil.rmtree(project / "dataset" / label)
    assert run(monkeypatch, "--grid") == 1
    checks = verdicts()
    assert checks["Class balance"] is False
    assert checks["Image counts"] is False
    assert not (project / "qa").exists(), "no sheet is drawn from nothing"


def test_a_missing_dataset_root_fails_without_crashing(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(monkeypatch, "--dir", "does_not_exist") == 1
    assert verdicts()["Class folders"] is False


def test_sheets_fill_gaps_when_a_class_has_few_or_broken_images(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "DATASET_DIR", str(project / "dataset"))
    monkeypatch.setattr(cd, "GRID_PATH", str(project / "qa" / "grid.jpg"))
    stats = cd.scan()
    stats["up"]["files"].append("vanished.jpg")  # listed, then deleted before drawing
    cd.write_sample_grid(stats)
    cd.write_updown_grid(stats)
    grid = cv2.imread(str(project / "qa" / "grid.jpg"))
    assert grid.shape[0] == len(cd.CLASSES) * cd.GRID_THUMB
