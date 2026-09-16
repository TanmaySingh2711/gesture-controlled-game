"""The lineage audit's offline parts: reading the split file and the subject-leakage arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from src import audit_hagrid_lineage
from src.audit_hagrid_lineage import leakage_report, load_splits


def entry(name: str) -> dict[str, object]:
    return {"class": "left", "label": 0, "path": f"dataset/left/{name}"}


def test_load_splits_ignores_the_counts_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: `counts` is also keyed train/val/test, sorts first, and holds only numbers."""
    split_file = tmp_path / "data_splits.json"
    split_file.write_text(
        json.dumps(
            {
                "counts": {"train": 2, "val": 1, "test": 1},
                "splits": {
                    "train": [entry("a.jpg"), entry("b.jpg")],
                    "val": [entry("c.jpg")],
                    "test": [entry("d.jpg")],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_hagrid_lineage, "SPLITS_FILE", split_file)
    assert load_splits() == {
        "train": ["dataset/left/a.jpg", "dataset/left/b.jpg"],
        "val": ["dataset/left/c.jpg"],
        "test": ["dataset/left/d.jpg"],
    }


def test_load_splits_refuses_a_file_without_real_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_file = tmp_path / "data_splits.json"
    split_file.write_text(json.dumps({"counts": {"train": 2, "val": 1, "test": 1}}))
    monkeypatch.setattr(audit_hagrid_lineage, "SPLITS_FILE", split_file)
    with pytest.raises(RuntimeError, match="no train/val/test section"):
        load_splits()


def test_leakage_report_counts_people_shared_across_splits() -> None:
    files = {
        "dataset/left/a.jpg": {"user_id": "ann"},
        "dataset/left/b.jpg": {"user_id": "bob"},
        "dataset/left/c.jpg": {"user_id": "ann"},
        "dataset/left/d.jpg": {"user_id": "cat"},
    }
    splits = {
        "train": ["dataset/left/a.jpg", "dataset/left/b.jpg"],
        "val": ["dataset/left/d.jpg"],
        "test": ["dataset/left/c.jpg", "dataset/left/missing.jpg"],
    }
    report = leakage_report(files, splits)
    assert report["people_per_split"] == {"train": 2, "val": 1, "test": 1}
    assert report["shared_between_splits"]["train/test"] == {
        "shared_people": 1,
        "test_images_from_shared_people": 1,
    }
    assert report["shared_between_splits"]["train/val"]["shared_people"] == 0
    assert report["split_images_without_lineage"] == 1
