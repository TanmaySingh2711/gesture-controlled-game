"""The real training pipeline, end to end on the GPU, redirected away from every frozen file.

Training normally writes `model/best_direction_model.pt`, so it can never be run casually. Here
its output paths are redirected into a temporary folder and both stages are cut to one epoch,
which exercises every step - seeding, both stages, checkpoint selection and reload, the
validation breakdown, history and curves - in well under a minute, and then proves the frozen
checkpoint was not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src import train_model

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_training_runs_end_to_end_without_touching_the_frozen_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_path = train_model.CHECKPOINT_PATH
    frozen_digest = train_model.file_sha256(frozen_path)

    monkeypatch.setattr(train_model, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(train_model, "CHECKPOINT_PATH", str(tmp_path / "trained.pt"))
    monkeypatch.setattr(train_model, "HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr(train_model, "CURVES_PATH", str(tmp_path / "curves.png"))
    monkeypatch.setattr(train_model, "STAGE1_EPOCHS", 1)
    monkeypatch.setattr(train_model, "STAGE2_EPOCHS", 1)

    best, batch_size = train_model.train(32)

    assert batch_size == 32
    assert 0.0 <= best["val_acc"] <= 1.0
    assert best["stage"] in ("stage1_head", "stage2_finetune")
    for name in ("trained.pt", "history.json", "curves.png"):
        assert (tmp_path / name).is_file(), name

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["stage_epochs"] == {"stage1_head": 1, "stage2_finetune": 1}
    assert history["test_split_used"] is False
    assert len(history["epochs"]) == 2
    assert history["validation_breakdown"]["total"] == 200

    _model, payload = train_model.load_direction_checkpoint(
        str(tmp_path / "trained.pt"), expected_sha256=None
    )
    assert payload["class_to_index"] == {"left": 0, "right": 1, "up": 2, "down": 3}

    assert train_model.file_sha256(frozen_path) == frozen_digest, "the frozen checkpoint changed"
