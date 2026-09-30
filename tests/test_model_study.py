"""The architecture study's plan, training run, summary and resume logic - on the CPU, in seconds.

The real study is 42 GPU runs over the full dataset. Here pretrained downloads are replaced by
randomly initialised networks, the dataset by the tiny stand-in, and the epochs by one each, so
every code path runs in CI without a GPU, a download or the dataset images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from src import data_pipeline, model_study
from src.model_study import ARCHITECTURES, RunConfig

pytestmark = pytest.mark.slow

CPU = torch.device("cpu")


@pytest.fixture
def no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every torchvision constructor builds its architecture with random weights instead."""
    for name in (
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "resnet18",
        "shufflenet_v2_x1_0",
    ):
        constructor = getattr(model_study.models, name)
        monkeypatch.setattr(
            model_study.models,
            name,
            lambda *_args, _build=constructor, **_kwargs: _build(weights=None),
        )


def test_the_quick_plan_is_one_run_per_architecture() -> None:
    plan = model_study.study_plan(quick=True)
    assert [config.architecture for config in plan] == list(ARCHITECTURES)
    assert {config.seed for config in plan} == {model_study.SEEDS[0]}


def test_the_full_plan_is_the_documented_42_runs_without_duplicates() -> None:
    plan = model_study.study_plan(quick=False)
    assert len(plan) == 42
    assert len({config.key for config in plan}) == 42
    # The P4 recipe is covered by the architecture study, never repeated as a sweep point.
    sweeps = [c for c in plan if c.unfreeze_from is not None]
    assert all(
        (c.stage2_lr, c.unfreeze_from) != (model_study.STAGE2_LR, model_study.UNFREEZE_FROM)
        for c in sweeps
    )


@pytest.mark.usefixtures("no_downloads")
@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_every_architecture_gets_a_four_way_head_and_a_finetune_block(name: str) -> None:
    architecture = ARCHITECTURES[name]
    model = architecture.build()
    output = model.eval()(torch.zeros(1, 3, 160, 160))
    assert output.shape == (1, model_study.NUM_CLASSES)
    assert list(architecture.head(model))
    assert architecture.finetune(model, None)


def test_a_head_that_is_not_linear_is_refused() -> None:
    with pytest.raises(TypeError, match="expected a Linear classifier"):
        model_study._new_head(torch.nn.Sequential(torch.nn.ReLU()), 0)


@pytest.mark.usefixtures("no_downloads")
def test_one_study_run_trains_selects_and_saves_on_the_cpu(
    tiny_dataset: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_study, "LATENCY_WARMUP", 1)
    monkeypatch.setattr(model_study, "LATENCY_RUNS", 3)
    train_set, val_set, _ = data_pipeline.get_datasets(tiny_dataset)
    loaders = (DataLoader(train_set, batch_size=8, shuffle=True), DataLoader(val_set, batch_size=8))
    save_to = tmp_path / "studies" / "run.pt"

    result = model_study.train_one(
        RunConfig("mobilenet_v2", 42, unfreeze_from=17),
        loaders,
        CPU,
        stage1_epochs=1,
        stage2_epochs=1,
        save_to=save_to,
    )

    assert result["epochs_run"] == 2
    assert result["best_stage"] in ("stage1_head", "stage2_finetune")
    assert 0.0 <= result["val_accuracy"] <= 1.0
    assert 0.0 <= result["val_ece"] <= 1.0
    assert result["latency_batch1_median_ms"] > 0.0
    assert result["peak_vram_mb"] == 0.0
    saved = torch.load(save_to, weights_only=True)
    assert saved["result"]["key"] == result["key"]


def fake_run(architecture: str, seed: int, loss: float, **extra: Any) -> dict[str, Any]:
    config = RunConfig(architecture, seed, **extra)
    return {
        "architecture": architecture,
        "seed": seed,
        "stage2_lr": config.stage2_lr,
        "unfreeze_from": config.unfreeze_from,
        "key": config.key,
        "val_loss": loss,
        "val_accuracy": 1.0 - loss,
        "val_ece": loss / 10,
        "latency_batch1_median_ms": 5.0,
        "train_seconds": 30.0,
        "parameters": 1000,
    }


def test_the_summary_separates_architectures_from_recipe_sweeps() -> None:
    runs = [
        fake_run("mobilenet_v2", 42, 0.04),
        fake_run("mobilenet_v2", 7, 0.06),
        fake_run("resnet18", 42, 0.05),
        fake_run("mobilenet_v2", 42, 0.08, stage2_lr=3e-4, unfreeze_from=10),
    ]
    summary = model_study.summarise(runs)
    assert set(summary["architectures"]) == {"mobilenet_v2", "resnet18"}
    loss = summary["architectures"]["mobilenet_v2"]["val_loss"]
    assert loss["mean"] == pytest.approx(0.05)
    assert loss["runs"] == 2
    assert summary["architectures"]["resnet18"]["val_loss"]["std"] == 0.0
    recipes = {(r["stage2_lr"], r["unfreeze_from"]) for r in summary["mobilenet_v2_recipes"]}
    assert recipes == {(model_study.STAGE2_LR, model_study.UNFREEZE_FROM), (3e-4, 10)}


def test_main_resumes_skips_finished_runs_and_keeps_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "model_study_quick.json"
    finished = fake_run("mobilenet_v2", 42, 0.04)
    report.write_text(
        json.dumps({"runs": [finished], "notes": ["kept across resumes"]}), encoding="utf-8"
    )
    trained: list[str] = []

    def fake_train_one(config: RunConfig, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        trained.append(config.architecture)
        return fake_run(config.architecture, config.seed, 0.05)

    monkeypatch.setattr(model_study, "QUICK_REPORT_PATH", report)
    monkeypatch.setattr(model_study, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(model_study, "require_cuda", lambda: None)
    monkeypatch.setattr(model_study, "get_dataloaders", lambda batch_size: (None, None, None))
    monkeypatch.setattr(model_study, "train_one", fake_train_one)

    assert model_study.main(["--quick"]) == 0

    assert trained == [name for name in ARCHITECTURES if name != "mobilenet_v2"]
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["notes"] == ["kept across resumes"]
    assert written["test_split_used"] is False
    assert len(written["runs"]) == len(ARCHITECTURES)
