"""The robustness evaluation: its corruptions, its scoring, and a full run on the CPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from src import evaluate_robustness as rob


def picture() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(40, 216, (64, 64, 3), np.uint8))


def mean(image: Image.Image) -> float:
    return float(np.asarray(image, dtype=np.float32).mean())


def test_every_condition_keeps_the_image_size_and_mode() -> None:
    source = picture()
    for condition in rob.CONDITIONS:
        result = source if condition.corrupt is None else condition.corrupt(source, 3)
        assert (result.size, result.mode) == ((64, 64), "RGB"), condition.name


def test_corruptions_move_the_image_the_way_they_claim() -> None:
    source = picture()
    assert (
        mean(rob.brightness(0.4)(source, 0)) < mean(source) < mean(rob.brightness(1.6)(source, 0))
    )
    warm = np.asarray(rob.colour_cast(1.25, 0.75)(source, 0), dtype=np.float32)
    plain = np.asarray(source, dtype=np.float32)
    assert warm[..., 0].mean() > plain[..., 0].mean()
    assert warm[..., 2].mean() < plain[..., 2].mean()
    blurred = np.asarray(rob.blur(3.0)(source, 0), dtype=np.float32)
    assert blurred.std() < plain.std()


def test_noise_is_seeded_per_image_so_reruns_match() -> None:
    source = picture()
    noisy = rob.noise(15.0)
    assert np.array_equal(np.asarray(noisy(source, 5)), np.asarray(noisy(source, 5)))
    assert not np.array_equal(np.asarray(noisy(source, 5)), np.asarray(noisy(source, 6)))


def test_entries_come_from_the_manifest_in_a_fixed_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"up/b.jpg": {}, "left/a.jpg": {}}}), "utf-8")
    assert rob.load_entries(manifest) == [
        {"path": "dataset/external/left/a.jpg", "label": 0},
        {"path": "dataset/external/up/b.jpg", "label": 2},
    ]
    manifest.write_text(json.dumps({"files": {"jump/a.jpg": {}}}), "utf-8")
    with pytest.raises(ValueError, match="unknown class 'jump'"):
        rob.load_entries(manifest)


@pytest.fixture
def external(tiny_dataset: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Path:
    """The tiny validation images laid out as dataset/external/ with a manifest."""
    root: Path = tiny_dataset["root"]
    files: dict[str, dict[str, Any]] = {}
    for number, entry in enumerate(tiny_dataset["splits"]["val"]):
        relative = f"{entry['class']}/{entry['class']}_{number:05d}.jpg"
        (root / "dataset" / "external" / entry["class"]).mkdir(parents=True, exist_ok=True)
        shutil.copy(root / entry["path"], root / "dataset" / "external" / relative)
        files[relative] = {}
    (root / "dataset" / "external" / "manifest.json").write_text(
        json.dumps({"files": files}), "utf-8"
    )
    monkeypatch.setattr(rob, "PROJECT_ROOT", root)
    monkeypatch.setattr(rob, "MANIFEST_PATH", root / "dataset" / "external" / "manifest.json")
    monkeypatch.setattr(rob, "REPORT_PATH", root / "reports" / "robustness.json")
    return root


@pytest.mark.slow
def test_a_full_run_scores_every_condition_on_the_cpu(
    external: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rob.main(["--device", "cpu"]) == 0
    report = json.loads((external / "reports" / "robustness.json").read_text("utf-8"))
    assert [row["condition"] for row in report["conditions"]] == [c.name for c in rob.CONDITIONS]
    for row in report["conditions"]:
        assert row["images"] == 8
        assert 0.0 <= row["wrong_and_accepted_at_0.90"] <= row["accepted_at_0.90"] <= 1.0
    accuracies = [row["accuracy"] for row in report["conditions"]]
    assert report["clean_accuracy"] == accuracies[0]
    assert report["worst_accuracy"] == min(accuracies)
    assert "worst:" in capsys.readouterr().out


def test_missing_images_or_manifest_stop_the_run(
    external: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    next((external / "dataset" / "external").rglob("*.jpg")).unlink()
    assert rob.main(["--device", "cpu"]) == 1
    assert "image(s) missing" in capsys.readouterr().out
    (external / "dataset" / "external" / "manifest.json").unlink()
    assert rob.main(["--device", "cpu"]) == 1
    assert "manifest.json not found" in capsys.readouterr().out
