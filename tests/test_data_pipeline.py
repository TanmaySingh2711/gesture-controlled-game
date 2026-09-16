"""The data pipeline on the CPU: the split, the orientation rule, transforms, dataset and loaders.

`src/check_data_pipeline.py` covers the same ground against the real dataset and CUDA. These
tests need neither, so they run in CI: the split is rebuilt from a generated dataset in a
temporary directory, and the committed split is checked against the real one when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.transforms import v2

from src import data_pipeline
from src.data_pipeline import (
    CLASS_TO_INDEX,
    CLASSES,
    IMAGE_SIZE,
    MAX_SAFE_ROTATION,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    VAL_PER_CLASS,
    GestureDataset,
    build_split,
    denormalize,
    eval_transform,
    get_dataloaders,
    get_datasets,
    inference_transform,
    load_split,
    train_transform,
)

ROOT = Path(__file__).resolve().parent.parent
PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS


def colour_image(size: tuple[int, int] = (24, 24), colour: tuple[int, int, int] = (90, 140, 200)):
    return Image.new("RGB", size, colour)


def top_bright_image() -> Image.Image:
    image = Image.new("RGB", (120, 120), (0, 0, 0))
    image.paste((255, 255, 255), (0, 0, 120, 60))
    return image


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A complete 4 x 500 dataset of small images, with every pipeline path redirected to it."""
    for label, name in enumerate(CLASSES):
        folder = tmp_path / "dataset" / name
        folder.mkdir(parents=True)
        for index in range(PER_CLASS):
            colour = (label * 60, index % 256, (index * 7) % 256)
            colour_image(colour=colour).save(folder / f"{name}_{index:05d}.jpg")
    monkeypatch.setattr(data_pipeline, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(data_pipeline, "DATASET_DIR", str(tmp_path / "dataset"))
    monkeypatch.setattr(data_pipeline, "SPLIT_PATH", str(tmp_path / "data_splits.json"))
    monkeypatch.setattr(data_pipeline, "MAPPING_PATH", str(tmp_path / "class_mapping.json"))
    return tmp_path


# --- split -------------------------------------------------------------------------------
def test_split_is_stratified_and_disjoint(tiny_dataset: Path) -> None:
    payload = build_split(force=True)
    splits = payload["splits"]
    assert payload["counts"] == {
        "train": TRAIN_PER_CLASS * 4,
        "val": VAL_PER_CLASS * 4,
        "test": TEST_PER_CLASS * 4,
    }
    expected = {"train": TRAIN_PER_CLASS, "val": VAL_PER_CLASS, "test": TEST_PER_CLASS}
    for split, per_class in expected.items():
        for name in CLASSES:
            entries = [e for e in splits[split] if e["class"] == name]
            assert len(entries) == per_class
            assert {e["label"] for e in entries} == {CLASS_TO_INDEX[name]}
    paths = [entry["path"] for split in splits.values() for entry in split]
    assert len(paths) == len(set(paths)) == PER_CLASS * 4


def test_split_rebuilds_byte_for_byte(tiny_dataset: Path) -> None:
    build_split(force=True)
    first = (tiny_dataset / "data_splits.json").read_bytes()
    build_split(force=True)
    assert (tiny_dataset / "data_splits.json").read_bytes() == first


def test_existing_split_is_reused_rather_than_rebuilt(tiny_dataset: Path) -> None:
    build_split(force=True)
    split_file = tiny_dataset / "data_splits.json"
    edited = json.loads(split_file.read_text(encoding="utf-8"))
    edited["seed"] = "sentinel"
    split_file.write_text(json.dumps(edited), encoding="utf-8")
    assert build_split()["seed"] == "sentinel"
    assert load_split()["seed"] == "sentinel"


def test_a_class_with_the_wrong_image_count_is_refused(tiny_dataset: Path) -> None:
    (tiny_dataset / "dataset" / "up" / "up_00000.jpg").unlink()
    with pytest.raises(RuntimeError, match=f"up: expected {PER_CLASS} images, found"):
        build_split(force=True)


def test_loading_a_missing_split_says_what_to_run(tiny_dataset: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_split"):
        load_split()


def test_mapping_file_records_the_frozen_classes(tiny_dataset: Path) -> None:
    build_split(force=True)
    mapping = json.loads((tiny_dataset / "class_mapping.json").read_text(encoding="utf-8"))
    assert mapping["class_to_index"] == {"left": 0, "right": 1, "up": 2, "down": 3}
    assert mapping["classes"] == ["left", "right", "up", "down"]
    assert mapping["index_to_class"] == {"0": "left", "1": "right", "2": "up", "3": "down"}


@pytest.mark.skipif(
    not (ROOT / "dataset" / "left").is_dir() or not any((ROOT / "dataset").rglob("*.jpg")),
    reason="dataset images are not tracked in git",
)
def test_committed_split_is_reproduced_from_the_real_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_pipeline, "SPLIT_PATH", str(tmp_path / "data_splits.json"))
    monkeypatch.setattr(data_pipeline, "MAPPING_PATH", str(tmp_path / "class_mapping.json"))
    build_split(force=True)
    for name in ("data_splits.json", "class_mapping.json"):
        committed = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        rebuilt = (tmp_path / name).read_bytes().replace(b"\r\n", b"\n")
        assert rebuilt == committed, f"{name} no longer matches a fresh rebuild"


# --- transforms --------------------------------------------------------------------------
def test_training_augmentation_never_flips_or_rotates_far() -> None:
    for transform in train_transform().transforms:
        assert not isinstance(transform, v2.RandomVerticalFlip | v2.RandomRotation)
        if isinstance(transform, v2.RandomAffine):
            assert max(abs(float(d)) for d in transform.degrees) <= MAX_SAFE_ROTATION


def test_augmentation_keeps_up_up() -> None:
    """Behavioural form of the orientation rule: a bright top half must stay on top."""
    torch.manual_seed(0)
    transform = train_transform()
    image = top_bright_image()
    for _ in range(50):
        pixels = denormalize(transform(image)).mean(dim=0)
        top, bottom = pixels[: IMAGE_SIZE // 2].mean(), pixels[IMAGE_SIZE // 2 :].mean()
        assert top > bottom + 0.3


def test_eval_preprocessing_is_deterministic_and_is_the_inference_contract() -> None:
    image = colour_image((200, 120))
    first = eval_transform()(image)
    assert first.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert first.dtype == torch.float32
    assert torch.equal(first, eval_transform()(image))
    assert torch.equal(first, inference_transform()(image))


def test_denormalize_undoes_normalisation() -> None:
    image = colour_image((IMAGE_SIZE, IMAGE_SIZE))
    raw = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)
    assert torch.allclose(denormalize(eval_transform()(image)), raw, atol=1e-5)


# --- dataset and loaders -----------------------------------------------------------------
def test_dataset_reads_images_and_labels(tiny_dataset: Path) -> None:
    entries = build_split(force=True)["splits"]["val"]
    dataset = GestureDataset(entries, eval_transform())
    image, label = dataset[len(dataset) - 1]
    assert len(dataset) == VAL_PER_CLASS * 4
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert label == entries[-1]["label"]


def test_datasets_use_augmentation_for_training_only(tiny_dataset: Path) -> None:
    train_set, val_set, test_set = get_datasets()
    assert len(train_set) == TRAIN_PER_CLASS * 4
    assert len(test_set) == TEST_PER_CLASS * 4
    augmenting = [type(t) for t in train_set.transform.transforms]
    assert v2.RandomHorizontalFlip in augmenting
    for split in (val_set, test_set):
        assert v2.RandomHorizontalFlip not in [type(t) for t in split.transform.transforms]


def test_loaders_shuffle_training_only(tiny_dataset: Path) -> None:
    build_split(force=True)
    train, val, test = get_dataloaders(batch_size=8, num_workers=0, pin_memory=False)
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    images, labels = next(iter(val))
    assert images.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert labels.tolist() == [CLASS_TO_INDEX["left"]] * 8
