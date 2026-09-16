"""The dataset cropping pipeline, tested offline against an in-memory archive of generated images."""

from __future__ import annotations

import hashlib
import io
import random
import zipfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

cv2 = pytest.importorskip("cv2")

from src.crop_hagrid_hands import (
    MIN_CROP,
    ROOT,
    SEED,
    CropRun,
    RangeFile,
    crop_class,
    crop_member,
    largest_box,
    seeded_order,
    square_crop,
)

CENTRE_BOX = [0.3, 0.3, 0.4, 0.4]


def member(gesture: str, uuid: str) -> str:
    return f"{ROOT}/hagrid_500k/train_val_{gesture}/{uuid}.jpg"


def jpeg(side: int, seed: int) -> bytes:
    image = np.random.default_rng(seed).integers(0, 255, (side, side, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded.tobytes())


def archive(files: dict[str, bytes]) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        for name, data in files.items():
            writer.writestr(name, data)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


# --- one image ----------------------------------------------------------------------------------
def test_a_good_image_crops_and_encodes() -> None:
    zf = archive({member("fist", "a"): jpeg(384, 1)})
    outcome = crop_member(zf, set(zf.namelist()), "fist", "a", CENTRE_BOX, set())
    assert outcome.status == "ok"
    assert outcome.side >= MIN_CROP
    assert outcome.data is not None
    assert outcome.digest == hashlib.md5(outcome.data, usedforsecurity=False).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(outcome.data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] == decoded.shape[1] == outcome.side


def test_each_skip_reason_is_reported() -> None:
    files = {
        member("fist", "tiny"): jpeg(100, 2),
        member("fist", "garbage"): b"this is not a jpeg",
        member("fist", "twin"): jpeg(384, 3),
    }
    zf = archive(files)
    members = set(zf.namelist())
    assert crop_member(zf, members, "fist", "absent", CENTRE_BOX, set()).status == "missing"
    assert crop_member(zf, members, "fist", "garbage", CENTRE_BOX, set()).status == "unreadable"
    assert crop_member(zf, members, "fist", "tiny", CENTRE_BOX, set()).status == "too_small"
    first = crop_member(zf, members, "fist", "twin", CENTRE_BOX, set())
    again = crop_member(zf, members, "fist", "twin", CENTRE_BOX, {first.digest})
    assert again.status == "duplicate"


# --- a whole class --------------------------------------------------------------------------------
def test_crop_class_writes_until_the_target_and_counts_every_skip(tmp_path: Path) -> None:
    files = {member("palm", uuid): jpeg(384, index) for index, uuid in enumerate(("a", "b", "c"))}
    zf = archive(files)
    (tmp_path / "right").mkdir()
    run = CropRun(zf=zf, members=set(zf.namelist()), target=2, fetched_mb=lambda: 0.0)
    run.used_uuids.add("a")  # already used by another class
    order = [
        ("palm", "a", CENTRE_BOX),
        ("palm", "missing", CENTRE_BOX),
        ("palm", "b", CENTRE_BOX),
        ("palm", "c", CENTRE_BOX),
    ]

    crop_class(run, "right", order, output_dir=str(tmp_path))

    assert sorted(p.name for p in (tmp_path / "right").iterdir()) == [
        "right_00000.jpg",
        "right_00001.jpg",
    ]
    assert run.counts["right"] == 2
    assert run.skipped["right"]["reused"] == 1
    assert run.skipped["right"]["missing"] == 1
    assert run.used_uuids == {"a", "b", "c"}
    assert len(run.seen_hashes) == 2


# --- geometry and ordering --------------------------------------------------------------------------
@given(
    width=st.integers(64, 640),
    height=st.integers(64, 640),
    x=st.floats(0.0, 0.9),
    y=st.floats(0.0, 0.9),
    w=st.floats(0.05, 1.0),
    h=st.floats(0.05, 1.0),
)
def test_square_crop_is_within_a_pixel_of_square_and_inside_the_image(
    width: int, height: int, x: float, y: float, w: float, h: float
) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    crop = square_crop(image, [x, y, min(w, 1.0 - x), min(h, 1.0 - y)])
    if crop is None:
        return
    assert abs(crop.shape[0] - crop.shape[1]) <= 1, "crops must be square to within the quirk"
    assert max(crop.shape[:2]) <= min(width, height)


def test_square_crop_rounding_quirk_is_pinned() -> None:
    """The frozen geometry's one-pixel overshoot, found by Hypothesis. See square_crop's docstring.

    Pinned so the quirk cannot change silently: the dataset and the lineage replay depend on
    this exact output. Changing it is a deliberate decision, not a side effect.
    """
    crop = square_crop(np.zeros((178, 135, 3), dtype=np.uint8), [0.5, 0.0, 0.5, 0.5])
    assert crop is not None
    assert crop.shape[:2] == (134, 133)


DATASET = Path(__file__).resolve().parent.parent / "dataset"


@pytest.mark.skipif(not any(DATASET.rglob("*.jpg")), reason="dataset images are not tracked in git")
def test_every_dataset_image_is_exactly_square() -> None:
    """The quirk above never reached the dataset: every accepted crop is exactly square."""
    from PIL import Image

    skewed = []
    for path in sorted(DATASET.rglob("*.jpg")):
        with Image.open(path) as image:
            if image.width != image.height:
                skewed.append((path.name, image.size))
    assert skewed == []


def test_largest_box_takes_the_biggest_box_with_the_wanted_label() -> None:
    record = {
        "bboxes": [[0, 0, 0.1, 0.1], [0, 0, 0.5, 0.5], [0, 0, 0.3, 0.3]],
        "labels": ["fist", "no_gesture", "fist"],
    }
    assert largest_box(record, "fist") == [0, 0, 0.3, 0.3]
    assert largest_box(record, "palm") is None


def test_seeded_order_shares_one_random_stream_across_classes() -> None:
    candidates = {
        label: [(label, str(i), CENTRE_BOX) for i in range(20)] for label in ("left", "up")
    }
    order = seeded_order(candidates, ["left", "up"])
    rng = random.Random(SEED)
    expected_left = list(candidates["left"])
    rng.shuffle(expected_left)
    expected_up = list(candidates["up"])
    rng.shuffle(expected_up)
    assert order == {"left": expected_left, "up": expected_up}
    assert seeded_order(candidates, ["up"])["up"] != expected_up, "the order depends on the run"


def test_range_reader_refuses_non_https_urls() -> None:
    with pytest.raises(ValueError, match="non-https"):
        RangeFile("http://example.com/archive.zip", 10)
