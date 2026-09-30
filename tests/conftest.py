"""Shared pytest configuration: a headless display, and automatic skipping of hardware tests.

Tests needing the CUDA model are marked ``gpu`` and skip when CUDA is unavailable, so the suite
also runs on machines without an NVIDIA GPU. Tests needing a physical webcam are marked
``webcam`` and run only when ``GESTURE_WEBCAM_TESTS=1`` is set: a webcam can be busy or absent
for reasons that have nothing to do with the code, and that must not fail an ordinary run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

if TYPE_CHECKING:
    from game.engine import Game


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    webcam_enabled = os.environ.get("GESTURE_WEBCAM_TESTS") == "1"
    cuda: bool | None = None
    for item in items:
        if "webcam" in item.keywords and not webcam_enabled:
            item.add_marker(pytest.mark.skip(reason="set GESTURE_WEBCAM_TESTS=1 to run"))
        if "gpu" in item.keywords:
            if cuda is None:
                cuda = _cuda_available()
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="CUDA is not available"))


@pytest.fixture
def playing_game() -> Game:
    """A headless game already past its READY pause, ready to be driven frame by frame."""
    from game.engine import PLAYING, Game

    instance = Game(headless=True)
    instance.state = PLAYING
    instance.state_timer = 0.0
    return instance


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A throwaway project root with a few images per class, and a split over them.

    The real dataset images are not in git, so CI has none. This stands in for them: every
    module that resolves image paths against the project root is pointed at the temporary
    folder, so the real training and evaluation code runs unchanged on these images.
    Each class gets a distinct colour so a model can actually learn something from them.
    """
    pytest.importorskip("torch")
    import numpy as np
    from PIL import Image

    from src import data_pipeline

    colours = {
        "left": (200, 40, 40),
        "right": (40, 200, 40),
        "up": (40, 40, 200),
        "down": (200, 200, 40),
    }
    rng = np.random.default_rng(7)
    per_split = {"train": 4, "val": 2, "test": 2}
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in per_split}
    for name, index in data_pipeline.CLASS_TO_INDEX.items():
        folder = tmp_path / "dataset" / name
        folder.mkdir(parents=True)
        number = 0
        for split, count in per_split.items():
            for _ in range(count):
                noise = rng.integers(-30, 30, (64, 64, 3))
                pixels = np.clip(np.array(colours[name]) + noise, 0, 255).astype(np.uint8)
                file_name = f"{name}_{number:05d}.jpg"
                Image.fromarray(pixels).save(folder / file_name)
                splits[split].append(
                    {"path": f"dataset/{name}/{file_name}", "label": index, "class": name}
                )
                number += 1

    monkeypatch.setattr(data_pipeline, "PROJECT_ROOT", str(tmp_path))
    return {
        "root": tmp_path,
        "class_to_index": dict(data_pipeline.CLASS_TO_INDEX),
        "splits": splits,
    }


@pytest.fixture(scope="session", autouse=True)
def _pygame_session() -> Iterator[None]:
    yield
    import pygame

    pygame.quit()
