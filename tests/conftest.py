"""Shared pytest configuration: a headless display, and automatic skipping of hardware tests.

Tests needing the CUDA model are marked ``gpu`` and skip when CUDA is unavailable, so the suite
also runs on machines without an NVIDIA GPU. Tests needing a physical webcam are marked
``webcam`` and run only when ``GESTURE_WEBCAM_TESTS=1`` is set: a webcam can be busy or absent
for reasons that have nothing to do with the code, and that must not fail an ordinary run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

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


@pytest.fixture(scope="session", autouse=True)
def _pygame_session() -> Iterator[None]:
    yield
    import pygame

    pygame.quit()
