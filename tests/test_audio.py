"""Sound is optional: synthesis must be well-formed, and a missing device must mean silence."""

from __future__ import annotations

import logging
from typing import ClassVar

import pygame
import pytest

from game.audio import RECIPES, SAMPLE_RATE, Audio, synthesise


@pytest.mark.parametrize("name", list(RECIPES))
def test_every_effect_synthesises_valid_16_bit_audio(name: str) -> None:
    parts = RECIPES[name]
    samples = synthesise(parts)
    assert len(samples) == sum(max(1, int(SAMPLE_RATE * part[2])) for part in parts)
    assert all(-32768 <= value <= 32767 for value in samples)
    assert max(abs(value) for value in samples) > 1000, "effect is effectively silent"


def test_effects_fade_in_and_out_so_they_do_not_click() -> None:
    samples = synthesise([(440.0, 440.0, 0.2, "square", 0.5)])
    assert abs(samples[0]) < 500
    assert abs(samples[-1]) < 500


def test_disabled_audio_is_a_silent_no_op() -> None:
    audio = Audio(enabled=False)
    assert not audio.available
    audio.play("power")
    audio.waka()
    assert audio.toggle_mute() is True
    assert audio.toggle_mute() is False


def test_missing_audio_device_degrades_to_silence(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def no_device(*_args: object, **_kwargs: object) -> None:
        raise pygame.error("no available audio device")

    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)
    monkeypatch.setattr(pygame.mixer, "init", no_device)
    with caplog.at_level(logging.WARNING, logger="game.audio"):
        audio = Audio(enabled=True)
    assert not audio.available
    assert "audio unavailable" in caplog.text
    audio.play("death")  # must not raise


def test_a_mixer_that_silently_fails_to_open_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: None)
    monkeypatch.setattr(pygame.mixer, "init", lambda **_kwargs: None)
    with caplog.at_level(logging.WARNING, logger="game.audio"):
        audio = Audio(enabled=True)
    assert not audio.available
    assert "mixer did not initialise" in caplog.text


class FakeSound:
    played: ClassVar[list[int]] = []

    def __init__(self, *, buffer: bytes) -> None:
        self.size = len(buffer)

    def play(self) -> None:
        FakeSound.played.append(self.size)


def test_a_working_mixer_builds_every_effect_and_plays_unless_muted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSound.played = []
    monkeypatch.setattr(pygame.mixer, "get_init", lambda: (SAMPLE_RATE, -16, 2))
    monkeypatch.setattr(pygame.mixer, "Sound", FakeSound)
    audio = Audio(enabled=True)
    assert audio.available

    audio.play("power")
    # two channels of signed 16-bit samples: every mono sample becomes four bytes
    assert FakeSound.played == [4 * len(synthesise(RECIPES["power"]))]

    audio.play("no-such-effect")
    audio.muted = True
    audio.play("power")
    assert len(FakeSound.played) == 1


def test_pellet_chirp_alternates_between_its_two_tones(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = Audio(enabled=False)
    calls: list[str] = []
    monkeypatch.setattr(audio, "play", calls.append)
    for _ in range(3):
        audio.waka()
    assert calls == ["waka_a", "waka_b", "waka_a"]
