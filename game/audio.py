"""Procedurally synthesised sound effects - no audio files, no extra dependencies.

Sound here is a usability aid, not decoration. When a player is watching their hand in the
camera panel, the pellet chirp confirms Pac-Man is still moving, and distinct cues mark a power
pellet, an eaten ghost, a lost life and a cleared round without needing to look at the maze.

Every effect is generated at start-up from simple sweeps, so there are no asset files to ship
and nothing with a licence. Press ``M`` in game to mute.

If the machine has no usable audio device - a CI runner, a remote session, the dummy SDL
driver the tests use - opening the mixer fails. That is logged once and every sound becomes a
silent no-op; the game itself never depends on audio being available.
"""

from __future__ import annotations

import logging
import math
from array import array
from typing import Final

import pygame

log = logging.getLogger(__name__)

SAMPLE_RATE: Final = 22_050
_FADE_IN: Final = 0.004  # seconds - short fades stop every effect from clicking
_FADE_OUT: Final = 0.012

# name -> list of (start Hz, end Hz, seconds, waveform, volume)
Part = tuple[float, float, float, str, float]
RECIPES: Final[dict[str, list[Part]]] = {
    "waka_a": [(360.0, 620.0, 0.055, "square", 0.16)],
    "waka_b": [(620.0, 360.0, 0.055, "square", 0.16)],
    "power": [(260.0, 900.0, 0.28, "sine", 0.35)],
    "ghost": [(1400.0, 240.0, 0.26, "square", 0.22)],
    "fruit": [(900.0, 1500.0, 0.09, "sine", 0.35), (1500.0, 1500.0, 0.07, "sine", 0.3)],
    "death": [(820.0, 90.0, 0.95, "square", 0.22)],
    "round": [
        (523.0, 523.0, 0.1, "sine", 0.35),
        (659.0, 659.0, 0.1, "sine", 0.35),
        (784.0, 784.0, 0.1, "sine", 0.35),
        (1047.0, 1047.0, 0.18, "sine", 0.35),
    ],
    "life": [(880.0, 880.0, 0.11, "sine", 0.35), (1175.0, 1175.0, 0.16, "sine", 0.35)],
}


def synthesise(parts: list[Part], rate: int = SAMPLE_RATE) -> array[int]:
    """Signed 16-bit mono samples for a sequence of frequency sweeps."""
    samples = array("h")
    for start_hz, end_hz, seconds, waveform, volume in parts:
        count = max(1, int(rate * seconds))
        fade_in = max(1, int(rate * _FADE_IN))
        fade_out = max(1, int(rate * _FADE_OUT))
        phase = 0.0
        for index in range(count):
            progress = index / count
            phase += 2.0 * math.pi * (start_hz + (end_hz - start_hz) * progress) / rate
            value = math.sin(phase)
            if waveform == "square":
                value = 1.0 if value >= 0.0 else -1.0
            envelope = min(1.0, index / fade_in, (count - index) / fade_out)
            samples.append(int(value * envelope * volume * 32767))
    return samples


class Audio:
    """Named sound effects that fail safe: no device means silence, never an exception."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.muted = False
        self.available = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._waka_toggle = False
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
            init = pygame.mixer.get_init()
            if not init:
                raise pygame.error("mixer did not initialise")
            channels = init[2]
            for name, parts in RECIPES.items():
                mono = synthesise(parts, init[0])
                if channels > 1:
                    mono = array("h", (value for value in mono for _ in range(channels)))
                self._sounds[name] = pygame.mixer.Sound(buffer=mono.tobytes())
        except pygame.error as error:
            log.warning("audio unavailable (%s); sound effects are disabled", error)
            self._sounds.clear()
            return
        self.available = True

    def play(self, name: str) -> None:
        if not self.available or self.muted:
            return
        sound = self._sounds.get(name)
        if sound is not None:
            sound.play()

    def waka(self) -> None:
        """The alternating two-tone chirp of eating pellets."""
        self._waka_toggle = not self._waka_toggle
        self.play("waka_a" if self._waka_toggle else "waka_b")

    def toggle_mute(self) -> bool:
        """Flip mute and return the new state."""
        self.muted = not self.muted
        return self.muted
