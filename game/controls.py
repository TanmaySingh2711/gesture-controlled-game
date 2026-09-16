"""The generic control seam between any input source and the game's movement code.

Nothing in the game reads the keyboard. Input of any kind - a key press or a CNN
`stable_command` - becomes a *direction request*, and the player reads only that. This is the
whole reason the seam exists: swapping the input source must not require touching movement
physics.

    controls.request_direction("left")

Requests are buffered, not instantaneous. A request that is illegal right now is remembered
for REQUEST_GRACE seconds and applied at the first tile where it becomes legal, the same
pre-turn forgiveness the arcade game had. The grace window matters more here than it did in
the arcade: P6 measured the gesture recognizer at roughly 143 ms mean and 217 ms p95 to
stabilise a command, so a player physically cannot hit a one-frame turn window.
"""

from __future__ import annotations

from typing import Final

LEFT: Final = "left"
RIGHT: Final = "right"
UP: Final = "up"
DOWN: Final = "down"

DIRECTIONS: Final = (LEFT, RIGHT, UP, DOWN)

# (row delta, column delta) - the grid's row axis points down the screen.
DELTA: Final[dict[str, tuple[int, int]]] = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

OPPOSITE: Final[dict[str, str]] = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

# How long a request that cannot be obeyed yet stays alive.
#
# 0.35 s covers the recognizer's measured p95 of 217 ms with about 130 ms of human timing
# error left over, so a gesture aimed at an intersection still turns there. At the player's
# 6.2 tiles/second it spans roughly 2.2 tiles of travel - long enough to be forgiving, short
# enough that a stale request cannot surprise the player by firing at a much later junction.
#
# The first self-test run used 0.30 s and a turn two tiles away missed by a few hundredths of
# a second, which is exactly the kind of near-miss a gesture player would hit constantly.
REQUEST_GRACE: Final = 0.35


class Controls:
    """Holds the pending direction request. No keyboard, no pygame, no CNN."""

    def __init__(self, grace: float = REQUEST_GRACE) -> None:
        self.grace = grace
        self._direction: str | None = None
        self._age = 0.0

    # --- input side ----------------------------------------------------------------------
    def request_direction(self, direction: str) -> None:
        """Ask for a direction. The newest request replaces any older pending one."""
        if direction not in DELTA:
            raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")
        self._direction = direction
        self._age = 0.0

    # --- movement side -------------------------------------------------------------------
    @property
    def pending(self) -> str | None:
        """The buffered request, or None if there is nothing to apply."""
        return self._direction

    def consume(self) -> str | None:
        """Take the pending request and clear it. Returns the direction, or None."""
        direction, self._direction = self._direction, None
        self._age = 0.0
        return direction

    def tick(self, dt: float) -> None:
        """Age the buffered request and drop it once it is stale."""
        if self._direction is None:
            return
        self._age += dt
        if self._age >= self.grace:
            self.clear()

    def clear(self) -> None:
        self._direction = None
        self._age = 0.0

    # --- introspection, for tests and the debug overlay -----------------------------------
    @property
    def age(self) -> float:
        return self._age

    def __repr__(self) -> str:
        if self._direction is None:
            return "<Controls pending=None>"
        return f"<Controls pending={self._direction} age={self._age:.2f}s>"
