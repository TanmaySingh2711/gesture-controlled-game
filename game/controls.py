"""The generic control seam between any input source and the game's movement code.

Nothing in the game reads the keyboard. Input of any kind - a key press now, a CNN
`stable_command` in P8 - becomes a *direction request*, and the player reads only that. This
is the whole reason the seam exists: swapping the input source must not require touching
movement physics.

    controls.request_direction("left")

Requests are buffered, not instantaneous. A request that is illegal right now is remembered
for REQUEST_GRACE seconds and applied at the first tile where it becomes legal, the same
pre-turn forgiveness the arcade game had. The grace window matters more here than it did in
the arcade: P6 measured the gesture recognizer at roughly 143 ms mean and 217 ms p95 to
stabilise a command, so a player physically cannot hit a one-frame turn window.
"""

LEFT = "left"
RIGHT = "right"
UP = "up"
DOWN = "down"

DIRECTIONS = (LEFT, RIGHT, UP, DOWN)

# (row delta, column delta) - the grid's row axis points down the screen.
DELTA = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}

# How long a request that cannot be obeyed yet stays alive.
#
# 0.35 s covers the recognizer's measured p95 of 217 ms with about 130 ms of human timing
# error left over, so a gesture aimed at an intersection still turns there. At the player's
# 6.2 tiles/second it spans roughly 2.2 tiles of travel - long enough to be forgiving, short
# enough that a stale request cannot surprise the player by firing at a much later junction.
#
# The first self-test run used 0.30 s and a turn two tiles away missed by a few hundredths of
# a second, which is exactly the kind of near-miss a gesture player would hit constantly.
REQUEST_GRACE = 0.35


class Controls:
    """Holds the pending direction request. No keyboard, no pygame, no CNN."""

    def __init__(self, grace=REQUEST_GRACE):
        self.grace = grace
        self._direction = None
        self._age = 0.0

    # --- input side ----------------------------------------------------------------------
    def request_direction(self, direction):
        """Ask for a direction. The newest request replaces any older pending one."""
        if direction not in DELTA:
            raise ValueError(f"unknown direction {direction!r}; expected one of {DIRECTIONS}")
        self._direction = direction
        self._age = 0.0

    # --- movement side -------------------------------------------------------------------
    @property
    def pending(self):
        """The buffered request, or None if there is nothing to apply."""
        return self._direction

    def consume(self):
        """Take the pending request and clear it. Returns the direction, or None."""
        direction, self._direction = self._direction, None
        self._age = 0.0
        return direction

    def tick(self, dt):
        """Age the buffered request and drop it once it is stale."""
        if self._direction is None:
            return
        self._age += dt
        if self._age >= self.grace:
            self._direction = None
            self._age = 0.0

    def clear(self):
        self._direction = None
        self._age = 0.0

    # --- introspection, for tests and the debug overlay -----------------------------------
    @property
    def age(self):
        return self._age

    def __repr__(self):
        if self._direction is None:
            return "<Controls pending=None>"
        return f"<Controls pending={self._direction} age={self._age:.2f}s>"
