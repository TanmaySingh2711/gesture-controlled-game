"""Pac-Man: continuous movement driven entirely by buffered direction requests.

The player never reads input. It reads `Controls.pending`, which the keyboard handler and the
gesture recognizer both fill in without either side knowing about the other.

Two turn rules, matching the arcade feel:

* **Reversal is immediate.** Turning back the way you came is legal anywhere in a corridor,
  because the corridor behind you is by definition open. It does not wait for a centre.
* **Every other turn waits for a tile centre**, and the request survives in the buffer until
  it becomes legal or goes stale, so the player can ask for a turn slightly early.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from .controls import OPPOSITE
from .entity import Entity
from .maze import PLAYER_SPAWN, TILE

if TYPE_CHECKING:
    from .controls import Controls
    from .maze import Maze

# 6.2 tiles/second. Deliberately modest: ghosts are capped below it, and a slower player makes
# the ~150 ms gesture latency comfortable rather than punishing.
PLAYER_SPEED: Final = 6.2 * TILE

MOUTH_CYCLE: Final = 0.22  # seconds for a full open-close of the mouth

FACING_ANGLE: Final = {"right": 0.0, "up": 90.0, "left": 180.0, "down": 270.0}


class Player(Entity):
    def __init__(self, maze: Maze, speed: float = PLAYER_SPEED) -> None:
        super().__init__(maze, PLAYER_SPAWN, direction=None, speed=speed)
        self.spawn_direction = "left"
        self.mouth_timer = 0.0
        self.controls: Controls | None = None
        self.reset()

    def reset(self) -> None:
        """Back to the spawn tile, facing the spawn direction, mouth animation restarted."""
        self.reset_to(self.spawn_tile, self.spawn_direction)
        self.mouth_timer = 0.0

    # --- movement ------------------------------------------------------------------------
    def update(self, dt: float, controls: Controls) -> None:
        """Advance one frame. `controls` supplies the buffered direction request."""
        self.controls = controls
        controls.tick(dt)

        # Reversal does not need an intersection, so handle it before moving.
        pending = controls.pending
        if (
            pending is not None
            and self.direction is not None
            and pending == OPPOSITE[self.direction]
        ):
            self.direction = pending
            controls.consume()

        self.move(dt)
        self.mouth_timer = (self.mouth_timer + dt) % MOUTH_CYCLE
        self.controls = None

    def on_center(self) -> None:
        """At a tile centre: take the buffered turn if it is legal, else carry straight on."""
        if self._take_pending_turn():
            return
        if self.direction is None or not self.can_move(self.direction):
            self.direction = None  # nose against a wall: stop until asked to turn

    def on_stopped(self) -> None:
        """Standing still against a wall - start again as soon as a legal request arrives."""
        self._take_pending_turn()

    def _take_pending_turn(self) -> bool:
        if self.controls is None:
            return False
        pending = self.controls.pending
        if pending is None or not self.can_move(pending):
            return False
        self.direction = pending
        self.controls.consume()
        return True

    # --- drawing state -------------------------------------------------------------------
    @property
    def mouth_openness(self) -> float:
        """0.0 closed to 1.0 wide open, as a simple triangle wave."""
        phase = self.mouth_timer / (MOUTH_CYCLE / 2.0)
        return phase if phase <= 1.0 else 2.0 - phase

    @property
    def facing_angle(self) -> float:
        """Degrees anticlockwise from east, for drawing the mouth wedge."""
        return FACING_ANGLE.get(self.direction or "left", 180.0)
