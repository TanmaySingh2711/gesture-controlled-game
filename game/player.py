"""Pac-Man: continuous movement driven entirely by buffered direction requests.

The player never reads input. It reads `Controls.pending`, which a keyboard handler fills in
P7 and the gesture recognizer will fill in P8 without either side knowing about the other.

Two turn rules, matching the arcade feel:

* **Reversal is immediate.** Turning back the way you came is legal anywhere in a corridor,
  because the corridor behind you is by definition open. It does not wait for a centre.
* **Every other turn waits for a tile centre**, and the request survives in the buffer until
  it becomes legal or goes stale, so the player can ask for a turn slightly early.
"""

from controls import DELTA, OPPOSITE
from entity import Entity
from maze import PLAYER_SPAWN, TILE

# 6.2 tiles/second. Deliberately modest: ghosts are capped below it, and a slower player
# makes the ~150 ms gesture latency of P8 comfortable rather than punishing.
PLAYER_SPEED = 6.2 * TILE

MOUTH_CYCLE = 0.22          # seconds for a full open-close of the mouth


class Player(Entity):
    def __init__(self, maze, speed=PLAYER_SPEED):
        super().__init__(maze, PLAYER_SPAWN, direction=None, speed=speed)
        self.spawn_direction = "left"
        self.mouth_timer = 0.0
        self.controls = None
        self.reset()

    def reset(self):
        """Back to the spawn tile, facing the spawn direction, mouth animation restarted."""
        self.reset_to(self.spawn_tile, self.spawn_direction)
        self.mouth_timer = 0.0

    # --- movement ------------------------------------------------------------------------
    def update(self, dt, controls):
        """Advance one frame. `controls` supplies the buffered direction request."""
        self.controls = controls
        controls.tick(dt)

        # Reversal does not need an intersection, so handle it before moving.
        pending = controls.pending
        if (pending is not None and self.direction is not None
                and pending == OPPOSITE[self.direction]):
            self.direction = pending
            controls.consume()

        super().update(dt)
        self.mouth_timer = (self.mouth_timer + dt) % MOUTH_CYCLE
        self.controls = None

    def on_center(self):
        """At a tile centre: take the buffered turn if it is legal, else carry straight on."""
        if self.controls is not None:
            pending = self.controls.pending
            if pending is not None and self.can_move(pending):
                self.direction = pending
                self.controls.consume()
                return

        if self.direction is None or not self.can_move(self.direction):
            self.direction = None          # nose against a wall: stop until asked to turn

    def on_stopped(self):
        """Standing still against a wall - start again as soon as a legal request arrives."""
        if self.controls is None:
            return
        pending = self.controls.pending
        if pending is not None and self.can_move(pending):
            self.direction = pending
            self.controls.consume()

    # --- drawing state -------------------------------------------------------------------
    @property
    def mouth_openness(self):
        """0.0 closed to 1.0 wide open, as a simple triangle wave."""
        half = MOUTH_CYCLE / 2.0
        phase = self.mouth_timer / half
        return phase if phase <= 1.0 else 2.0 - phase

    @property
    def facing_angle(self):
        """Degrees anticlockwise from east, for drawing the mouth wedge."""
        return {"right": 0.0, "up": 90.0, "left": 180.0, "down": 270.0}.get(
            self.direction or "left", 180.0)
