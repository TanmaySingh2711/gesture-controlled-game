"""Shared grid-aligned movement for everything that walks the maze.

Both the player and the ghosts move the same way: continuously along a corridor, turning only
at tile centres, wrapping through the tunnel. Only the *decision* made at each centre differs
- the player consults a buffered request, a ghost consults its target - so that decision is
the one method subclasses override.

Why turns happen at centres
---------------------------
Movement is float-based, but a turn is only legal from a tile centre. Rather than test for
exact float equality, a step that would carry the character past the centre is split: it
advances exactly to the centre, snaps to it, decides, and spends the remainder in whatever
direction came out of that decision. The snap keeps tiny floating-point error from
accumulating, so a character can never drift off the corridor line or clip a wall corner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from .controls import DELTA, OPPOSITE
from .maze import TILE, Tile

if TYPE_CHECKING:
    from .maze import Maze

# A frame is split into sub-steps no longer than half a tile, so a centre is never skipped.
MAX_SUB_STEP: Final = TILE / 2.0
MAX_SUB_STEPS: Final = 64  # guards against a pathological dt ever looping forever

# Fixed order, so every choice between equally good directions is deterministic.
DIRECTION_ORDER: Final = ("up", "left", "down", "right")


class Entity:
    """A character that moves along maze corridors on the tile grid."""

    def __init__(
        self, maze: Maze, tile: Tile, direction: str | None = None, speed: float = 100.0
    ) -> None:
        self.maze = maze
        self.direction: str | None = direction
        self.speed = speed
        self.spawn_tile: Tile = tile
        self.x, self.y = maze.tile_center(*tile)

    # --- geometry ------------------------------------------------------------------------
    @property
    def tile(self) -> Tile:
        return int(self.y // TILE), int(self.x // TILE)

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def at_center(self, tolerance: float = 0.6) -> bool:
        centre_x, centre_y = self.maze.tile_center(*self.tile)
        return abs(self.x - centre_x) < tolerance and abs(self.y - centre_y) < tolerance

    def distance_to(self, other: Entity) -> float:
        return float(((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)

    def reset_to(self, tile: Tile, direction: str | None = None) -> None:
        self.x, self.y = self.maze.tile_center(*tile)
        self.direction = direction

    # --- permissions: ghosts may use the door and the house, the player may not -----------
    def can_pass_doors(self) -> bool:
        return False

    def can_enter_house(self) -> bool:
        return False

    def can_walk(self, row: int, column: int) -> bool:
        return self.maze.is_walkable(
            row, column, doors=self.can_pass_doors(), house=self.can_enter_house()
        )

    def tile_ahead(self, direction: str, steps: int = 1) -> Tile:
        """The tile `steps` away in `direction`, tunnel wrapping applied."""
        row, column = self.tile
        d_row, d_column = DELTA[direction]
        return self.maze.wrap(row + d_row * steps, column + d_column * steps)

    def can_move(self, direction: str) -> bool:
        return self.can_walk(*self.tile_ahead(direction))

    def legal_directions(self, *, allow_reverse: bool = True) -> list[str]:
        """Directions with a walkable tile next to them, in a fixed order for determinism."""
        behind = OPPOSITE[self.direction] if self.direction else None
        return [
            direction
            for direction in DIRECTION_ORDER
            if (allow_reverse or direction != behind) and self.can_move(direction)
        ]

    # --- movement ------------------------------------------------------------------------
    def move(self, dt: float) -> None:
        """Advance by speed * dt, split so a tile centre is never skipped over."""
        remaining = self.speed * dt
        for _ in range(MAX_SUB_STEPS):
            if remaining <= 1e-9:
                break
            step = min(remaining, MAX_SUB_STEP)
            self._advance(step)
            remaining -= step

    def _advance(self, step: float) -> None:
        if self.direction is None:
            self.on_stopped()
            if self.direction is None:
                return

        d_row, d_column = DELTA[self.direction]
        centre_x, centre_y = self.maze.tile_center(*self.tile)

        # Distance still to travel before reaching this tile's centre, measured along the
        # direction of travel. Negative means the centre is already behind us.
        to_centre = (centre_x - self.x) * d_column if d_column else (centre_y - self.y) * d_row

        if 0 <= to_centre < step:
            self.x, self.y = centre_x, centre_y  # snap: no accumulated drift
            leftover = step - to_centre
            self.on_center()
            if self.direction is None:
                return
            d_row, d_column = DELTA[self.direction]
            self.x += d_column * leftover
            self.y += d_row * leftover
        else:
            self.x += d_column * step
            self.y += d_row * step

        self._wrap_tunnel()

    def _wrap_tunnel(self) -> None:
        width = self.maze.pixel_width
        if self.x < 0:
            self.x += width
        elif self.x >= width:
            self.x -= width

    def reverse(self) -> None:
        """Turn around on the spot. Always legal - the way we came is by definition open."""
        if self.direction is not None:
            self.direction = OPPOSITE[self.direction]

    # --- hooks ---------------------------------------------------------------------------
    def on_center(self) -> None:
        """Called standing exactly on a tile centre. Subclasses choose a direction here."""
        if self.direction is None or not self.can_move(self.direction):
            self.direction = None

    def on_stopped(self) -> None:
        """Called when there is no direction at all. Subclasses may start moving again."""
