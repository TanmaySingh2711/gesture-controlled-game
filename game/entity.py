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

from controls import DELTA, OPPOSITE
from maze import TILE


class Entity:
    """A character that moves along maze corridors on the tile grid."""

    def __init__(self, maze, tile, direction=None, speed=100.0):
        self.maze = maze
        self.direction = direction
        self.speed = speed
        self.spawn_tile = tile
        self.x, self.y = maze.tile_center(*tile)

    # --- geometry ------------------------------------------------------------------------
    @property
    def tile(self):
        row = int(self.y // TILE)
        column = int(self.x // TILE)
        return row, column

    @property
    def position(self):
        return self.x, self.y

    def at_center(self, tolerance=0.6):
        centre_x, centre_y = self.maze.tile_center(*self.tile)
        return abs(self.x - centre_x) < tolerance and abs(self.y - centre_y) < tolerance

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def reset_to(self, tile, direction=None):
        self.x, self.y = self.maze.tile_center(*tile)
        self.direction = direction

    # --- permissions: ghosts may use the door and the house, the player may not -----------
    def can_pass_doors(self):
        return False

    def can_enter_house(self):
        return False

    def can_walk(self, row, column):
        return self.maze.is_walkable(row, column, doors=self.can_pass_doors(),
                                     house=self.can_enter_house())

    def tile_ahead(self, direction, steps=1):
        """The tile `steps` away in `direction`, tunnel wrapping applied."""
        row, column = self.tile
        d_row, d_column = DELTA[direction]
        return self.maze.wrap(row + d_row * steps, column + d_column * steps)

    def can_move(self, direction):
        return self.can_walk(*self.tile_ahead(direction))

    def legal_directions(self, *, allow_reverse=True):
        """Directions with a walkable tile next to them, in a fixed order for determinism."""
        found = []
        for direction in ("up", "left", "down", "right"):
            if not allow_reverse and self.direction and direction == OPPOSITE[self.direction]:
                continue
            if self.can_move(direction):
                found.append(direction)
        return found

    # --- movement ------------------------------------------------------------------------
    def update(self, dt):
        """Advance by speed * dt, split so a centre is never skipped over."""
        remaining = self.speed * dt
        guard = 0
        while remaining > 1e-9 and guard < 64:
            guard += 1
            step = min(remaining, TILE / 2.0)
            self._advance(step)
            remaining -= step

    def _advance(self, step):
        if self.direction is None:
            self.on_stopped()
            if self.direction is None:
                return

        d_row, d_column = DELTA[self.direction]
        centre_x, centre_y = self.maze.tile_center(*self.tile)

        # Distance still to travel before reaching this tile's centre, measured along the
        # direction of travel. Negative means the centre is already behind us.
        if d_column:
            to_centre = (centre_x - self.x) * d_column
        else:
            to_centre = (centre_y - self.y) * d_row

        if 0 <= to_centre < step:
            self.x, self.y = centre_x, centre_y          # snap: no accumulated drift
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

    def _wrap_tunnel(self):
        width = self.maze.pixel_width
        if self.x < 0:
            self.x += width
        elif self.x >= width:
            self.x -= width

    def reverse(self):
        """Turn around on the spot. Always legal - the way we came is by definition open."""
        if self.direction is not None:
            self.direction = OPPOSITE[self.direction]

    # --- hooks ---------------------------------------------------------------------------
    def on_center(self):
        """Called standing exactly on a tile centre. Subclasses choose a direction here."""
        if not self.can_move(self.direction):
            self.direction = None

    def on_stopped(self):
        """Called when there is no direction at all. Subclasses may start moving again."""
