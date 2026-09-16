"""Maze grid for the Pac-Man-style game: layout, tile queries and geometry helpers.

The maze is 28 columns x 31 rows, the classic arcade proportions, but the layout itself is
our own. It is written as 31 left halves of 14 characters each and mirrored, which makes the
maze symmetric by construction and makes it impossible to author a row of the wrong width.

Layout characters
-----------------
    #   wall
    .   pellet
    o   power pellet
    ' ' open path with no pellet
    -   ghost-house door   (ghosts may pass, Pac-Man may not)
    G   ghost-house interior

Everything about movement and navigation is driven by this grid. Nothing in the game does
pixel-level collision against wall rectangles: a move is legal if the destination tile is
walkable, which is what keeps the player and the ghosts on the corridor centre lines.
"""

from __future__ import annotations

from collections import deque
from typing import Final

Tile = tuple[int, int]
"""A (row, column) grid position."""

WALL: Final = "#"
PELLET: Final = "."
POWER: Final = "o"
EMPTY: Final = " "
DOOR: Final = "-"
HOUSE: Final = "G"

COLUMNS: Final = 28
ROWS: Final = 31
TILE: Final = 20  # pixels per tile

# Each string is the left half of a row; the right half is its mirror image.
LEFT_HALF: Final[list[str]] = [
    "##############",  # 0
    "#............#",  # 1
    "#.##.####.##.#",  # 2
    "#o##.####.##.#",  # 3   power pellet, top corners
    "#.##.####.##.#",  # 4
    "#............#",  # 5
    "#.##.##.##.##.",  # 6   narrow pillars, centre open
    "#.##.##.##.##.",  # 7
    "#.............",  # 8   open row linking every pillar gap
    "######.#####.#",  # 9
    "######.#####.#",  # 10
    "######........",  # 11  corridor above the ghost house
    "######.######-",  # 12  house door at the centre
    "######.#####GG",  # 13  house interior
    ".......#####GG",  # 14  tunnel row, open at the maze edge
    "######.#####GG",  # 15  house interior
    "######.#######",  # 16  house floor
    "######........",  # 17  corridor below the ghost house
    "######.#####.#",  # 18
    "######.#####.#",  # 19
    "#............#",  # 20
    "#.##.####.##.#",  # 21
    "#o##..........",  # 22  power pellet; the player spawns at the centre of this row
    "#.##.####.##.#",  # 23
    "#......##....#",  # 24
    "#.####.##.####",  # 25
    "#.####.##.####",  # 26
    "#............#",  # 27
    "#.##########.#",  # 28
    "#............#",  # 29
    "##############",  # 30
]

TUNNEL_ROW: Final = 14
PLAYER_SPAWN: Final[Tile] = (22, 13)
HOUSE_DOOR: Final[Tile] = (12, 13)  # the door tile ghosts pass through on the way out
HOUSE_EXIT: Final[Tile] = (11, 13)  # the first tile outside the house
FRUIT_TILE: Final[Tile] = (17, 13)  # just below the house, where bonus fruit appears

# Where each ghost starts. The Chaser begins outside so there is pressure immediately.
GHOST_SPAWN: Final[dict[str, Tile]] = {
    "chaser": (11, 13),
    "ambusher": (14, 13),
    "flanker": (14, 12),
    "drifter": (14, 15),
}

# Scatter corners, one per ghost. They sit outside the playable area on purpose: a target a
# ghost can never reach makes it circle its corner instead of settling on one tile.
SCATTER_TARGETS: Final[dict[str, Tile]] = {
    "chaser": (-2, COLUMNS - 1),  # top right
    "ambusher": (-2, 2),  # top left
    "flanker": (ROWS + 1, COLUMNS - 1),  # bottom right
    "drifter": (ROWS + 1, 2),  # bottom left
}

_NEIGHBOUR_STEPS: Final = ((-1, 0), (1, 0), (0, -1), (0, 1))


def build_rows() -> list[str]:
    """Full 28-character rows, each left half mirrored onto its right half."""
    return [half + half[::-1] for half in LEFT_HALF]


class Maze:
    """The tile grid plus every query the rest of the game needs to ask about it."""

    def __init__(self) -> None:
        self.rows: int = ROWS
        self.columns: int = COLUMNS
        self.layout: list[str] = build_rows()
        self.walls: list[list[bool]] = [[cell == WALL for cell in row] for row in self.layout]
        self.doors: list[list[bool]] = [[cell == DOOR for cell in row] for row in self.layout]
        self.house: list[list[bool]] = [[cell == HOUSE for cell in row] for row in self.layout]
        self.pellets: set[Tile] = set()
        self.power_pellets: set[Tile] = set()
        self.reset_pellets()

    # --- pellets -------------------------------------------------------------------------
    def reset_pellets(self) -> None:
        """Refill every pellet. Called at the start of the game and of each new round."""
        self.pellets.clear()
        self.power_pellets.clear()
        for row, line in enumerate(self.layout):
            for column, cell in enumerate(line):
                if cell == PELLET:
                    self.pellets.add((row, column))
                elif cell == POWER:
                    self.power_pellets.add((row, column))

    @property
    def pellets_remaining(self) -> int:
        return len(self.pellets) + len(self.power_pellets)

    @property
    def pellets_total(self) -> int:
        """How many pellets a full board holds; the fruit schedule counts against it."""
        return sum(line.count(PELLET) + line.count(POWER) for line in self.layout)

    def eat(self, tile: Tile) -> str | None:
        """Remove a pellet at `tile`. Returns "pellet", "power" or None."""
        if tile in self.pellets:
            self.pellets.discard(tile)
            return "pellet"
        if tile in self.power_pellets:
            self.power_pellets.discard(tile)
            return "power"
        return None

    # --- tile queries --------------------------------------------------------------------
    def in_bounds(self, row: int, column: int) -> bool:
        return 0 <= row < self.rows and 0 <= column < self.columns

    def is_wall(self, row: int, column: int) -> bool:
        if not self.in_bounds(row, column):
            return True
        return self.walls[row][column]

    def is_walkable(
        self, row: int, column: int, *, doors: bool = False, house: bool = False
    ) -> bool:
        """Can a character stand here?

        `doors` and `house` are opened up for ghosts; Pac-Man is never allowed either, which
        is what keeps the player out of the ghost house without a special case elsewhere.
        """
        if not self.in_bounds(row, column) or self.walls[row][column]:
            return False
        if self.doors[row][column] and not doors:
            return False
        return house or not self.house[row][column]

    def neighbours(
        self, row: int, column: int, *, doors: bool = False, house: bool = False
    ) -> list[Tile]:
        """Walkable orthogonal neighbours, with tunnel wrapping applied."""
        found: list[Tile] = []
        for d_row, d_column in _NEIGHBOUR_STEPS:
            next_row, next_column = self.wrap(row + d_row, column + d_column)
            if self.is_walkable(next_row, next_column, doors=doors, house=house):
                found.append((next_row, next_column))
        return found

    def wrap(self, row: int, column: int) -> Tile:
        """Tunnel wrapping: stepping off one side of the maze arrives at the other."""
        if column < 0:
            column = self.columns - 1
        elif column >= self.columns:
            column = 0
        return row, column

    # --- geometry ------------------------------------------------------------------------
    @staticmethod
    def tile_center(row: int, column: int) -> tuple[float, float]:
        """Pixel centre of a tile, in maze-local coordinates."""
        return (column * TILE + TILE / 2.0, row * TILE + TILE / 2.0)

    @staticmethod
    def world_to_tile(x: float, y: float) -> Tile:
        """Which tile a pixel position falls in."""
        return int(y // TILE), int(x // TILE)

    @property
    def pixel_width(self) -> int:
        return self.columns * TILE

    @property
    def pixel_height(self) -> int:
        return self.rows * TILE

    # --- validation ----------------------------------------------------------------------
    def reachable_from(self, start: Tile, *, doors: bool = False, house: bool = False) -> set[Tile]:
        """Every tile reachable from `start` by legal moves. Breadth-first."""
        seen = {start}
        queue = deque([start])
        while queue:
            row, column = queue.popleft()
            for neighbour in self.neighbours(row, column, doors=doors, house=house):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return seen

    def unreachable_pellets(self, start: Tile = PLAYER_SPAWN) -> list[Tile]:
        """Pellets the player can never eat. Any of these makes a round unwinnable."""
        reachable = self.reachable_from(start)
        return sorted((self.pellets | self.power_pellets) - reachable)

    def dead_ends(self) -> list[Tile]:
        """Walkable tiles with exactly one exit."""
        return [
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
            if self.is_walkable(row, column) and len(self.neighbours(row, column)) == 1
        ]
