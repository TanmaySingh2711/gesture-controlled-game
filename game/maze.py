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

from collections import deque

WALL = "#"
PELLET = "."
POWER = "o"
EMPTY = " "
DOOR = "-"
HOUSE = "G"

COLUMNS = 28
ROWS = 31
TILE = 20                       # pixels per tile

# Each string is the left half of a row; the right half is its mirror image.
LEFT_HALF = [
    "##############",   # 0
    "#............#",   # 1
    "#.##.####.##.#",   # 2
    "#o##.####.##.#",   # 3   power pellet, top corners
    "#.##.####.##.#",   # 4
    "#............#",   # 5
    "#.##.##.##.##.",   # 6   narrow pillars, centre open
    "#.##.##.##.##.",   # 7
    "#.............",   # 8   open row linking every pillar gap
    "######.#####.#",   # 9
    "######.#####.#",   # 10
    "######........",   # 11  corridor above the ghost house
    "######.######-",   # 12  house door at the centre
    "######.#####GG",   # 13  house interior
    ".......#####GG",   # 14  tunnel row, open at the maze edge
    "######.#####GG",   # 15  house interior
    "######.#######",   # 16  house floor
    "######........",   # 17  corridor below the ghost house
    "######.#####.#",   # 18
    "######.#####.#",   # 19
    "#............#",   # 20
    "#.##.####.##.#",   # 21
    "#o##..........",   # 22  power pellet; the player spawns at the centre of this row
    "#.##.####.##.#",   # 23
    "#......##....#",   # 24
    "#.####.##.####",   # 25
    "#.####.##.####",   # 26
    "#............#",   # 27
    "#.##########.#",   # 28
    "#............#",   # 29
    "##############",   # 30
]

TUNNEL_ROW = 14
PLAYER_SPAWN = (22, 13)         # row, column
HOUSE_DOOR = (12, 13)           # the door tile ghosts pass through on the way out
HOUSE_EXIT = (11, 13)           # the first tile outside the house

# Where each ghost starts. The Chaser begins outside so there is pressure immediately.
GHOST_SPAWN = {
    "chaser": (11, 13),
    "ambusher": (14, 13),
    "flanker": (14, 12),
    "drifter": (14, 15),
}

# Scatter corners, one per ghost. They sit outside the playable area on purpose: a target a
# ghost can never reach makes it circle its corner instead of settling on one tile.
SCATTER_TARGETS = {
    "chaser": (-2, COLUMNS - 1),        # top right
    "ambusher": (-2, 2),                # top left
    "flanker": (ROWS + 1, COLUMNS - 1),  # bottom right
    "drifter": (ROWS + 1, 2),           # bottom left
}


def build_rows():
    """Full 28-character rows, each left half mirrored onto its right half."""
    rows = []
    for half in LEFT_HALF:
        rows.append(half + half[::-1])
    return rows


class Maze:
    """The tile grid plus every query the rest of the game needs to ask about it."""

    def __init__(self):
        self.rows = ROWS
        self.columns = COLUMNS
        self.layout = build_rows()
        self.walls = [[cell == WALL for cell in row] for row in self.layout]
        self.doors = [[cell == DOOR for cell in row] for row in self.layout]
        self.house = [[cell == HOUSE for cell in row] for row in self.layout]
        self.pellets = set()
        self.power_pellets = set()
        self.reset_pellets()

    # --- pellets -------------------------------------------------------------------------
    def reset_pellets(self):
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
    def pellets_remaining(self):
        return len(self.pellets) + len(self.power_pellets)

    def eat(self, tile):
        """Remove a pellet at `tile`. Returns "pellet", "power" or None."""
        if tile in self.pellets:
            self.pellets.discard(tile)
            return "pellet"
        if tile in self.power_pellets:
            self.power_pellets.discard(tile)
            return "power"
        return None

    # --- tile queries --------------------------------------------------------------------
    def in_bounds(self, row, column):
        return 0 <= row < self.rows and 0 <= column < self.columns

    def is_wall(self, row, column):
        if not self.in_bounds(row, column):
            return True
        return self.walls[row][column]

    def is_door(self, row, column):
        return self.in_bounds(row, column) and self.doors[row][column]

    def is_house(self, row, column):
        return self.in_bounds(row, column) and self.house[row][column]

    def is_walkable(self, row, column, *, doors=False, house=False):
        """Can a character stand here?

        `doors` and `house` are opened up for ghosts; Pac-Man is never allowed either, which
        is what keeps the player out of the ghost house without a special case elsewhere.
        """
        if not self.in_bounds(row, column):
            return False
        if self.walls[row][column]:
            return False
        if self.doors[row][column] and not doors:
            return False
        if self.house[row][column] and not house:
            return False
        return True

    def neighbours(self, row, column, *, doors=False, house=False):
        """Walkable orthogonal neighbours, with tunnel wrapping applied."""
        found = []
        for d_row, d_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_column = self.wrap(row + d_row, column + d_column)
            if self.is_walkable(next_row, next_column, doors=doors, house=house):
                found.append((next_row, next_column))
        return found

    def wrap(self, row, column):
        """Tunnel wrapping: stepping off one side of the maze arrives at the other."""
        if column < 0:
            column = self.columns - 1
        elif column >= self.columns:
            column = 0
        return row, column

    # --- geometry ------------------------------------------------------------------------
    @staticmethod
    def tile_center(row, column):
        """Pixel centre of a tile, in maze-local coordinates."""
        return (column * TILE + TILE / 2.0, row * TILE + TILE / 2.0)

    @staticmethod
    def world_to_tile(x, y):
        """Which tile a pixel position falls in."""
        return int(y // TILE), int(x // TILE)

    @property
    def pixel_width(self):
        return self.columns * TILE

    @property
    def pixel_height(self):
        return self.rows * TILE

    # --- validation ----------------------------------------------------------------------
    def reachable_from(self, start, *, doors=False, house=False):
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

    def unreachable_pellets(self, start=PLAYER_SPAWN):
        """Pellets the player can never eat. Any of these makes a round unwinnable."""
        reachable = self.reachable_from(start)
        return sorted((self.pellets | self.power_pellets) - reachable)

    def dead_ends(self):
        """Walkable tiles with exactly one exit."""
        found = []
        for row in range(self.rows):
            for column in range(self.columns):
                if self.is_walkable(row, column) and len(self.neighbours(row, column)) == 1:
                    found.append((row, column))
        return found
