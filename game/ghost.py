"""The four ghosts: distinct personalities over one shared state machine.

Every ghost navigates the same way - at each tile centre it looks at the tiles it could move
to, rejects the one behind it, and takes whichever remaining tile is closest in straight-line
distance to its current *target tile*. All four therefore share one navigation routine, and
what makes them behave differently is only how each computes that target:

    Chaser     the player's tile. Relentless, arrives from behind. Speeds up ("Elroy") when
               few pellets remain, so the end of a round never goes slack.
    Ambusher   four tiles ahead of the player. Cuts you off at the junction you are heading for.
    Flanker    the Chaser's position reflected through a point two tiles ahead of the player,
               so it closes the pincer opposite whichever side the Chaser is on.
    Drifter    the player's tile while far away, its own corner once within eight tiles, so it
               keeps breaking off and wandering home.

This reproduces the *personalities* of the arcade ghosts without reproducing the arcade's
exact target arithmetic - in particular the original's up-direction overflow bug is not
recreated, which is a deliberate simplification recorded in PROJECT_SPEC.md.

States
------
    HOUSE       inside the ghost house, bobbing, waiting for its timed release
    SCATTER     heading for its own corner, set by the global schedule
    CHASE       hunting, using the role target above
    FRIGHTENED  edible, slower, wandering at random
    EATEN       a pair of eyes travelling quickly back to the house
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Final

from .controls import DELTA
from .entity import Entity
from .maze import GHOST_SPAWN, HOUSE_EXIT, SCATTER_TARGETS, TILE, Tile

if TYPE_CHECKING:
    from .maze import Maze

HOUSE: Final = "house"
SCATTER: Final = "scatter"
CHASE: Final = "chase"
FRIGHTENED: Final = "frightened"
EATEN: Final = "eaten"

DANGEROUS_STATES: Final = (SCATTER, CHASE)

ROLES: Final = ("chaser", "ambusher", "flanker", "drifter")

# Speeds in tiles/second. The base is below the player's 6.2 so a straight chase down a
# corridor can always be outrun; frightened ghosts are much slower so they can be caught;
# eyes return fast so a ghost is not out of play for long.
GHOST_BASE_SPEED: Final = 5.4
GHOST_SPEED_PER_ROUND: Final = 0.2
GHOST_MAX_SPEED: Final = 6.0  # stays under the player's 6.2, at every round
FRIGHTENED_SPEED: Final = 3.1
EATEN_SPEED: Final = 11.0
HOUSE_SPEED: Final = 2.6

# "Elroy": the Chaser's end-of-round surge. Two stages, still capped below the player's 6.2
# so the game stays winnable by skill - the arcade let Elroy outrun Pac-Man, which is exactly
# the unfairness a ~150 ms gesture latency cannot absorb.
ELROY_PELLETS: Final = ((20, 0.3), (10, 0.5))  # (pellets remaining at or below, extra speed)
ELROY_MAX_SPEED: Final = 6.1

# Seconds before each ghost leaves the house at the start of a life. The Chaser starts
# outside, so there is pressure immediately without all four arriving at once.
RELEASE_DELAY: Final = {"chaser": 0.0, "ambusher": 2.0, "flanker": 6.0, "drifter": 10.0}

RESPAWN_DELAY: Final = 1.2  # how long a returned ghost sits in the house before leaving

DRIFTER_RETREAT_TILES: Final = 8
AMBUSH_TILES: Final = 4
FLANK_TILES: Final = 2


def tile_distance(a: Tile, b: Tile) -> int:
    """Squared straight-line distance. Squared is enough for comparisons and avoids a sqrt."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class Ghost(Entity):
    def __init__(self, maze: Maze, role: str, speed_scale: float = 1.0, seed: int = 0) -> None:
        self.role = role
        self.scatter_target: Tile = SCATTER_TARGETS[role]
        # A stable per-role offset. The first version used `hash(role)`, which Python
        # randomises per process for strings - so frightened ghosts wandered differently on
        # every launch despite the promise of determinism.
        self.rng = random.Random(seed * len(ROLES) + ROLES.index(role))
        super().__init__(maze, GHOST_SPAWN[role], direction="up", speed=GHOST_BASE_SPEED * TILE)
        self.state: str = HOUSE
        self.release_timer: float = RELEASE_DELAY[role]
        self.released = False
        self.base_speed: float = min(GHOST_MAX_SPEED, GHOST_BASE_SPEED + speed_scale)
        self.elroy_bonus = 0.0
        self.frightened_flash = False
        self._bob_direction = "up"
        self._player: Entity | None = None
        self._chaser: Ghost | None = None
        self._pending_mode: str | None = None
        self.reset()

    # --- lifecycle -----------------------------------------------------------------------
    def reset(self) -> None:
        """Back to the start of a life: spawn tile, house state, release timer rewound."""
        self.reset_to(GHOST_SPAWN[self.role], "up" if self.role != "chaser" else "left")
        self.release_timer = RELEASE_DELAY[self.role]
        self.released = self.release_timer <= 0.0
        self.state = HOUSE
        self.frightened_flash = False
        if self.role == "chaser":
            # Starts outside the house, so it joins the global mode straight away.
            self.state = SCATTER

    def set_round(self, round_number: int) -> None:
        """Mild per-round speed increase, capped so the player is always slightly faster."""
        self.base_speed = min(
            GHOST_MAX_SPEED, GHOST_BASE_SPEED + GHOST_SPEED_PER_ROUND * (round_number - 1)
        )
        self.elroy_bonus = 0.0

    def set_pellets_remaining(self, remaining: int) -> None:
        """Only the Chaser reacts: it surges as the board empties."""
        if self.role != "chaser":
            return
        self.elroy_bonus = 0.0
        for threshold, bonus in ELROY_PELLETS:
            if remaining <= threshold:
                self.elroy_bonus = bonus

    # --- state ---------------------------------------------------------------------------
    @property
    def is_dangerous(self) -> bool:
        return self.state in DANGEROUS_STATES

    @property
    def is_edible(self) -> bool:
        return self.state == FRIGHTENED

    @property
    def in_house(self) -> bool:
        return self.state == HOUSE

    def set_mode(self, mode: str, *, reverse: bool = True) -> None:
        """Apply a global scatter/chase change. Ghosts reverse on a mode switch, as in arcade."""
        if self.state in (HOUSE, EATEN, FRIGHTENED):
            return
        if self.state != mode:
            self.state = mode
            if reverse:
                self.reverse()

    def frighten(self) -> bool:
        """Turn edible. A ghost already heading home as eyes is not affected."""
        if self.state in (EATEN, HOUSE):
            return False
        self.state = FRIGHTENED
        self.frightened_flash = False
        self.reverse()
        return True

    def unfrighten(self, mode: str) -> None:
        if self.state == FRIGHTENED:
            self.state = mode
            self.frightened_flash = False

    def get_eaten(self) -> None:
        self.state = EATEN
        self.frightened_flash = False

    # --- permissions ---------------------------------------------------------------------
    def can_pass_doors(self) -> bool:
        # The door is one-way in spirit: used on the way out of the house and on the way back.
        return self.state in (HOUSE, EATEN)

    def can_enter_house(self) -> bool:
        return self.state in (HOUSE, EATEN)

    # --- targeting -----------------------------------------------------------------------
    def target_tile(self, player: Entity, chaser: Ghost | None) -> Tile | None:
        if self.state == SCATTER:
            return self.scatter_target
        if self.state == EATEN:
            return GHOST_SPAWN[self.role]
        if self.state == HOUSE:
            return HOUSE_EXIT
        if self.state == FRIGHTENED:
            return None  # frightened ghosts do not aim at anything
        return self.chase_target(player, chaser)

    def chase_target(self, player: Entity, chaser: Ghost | None) -> Tile:
        """The role-specific target. This is the only place the four ghosts really differ."""
        player_tile = player.tile
        d_row, d_column = DELTA[player.direction or "left"]

        if self.role == "chaser":
            return player_tile

        if self.role == "ambusher":
            return (player_tile[0] + d_row * AMBUSH_TILES, player_tile[1] + d_column * AMBUSH_TILES)

        if self.role == "flanker":
            pivot = (player_tile[0] + d_row * FLANK_TILES, player_tile[1] + d_column * FLANK_TILES)
            chaser_tile = chaser.tile if chaser is not None else pivot
            # Reflect the Chaser through the pivot: the further the Chaser is on one side,
            # the further this ghost swings round to the other.
            return (2 * pivot[0] - chaser_tile[0], 2 * pivot[1] - chaser_tile[1])

        # drifter: bold at a distance, shy up close
        if tile_distance(self.tile, player_tile) > DRIFTER_RETREAT_TILES**2:
            return player_tile
        return self.scatter_target

    # --- navigation ----------------------------------------------------------------------
    def current_speed(self) -> float:
        if self.state == FRIGHTENED:
            return FRIGHTENED_SPEED * TILE
        if self.state == EATEN:
            return EATEN_SPEED * TILE
        if self.state == HOUSE:
            return HOUSE_SPEED * TILE
        if self.elroy_bonus and self.state == CHASE:
            return min(ELROY_MAX_SPEED, self.base_speed + self.elroy_bonus) * TILE
        return self.base_speed * TILE

    def update(self, dt: float, player: Entity, chaser: Ghost | None) -> None:
        self._player = player
        self._chaser = chaser
        self.speed = self.current_speed()

        if self.state == HOUSE and not self.released:
            self.release_timer -= dt
            if self.release_timer <= 0.0:
                self.released = True
            else:
                self._bob(dt)
                self._player = self._chaser = None
                return

        self.move(dt)
        self._arrival_checks()
        self._player = self._chaser = None

    def _bob(self, dt: float) -> None:
        """Idle drift inside the house while waiting for the release timer."""
        centre_y = self.maze.tile_center(*GHOST_SPAWN[self.role])[1]
        limit = TILE * 0.35
        step = HOUSE_SPEED * TILE * dt
        self.y += step if self._bob_direction == "down" else -step
        if self.y > centre_y + limit:
            self.y = centre_y + limit
            self._bob_direction = "up"
        elif self.y < centre_y - limit:
            self.y = centre_y - limit
            self._bob_direction = "down"

    def _arrival_checks(self) -> None:
        """State transitions that depend on where the ghost has got to."""
        if self.state == HOUSE and self.released and self.tile == HOUSE_EXIT:
            self.state = self._pending_mode or CHASE
            self._pending_mode = None
        elif self.state == EATEN and self.tile == GHOST_SPAWN[self.role]:
            self.state = HOUSE
            self.released = False
            self.release_timer = RESPAWN_DELAY
            self.direction = "up"

    def rejoin_mode(self, mode: str) -> None:
        """Told by the game which global mode to adopt once it reaches the house exit."""
        self._pending_mode = mode

    def on_center(self) -> None:
        """Pick the exit that gets closest to the target. This is the whole navigation rule."""
        options = self.legal_directions(allow_reverse=False) or self.legal_directions()
        if not options:
            # Boxed in completely; cannot happen in a valid maze, but never crash on it.
            self.direction = None
            return

        if self.state == FRIGHTENED:
            self.direction = self.rng.choice(options)
            return

        player = self._player
        target = self.target_tile(player, self._chaser) if player else self.scatter_target
        if target is None:
            self.direction = self.rng.choice(options)
            return

        # `options` is in a fixed order, so ties always resolve the same way.
        self.direction = min(options, key=lambda d: tile_distance(self.tile_ahead(d), target))

    def on_stopped(self) -> None:
        options = self.legal_directions()
        if options:
            self.direction = options[0]


def make_ghosts(maze: Maze, seed: int = 0) -> list[Ghost]:
    """The four ghosts, in a fixed order so tests and rendering are deterministic."""
    return [Ghost(maze, role, seed=seed) for role in ROLES]
