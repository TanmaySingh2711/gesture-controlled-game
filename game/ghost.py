"""The four ghosts: distinct personalities over one shared state machine.

Every ghost navigates the same way - at each tile centre it looks at the tiles it could move
to, rejects the one behind it, and takes whichever remaining tile is closest in straight-line
distance to its current *target tile*. All four therefore share one navigation routine, and
what makes them behave differently is only how each computes that target:

    Chaser     the player's tile. Relentless, arrives from behind.
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

import random

from controls import DELTA, OPPOSITE
from entity import Entity
from maze import GHOST_SPAWN, HOUSE_DOOR, HOUSE_EXIT, SCATTER_TARGETS, TILE

HOUSE = "house"
SCATTER = "scatter"
CHASE = "chase"
FRIGHTENED = "frightened"
EATEN = "eaten"

DANGEROUS_STATES = (SCATTER, CHASE)

# Speeds in tiles/second. The base is below the player's 6.2 so a straight chase down a
# corridor can always be outrun; frightened ghosts are much slower so they can be caught;
# eyes return fast so a ghost is not out of play for long.
GHOST_BASE_SPEED = 5.4
GHOST_SPEED_PER_ROUND = 0.2
GHOST_MAX_SPEED = 6.0           # stays under the player's 6.2, at every round
FRIGHTENED_SPEED = 3.1
EATEN_SPEED = 11.0
HOUSE_SPEED = 2.6

# Seconds before each ghost leaves the house at the start of a life. The Chaser starts
# outside, so there is pressure immediately without all four arriving at once.
RELEASE_DELAY = {"chaser": 0.0, "ambusher": 2.0, "flanker": 6.0, "drifter": 10.0}

RESPAWN_DELAY = 1.2             # how long a returned ghost sits in the house before leaving

COLORS = {
    "chaser": (255, 82, 82),        # red
    "ambusher": (255, 170, 215),    # pink
    "flanker": (105, 220, 235),     # cyan
    "drifter": (255, 178, 82),      # orange
}

FRIGHTENED_COLOR = (60, 80, 225)
FRIGHTENED_FLASH_COLOR = (235, 235, 245)
EYE_COLOR = (235, 235, 245)

DRIFTER_RETREAT_TILES = 8
AMBUSH_TILES = 4
FLANK_TILES = 2


def tile_distance(a, b):
    """Squared straight-line distance. Squared is enough for comparisons and avoids a sqrt."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class Ghost(Entity):
    def __init__(self, maze, role, speed_scale=1.0, seed=0):
        self.role = role
        self.color = COLORS[role]
        self.scatter_target = SCATTER_TARGETS[role]
        self.rng = random.Random(seed + hash(role) % 1000)
        super().__init__(maze, GHOST_SPAWN[role], direction="up", speed=GHOST_BASE_SPEED * TILE)
        self.state = HOUSE
        self.release_timer = RELEASE_DELAY[role]
        self.released = False
        self.base_speed = min(GHOST_MAX_SPEED, GHOST_BASE_SPEED + speed_scale)
        self.frightened_flash = False
        self._bob_direction = "up"
        self.reset()

    # --- lifecycle -----------------------------------------------------------------------
    def reset(self):
        """Back to the start of a life: spawn tile, house state, release timer rewound."""
        self.reset_to(GHOST_SPAWN[self.role], "up" if self.role != "chaser" else "left")
        self.release_timer = RELEASE_DELAY[self.role]
        self.released = self.release_timer <= 0.0
        self.state = HOUSE
        self.frightened_flash = False
        if self.role == "chaser":
            # Starts outside the house, so it joins the global mode straight away.
            self.state = SCATTER

    def set_round(self, round_number):
        """Mild per-round speed increase, capped so the player is always slightly faster."""
        self.base_speed = min(GHOST_MAX_SPEED,
                              GHOST_BASE_SPEED + GHOST_SPEED_PER_ROUND * (round_number - 1))

    # --- state ---------------------------------------------------------------------------
    @property
    def is_dangerous(self):
        return self.state in DANGEROUS_STATES

    @property
    def is_edible(self):
        return self.state == FRIGHTENED

    @property
    def in_house(self):
        return self.state == HOUSE

    def set_mode(self, mode, *, reverse=True):
        """Apply a global scatter/chase change. Ghosts reverse on a mode switch, as in arcade."""
        if self.state in (HOUSE, EATEN, FRIGHTENED):
            return
        if self.state != mode:
            self.state = mode
            if reverse:
                self.reverse()

    def frighten(self):
        """Turn edible. A ghost already heading home as eyes is not affected."""
        if self.state in (EATEN, HOUSE):
            return False
        self.state = FRIGHTENED
        self.frightened_flash = False
        self.reverse()
        return True

    def unfrighten(self, mode):
        if self.state == FRIGHTENED:
            self.state = mode
            self.frightened_flash = False

    def get_eaten(self):
        self.state = EATEN
        self.frightened_flash = False

    # --- permissions ---------------------------------------------------------------------
    def can_pass_doors(self):
        # The door is one-way in spirit: used on the way out of the house and on the way back.
        return self.state in (HOUSE, EATEN)

    def can_enter_house(self):
        return self.state in (HOUSE, EATEN)

    # --- targeting -----------------------------------------------------------------------
    def target_tile(self, player, chaser):
        if self.state == SCATTER:
            return self.scatter_target
        if self.state == EATEN:
            return GHOST_SPAWN[self.role]
        if self.state == HOUSE:
            return HOUSE_EXIT
        if self.state == FRIGHTENED:
            return None                      # frightened ghosts do not aim at anything
        return self.chase_target(player, chaser)

    def chase_target(self, player, chaser):
        """The role-specific target. This is the only place the four ghosts really differ."""
        player_tile = player.tile
        facing = player.direction or "left"

        if self.role == "chaser":
            return player_tile

        if self.role == "ambusher":
            d_row, d_column = DELTA[facing]
            return (player_tile[0] + d_row * AMBUSH_TILES,
                    player_tile[1] + d_column * AMBUSH_TILES)

        if self.role == "flanker":
            d_row, d_column = DELTA[facing]
            pivot = (player_tile[0] + d_row * FLANK_TILES,
                     player_tile[1] + d_column * FLANK_TILES)
            chaser_tile = chaser.tile if chaser is not None else pivot
            # Reflect the Chaser through the pivot: the further the Chaser is on one side,
            # the further this ghost swings round to the other.
            return (2 * pivot[0] - chaser_tile[0], 2 * pivot[1] - chaser_tile[1])

        # drifter: bold at a distance, shy up close
        if tile_distance(self.tile, player_tile) > DRIFTER_RETREAT_TILES ** 2:
            return player_tile
        return self.scatter_target

    # --- navigation ----------------------------------------------------------------------
    def current_speed(self):
        if self.state == FRIGHTENED:
            return FRIGHTENED_SPEED * TILE
        if self.state == EATEN:
            return EATEN_SPEED * TILE
        if self.state == HOUSE:
            return HOUSE_SPEED * TILE
        return self.base_speed * TILE

    def update(self, dt, player, chaser):
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

        super().update(dt)
        self._arrival_checks()
        self._player = self._chaser = None

    def _bob(self, dt):
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

    def _arrival_checks(self):
        """State transitions that depend on where the ghost has got to."""
        if self.state == HOUSE and self.released and self.tile == HOUSE_EXIT:
            self.state = self._pending_mode or CHASE
            self._pending_mode = None
        elif self.state == EATEN and self.tile == GHOST_SPAWN[self.role]:
            self.state = HOUSE
            self.released = False
            self.release_timer = RESPAWN_DELAY
            self.direction = "up"

    _pending_mode = None

    def rejoin_mode(self, mode):
        """Told by the game which global mode to adopt once it reaches the house exit."""
        self._pending_mode = mode

    def on_center(self):
        """Pick the exit that gets closest to the target. This is the whole navigation rule."""
        player = getattr(self, "_player", None)
        chaser = getattr(self, "_chaser", None)

        options = self.legal_directions(allow_reverse=False)
        if not options:
            # A cul-de-sac or a state change that closed the way ahead: turning back is the
            # only legal move, and is always available.
            options = self.legal_directions(allow_reverse=True)
            if not options:
                self.direction = None
                return

        if self.state == FRIGHTENED:
            self.direction = self.rng.choice(options)
            return

        target = self.target_tile(player, chaser) if player else self.scatter_target
        if target is None:
            self.direction = self.rng.choice(options)
            return

        best, best_distance = None, None
        for direction in options:                     # fixed order gives deterministic ties
            candidate = self.tile_ahead(direction)
            distance = tile_distance(candidate, target)
            if best_distance is None or distance < best_distance:
                best, best_distance = direction, distance
        self.direction = best

    def on_stopped(self):
        options = self.legal_directions(allow_reverse=True)
        if options:
            self.direction = options[0]

    # --- drawing state -------------------------------------------------------------------
    def body_color(self):
        if self.state == EATEN:
            return None                                # eyes only
        if self.state == FRIGHTENED:
            return FRIGHTENED_FLASH_COLOR if self.frightened_flash else FRIGHTENED_COLOR
        return self.color


def make_ghosts(maze, seed=0):
    """The four ghosts, in a fixed order so tests and rendering are deterministic."""
    return [Ghost(maze, role, seed=seed)
            for role in ("chaser", "ambusher", "flanker", "drifter")]
