"""Ghost state machine, role targeting, speeds and navigation, driven on the real maze."""

from __future__ import annotations

import pytest

from game.entity import Entity
from game.ghost import (
    CHASE,
    DRIFTER_RETREAT_TILES,
    EATEN,
    EATEN_SPEED,
    ELROY_MAX_SPEED,
    FRIGHTENED,
    FRIGHTENED_COLOR,
    FRIGHTENED_FLASH_COLOR,
    FRIGHTENED_SPEED,
    GHOST_BASE_SPEED,
    GHOST_MAX_SPEED,
    HOUSE,
    HOUSE_SPEED,
    RELEASE_DELAY,
    RESPAWN_DELAY,
    ROLES,
    SCATTER,
    Ghost,
    make_ghosts,
    tile_distance,
)
from game.maze import GHOST_SPAWN, HOUSE_EXIT, PLAYER_SPAWN, SCATTER_TARGETS, TILE, Maze
from game.player import PLAYER_SPEED

FRAME = 1.0 / 60.0


@pytest.fixture
def maze() -> Maze:
    return Maze()


def ghost_named(maze: Maze, role: str) -> Ghost:
    return next(ghost for ghost in make_ghosts(maze) if ghost.role == role)


def player_at(maze: Maze, tile: tuple[int, int], direction: str | None = "left") -> Entity:
    return Entity(maze, tile, direction=direction)


# --- lifecycle ---------------------------------------------------------------------------
def test_tile_distance_is_squared_euclidean() -> None:
    assert tile_distance((0, 0), (3, 4)) == 25
    assert tile_distance((5, 5), (5, 5)) == 0


def test_make_ghosts_returns_every_role_in_a_fixed_order(maze: Maze) -> None:
    assert [ghost.role for ghost in make_ghosts(maze)] == list(ROLES)


def test_chaser_starts_outside_and_the_rest_wait_in_the_house(maze: Maze) -> None:
    for ghost in make_ghosts(maze):
        assert ghost.tile == GHOST_SPAWN[ghost.role]
        assert ghost.release_timer == RELEASE_DELAY[ghost.role]
        if ghost.role == "chaser":
            assert (ghost.state, ghost.released, ghost.direction) == (SCATTER, True, "left")
        else:
            assert (ghost.state, ghost.released, ghost.direction) == (HOUSE, False, "up")


def test_reset_rewinds_state_position_and_timer(maze: Maze) -> None:
    ghost = ghost_named(maze, "ambusher")
    ghost.reset_to((1, 1), "right")
    ghost.state = FRIGHTENED
    ghost.frightened_flash = True
    ghost.release_timer = 0.0
    ghost.reset()
    assert ghost.tile == GHOST_SPAWN["ambusher"]
    assert (ghost.state, ghost.frightened_flash) == (HOUSE, False)
    assert ghost.release_timer == RELEASE_DELAY["ambusher"]


# --- speed -------------------------------------------------------------------------------
@pytest.mark.parametrize("round_number", [1, 2, 5, 10, 50])
def test_round_speed_rises_but_never_reaches_the_player(maze: Maze, round_number: int) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.set_round(round_number)
    expected = min(GHOST_MAX_SPEED, GHOST_BASE_SPEED + 0.2 * (round_number - 1))
    assert ghost.base_speed == pytest.approx(expected)
    assert ghost.base_speed * TILE < PLAYER_SPEED


@pytest.mark.parametrize(
    ("remaining", "bonus"), [(330, 0.0), (21, 0.0), (20, 0.3), (11, 0.3), (10, 0.5), (0, 0.5)]
)
def test_elroy_bonus_follows_the_pellet_thresholds(
    maze: Maze, remaining: int, bonus: float
) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.set_pellets_remaining(remaining)
    assert chaser.elroy_bonus == bonus


@pytest.mark.parametrize("role", [role for role in ROLES if role != "chaser"])
def test_only_the_chaser_surges(maze: Maze, role: str) -> None:
    ghost = ghost_named(maze, role)
    ghost.set_pellets_remaining(0)
    assert ghost.elroy_bonus == 0.0


def test_elroy_applies_in_chase_only_and_stays_below_the_player(maze: Maze) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.set_round(50)
    chaser.set_pellets_remaining(0)
    chaser.state = SCATTER
    assert chaser.current_speed() == pytest.approx(chaser.base_speed * TILE)
    chaser.state = CHASE
    assert chaser.current_speed() == pytest.approx(ELROY_MAX_SPEED * TILE)
    assert chaser.current_speed() < PLAYER_SPEED


def test_a_new_round_clears_the_surge(maze: Maze) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.set_pellets_remaining(0)
    chaser.set_round(2)
    assert chaser.elroy_bonus == 0.0


@pytest.mark.parametrize(
    ("state", "tiles_per_second"),
    [(FRIGHTENED, FRIGHTENED_SPEED), (EATEN, EATEN_SPEED), (HOUSE, HOUSE_SPEED)],
)
def test_state_speeds(maze: Maze, state: str, tiles_per_second: float) -> None:
    ghost = ghost_named(maze, "flanker")
    ghost.state = state
    assert ghost.current_speed() == pytest.approx(tiles_per_second * TILE)


# --- state transitions -------------------------------------------------------------------
def test_mode_switch_reverses_unless_told_not_to(maze: Maze) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.set_mode(CHASE)
    assert (chaser.state, chaser.direction) == (CHASE, "right")
    chaser.set_mode(SCATTER, reverse=False)
    assert (chaser.state, chaser.direction) == (SCATTER, "right")
    chaser.set_mode(SCATTER)
    assert chaser.direction == "right", "re-applying the current mode must not reverse"


@pytest.mark.parametrize("state", [HOUSE, EATEN, FRIGHTENED])
def test_mode_switch_is_ignored_outside_scatter_and_chase(maze: Maze, state: str) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.state = state
    ghost.set_mode(CHASE)
    assert ghost.state == state


def test_frighten_reverses_a_dangerous_ghost_and_makes_it_edible(maze: Maze) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.frightened_flash = True
    assert chaser.frighten() is True
    assert (chaser.state, chaser.direction, chaser.frightened_flash) == (FRIGHTENED, "right", False)
    assert chaser.is_edible
    assert not chaser.is_dangerous


@pytest.mark.parametrize("state", [HOUSE, EATEN])
def test_frighten_leaves_house_and_eaten_ghosts_alone(maze: Maze, state: str) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.state = state
    assert ghost.frighten() is False
    assert ghost.state == state


def test_unfrighten_only_changes_a_frightened_ghost(maze: Maze) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.frighten()
    ghost.frightened_flash = True
    ghost.unfrighten(CHASE)
    assert (ghost.state, ghost.frightened_flash) == (CHASE, False)
    ghost.state = EATEN
    ghost.unfrighten(CHASE)
    assert ghost.state == EATEN


def test_eaten_ghost_is_neither_edible_nor_dangerous(maze: Maze) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.frighten()
    ghost.frightened_flash = True
    ghost.get_eaten()
    assert (ghost.state, ghost.frightened_flash) == (EATEN, False)
    assert not ghost.is_edible
    assert not ghost.is_dangerous
    assert not ghost.in_house


@pytest.mark.parametrize(
    ("state", "allowed"), [(HOUSE, True), (EATEN, True), (SCATTER, False), (CHASE, False)]
)
def test_only_house_and_eaten_ghosts_use_the_door(maze: Maze, state: str, allowed: bool) -> None:
    ghost = ghost_named(maze, "ambusher")
    ghost.state = state
    assert ghost.can_pass_doors() is allowed
    assert ghost.can_enter_house() is allowed


def test_body_colour_by_state(maze: Maze) -> None:
    ghost = ghost_named(maze, "flanker")
    ghost.state = CHASE
    assert ghost.body_color() == ghost.color
    ghost.state = FRIGHTENED
    assert ghost.body_color() == FRIGHTENED_COLOR
    ghost.frightened_flash = True
    assert ghost.body_color() == FRIGHTENED_FLASH_COLOR
    ghost.state = EATEN
    assert ghost.body_color() is None


# --- targeting ---------------------------------------------------------------------------
def test_target_by_state(maze: Maze) -> None:
    player = player_at(maze, PLAYER_SPAWN)
    ghost = ghost_named(maze, "ambusher")
    ghost.state = SCATTER
    assert ghost.target_tile(player, None) == SCATTER_TARGETS["ambusher"]
    ghost.state = EATEN
    assert ghost.target_tile(player, None) == GHOST_SPAWN["ambusher"]
    ghost.state = HOUSE
    assert ghost.target_tile(player, None) == HOUSE_EXIT
    ghost.state = FRIGHTENED
    assert ghost.target_tile(player, None) is None
    ghost.state = CHASE
    assert ghost.target_tile(player, None) == ghost.chase_target(player, None)


def test_chaser_targets_the_player_tile(maze: Maze) -> None:
    player = player_at(maze, (22, 13), "left")
    assert ghost_named(maze, "chaser").chase_target(player, None) == (22, 13)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("left", (22, 9)), ("right", (22, 17)), ("up", (18, 13)), ("down", (26, 13)), (None, (22, 9))],
)
def test_ambusher_targets_four_tiles_ahead(
    maze: Maze, direction: str | None, expected: tuple[int, int]
) -> None:
    player = player_at(maze, (22, 13), direction)
    assert ghost_named(maze, "ambusher").chase_target(player, None) == expected


def test_flanker_reflects_the_chaser_through_the_pivot(maze: Maze) -> None:
    ghosts = make_ghosts(maze)
    chaser, flanker = ghosts[0], ghosts[2]
    player = player_at(maze, (22, 13), "left")
    # pivot is two tiles ahead of the player: (22, 11); the chaser sits at (11, 13)
    assert flanker.chase_target(player, chaser) == (33, 9)
    assert flanker.chase_target(player, None) == (22, 11)


def test_drifter_chases_from_far_and_retreats_up_close(maze: Maze) -> None:
    drifter = ghost_named(maze, "drifter")
    player = player_at(maze, (22, 13))
    drifter.reset_to((14, 15))  # 8^2 + 2^2 = 68, just outside the retreat radius
    assert drifter.chase_target(player, None) == (22, 13)
    drifter.reset_to((22 - DRIFTER_RETREAT_TILES, 13))  # exactly on the radius
    assert drifter.chase_target(player, None) == SCATTER_TARGETS["drifter"]


# --- movement and navigation -------------------------------------------------------------
def test_house_ghost_bobs_in_place_until_released(maze: Maze) -> None:
    ghost = ghost_named(maze, "ambusher")
    player = player_at(maze, PLAYER_SPAWN)
    centre_x, centre_y = maze.tile_center(*GHOST_SPAWN["ambusher"])
    heights = []
    for _ in range(int(RELEASE_DELAY["ambusher"] / FRAME) - 2):
        ghost.update(FRAME, player, None)
        heights.append(ghost.y)
        assert ghost.x == centre_x
    assert not ghost.released
    assert min(heights) == pytest.approx(centre_y - TILE * 0.35)
    assert max(heights) == pytest.approx(centre_y + TILE * 0.35)
    for _ in range(5):
        ghost.update(FRAME, player, None)
    assert ghost.released


def run_until(ghost: Ghost, player: Entity, done: object, seconds: float) -> None:
    for _ in range(int(seconds / FRAME)):
        ghost.update(FRAME, player, None)
        if callable(done) and done():
            return
    raise AssertionError(f"{ghost.role} did not reach the expected state within {seconds} s")


@pytest.mark.parametrize(("pending", "expected"), [(SCATTER, SCATTER), (None, CHASE)])
def test_released_ghost_leaves_through_the_door_and_joins_the_mode(
    maze: Maze, pending: str | None, expected: str
) -> None:
    ghost = ghost_named(maze, "ambusher")
    player = player_at(maze, PLAYER_SPAWN)
    if pending is not None:
        ghost.rejoin_mode(pending)
    run_until(ghost, player, lambda: ghost.state != HOUSE, RELEASE_DELAY["ambusher"] + 3.0)
    assert ghost.state == expected
    assert ghost.tile == HOUSE_EXIT


def test_eaten_ghost_returns_home_and_waits_to_respawn(maze: Maze) -> None:
    ghost = ghost_named(maze, "ambusher")
    ghost.reset_to(HOUSE_EXIT, "left")
    ghost.state = EATEN
    player = player_at(maze, PLAYER_SPAWN)
    run_until(ghost, player, lambda: ghost.state == HOUSE, 2.0)
    assert ghost.tile == GHOST_SPAWN["ambusher"]
    assert (ghost.released, ghost.release_timer, ghost.direction) == (False, RESPAWN_DELAY, "up")


def test_navigation_without_a_player_heads_for_the_scatter_corner(maze: Maze) -> None:
    chaser = ghost_named(maze, "chaser")
    chaser.reset_to((1, 4), "left")
    chaser.on_center()
    # right is behind; of left (1,3) and down (2,4), down is nearer the top-right corner
    assert chaser.direction == "down"


def test_boxed_in_ghost_stops_instead_of_crashing(maze: Maze) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.reset_to((3, 6), "left")  # a wall tile with walls on every side
    ghost.on_center()
    assert ghost.direction is None
    ghost.on_stopped()
    assert ghost.direction is None


def test_stopped_ghost_restarts_in_the_fixed_direction_order(maze: Maze) -> None:
    ghost = ghost_named(maze, "chaser")
    ghost.reset_to((1, 1), None)  # up and left are walls; down comes before right
    ghost.on_stopped()
    assert ghost.direction == "down"


def frightened_path(maze: Maze, seed: int) -> list[tuple[int, int]]:
    ghost = Ghost(maze, "flanker", seed=seed)
    ghost.reset_to((8, 6), "left")
    ghost.state = FRIGHTENED
    player = player_at(maze, PLAYER_SPAWN)
    path = []
    for _ in range(600):
        ghost.update(FRAME, player, None)
        path.append(ghost.tile)
    return path


def test_frightened_wandering_is_reproducible_for_a_seed(maze: Maze) -> None:
    assert frightened_path(maze, 3) == frightened_path(Maze(), 3)


def test_ghosts_never_leave_the_corridors(maze: Maze) -> None:
    ghosts = make_ghosts(maze, seed=11)
    player = player_at(maze, PLAYER_SPAWN)
    for ghost in ghosts:
        ghost.set_mode(CHASE)
    for frame in range(60 * 30):
        if frame == 60 * 10:
            for ghost in ghosts:
                ghost.frighten()
        for ghost in ghosts:
            ghost.update(FRAME, player, ghosts[0])
            row, column = ghost.tile
            assert ghost.can_walk(row, column), f"{ghost.role} inside a wall at {ghost.tile}"
            centre_x, centre_y = maze.tile_center(row, column)
            assert ghost.x == pytest.approx(centre_x) or ghost.y == pytest.approx(centre_y)
