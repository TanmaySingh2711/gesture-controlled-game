"""Player turning rules and the shared grid movement underneath every character."""

from __future__ import annotations

import pytest

from game.controls import REQUEST_GRACE, Controls
from game.entity import MAX_SUB_STEP, MAX_SUB_STEPS, Entity
from game.maze import PLAYER_SPAWN, TILE, Maze
from game.player import MOUTH_CYCLE, PLAYER_SPEED, Player

FRAME = 1.0 / 60.0


@pytest.fixture
def maze() -> Maze:
    return Maze()


def run(player: Player, controls: Controls, seconds: float) -> None:
    for _ in range(round(seconds / FRAME)):
        player.update(FRAME, controls)


# --- player ------------------------------------------------------------------------------
def test_player_spawns_facing_left_with_a_closed_mouth(maze: Maze) -> None:
    player = Player(maze)
    assert (player.tile, player.direction, player.mouth_timer) == (PLAYER_SPAWN, "left", 0.0)


def test_reset_returns_the_player_to_spawn(maze: Maze) -> None:
    player = Player(maze)
    run(player, Controls(), 0.3)
    player.reset()
    assert player.position == maze.tile_center(*PLAYER_SPAWN)
    assert (player.direction, player.mouth_timer) == ("left", 0.0)


def test_player_moves_at_its_speed(maze: Maze) -> None:
    player = Player(maze)
    start_x = player.x
    player.update(0.1, Controls())
    assert start_x - player.x == pytest.approx(PLAYER_SPEED * 0.1)


def test_reversal_is_immediate_mid_corridor(maze: Maze) -> None:
    player = Player(maze)
    controls = Controls()
    player.update(0.05, controls)
    assert not player.at_center()
    controls.request_direction("right")
    player.update(FRAME, controls)
    assert player.direction == "right"
    assert controls.pending is None


def test_a_turn_requested_early_is_taken_at_the_junction(maze: Maze) -> None:
    player = Player(maze)
    player.reset_to((22, 10), "left")  # the junction at column 9 opens upward
    controls = Controls()
    controls.request_direction("up")
    run(player, controls, 0.3)
    assert player.direction == "up"
    assert player.tile[1] == 9
    assert controls.pending is None


def test_a_stale_request_expires_without_turning(maze: Maze) -> None:
    player = Player(maze)
    player.reset_to((22, 8), "left")  # the next upward opening, column 4, is four tiles away
    controls = Controls()
    controls.request_direction("up")
    run(player, controls, REQUEST_GRACE + 0.05)
    assert controls.pending is None
    assert player.direction == "left"


def test_player_stops_at_a_wall_and_restarts_on_a_legal_request(maze: Maze) -> None:
    player = Player(maze)
    player.reset_to((22, 5), "left")  # (22, 3) is a wall
    controls = Controls()
    run(player, controls, 0.5)
    assert player.direction is None
    assert player.position == maze.tile_center(22, 4)

    controls.request_direction("left")  # still into the wall: ignored, but kept buffered
    player.update(FRAME, controls)
    assert (player.direction, controls.pending) == (None, "left")

    controls.request_direction("down")
    player.update(FRAME, controls)
    assert player.direction == "down"


def test_player_wraps_through_the_tunnel(maze: Maze) -> None:
    player = Player(maze)
    player.reset_to((14, 0), "left")
    player.update(0.1, Controls())
    assert player.tile == (14, 27)
    assert 0 <= player.x < maze.pixel_width


@pytest.mark.parametrize(
    ("fraction", "openness"), [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 0.5)]
)
def test_mouth_is_a_triangle_wave(maze: Maze, fraction: float, openness: float) -> None:
    player = Player(maze)
    player.mouth_timer = MOUTH_CYCLE * fraction
    assert player.mouth_openness == pytest.approx(openness)


def test_mouth_timer_wraps_each_cycle(maze: Maze) -> None:
    player = Player(maze)
    run(player, Controls(), 1.0)
    assert 0.0 <= player.mouth_timer < MOUTH_CYCLE


@pytest.mark.parametrize(
    ("direction", "angle"),
    [("right", 0.0), ("up", 90.0), ("left", 180.0), ("down", 270.0), (None, 180.0)],
)
def test_facing_angle(maze: Maze, direction: str | None, angle: float) -> None:
    player = Player(maze)
    player.direction = direction
    assert player.facing_angle == angle


# --- entity ------------------------------------------------------------------------------
def test_entity_geometry(maze: Maze) -> None:
    a = Entity(maze, (1, 1))
    b = Entity(maze, (1, 4))
    assert a.position == maze.tile_center(1, 1)
    assert a.distance_to(b) == pytest.approx(3 * TILE)
    assert a.at_center()
    a.x += 1.0
    assert not a.at_center()


def test_legal_directions_can_exclude_the_way_back(maze: Maze) -> None:
    entity = Entity(maze, (1, 4), direction="left")
    assert entity.legal_directions() == ["left", "down", "right"]
    assert entity.legal_directions(allow_reverse=False) == ["left", "down"]


def test_base_entity_stops_at_a_wall_and_stays_stopped(maze: Maze) -> None:
    entity = Entity(maze, (22, 5), direction="left", speed=PLAYER_SPEED)
    entity.move(1.0)
    assert entity.direction is None
    assert entity.position == maze.tile_center(22, 4)
    entity.move(1.0)
    assert entity.position == maze.tile_center(22, 4)


def test_reverse_without_a_direction_is_a_no_op(maze: Maze) -> None:
    entity = Entity(maze, (1, 1))
    entity.reverse()
    assert entity.direction is None


def test_a_huge_time_step_is_bounded_and_stays_on_the_corridor_line(maze: Maze) -> None:
    entity = Entity(maze, (29, 26), direction="left", speed=PLAYER_SPEED)
    start_x = entity.x
    entity.move(1000.0)
    assert start_x - entity.x <= MAX_SUB_STEP * MAX_SUB_STEPS
    assert entity.y == maze.tile_center(29, 26)[1]


def test_humans_and_entities_cannot_use_the_ghost_door(maze: Maze) -> None:
    entity = Entity(maze, (11, 13), direction="down")
    assert not entity.can_pass_doors()
    assert not entity.can_enter_house()
    assert not entity.can_move("down")
