"""The maze grid: shape, fairness, walkability rules and geometry round-trips."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from game.maze import (
    COLUMNS,
    FRUIT_TILE,
    GHOST_SPAWN,
    HOUSE_DOOR,
    HOUSE_EXIT,
    PLAYER_SPAWN,
    ROWS,
    TUNNEL_ROW,
    Maze,
)


def test_dimensions_and_mirror_symmetry() -> None:
    maze = Maze()
    assert len(maze.layout) == ROWS
    assert all(len(row) == COLUMNS for row in maze.layout)
    assert all(row == row[::-1] for row in maze.layout)


def test_every_pellet_is_reachable_and_there_are_no_dead_ends() -> None:
    maze = Maze()
    assert maze.unreachable_pellets() == []
    assert maze.dead_ends() == []


def test_pellet_totals_agree() -> None:
    maze = Maze()
    assert maze.pellets_total == maze.pellets_remaining == 334
    assert len(maze.power_pellets) == 4


def test_player_is_kept_out_of_the_ghost_house_but_ghosts_are_not() -> None:
    maze = Maze()
    assert not maze.is_walkable(*HOUSE_DOOR)
    assert maze.is_walkable(*HOUSE_DOOR, doors=True)
    inside = GHOST_SPAWN["ambusher"]
    assert not maze.is_walkable(*inside)
    assert maze.is_walkable(*inside, house=True)


def test_special_tiles_are_walkable() -> None:
    maze = Maze()
    for tile in (PLAYER_SPAWN, HOUSE_EXIT, FRUIT_TILE):
        assert maze.is_walkable(*tile), tile


def test_tunnel_wraps_in_both_directions() -> None:
    maze = Maze()
    assert maze.wrap(TUNNEL_ROW, -1) == (TUNNEL_ROW, COLUMNS - 1)
    assert maze.wrap(TUNNEL_ROW, COLUMNS) == (TUNNEL_ROW, 0)
    assert (TUNNEL_ROW, COLUMNS - 1) in maze.neighbours(TUNNEL_ROW, 0)


def test_eating_removes_a_pellet_exactly_once() -> None:
    maze = Maze()
    tile = next(iter(maze.pellets))
    assert maze.eat(tile) == "pellet"
    assert maze.eat(tile) is None
    power = next(iter(maze.power_pellets))
    assert maze.eat(power) == "power"
    maze.reset_pellets()
    assert maze.pellets_remaining == 334


def test_out_of_bounds_is_wall_and_never_walkable() -> None:
    maze = Maze()
    for tile in ((-1, 5), (ROWS, 5), (5, -1), (5, COLUMNS)):
        assert maze.is_wall(*tile)
        assert not maze.is_walkable(*tile, doors=True, house=True)


@given(st.integers(0, ROWS - 1), st.integers(0, COLUMNS - 1))
def test_tile_centre_round_trips(row: int, column: int) -> None:
    assert Maze.world_to_tile(*Maze.tile_center(row, column)) == (row, column)
