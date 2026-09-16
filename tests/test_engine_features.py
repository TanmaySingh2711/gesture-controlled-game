"""Engine features added for playability and accessibility, and the fixes that came with them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pygame
import pytest

from game.engine import (
    DYING,
    FRUIT_POINTS,
    FRUIT_SECONDS,
    GAME_OVER,
    PLAYING,
    POPUP_SECONDS,
    READY,
    ROUND_CLEAR,
    WALL_INSET,
    Game,
)
from game.ghost import CHASE, EATEN, FRIGHTENED
from game.maze import FRUIT_TILE, TILE
from game.player import PLAYER_SPEED
from game.profile import ProfileStore
from game.theme import CLASSIC, THEMES

STEP = 1.0 / 60.0


def rgb(surface: pygame.Surface, x: int, y: int) -> tuple[int, int, int]:
    red, green, blue, _alpha = surface.get_at((x, y))
    return (red, green, blue)


def key(game: Game, code: int) -> None:
    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=code))


def snapshot(game: Game) -> tuple[object, ...]:
    return (
        game.player.position,
        tuple(g.position for g in game.ghosts),
        game.mode_timer,
        game.frightened_timer,
        game.score,
    )


# --- pause and help ------------------------------------------------------------------------
def test_pause_freezes_everything_and_drops_the_buffered_turn(playing_game: Game) -> None:
    playing_game.request_direction("up")
    key(playing_game, pygame.K_p)
    assert playing_game.paused
    assert playing_game.controls.pending is None
    frozen = snapshot(playing_game)
    for _ in range(90):
        playing_game.update(STEP)
    assert snapshot(playing_game) == frozen
    assert playing_game.banner_lines() == ("PAUSED", ["Press P to resume"])

    key(playing_game, pygame.K_p)
    for _ in range(10):
        playing_game.update(STEP)
    assert playing_game.player.position != frozen[0]


def test_game_over_cannot_be_paused(playing_game: Game) -> None:
    playing_game.state = GAME_OVER
    playing_game.toggle_pause()
    assert not playing_game.paused


@pytest.mark.parametrize("code", [pygame.K_h, pygame.K_F1])
def test_help_overlay_toggles_and_freezes_play(playing_game: Game, code: int) -> None:
    key(playing_game, code)
    assert playing_game.show_help
    frozen = snapshot(playing_game)
    for _ in range(30):
        playing_game.update(STEP)
    playing_game.draw()
    assert snapshot(playing_game) == frozen
    key(playing_game, code)
    assert not playing_game.show_help


# --- theme and sound -----------------------------------------------------------------------
def test_theme_cycles_rebuilds_the_maze_once_and_is_remembered(
    playing_game: Game, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[str] = []
    original = playing_game._build_maze_surface

    def counting() -> pygame.Surface:
        builds.append(playing_game.theme.name)
        return original()

    monkeypatch.setattr(playing_game, "_build_maze_surface", counting)
    for _ in range(20):
        playing_game.draw()
    assert builds == [], "the static maze must not be redrawn every frame"

    key(playing_game, pygame.K_c)
    assert playing_game.theme.name == "high-contrast"
    assert playing_game.profile.theme == "high-contrast"
    assert builds == ["high-contrast"]


def test_mute_is_recorded_in_the_profile(playing_game: Game) -> None:
    key(playing_game, pygame.K_m)
    assert playing_game.audio.muted and playing_game.profile.muted
    key(playing_game, pygame.K_m)
    assert not playing_game.audio.muted and not playing_game.profile.muted


def test_walls_render_as_connected_outlines() -> None:
    surface = Game(headless=True)._build_maze_surface()
    # Tile (0, 1) is wall with open corridor below it: its outline runs along its lower edge,
    # inset from the border, and joins the neighbouring wall tiles instead of stopping short.
    edge_y = TILE - WALL_INSET - 1
    for x in (TILE, TILE + 10, 2 * TILE - 1):
        assert rgb(surface, x, edge_y) == CLASSIC.wall_edge
    assert rgb(surface, TILE + 10, 5) == CLASSIC.wall_fill
    assert rgb(surface, TILE + 10, TILE + 10) == CLASSIC.background  # corridor


@pytest.mark.parametrize("theme_name", list(THEMES))
@pytest.mark.parametrize("state", [READY, PLAYING, DYING, ROUND_CLEAR, GAME_OVER])
def test_every_state_draws_in_every_theme(playing_game: Game, theme_name: str, state: str) -> None:
    playing_game.set_theme(THEMES[theme_name])
    playing_game.state = state
    playing_game.state_timer = 0.5
    playing_game.ghosts[1].state = FRIGHTENED
    playing_game.ghosts[2].state = EATEN
    playing_game.fruit_timer = 3.0
    playing_game._popup("400", 100, 100)
    playing_game.draw()


# --- bonus fruit ------------------------------------------------------------------------------
def _eat_until(game: Game, count: int) -> None:
    pellets = sorted(game.maze.pellets - {FRUIT_TILE, game.player.tile})
    for tile in pellets[: count - 1]:
        game.maze.pellets.discard(tile)
    game.player.reset_to(pellets[count - 1], "left")
    game._eat_pellet()


def test_fruit_appears_after_seventy_pellets_and_scores_when_eaten(playing_game: Game) -> None:
    _eat_until(playing_game, 70)
    assert playing_game.fruit_timer == FRUIT_SECONDS

    score = playing_game.score
    playing_game.player.reset_to(FRUIT_TILE, "left")
    playing_game._update_fruit(STEP)
    assert playing_game.score == score + FRUIT_POINTS[0]
    assert playing_game.fruit_timer == 0.0
    assert playing_game.popups[-1].text == str(FRUIT_POINTS[0])


def test_each_fruit_threshold_fires_only_once(playing_game: Game) -> None:
    _eat_until(playing_game, 70)
    playing_game.fruit_timer = 1.0
    playing_game._maybe_spawn_fruit()
    assert playing_game.fruit_timer == 1.0


def test_uneaten_fruit_disappears(playing_game: Game) -> None:
    playing_game.fruit_timer = FRUIT_SECONDS
    for _ in range(int((FRUIT_SECONDS + 0.2) / STEP)):
        playing_game._update_fruit(STEP)
    assert playing_game.fruit_timer == 0.0


def test_fruit_value_rises_with_the_round_and_then_holds(playing_game: Game) -> None:
    values = []
    for round_number in range(1, 12):
        playing_game.round = round_number
        values.append(playing_game.fruit_points())
    assert values[: len(FRUIT_POINTS)] == list(FRUIT_POINTS)
    assert set(values[len(FRUIT_POINTS) :]) == {FRUIT_POINTS[-1]}


def test_a_death_removes_the_fruit(playing_game: Game) -> None:
    playing_game.fruit_timer = FRUIT_SECONDS
    playing_game._reset_positions()
    assert playing_game.fruit_timer == 0.0


# --- the Chaser's end-of-round surge -------------------------------------------------------------
def test_chaser_surges_near_the_end_but_never_outruns_pacman(playing_game: Game) -> None:
    chaser = next(g for g in playing_game.ghosts if g.role == "chaser")
    chaser.state = CHASE
    normal = chaser.current_speed()
    chaser.set_pellets_remaining(20)
    first = chaser.current_speed()
    chaser.set_pellets_remaining(10)
    second = chaser.current_speed()
    assert normal < first < second < PLAYER_SPEED

    for round_number in range(1, 40):
        chaser.set_round(round_number)
        chaser.set_pellets_remaining(1)
        assert chaser.current_speed() < PLAYER_SPEED

    chaser.state = FRIGHTENED
    assert chaser.current_speed() < normal


def test_only_the_chaser_surges(playing_game: Game) -> None:
    for ghost in playing_game.ghosts:
        if ghost.role == "chaser":
            continue
        ghost.state = CHASE
        ghost.set_pellets_remaining(1)
        assert ghost.current_speed() == ghost.base_speed * TILE


# --- scores ---------------------------------------------------------------------------------------
def _finish_game(game: Game, score: int) -> None:
    game.state = PLAYING
    game.score = score
    game.lives = 1
    game._lose_life()
    game.state_timer = 0.0
    game.update(0.0)
    assert game.state == GAME_OVER


def test_high_score_persists_across_games(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.json")
    first = Game(headless=True, profile=store)
    _finish_game(first, 1234)
    assert "NEW HIGH SCORE!" in first.banner_lines()[1]  # type: ignore[index]

    second = Game(headless=True, profile=store)
    assert second.hud_text()["high_score"] == "HIGH 1234"
    _finish_game(second, 50)
    assert "NEW HIGH SCORE!" not in second.banner_lines()[1]  # type: ignore[index]
    assert ProfileStore(tmp_path / "profile.json").load().high_score == 1234


def test_eating_a_ghost_shows_its_points_where_it_happened(playing_game: Game) -> None:
    ghost = playing_game.ghosts[0]
    ghost.state = FRIGHTENED
    ghost.reset_to(playing_game.player.tile)
    playing_game._handle_collisions()
    assert playing_game.popups[-1].text == "200"
    assert (playing_game.popups[-1].x, playing_game.popups[-1].y) == ghost.position


def test_popups_expire(playing_game: Game) -> None:
    playing_game.state = READY
    playing_game._popup("100", 10, 10)
    playing_game.update(POPUP_SECONDS + 0.01)
    assert playing_game.popups == []


# --- isolation and determinism -------------------------------------------------------------------
def test_dummy_display_never_plays_sound_or_writes_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GESTURE_PACMAN_HOME", str(tmp_path))
    game = Game()  # a windowed game, but on the dummy video driver the tests use
    assert not game.interactive
    assert not game.audio.available
    assert game.profile_store.path is None
    game.add_score(500)
    game.cycle_theme()
    game.toggle_mute()
    assert list(tmp_path.iterdir()) == []


def test_ghost_randomness_is_identical_across_processes() -> None:
    """Regression: ghosts were seeded with hash(role), which differs per Python process."""
    code = (
        "from game.maze import Maze; from game.ghost import make_ghosts; "
        "print([round(g.rng.random(), 12) for g in make_ghosts(Maze(), seed=3)])"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        ).stdout
        for hash_seed in (1, 2, 3)
    }
    assert len(outputs) == 1, outputs
