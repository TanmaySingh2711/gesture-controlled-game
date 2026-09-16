"""Core game rules: state flow, the mode schedule, frightened mode, collisions and input."""

from __future__ import annotations

import math
from pathlib import Path

import pygame
import pytest

from game.engine import (
    DYING,
    DYING_SECONDS,
    EXTRA_LIFE_SCORE,
    GAME_OVER,
    MODE_SCHEDULE,
    PLAYING,
    POWER_PELLET_SCORE,
    READY,
    READY_SECONDS,
    ROUND_CLEAR,
    ROUND_CLEAR_SECONDS,
    START_LIVES,
    Game,
)
from game.ghost import CHASE, EATEN, FRIGHTENED, GHOST_BASE_SPEED, HOUSE, SCATTER
from game.maze import PLAYER_SPAWN

STEP = 1.0 / 60.0


def key(game: Game, code: int) -> None:
    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=code))


# --- scoring -----------------------------------------------------------------------------
@pytest.mark.parametrize(("round_number", "seconds"), [(1, 6.0), (3, 5.0), (9, 2.0), (30, 2.0)])
def test_frightened_time_shrinks_each_round_to_a_floor(
    playing_game: Game, round_number: int, seconds: float
) -> None:
    playing_game.round = round_number
    assert playing_game.frightened_duration() == seconds


def test_extra_life_is_awarded_exactly_once(playing_game: Game) -> None:
    playing_game.add_score(EXTRA_LIFE_SCORE - 10)
    assert playing_game.lives == START_LIVES
    playing_game.add_score(10)
    assert playing_game.lives == START_LIVES + 1
    playing_game.add_score(EXTRA_LIFE_SCORE * 3)
    assert playing_game.lives == START_LIVES + 1


def test_high_score_is_the_better_of_the_record_and_the_current_game(playing_game: Game) -> None:
    playing_game.profile.high_score = 500
    playing_game.score = 200
    assert playing_game.high_score == 500
    playing_game.score = 900
    assert playing_game.high_score == 900


# --- state flow --------------------------------------------------------------------------
def test_ready_counts_down_into_play() -> None:
    game = Game(headless=True)
    assert game.state == READY
    game.update(READY_SECONDS - 0.2)
    assert game.state == READY
    game.update(0.3)
    assert game.state == PLAYING


def test_a_lost_life_resets_positions_but_keeps_pellets_and_score(playing_game: Game) -> None:
    playing_game.player.reset_to((1, 1), "right")
    playing_game.maze.pellets.discard((1, 2))
    remaining = playing_game.maze.pellets_remaining
    playing_game.score = 750
    playing_game._lose_life()
    assert (playing_game.state, playing_game.lives) == (DYING, START_LIVES - 1)

    playing_game.update(DYING_SECONDS + 0.01)
    assert (playing_game.state, playing_game.state_timer) == (READY, READY_SECONDS)
    assert playing_game.player.tile == PLAYER_SPAWN
    assert playing_game.maze.pellets_remaining == remaining
    assert playing_game.score == 750


def test_clearing_the_board_starts_a_faster_round_with_fresh_pellets(playing_game: Game) -> None:
    playing_game.maze.pellets.clear()
    playing_game.maze.power_pellets.clear()
    playing_game._check_round_clear()
    assert playing_game.state == ROUND_CLEAR

    playing_game.update(ROUND_CLEAR_SECONDS + 0.01)
    assert (playing_game.round, playing_game.state) == (2, READY)
    assert playing_game.maze.pellets_remaining == playing_game.maze.pellets_total
    assert all(g.base_speed == pytest.approx(GHOST_BASE_SPEED + 0.2) for g in playing_game.ghosts)


def test_nothing_moves_after_game_over(playing_game: Game) -> None:
    playing_game.state = GAME_OVER
    before = (playing_game.player.position, [g.position for g in playing_game.ghosts])
    for _ in range(60):
        playing_game.update(STEP)
    assert (playing_game.player.position, [g.position for g in playing_game.ghosts]) == before


# --- scatter / chase schedule ------------------------------------------------------------
def test_schedule_switches_mode_and_every_ghost_follows(playing_game: Game) -> None:
    chaser = playing_game.ghosts[0]
    assert (playing_game.mode, chaser.state) == (SCATTER, SCATTER)
    playing_game.mode_timer = 0.01
    playing_game._update_modes(0.02)
    assert (playing_game.mode, playing_game.mode_index) == (CHASE, 1)
    assert playing_game.mode_timer == MODE_SCHEDULE[1][1]
    assert chaser.state == CHASE
    assert all(g.state == HOUSE for g in playing_game.ghosts[1:]), "housed ghosts wait"


def test_schedule_ends_in_permanent_chase(playing_game: Game) -> None:
    playing_game.mode_index = len(MODE_SCHEDULE) - 2
    playing_game.mode_timer = 0.01
    playing_game._update_modes(0.02)
    assert (playing_game.mode, playing_game.mode_timer) == (CHASE, math.inf)

    playing_game.mode_timer = 5.0
    playing_game._update_modes(1000.0)
    assert (playing_game.mode_index, playing_game.mode_timer) == (len(MODE_SCHEDULE) - 1, 5.0)


def test_frightened_mode_pauses_the_schedule_flashes_then_ends(playing_game: Game) -> None:
    chaser = playing_game.ghosts[0]
    schedule_timer = playing_game.mode_timer
    playing_game._start_frightened()
    assert chaser.state == FRIGHTENED
    assert playing_game.ghosts[1].state == HOUSE, "a housed ghost cannot be frightened"

    playing_game._update_modes(1.0)
    assert not chaser.frightened_flash
    assert playing_game.mode_timer == schedule_timer

    playing_game._update_modes(3.6)  # 1.4 s left: inside the flashing window, on a lit phase
    assert chaser.frightened_flash

    playing_game.ghost_chain = 2
    playing_game._update_modes(2.0)
    assert playing_game.frightened_timer == 0.0
    assert playing_game.ghost_chain == 0
    assert (chaser.state, chaser.frightened_flash) == (playing_game.mode, False)


def test_power_pellet_scores_and_frightens(playing_game: Game) -> None:
    playing_game.player.reset_to((3, 1), "up")
    playing_game.ghost_chain = 3
    playing_game._eat_pellet()
    assert playing_game.score == POWER_PELLET_SCORE
    assert playing_game.frightened_timer == playing_game.frightened_duration()
    assert playing_game.ghost_chain == 0


# --- collisions --------------------------------------------------------------------------
def test_touching_a_dangerous_ghost_costs_a_life(playing_game: Game) -> None:
    chaser = playing_game.ghosts[0]
    chaser.reset_to(playing_game.player.tile)
    playing_game._handle_collisions()
    assert (playing_game.state, playing_game.lives) == (DYING, START_LIVES - 1)


@pytest.mark.parametrize("state", [EATEN, HOUSE])
def test_eyes_and_housed_ghosts_are_harmless(playing_game: Game, state: str) -> None:
    chaser = playing_game.ghosts[0]
    chaser.state = state
    chaser.reset_to(playing_game.player.tile)
    playing_game._handle_collisions()
    assert (playing_game.state, playing_game.lives, playing_game.score) == (
        PLAYING,
        START_LIVES,
        0,
    )


def test_eating_ghosts_in_one_frightened_spell_doubles_the_reward(playing_game: Game) -> None:
    for ghost in playing_game.ghosts[:3]:
        ghost.state = FRIGHTENED
        ghost.reset_to(playing_game.player.tile)
    playing_game._handle_collisions()
    assert playing_game.score == 200 + 400 + 800
    assert [popup.text for popup in playing_game.popups] == ["200", "400", "800"]


# --- input -------------------------------------------------------------------------------
@pytest.mark.parametrize(("code", "direction"), sorted(Game.KEY_DIRECTIONS.items()))
def test_direction_keys_only_submit_a_request(
    playing_game: Game, code: int, direction: str
) -> None:
    position = playing_game.player.position
    key(playing_game, code)
    assert playing_game.controls.pending == direction
    assert playing_game.player.position == position


def test_quit_and_escape_stop_the_game(playing_game: Game) -> None:
    playing_game.handle_event(pygame.event.Event(pygame.QUIT))
    assert not playing_game.running
    playing_game.running = True
    key(playing_game, pygame.K_ESCAPE)
    assert not playing_game.running


def test_non_key_events_are_ignored(playing_game: Game) -> None:
    playing_game.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_LEFT))
    playing_game.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(5, 5)))
    assert playing_game.controls.pending is None
    assert playing_game.running


def test_restart_only_works_after_game_over(playing_game: Game) -> None:
    playing_game.score = 420
    key(playing_game, pygame.K_r)
    assert playing_game.score == 420

    playing_game.state = GAME_OVER
    playing_game.lives = 0
    key(playing_game, pygame.K_r)
    assert (playing_game.score, playing_game.lives, playing_game.state) == (0, START_LIVES, READY)


def test_sound_status_reports_mute_when_a_device_exists(playing_game: Game) -> None:
    playing_game.audio.available = True
    assert playing_game.sound_status == "on"
    playing_game.audio.muted = True
    assert playing_game.sound_status == "off"


def test_main_loop_runs_until_quit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GESTURE_PACMAN_HOME", str(tmp_path))
    game = Game()  # a real display surface on the dummy video driver
    frames: list[float] = []
    original_update = game.update

    def counting_update(dt: float) -> None:
        frames.append(dt)
        original_update(dt)

    monkeypatch.setattr(game, "update", counting_update)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    game.run()
    assert not game.running
    assert len(frames) == 1
    assert 0.0 <= frames[0] <= 0.05, "a stalled frame must be clamped"
