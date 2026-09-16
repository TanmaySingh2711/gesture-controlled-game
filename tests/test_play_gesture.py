"""The gesture application's interface: wording for every state, validation, themed rendering."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pygame
import pytest

from game.theme import THEMES, Theme
from src.game_integration import GestureController, SharedState, Snapshot
from src.play_gesture import (
    FRAME_HISTORY,
    GESTURE_HELP,
    PANEL_MARGIN,
    PANEL_WIDTH,
    GesturePacman,
    build_parser,
    command_color,
    command_text,
    confidence_text,
    control_status,
    loading_text,
    main,
    positive_int,
    probability,
    status_color,
)

STATUSES = [
    "starting",
    "loading model",
    "opening camera",
    "ready",
    "reconnecting",
    "camera error",
    "stopped",
]
INNER_WIDTH = PANEL_WIDTH - 2 * PANEL_MARGIN


def snap(status: str = "ready", command: str | None = "left") -> Snapshot:
    return Snapshot(
        sequence=1,
        timestamp=time.perf_counter(),
        stable_command=command,
        raw_direction="left",
        raw_confidence=0.97,
        camera_ok=status == "ready",
        status=status,
    )


@pytest.fixture
def app() -> GesturePacman:
    """The real application with a controller but no worker: real UI, faked recognition."""
    application = GesturePacman(use_camera=False)
    application.use_camera = True
    application.controller = GestureController(SharedState())
    return application


# --- wording -------------------------------------------------------------------------------
def test_headline_command_covers_every_worker_status() -> None:
    assert {status: command_text(snap(status), True, True) for status in STATUSES} == {
        "starting": "STARTING",
        "loading model": "STARTING",
        "opening camera": "STARTING",
        "ready": "LEFT",
        "reconnecting": "RECONNECTING",
        "camera error": "CAMERA ERROR",
        "stopped": "GESTURE OFF",
    }


def test_health_line_covers_every_worker_status() -> None:
    assert {status: control_status(snap(status), True, True) for status in STATUSES} == {
        "starting": "GESTURE CONTROL: STARTING",
        "loading model": "GESTURE CONTROL: STARTING",
        "opening camera": "GESTURE CONTROL: STARTING",
        "ready": "GESTURE CONTROL: READY",
        "reconnecting": "CAMERA RECONNECTING",
        "camera error": "CAMERA ERROR",
        "stopped": "GESTURE CONTROL: OFF",
    }
    assert control_status(snap("ready"), False, True) == "GESTURE CONTROL: WAITING"


def test_keyboard_only_mode_is_off_never_an_error() -> None:
    for status in STATUSES:
        assert command_text(snap(status), True, False) == "GESTURE OFF"
        assert control_status(snap(status), True, False) == "GESTURE CONTROL: OFF"


def test_confidence_only_accompanies_a_command_that_is_steering() -> None:
    assert confidence_text(snap(command="up"), True, True) == "Confidence: 97.0%"
    assert confidence_text(snap(command=None), True, True) is None
    assert confidence_text(snap(command="up"), False, True) is None
    assert confidence_text(snap(command="up"), True, False) is None


@pytest.mark.parametrize("theme", list(THEMES.values()), ids=list(THEMES))
def test_status_colours_come_from_the_active_theme(theme: Theme) -> None:
    assert command_color("LEFT", theme) == theme.directions["left"]
    assert command_color("CAMERA ERROR", theme) == theme.error
    assert command_color("RECONNECTING", theme) == theme.warn
    assert command_color("NO COMMAND", theme) == theme.dim_text
    assert status_color("GESTURE CONTROL: READY", theme) == theme.ok
    assert status_color("CAMERA ERROR", theme) == theme.error
    assert status_color("GESTURE CONTROL: OFF", theme) == theme.dim_text
    assert loading_text(snap("reconnecting"), True, theme) == (
        "Reconnecting to the camera...",
        theme.warn,
    )


# --- command-line validation -----------------------------------------------------------------
@pytest.mark.parametrize("text", ["0.5", "1", "0.9"])
def test_threshold_accepts_probabilities(text: str) -> None:
    assert probability(text) == float(text)


@pytest.mark.parametrize("text", ["0", "-0.1", "1.01", "abc", "nan"])
def test_threshold_rejects_anything_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        probability(text)


@pytest.mark.parametrize("text", ["0", "-5", "2.5", "many"])
def test_benchmark_frames_must_be_a_positive_whole_number(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(text)


def test_parser_exits_cleanly_on_a_bad_threshold() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--threshold", "2"])


# --- rendering ---------------------------------------------------------------------------------
@pytest.mark.parametrize("theme_name", list(THEMES))
def test_every_panel_string_fits_inside_the_panel(app: GesturePacman, theme_name: str) -> None:
    app.game.set_theme(THEMES[theme_name])
    for surface in app._ensure_static().values():
        assert surface.get_width() <= INNER_WIDTH
    for status in STATUSES:
        for fresh in (True, False):
            for command in ("left", "right", "up", "down", None):
                shown = snap(status, command)
                headline = command_text(shown, fresh, True)
                assert app.command_font.size(headline)[0] <= INNER_WIDTH, headline
                health = control_status(shown, fresh, True)
                assert app.font.size(health)[0] <= INNER_WIDTH, health


def test_static_text_is_rebuilt_only_when_the_theme_changes(app: GesturePacman) -> None:
    first = app._ensure_static()
    assert app._ensure_static() is first
    app.game.cycle_theme()
    assert app._ensure_static() is not first


@pytest.mark.parametrize("theme_name", list(THEMES))
@pytest.mark.parametrize("status", STATUSES)
def test_panel_and_start_screen_render_in_every_theme_and_state(
    app: GesturePacman, theme_name: str, status: str
) -> None:
    app.game.set_theme(THEMES[theme_name])
    assert app.controller is not None
    app.controller.state.publish(
        sequence=1,
        timestamp=time.perf_counter(),
        status=status,
        camera_ok=status == "ready",
        stable_command="up",
        preview=np.zeros((300, 300, 3), dtype=np.uint8),
    )
    app.draw_start()
    app.draw()


def test_help_overlay_lists_the_gestures_first(app: GesturePacman) -> None:
    assert app.game.help_lines[: len(GESTURE_HELP)] == list(GESTURE_HELP)


def test_theme_and_sound_can_be_changed_from_the_start_screen(app: GesturePacman) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
    app.handle_start_events()
    assert app.theme.name == "high-contrast"
    assert app.game.audio.muted
    assert app.phase == "start"


def test_frame_history_is_bounded(app: GesturePacman) -> None:
    assert app.frame_times.maxlen == FRAME_HISTORY


# --- reporting --------------------------------------------------------------------------------
def test_report_includes_capture_to_request_latency(
    app: GesturePacman, capsys: pytest.CaptureFixture[str]
) -> None:
    assert app.controller is not None
    app.frame_times.extend([1 / 60] * 120)
    now = time.perf_counter()
    for sequence in range(1, 6):
        app.controller.state.publish(
            sequence=sequence,
            timestamp=now,
            captured_at=now - 0.040,
            stable_command="left",
            camera_ok=True,
            status="ready",
        )
        app.controller.apply_to(app.game, now)
    app._report(released=True)
    output = capsys.readouterr().out
    assert "game: 120 frames | 60.0 FPS" in output
    assert "camera frame -> direction request: median 40.0 ms" in output
    assert "worker stopped cleanly: True" in output


def test_main_runs_a_keyboard_only_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--no-camera", "--benchmark", "5", "--log-level", "WARNING"]) == 0
    assert "game: 5 frames" in capsys.readouterr().out
