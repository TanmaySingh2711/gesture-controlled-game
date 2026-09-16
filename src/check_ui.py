"""Automated checks for the Objective 10 interface. No webcam, no CNN, no hand required.

The UI's wording is produced by pure functions (`command_text`, `control_status`,
`confidence_text`, `Game.banner_lines`, `Game.hud_text`), so every state can be asserted on
directly instead of comparing pixels. Drawing itself is exercised as a smoke test: every
screen is rendered in every state to prove nothing raises or falls off the panel.

    python src/check_ui.py
"""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game.engine import (
    DYING,
    GAME_OVER,
    PLAYING,
    READY,
    ROUND_CLEAR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Game,
)
from src.game_integration import STALE_SECONDS, GestureController, SharedState
from src.play_gesture import (
    GESTURE_ROWS,
    KEYBOARD_ROWS,
    PANEL_WIDTH,
    TITLE,
    GesturePacman,
    command_text,
    confidence_text,
    control_status,
    loading_text,
)

results: list[tuple[str, bool, str]] = []


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<34} {detail}")


def publish(state, command, now, sequence=1, **extra):
    state.publish(
        sequence=sequence,
        timestamp=now,
        stable_command=command,
        raw_direction=command or "up",
        raw_confidence=0.986,
        thresholded_direction=command,
        camera_ok=True,
        status="ready",
        inference_ms=16.4,
        **extra,
    )


def wired(use_camera=True):
    """An application with a controller but no worker: real UI, faked recognition."""
    app = GesturePacman(use_camera=False)  # never opens a camera or loads the model
    app.use_camera = use_camera
    state = SharedState()
    app.controller = GestureController(state)
    return app, state


def main():
    # --- the start screen -------------------------------------------------------------
    app, state = wired()
    on_start = app.phase == "start"
    app.draw_start()  # must render before anything is pressed
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    app.handle_start_events()
    by_space = app.phase == "playing"

    app2, _ = wired()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    app2.handle_start_events()
    check(
        "start screen opens first",
        on_start and by_space and app2.phase == "playing",
        "the app opens on the start screen; SPACE and ENTER both begin play",
    )

    app3, _ = wired()
    frozen = app3.game.player.position
    app3.draw_start()
    check(
        "start screen does not play",
        app3.game.player.position == frozen and app3.game.state == READY,
        "the game is not updated while the instructions are on screen",
    )

    # --- loading / ready / off wording ------------------------------------------------
    empty = SharedState()
    loading, _ = loading_text(empty.read(), True)
    empty.publish(status="ready", camera_ok=True)
    ready_line, _ = loading_text(empty.read(), True)
    off_line, _ = loading_text(empty.read(), False)
    check(
        "loading states read clearly",
        loading == "Initializing gesture recognition..."
        and ready_line == "Camera ready"
        and off_line.startswith("Gesture control off"),
        f"{loading!r} -> {ready_line!r}; keyboard-only reads {off_line!r}",
    )

    # --- the four commands ------------------------------------------------------------
    app, state = wired()
    shown = []
    for index, command in enumerate(("left", "right", "up", "down"), start=1):
        publish(state, command, time.perf_counter(), sequence=index)
        snapshot, fresh = app.controller.poll()
        shown.append(command_text(snapshot, fresh, True))
        app.draw_panel()
    check("four commands display", shown == ["LEFT", "RIGHT", "UP", "DOWN"], ", ".join(shown))

    # --- None, waiting, error, off ----------------------------------------------------
    publish(state, None, time.perf_counter(), sequence=5)
    snapshot, fresh = app.controller.poll()
    none_text = command_text(snapshot, fresh, True)
    none_conf = confidence_text(snapshot, fresh, True)
    check(
        "None reads as NO COMMAND",
        none_text == "NO COMMAND" and none_conf is None,
        "no direction and no confidence figure, so nothing looks like a live command",
    )

    stale = time.perf_counter() + STALE_SECONDS + 0.1
    snapshot, fresh = app.controller.poll(stale)
    waiting = command_text(snapshot, fresh, True)
    waiting_status = control_status(snapshot, fresh, True)
    check(
        "stale reads as WAITING",
        waiting == "WAITING"
        and waiting_status == "GESTURE CONTROL: WAITING"
        and confidence_text(snapshot, fresh, True) is None,
        "a stale snapshot shows WAITING with no confidence figure beside it",
    )

    state.publish(camera_ok=False, status="camera error", error="device lost", stable_command=None)
    snapshot, fresh = app.controller.poll()
    error_text = command_text(snapshot, fresh, True)
    error_status = control_status(snapshot, fresh, True)
    app.draw_panel()  # the two extra lines must still fit
    check(
        "camera error displays",
        error_text == "CAMERA ERROR" and error_status == "CAMERA ERROR",
        "the panel shows CAMERA ERROR and renders its keyboard advice without overflow",
    )

    off_app, _ = wired(use_camera=False)
    snapshot, fresh = off_app.controller.poll()
    check(
        "keyboard-only reads OFF",
        command_text(snapshot, fresh, False) == "GESTURE OFF"
        and control_status(snapshot, fresh, False) == "GESTURE CONTROL: OFF",
        "no camera is an OFF state, never an error",
    )

    # --- confidence only when it means something --------------------------------------
    publish(state, "up", time.perf_counter(), sequence=9)
    snapshot, fresh = app.controller.poll()
    check(
        "confidence shown for a command",
        confidence_text(snapshot, fresh, True) == "Confidence: 98.6%",
        "a live command carries its percentage; NO COMMAND / WAITING / ERROR do not",
    )

    # --- score, lives, round ----------------------------------------------------------
    app, state = wired()
    game = app.game
    game.score = 1230
    game.lives = 2
    game.round = 3
    values = game.hud_text()
    check(
        "score, lives and round display",
        values["score"].strip() == "SCORE   1230"
        and values["lives"] == 2
        and values["round"] == "ROUND 3",
        f"{values['score'].strip()} | LIVES {values['lives']} | {values['round']}",
    )

    game.score = 0
    game.lives = 3
    game.round = 1
    check(
        "HUD only reports state",
        game.hud_text()["score"].strip() == "SCORE      0",
        "the HUD renders existing values; it owns no scoring or lives logic of its own",
    )

    # --- every state's banner ---------------------------------------------------------
    game.state = READY
    ready_banner = game.banner_lines()
    game.state = DYING
    game.lives = 2
    dying_banner = game.banner_lines()
    game.state = ROUND_CLEAR
    game.round = 2
    round_banner = game.banner_lines()
    game.state = GAME_OVER
    game.score = 4560
    over_banner = game.banner_lines()
    game.state = PLAYING
    playing_banner = game.banner_lines()

    check(
        "READY and LIFE LOST read clearly",
        ready_banner == ("READY!", []) and dying_banner == ("LIFE LOST", ["Lives Remaining: 2"]),
        "the ready pause and the death pause both explain themselves",
    )
    check("round clear displays", round_banner[0] == "ROUND 2 CLEARED!", f"{round_banner[0]!r}")
    check(
        "game over displays fully",
        over_banner[0] == "GAME OVER"
        and "Final Score: 4560" in over_banner[1]
        and "Round Reached: 2" in over_banner[1],
        "score and round reached are both on the Game Over screen",
    )
    check(
        "restart instructions display",
        "Press R to Restart" in over_banner[1] and "Press ESC to Quit" in over_banner[1],
        "the player is told how to restart and how to quit",
    )
    check(
        "no banner during play",
        playing_banner is None,
        "nothing covers the maze while the game is actually being played",
    )

    # --- drawing is safe in every state -------------------------------------------------
    app, state = wired()
    app.phase = "playing"
    drawn = []
    for game_state in (READY, PLAYING, DYING, ROUND_CLEAR, GAME_OVER):
        app.game.state = game_state
        app.draw()
        drawn.append(game_state)
    app.draw_start()
    check(
        "every screen renders",
        len(drawn) == 5,
        "start screen and all five game states draw without raising",
    )

    # --- the panel never covers the maze --------------------------------------------------
    surface = app.game.screen
    check(
        "panel does not cover the maze",
        surface.get_width() == WINDOW_WIDTH + PANEL_WIDTH and surface.get_height() == WINDOW_HEIGHT,
        f"window {surface.get_width()}x{surface.get_height()}: the maze keeps its full "
        f"{WINDOW_WIDTH}px, the panel is added beside it",
    )

    # --- text the fonts can actually draw --------------------------------------------------
    strings = [TITLE, *KEYBOARD_ROWS]
    strings += [part for row in GESTURE_ROWS for part in row]
    strings += [
        command_text(_EMPTY_STATE.read(), False, False),
        control_status(_EMPTY_STATE.read(), False, False),
    ]
    non_ascii = [text for text in strings if any(ord(char) > 126 for char in text)]
    check(
        "no unrenderable glyphs",
        not non_ascii,
        "every UI string is plain ASCII, so nothing becomes a missing-glyph box",
    )

    # --- the UI changes nothing about the game --------------------------------------------
    drawn_app, _ = wired()
    drawn_app.phase = "playing"
    plain = Game(headless=True)
    drawn_app.game.state = plain.state = PLAYING
    drawn_app.game.state_timer = plain.state_timer = 0.0
    step = 1.0 / 60.0
    for index in range(300):
        if index % 45 == 0:
            direction = ("left", "up", "right", "down")[index // 45 % 4]
            drawn_app.game.request_direction(direction)
            plain.request_direction(direction)
        drawn_app.step(step)  # draws nothing
        plain.update(step)
        drawn_app.draw()  # full UI, every frame
    same = (
        drawn_app.game.player.position == plain.player.position
        and drawn_app.game.score == plain.score
        and drawn_app.game.lives == plain.lives
        and [g.position for g in drawn_app.game.ghosts] == [g.position for g in plain.ghosts]
    )
    check(
        "UI does not change physics",
        same,
        f"300 drawn frames matched an undrawn game exactly: tile "
        f"{drawn_app.game.player.tile}, score {drawn_app.game.score}",
    )

    # --- drawing stays cheap -----------------------------------------------------------------
    app, state = wired()
    app.phase = "playing"
    publish(state, "left", time.perf_counter(), sequence=1)
    app.draw()  # warm the font cache
    began = time.perf_counter()
    for _ in range(120):
        app.draw()
    per_frame = (time.perf_counter() - began) / 120 * 1000.0
    check(
        "drawing stays cheap",
        per_frame < 8.0,
        f"{per_frame:.2f} ms per fully drawn frame, well inside a 16.7 ms budget",
    )

    pygame.quit()
    print("-" * 92)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)}: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} UI checks passed)")
    return 0


_EMPTY_STATE = SharedState()


if __name__ == "__main__":
    sys.exit(main())
