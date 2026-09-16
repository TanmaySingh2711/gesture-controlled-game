"""Objective 11 - final end-to-end checks on the complete application.

This suite deliberately does not repeat P7's rule tests or P6's recognition tests. It checks
the contract *between* the subsystems: that the frozen checkpoint is the one the application
loads, that a recognition snapshot travels the one canonical path

    snapshot -> GestureController -> game.request_direction -> Pac-Man moves

and that every whole-application flow a player can reach - eating pellets, frightening and
eating a ghost, losing a life, Game Over, restart, clearing a round, the tunnel, the extra
life - still behaves when the game is being driven by gestures rather than by keys.

Recognition is faked by publishing snapshots, exactly as the real worker does, so the whole
suite is deterministic and needs no webcam, no hand and no CUDA.

    python src/check_final_application.py
"""

import hashlib
import os
import sys
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game.controls import REQUEST_GRACE
from game.engine import (
    DYING,
    EXTRA_LIFE_SCORE,
    GAME_OVER,
    GHOST_CHAIN,
    PELLET_SCORE,
    PLAYING,
    POWER_PELLET_SCORE,
    READY,
    ROUND_CLEAR,
    START_LIVES,
    Game,
)
from game.ghost import CHASE, EATEN, FRIGHTENED, SCATTER
from game.maze import TUNNEL_ROW
from src.game_integration import GestureController, RecognitionWorker, SharedState, Snapshot
from src.play_gesture import GesturePacman, command_text, control_status

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, "model", "best_direction_model.pt")

# Run in a fresh interpreter: a keyboard-only launch must never import the CNN stack.
KEYBOARD_PROBE = (
    "import sys; "
    "from src.play_gesture import GesturePacman; "
    "a = GesturePacman(use_camera=False); a.phase = 'playing'; "
    "[(a.step(1 / 60.0), a.draw()) for _ in range(120)]; "
    "print('LOADED', sorted(m for m in ('torch', 'torchvision', 'cv2') "
    "if m in sys.modules)); "
    "print('SCORE', a.game.score)"
)

results: list[tuple[str, bool, str]] = []


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<36} {detail}")


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest().upper()


def publish(state, command, now=None, sequence=1):
    state.publish(
        sequence=sequence,
        timestamp=now if now is not None else time.perf_counter(),
        stable_command=command,
        raw_direction=command or "up",
        raw_confidence=0.97,
        thresholded_direction=command,
        camera_ok=True,
        status="ready",
        inference_ms=16.0,
    )


def build():
    """A game already in play, with a controller fed by hand-published snapshots."""
    state = SharedState()
    controller = GestureController(state)
    game = Game(headless=True)
    game.state = PLAYING
    game.state_timer = 0.0
    return state, controller, game


def gesture_run(state, controller, game, command, frames, sequence=1, step=1.0 / 60.0):
    """Drive the game purely through the gesture path for N frames, as the real loop does."""
    now = time.perf_counter()
    for index in range(frames):
        now += step
        if index % 2 == 0:  # ~30 FPS recognizer against a 60 FPS game
            sequence += 1
            publish(state, command, now, sequence)
        controller.apply_to(game, now)
        game.update(step)
    return sequence


def main():
    before_hash = digest(CHECKPOINT)
    before_mtime = os.path.getmtime(CHECKPOINT)
    before_size = os.path.getsize(CHECKPOINT)

    # --- 1. the checkpoint the application actually loads -------------------------------
    import torch

    from src.data_pipeline import CLASS_TO_INDEX

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    mapping = payload.get("class_to_index")
    check(
        "final checkpoint identity",
        payload.get("task") == "pacman_direction_v1" and payload.get("num_classes") == 4,
        f"task {payload.get('task')!r}, {payload.get('num_classes')} classes, "
        f"{payload.get('architecture', 'mobilenet_v2')}",
    )
    check(
        "frozen class mapping",
        mapping == CLASS_TO_INDEX == {"left": 0, "right": 1, "up": 2, "down": 3},
        f"{mapping} - and no jump/neutral anywhere in it",
    )

    from src.train_model import FROZEN_CHECKPOINT_SHA256, file_sha256

    check(
        "checkpoint matches its pinned digest",
        file_sha256(CHECKPOINT) == FROZEN_CHECKPOINT_SHA256,
        f"sha256 {FROZEN_CHECKPOINT_SHA256[:16]}... - the loader refuses any other file",
    )

    from src.gesture_recognizer import (
        DEFAULT_MIN_AGREEMENT,
        DEFAULT_THRESHOLD,
        DEFAULT_WINDOW,
        ROI_X1,
        ROI_X2,
        ROI_Y1,
        ROI_Y2,
    )

    check(
        "frozen recognition settings",
        (DEFAULT_THRESHOLD, DEFAULT_WINDOW, DEFAULT_MIN_AGREEMENT) == (0.90, 5, 3)
        and (ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1) == (300, 300),
        f"threshold {DEFAULT_THRESHOLD}, {DEFAULT_MIN_AGREEMENT}-of-{DEFAULT_WINDOW}, "
        f"ROI {ROI_X2 - ROI_X1}x{ROI_Y2 - ROI_Y1} at x[{ROI_X1}:{ROI_X2}] y[{ROI_Y1}:{ROI_Y2}]",
    )

    # --- 2. startup state ----------------------------------------------------------------
    app = GesturePacman(use_camera=False)
    check(
        "application starts on the start screen",
        app.phase == "start" and app.game.state == READY and app.game.running,
        "a fresh launch shows instructions and has not begun play",
    )

    class IdleWorker(RecognitionWorker):
        """The real start/stop lifecycle, minus the camera and model, so no device is touched."""

        def run(self):
            self._stop_event.wait()

    state = SharedState()
    worker: RecognitionWorker = IdleWorker(state)
    controller = GestureController(state, worker)
    before_threads = threading.active_count()
    controller.start()
    controller.start()  # a second call must not start a second one
    started = threading.active_count() - before_threads
    released = controller.stop(timeout=2.0)
    check(
        "one worker, started once",
        started == 1 and released,
        f"two start() calls produced {started} worker thread - the only thread that ever "
        "opens the camera or loads the model",
    )

    # --- 3. the canonical control path ---------------------------------------------------
    state, controller, game = build()
    game.player.reset_to((22, 13), "left")
    start_tile = game.player.tile
    gesture_run(state, controller, game, "right", 60)
    check(
        "gesture drives the player",
        game.player.direction == "right" and game.player.tile != start_tile,
        f"a published snapshot moved Pac-Man from {start_tile} to {game.player.tile} "
        f"through request_direction alone",
    )

    # --- 4. None never stops the game ----------------------------------------------------
    # Down a clear corridor: running into a wall also stops Pac-Man, and that is correct
    # arcade behaviour, so the corridor has to be long enough for the wall not to be what
    # the check actually measures.
    state, controller, game = build()
    game.player.reset_to((22, 13), "left")
    requests_before = controller.applied_count
    moving = game.player.position
    gesture_run(state, controller, game, None, 60)
    check(
        "None does not stop Pac-Man",
        game.player.direction == "left"
        and game.player.position != moving
        and game.controls.pending is None
        and controller.applied_count == requests_before,
        "a second of no-command frames issued no request at all, left the "
        f"{REQUEST_GRACE:.2f}s buffer empty, and Pac-Man kept travelling left",
    )

    # --- 5. pellets and score -------------------------------------------------------------
    state, controller, game = build()
    game.player.reset_to((22, 13), "left")
    pellets_before = game.maze.pellets_remaining
    score_before = game.score
    gesture_run(state, controller, game, "left", 120)
    eaten = pellets_before - game.maze.pellets_remaining
    check(
        "pellets are eaten and scored",
        eaten > 0 and game.score == score_before + eaten * PELLET_SCORE,
        f"{eaten} pellets removed from the maze, score +{game.score - score_before}",
    )

    # --- 6. power pellet, frightened ghosts, eating one -----------------------------------
    state, controller, game = build()
    power = min(game.maze.power_pellets)
    game.player.reset_to(power, "left")
    score_before = game.score
    game.update(1.0 / 60.0)
    frightened = [g for g in game.ghosts if g.state == FRIGHTENED]
    check(
        "power pellet frightens the ghosts",
        power not in game.maze.power_pellets
        and game.score == score_before + POWER_PELLET_SCORE
        and game.frightened_timer > 0
        and len(frightened) >= 1,
        f"{len(frightened)} ghosts frightened for {game.frightened_timer:.1f}s, "
        f"score +{POWER_PELLET_SCORE}",
    )

    victim = frightened[0]
    victim.reset_to(game.player.tile)
    victim.state = FRIGHTENED
    score_before = game.score
    game.update(1.0 / 60.0)
    check(
        "a frightened ghost can be eaten",
        victim.state == EATEN and game.score == score_before + GHOST_CHAIN[0],
        f"contact scored {GHOST_CHAIN[0]} and left the ghost in the EATEN state, "
        "heading home, instead of costing a life",
    )

    # --- 7. life loss keeps the board ------------------------------------------------------
    state, controller, game = build()
    gesture_run(state, controller, game, "left", 90)
    score_kept = game.score
    pellets_kept = game.maze.pellets_remaining
    hunter = next(g for g in game.ghosts if g.state in (SCATTER, CHASE))
    hunter.reset_to(game.player.tile)
    game.update(1.0 / 60.0)
    lost_one = game.lives == START_LIVES - 1 and game.state == DYING
    for _ in range(120):  # let the death pause run out
        game.update(1.0 / 60.0)
    check(
        "life loss costs exactly one life",
        lost_one and game.state == READY,
        f"lives {START_LIVES} -> {game.lives}, then the round resumes at READY",
    )
    check(
        "the board survives a death",
        game.score == score_kept and game.maze.pellets_remaining == pellets_kept,
        f"score {game.score} and {game.maze.pellets_remaining} remaining pellets both "
        "carried through the death",
    )

    controller.reset()
    check(
        "respawn is not steered by a stale gesture",
        game.controls.pending is None and controller.last_applied is None,
        "the direction buffer and the integration's last command are both empty at respawn",
    )

    # --- 8. Game Over and restart ----------------------------------------------------------
    state, controller, game = build()
    game.score = 2340
    game.round = 2
    game.lives = 1
    hunter = next(g for g in game.ghosts if g.state in (SCATTER, CHASE))
    hunter.reset_to(game.player.tile)
    game.update(1.0 / 60.0)
    for _ in range(120):
        game.update(1.0 / 60.0)
    headline, details = game.banner_lines()
    check(
        "Game Over ends the game",
        game.state == GAME_OVER
        and headline == "GAME OVER"
        and f"Final Score: {game.score}" in details
        and f"Round Reached: {game.round}" in details
        and "Press R to Restart" in details
        and "Press ESC to Quit" in details,
        f"final score {game.score}, round {game.round}, and both the restart and quit "
        "instructions are on screen",
    )

    moved = game.player.position
    for _ in range(60):
        game.update(1.0 / 60.0)
    check(
        "Game Over stops play",
        game.player.position == moved,
        "nothing moves once the game is over; it waits for R",
    )

    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    controller.reset()
    fresh = (
        game.state == READY
        and game.score == 0
        and game.lives == START_LIVES
        and game.round == 1
        and game.controls.pending is None
    )
    game.state, game.state_timer = PLAYING, 0.0
    gesture_run(state, controller, game, "left", 60)
    check(
        "restart gives a clean game",
        fresh and game.maze.pellets_remaining > 300,
        f"score 0, {START_LIVES} lives, round 1, "
        f"{game.maze.pellets_remaining} pellets back on the board",
    )
    check(
        "gestures still steer after restart",
        game.player.direction == "left",
        "the restarted game is driven by the same recognizer, which was never reloaded",
    )

    # --- 9. round completion ----------------------------------------------------------------
    state, controller, game = build()
    game.score = 500
    game.lives = 2
    speeds_before = [g.base_speed for g in game.ghosts]
    last = (22, 12)
    game.maze.pellets = {last}  # one pellet left, eaten by the real code
    game.maze.power_pellets = set()
    game.player.reset_to((22, 13), "left")
    gesture_run(state, controller, game, "left", 40)
    cleared = game.state == ROUND_CLEAR and game.banner_lines()[0] == "ROUND 1 CLEARED!"
    for _ in range(150):  # run the round-clear pause out
        game.update(1.0 / 60.0)
    speeds_after = [g.base_speed for g in game.ghosts]
    check(
        "round completes and advances",
        cleared and game.round == 2 and game.state == READY and game.maze.pellets_remaining > 300,
        f"the last pellet cleared round 1; round {game.round} starts with "
        f"{game.maze.pellets_remaining} pellets restored",
    )
    check(
        "progress carries into the round",
        game.score >= 500 + PELLET_SCORE and game.lives == 2 and speeds_after > speeds_before,
        f"score {game.score} and {game.lives} lives carried over; ghost base speed rose "
        f"from {speeds_before[0]:.1f} to {speeds_after[0]:.1f} tiles/s",
    )

    # --- 10. the tunnel ----------------------------------------------------------------------
    state, controller, game = build()
    game.player.reset_to((TUNNEL_ROW, 1), "left")
    gesture_run(state, controller, game, "left", 60)
    wrapped = game.player.tile
    check(
        "the tunnel wraps under gesture control",
        wrapped[0] == TUNNEL_ROW and wrapped[1] > 20,
        f"Pac-Man left column 1 heading left and re-entered at column {wrapped[1]}, "
        "still gesture-controlled",
    )

    # --- 11. the extra life ------------------------------------------------------------------
    state, controller, game = build()
    game.score = EXTRA_LIFE_SCORE - 10
    game.add_score(10)
    after_first = game.lives
    game.add_score(EXTRA_LIFE_SCORE)
    check(
        "extra life is awarded once",
        after_first == START_LIVES + 1 and game.lives == after_first,
        f"crossing {EXTRA_LIFE_SCORE:,} gave one extra life; passing it again gave none",
    )

    # --- 12. keyboard fallback and coexistence ------------------------------------------------
    state, controller, game = build()
    game.player.reset_to((22, 13), "left")
    gesture_run(state, controller, game, "left", 20)
    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    keyboard_won = game.controls.pending == "right"
    publish(state, "up", time.perf_counter(), 99)
    controller.apply_to(game)
    gesture_won = game.controls.pending == "up"
    position = game.player.position
    for _ in range(30):
        game.update(1.0 / 60.0)
    check(
        "keyboard and gesture share one seam",
        keyboard_won and gesture_won and game.player.position != position,
        "each new request replaced the last; both arrive as a pending direction, and the "
        "player kept moving throughout",
    )

    # --- 13. keyboard-only mode ----------------------------------------------------------------
    off = GesturePacman(use_camera=False)
    off.phase = "playing"
    score_before = off.game.score
    for _ in range(120):
        off.step(1.0 / 60.0)
        off.draw()
    snapshot, fresh = Snapshot(status="off"), False
    check(
        "keyboard-only mode reads OFF",
        off.controller is None
        and command_text(snapshot, fresh, False) == "GESTURE OFF"
        and control_status(snapshot, fresh, False) == "GESTURE CONTROL: OFF"
        and off.game.state == PLAYING,
        "no controller and no webcam; the UI reads OFF rather than reporting an error",
    )

    # This suite loads torch and cv2 itself for the checkpoint check, so its own sys.modules
    # proves nothing. The claim is about a *fresh* keyboard-only launch, so ask one.
    import subprocess

    # Fixed argv: this interpreter plus a constant script - no untrusted input reaches it.
    probe = subprocess.run(  # noqa: S603
        [sys.executable, "-c", KEYBOARD_PROBE],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"},
    )
    loaded = [line for line in probe.stdout.splitlines() if line.startswith("LOADED")]
    played = [line for line in probe.stdout.splitlines() if line.startswith("SCORE")]
    check(
        "keyboard-only launch loads no CNN",
        probe.returncode == 0 and loaded == ["LOADED []"] and played,
        f"a fresh `--no-camera` process played 120 frames ({played[0].lower() if played else 'n/a'}) "
        "with torch, torchvision and cv2 never imported",
    )

    # --- 14. camera failure ----------------------------------------------------------------------
    state, controller, game = build()
    publish(state, "left", time.perf_counter(), 1)
    controller.apply_to(game)
    failed_worker = RecognitionWorker(state)
    failed_worker._fail("OSError: camera device 0 could not be opened")
    failed_worker._publish_exit()  # the real exit path
    applied = controller.apply_to(game)
    snapshot, fresh = controller.poll()
    game.request_direction("up")
    moving = game.player.position
    for _ in range(60):
        game.update(1.0 / 60.0)
    check(
        "camera failure degrades to keyboard",
        applied is None
        and command_text(snapshot, fresh, True) == "CAMERA ERROR"
        and control_status(snapshot, fresh, True) == "CAMERA ERROR"
        and game.player.position != moving,
        "gesture commands stop, the UI says CAMERA ERROR once, and the keyboard still plays",
    )

    # --- 15. clean shutdown ------------------------------------------------------------------------
    class LoopWorker(RecognitionWorker):
        """The real lifecycle with the camera and CNN removed, so shutdown is repeatable."""

        def run(self):
            try:
                self.state.publish(status="ready", camera_ok=True)
                while not self._stop_event.is_set():
                    self.frames += 1
                    self.state.publish(
                        sequence=self.frames,
                        timestamp=time.perf_counter(),
                        stable_command="up",
                        camera_ok=True,
                        status="ready",
                    )
                    time.sleep(0.002)
            finally:
                self._publish_exit()

    state = SharedState()
    worker = LoopWorker(state)
    controller = GestureController(state, worker)
    controller.start()
    time.sleep(0.08)
    published = worker.frames
    threads_before = threading.active_count()
    released = controller.stop(timeout=3.0)
    _, _, game = build()
    after = controller.apply_to(game)
    check(
        "shutdown is clean and complete",
        released
        and not worker.is_alive()
        and published > 0
        and after is None
        and threading.active_count() < threads_before,
        f"the worker published {published} snapshots, then stopped, joined and left no "
        "thread behind; no command survives it",
    )

    # --- 16. nothing about the model changed ----------------------------------------------------------
    check(
        "checkpoint untouched by testing",
        digest(CHECKPOINT) == before_hash
        and os.path.getmtime(CHECKPOINT) == before_mtime
        and os.path.getsize(CHECKPOINT) == before_size,
        f"{before_hash}, {before_size:,} bytes, timestamp unchanged across the whole suite",
    )

    pygame.quit()
    print("-" * 100)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)}: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} final-application checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
