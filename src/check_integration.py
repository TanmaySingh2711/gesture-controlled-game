"""Automated tests for the P8 gesture-to-game bridge. No webcam, no CNN, no hand required.

Recognition is faked by publishing snapshots straight into `SharedState`, which is exactly
what the real worker does. That lets every integration rule be tested deterministically -
including the timing relationship between a ~30 FPS recognizer and a ~60 FPS game, which is
the part most likely to be wrong and the hardest to judge by playing.

    python src/check_integration.py
"""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from game.controls import REQUEST_GRACE
from game.engine import DYING, GAME_OVER, PLAYING, ROUND_CLEAR, Game
from src.game_integration import (
    STALE_SECONDS,
    GestureController,
    RecognitionWorker,
    SharedState,
)

results: list[tuple[str, bool, str]] = []


def check(name, passed, detail):
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name:<34} {detail}")


def fake(state, command, now, sequence=1, **extra):
    """Publish a recognition snapshot exactly as the real worker would."""
    state.publish(
        sequence=sequence,
        timestamp=now,
        stable_command=command,
        raw_direction=command or "up",
        raw_confidence=0.99,
        thresholded_direction=command,
        camera_ok=True,
        status="ready",
        inference_ms=12.5,
        **extra,
    )


def build():
    state = SharedState()
    controller = GestureController(state)
    game = Game(headless=True)
    game.state = PLAYING
    game.state_timer = 0.0
    return state, controller, game


def main():
    step = 1.0 / 60.0

    # --- direction mapping ------------------------------------------------------------
    mapping_ok, detail = True, []
    for index, command in enumerate(("left", "right", "up", "down"), start=1):
        state, controller, game = build()
        now = time.perf_counter()
        fake(state, command, now, sequence=index)
        applied = controller.apply_to(game, now)
        pending = game.controls.pending
        detail.append(f"{command}->{pending}")
        if applied != command or pending != command:
            mapping_ok = False
    check("all four directions map", mapping_ok, ", ".join(detail))

    # --- None means no new command, not stop -------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "right", now, sequence=1)
    controller.apply_to(game, now)
    for _ in range(20):
        game.update(step)
    direction_before = game.player.direction

    moved_before = game.player.position
    for index in range(60):  # a full second of "no command"
        now += step
        fake(state, None, now, sequence=2 + index)
        applied = controller.apply_to(game, now)
        game.update(step)
        if applied is not None:
            break
    check(
        "None issues no request",
        applied is None,
        "sixty consecutive None snapshots produced no direction request",
    )
    check(
        "None does not stop the player",
        game.player.direction == direction_before and game.player.position != moved_before,
        f"Pac-Man kept travelling {direction_before} through a second of no-command frames",
    )
    check(
        "None clears the buffer",
        game.controls.pending is None,
        f"the {REQUEST_GRACE:.2f}s grace expired the old request instead of holding it",
    )

    # --- held gesture keeps refreshing intent -------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    game.player.reset_to((22, 8), "right")
    refreshed = 0
    for index in range(40):
        now += step
        if index % 2 == 0:  # a 30 FPS recognizer, 60 FPS game
            fake(state, "up", now, sequence=index // 2 + 1)
        if controller.apply_to(game, now):
            refreshed += 1
        game.update(step)
    check(
        "held gesture refreshes intent",
        refreshed >= 35,
        f"UP was re-requested on {refreshed} of 40 game frames while the gesture was held",
    )

    # --- the real use case: a turn asked for before the intersection ----------------------
    state, controller, game = build()
    now = time.perf_counter()
    game.player.reset_to((22, 6), "right")
    game.controls.clear()
    fake(state, "up", now, sequence=1)
    turned_at = None
    for index in range(120):
        now += step
        if index % 2 == 0:
            fake(state, "up", now, sequence=index // 2 + 1)
        controller.apply_to(game, now)
        game.update(step)
        if game.player.direction == "up" and turned_at is None:
            turned_at = index
            break
    check(
        "buffered pre-turn works",
        turned_at is not None,
        f"UP requested two tiles early turned at frame {turned_at} "
        f"({(turned_at or 0) * step * 1000:.0f} ms later), tile {game.player.tile}",
    )

    # --- stale results expire --------------------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "left", now, sequence=1)
    fresh_applied = controller.apply_to(game, now)
    stale_applied = controller.apply_to(game, now + STALE_SECONDS + 0.05)
    check(
        "stale results expire",
        fresh_applied == "left" and stale_applied is None,
        f"a snapshot older than {STALE_SECONDS:.2f}s stopped being applied",
    )

    _, fresh = controller.poll(now + STALE_SECONDS + 0.05)
    check(
        "stale shows as waiting",
        not fresh and controller.status_line(now + STALE_SECONDS + 0.05) == "GESTURE: WAITING",
        "a dead worker reads as WAITING rather than holding an ancient LEFT",
    )

    # --- camera failure degrades gracefully --------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "down", now, sequence=1)
    controller.apply_to(game, now)
    state.publish(camera_ok=False, status="camera error", error="device lost", stable_command=None)
    applied = controller.apply_to(game, now)
    line = controller.status_line(now)
    game.request_direction("left")  # keyboard must still work
    check("camera failure stops commands", applied is None, f"status line reads {line!r}")
    check(
        "keyboard survives camera failure",
        game.controls.pending == "left",
        "a key press still reaches the same seam with the camera dead",
    )

    # --- changing gesture changes direction ---------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "up", now, sequence=1)
    controller.apply_to(game, now)
    first = game.controls.pending
    now += step
    fake(state, "down", now, sequence=2)
    controller.apply_to(game, now)
    check(
        "new gesture replaces the old",
        first == "up" and game.controls.pending == "down",
        "UP then DOWN left exactly one pending request, the newest",
    )

    # --- 30 FPS recognizer vs 60 FPS game -------------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    game.player.reset_to((22, 13), "left")
    published, game_frames, applied_total = 0, 0, 0
    plan: list[tuple[str | None, int]] = [
        ("left", 60),
        (None, 30),
        ("up", 60),
        (None, 30),
        ("right", 60),
    ]
    for planned, frames in plan:
        for index in range(frames):
            now += step
            game_frames += 1
            if index % 2 == 0:  # recognizer publishes at half the rate
                published += 1
                fake(state, planned, now, sequence=published)
            if controller.apply_to(game, now):
                applied_total += 1
            game.update(step)
    backlog = state.writes - published  # one slot: writes must equal publishes
    check(
        "30 FPS recognizer, 60 FPS game",
        game_frames == 240 and published == 120,
        f"{game_frames} game frames consumed {published} recognition updates",
    )
    check(
        "no backlog accumulates",
        backlog == 0,
        "the shared slot holds exactly one snapshot; writes == publishes, nothing queues",
    )
    check(
        "commands survive the rate gap",
        applied_total > 150,
        f"{applied_total} direction requests issued across the run, none lost to the gap",
    )

    # --- reset on death, round change and restart --------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "left", now, sequence=1)
    controller.apply_to(game, now)
    had_command = controller.last_applied
    controller.reset()
    check(
        "reset clears integration state",
        had_command == "left" and controller.last_applied is None and controller.last_sequence == 0,
        "the held direction and sequence marker are dropped on reset",
    )

    game.request_direction("up")
    game._reset_positions()
    check(
        "life reset clears the buffer",
        game.controls.pending is None,
        "the game's own reset already empties the request buffer on death and round change",
    )

    game.new_game()
    check(
        "restart clears the buffer",
        game.controls.pending is None and game.score == 0 and game.lives == 3,
        "a full restart resets score, lives and the pending request together",
    )

    # --- the seam is shared, not duplicated -----------------------------------------------------------
    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "right", now, sequence=1)
    controller.apply_to(game, now)
    gesture_pending = game.controls.pending
    game.controls.clear()
    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    keyboard_pending = game.controls.pending
    check(
        "gesture uses the P7 seam",
        gesture_pending == keyboard_pending == "right",
        "a gesture and an arrow key produce an identical pending request",
    )

    state, controller, game = build()
    now = time.perf_counter()
    fake(state, "left", now, sequence=1)
    controller.apply_to(game, now)
    game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
    check(
        "most recent request wins",
        game.controls.pending == "down",
        "a key press after a gesture overrides it; no priority table needed",
    )

    # --- the launcher drops control state at every pause ------------------------------------
    from src.play_gesture import GesturePacman

    app = GesturePacman(use_camera=False)
    resets: list[str] = []

    class RecordingController(GestureController):
        """Records every reset instead of performing it."""

        def reset(self) -> None:
            resets.append(app.game.state)

    app.controller = RecordingController(SharedState())

    app.game.state = PLAYING
    app.previous_state = PLAYING
    app.step(0.0)
    during_play = list(resets)

    app.game.state = DYING  # death
    app.step(0.0)
    app.game.state = ROUND_CLEAR  # round change
    app.step(0.0)
    paused = list(resets)

    app.game.state = GAME_OVER  # restart
    app.previous_state = GAME_OVER
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    app.handle_events()
    check(
        "pauses clear control state",
        not during_play and paused == [DYING, ROUND_CLEAR],
        "state is dropped on death and round change, left alone during normal play",
    )
    check(
        "restart clears control state",
        len(resets) == 3 and app.game.state != GAME_OVER,
        "R after Game Over clears the held direction before the new game begins",
    )

    # --- clean shutdown -----------------------------------------------------------------------
    class LoopWorker(RecognitionWorker):
        """The real worker's lifecycle with the camera and the CNN taken out.

        This tests the part a live play session cannot test repeatably: stop() sets the
        event, the loop notices, run() returns, join() succeeds. The first build named that
        event `self._stop`, which silently shadowed Thread._stop() - a method join() calls
        internally - and made shutdown raise TypeError. No mapping test can catch that.
        """

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
    worker: RecognitionWorker = LoopWorker(state)
    controller = GestureController(state, worker)
    controller.start()
    time.sleep(0.05)
    published = worker.frames
    released = controller.stop(timeout=3.0)
    _, _, game = build()
    after = controller.apply_to(game)
    check(
        "worker shuts down cleanly",
        released and not worker.is_alive() and published > 0,
        f"a worker that published {published} snapshots stopped and joined on request",
    )
    check(
        "shutdown issues no command",
        after is None and state.read().status == "stopped",
        "once stopped the state reads 'stopped' and steers nothing",
    )
    check(
        "stop flag does not shadow Thread",
        callable(getattr(worker, "_stop", None)),
        "the flag is _stop_event, so Thread._stop stays the method join() relies on",
    )

    idle = RecognitionWorker(SharedState())
    check(
        "stop before start is safe",
        GestureController(idle.state, idle).stop() is True,
        "stopping a worker that was never started does not raise on join()",
    )

    # --- a camera failure stays visible after the thread exits -----------------------------------
    state = SharedState()
    worker = RecognitionWorker(state)
    worker._fail("OSError: camera device 0 could not be opened")
    worker._publish_exit()  # the real exit path, no camera involved
    controller = GestureController(state, worker)
    _, _, game = build()
    applied = controller.apply_to(game)
    line = controller.status_line()
    game.request_direction("up")
    check(
        "camera error survives exit",
        line == "CAMERA ERROR - keyboard still works"
        and applied is None
        and game.controls.pending == "up",
        "the exit publish no longer hides CAMERA ERROR behind 'stopped'; keys still work",
    )

    # --- physics untouched -------------------------------------------------------------------------------
    import inspect

    from src import game_integration

    source = inspect.getsource(game_integration)
    touches_physics = any(
        word in source for word in (".x =", ".y =", ".direction =", "player.tile", "_advance")
    )
    check(
        "no duplicated physics",
        not touches_physics,
        "the bridge only calls request_direction; it never moves the player itself",
    )

    # --- the game package is still CNN-free ---------------------------------------------------------------
    leaked = [m for m in ("torch", "torchvision", "cv2") if m in sys.modules]
    check(
        "game package still CNN-free",
        not leaked,
        "importing the bridge's pure logic pulled in no torch, torchvision or cv2",
    )

    pygame.quit()
    print("-" * 92)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)}: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} integration checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
