"""Entry point for the keyboard-controlled Pac-Man-style game.

    python game/main.py              play
    python game/main.py --selftest   headless checks on the rules, no window
    python game/main.py --fps 3000   headless frame-rate capability probe

No webcam, no CNN, no torch. The gesture application in `src/play_gesture.py` drives exactly
the same `request_direction` seam the keyboard uses here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.controls import Controls


def _headless() -> None:
    """Run pygame without opening a window, so the rules can be tested automatically."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def play() -> int:
    from game.engine import Game

    Game().run()
    return 0


# ---------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------
def selftest() -> int:
    _headless()
    import pygame

    from game import controls as controls_module
    from game import engine as game_module
    from game import ghost as ghost_module
    from game.controls import DELTA, Controls
    from game.engine import (
        EXTRA_LIFE_SCORE,
        GAME_OVER,
        GHOST_CHAIN,
        MODE_SCHEDULE,
        PELLET_SCORE,
        POWER_PELLET_SCORE,
        READY,
        ROUND_CLEAR,
        START_LIVES,
        Game,
    )
    from game.ghost import CHASE, EATEN, FRIGHTENED, GHOST_MAX_SPEED, HOUSE, SCATTER
    from game.maze import COLUMNS, GHOST_SPAWN, HOUSE_EXIT, PLAYER_SPAWN, ROWS, TILE, TUNNEL_ROW
    from game.player import PLAYER_SPEED

    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        results.append((name, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name:<30} {detail}")

    game = Game(headless=True)
    maze = game.maze
    step = 1.0 / 60.0

    def play_frames(count: int, target: str | None = None) -> None:
        """Advance the simulation, optionally holding a direction request each frame."""
        for _ in range(count):
            if target:
                game.request_direction(target)
            game.update(step)

    def skip_ready() -> None:
        game.state = game_module.PLAYING
        game.state_timer = 0.0

    # --- maze ---------------------------------------------------------------------------
    check(
        "maze dimensions",
        maze.rows == ROWS
        and maze.columns == COLUMNS
        and all(len(r) == COLUMNS for r in maze.layout),
        f"{maze.columns} x {maze.rows}, every row {COLUMNS} tiles",
    )

    check(
        "player spawn walkable",
        maze.is_walkable(*PLAYER_SPAWN),
        f"spawn {PLAYER_SPAWN} is a walkable tile",
    )

    house_ok = all(maze.is_walkable(*pos, doors=True, house=True) for pos in GHOST_SPAWN.values())
    check(
        "ghost house valid",
        house_ok and maze.is_walkable(*HOUSE_EXIT),
        f"4 spawn tiles reachable by ghosts, exit {HOUSE_EXIT} walkable",
    )

    check(
        "player cannot enter house",
        not maze.is_walkable(*GHOST_SPAWN["ambusher"]),
        "house interior and door are closed to Pac-Man",
    )

    check("pellets exist", len(maze.pellets) > 200, f"{len(maze.pellets)} normal pellets")
    check(
        "exactly four power pellets", len(maze.power_pellets) == 4, f"{sorted(maze.power_pellets)}"
    )

    check(
        "tunnel endpoints",
        maze.is_walkable(TUNNEL_ROW, 0) and maze.is_walkable(TUNNEL_ROW, COLUMNS - 1),
        f"row {TUNNEL_ROW} open at both maze edges",
    )

    unreachable = maze.unreachable_pellets()
    check(
        "every pellet reachable",
        not unreachable,
        f"BFS from spawn reaches all {maze.pellets_remaining} pellets"
        if not unreachable
        else f"{len(unreachable)} unreachable: {unreachable[:6]}",
    )

    check("no dead ends", not maze.dead_ends(), f"{len(maze.dead_ends())} tiles with a single exit")

    # --- player movement ----------------------------------------------------------------
    game.new_game()
    skip_ready()
    start = game.player.position
    play_frames(30, "left")
    check(
        "continuous movement",
        game.player.position != start,
        f"moved from {tuple(round(v) for v in start)} to "
        f"{tuple(round(v) for v in game.player.position)}",
    )

    game.new_game()
    skip_ready()
    game.request_direction("right")
    play_frames(20)
    check(
        "direction request applies",
        game.player.direction == "right",
        "a legal request is obeyed at the next tile centre",
    )

    # Walls: drive into one and confirm the player stops rather than passing through.
    game.new_game()
    skip_ready()
    play_frames(240, "up")  # spawn row has walls above it
    tile = game.player.tile
    check(
        "cannot pass through walls",
        maze.is_walkable(*tile),
        f"after 4s of holding UP the player is on walkable tile {tile}",
    )

    # Buffered turn: ask for a direction that is illegal now and legal a few tiles later.
    game.new_game()
    skip_ready()
    game.player.reset_to((22, 8), "right")
    game.controls.clear()
    game.request_direction("up")  # blocked at (22,7)
    buffered_now = not game.player.can_move("up") and game.controls.pending == "up"
    for _ in range(60):
        game.update(step)
        if game.player.direction == "up":
            break
    check(
        "illegal request is buffered",
        buffered_now,
        "UP was impossible at the moment it was requested and stayed in the buffer",
    )
    check(
        "buffered turn executes",
        game.player.direction == "up",
        f"the buffered UP fired at the first legal tile, now at {game.player.tile}",
    )

    game.controls.clear()
    game.request_direction("down")
    game.update(step)
    check(
        "buffer expires",
        game.controls.grace == controls_module.REQUEST_GRACE and _expires(Controls()),
        f"a request is dropped after {controls_module.REQUEST_GRACE:.2f}s",
    )

    # Reversal without an intersection.
    game.new_game()
    skip_ready()
    game.player.reset_to((22, 10), "right")
    play_frames(4)
    game.request_direction("left")
    game.update(step)
    check(
        "reversal is immediate",
        game.player.direction == "left",
        "turning back happened mid-corridor, with no wait for a tile centre",
    )

    # No diagonal movement: only one axis may change in a frame.
    game.new_game()
    skip_ready()
    diagonal = False
    previous = game.player.position
    for index in range(300):
        game.request_direction(("left", "up", "right", "down")[index % 4])
        game.update(step)
        now = game.player.position
        if abs(now[0] - previous[0]) > 1e-6 and abs(now[1] - previous[1]) > 1e-6:
            diagonal = True
        previous = now
    check(
        "no diagonal movement",
        not diagonal,
        "300 frames of rotating requests never moved both axes at once",
    )

    # Tunnel wrap.
    game.new_game()
    skip_ready()
    game.player.reset_to((TUNNEL_ROW, 1), "left")
    for _ in range(120):
        game.request_direction("left")
        game.update(step)
        if game.player.tile[1] > COLUMNS - 4:
            break
    check(
        "tunnel wrap",
        game.player.tile[1] > COLUMNS - 4,
        f"walking off the left edge arrived at column {game.player.tile[1]}",
    )

    # --- pellets and scoring --------------------------------------------------------------
    game.new_game()
    skip_ready()
    target_tile = (22, 12)
    maze.pellets.add(target_tile)
    game.player.reset_to((22, 13), "left")
    game.score = 0
    play_frames(40, "left")
    check(
        "pellet collected",
        target_tile not in maze.pellets,
        "the pellet under the player disappeared",
    )
    check(
        "pellet scores 10",
        game.score >= PELLET_SCORE and game.score % 10 == 0,
        f"score {game.score} after eating {game.score // PELLET_SCORE} pellet(s)",
    )

    game.new_game()
    skip_ready()
    game.score = 0
    game.player.reset_to((22, 2), "left")
    play_frames(40, "left")  # (22,1) holds a power pellet
    check(
        "power pellet scores 50",
        game.score >= POWER_PELLET_SCORE,
        f"score {game.score}, frightened timer {game.frightened_timer:.1f}s started",
    )

    frightened_count = sum(1 for g in game.ghosts if g.state == FRIGHTENED)
    check(
        "power pellet frightens",
        frightened_count >= 1 and game.frightened_timer > 0,
        f"{frightened_count} ghost(s) became frightened",
    )

    # Ghost chain 200/400/800/1600.
    game.new_game()
    skip_ready()
    game.score = 0
    game.frightened_timer = 10.0
    scores = []
    for ghost in game.ghosts:
        ghost.state = FRIGHTENED
        ghost.x, ghost.y = game.player.x, game.player.y
        game._handle_collisions()
        scores.append(game.score)
    gains = [scores[0]] + [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    check(
        "ghost chain scoring",
        tuple(gains) == GHOST_CHAIN,
        f"{gains} points for successive ghosts in one frightened period",
    )

    check(
        "eating a ghost is safe",
        game.lives == START_LIVES and all(g.state == EATEN for g in game.ghosts),
        "no life lost, all four ghosts are now EATEN",
    )

    # --- round completion -------------------------------------------------------------------
    game.new_game()
    skip_ready()
    game.score = 500
    game.maze.pellets.clear()
    game.maze.power_pellets.clear()
    game.update(step)
    check("round completes", game.state == ROUND_CLEAR, "clearing every pellet ended the round")

    lives_before, score_before, round_before = game.lives, game.score, game.round
    game.state_timer = 0.0
    game.update(step)
    check(
        "next round starts",
        game.round == round_before + 1 and game.maze.pellets_remaining > 300,
        f"round {game.round}, {game.maze.pellets_remaining} pellets refilled",
    )
    check(
        "round preserves progress",
        game.score == score_before and game.lives == lives_before,
        f"score {game.score} and {game.lives} lives carried over",
    )

    speeds = {g.role: g.base_speed for g in game.ghosts}
    check(
        "difficulty increases",
        all(v > ghost_module.GHOST_BASE_SPEED for v in speeds.values()),
        f"ghost speed now {max(speeds.values()):.2f} tiles/s (was {ghost_module.GHOST_BASE_SPEED})",
    )

    game.round = 40
    for ghost in game.ghosts:
        ghost.set_round(game.round)
    capped = max(g.base_speed for g in game.ghosts)
    check(
        "difficulty capped",
        capped <= GHOST_MAX_SPEED < PLAYER_SPEED / TILE,
        f"ghosts cap at {capped:.2f} tiles/s, below the player's {PLAYER_SPEED / TILE:.2f}",
    )

    frightened_late = game.frightened_duration()
    check(
        "frightened duration floor",
        abs(frightened_late - 2.0) < 1e-6,
        f"round {game.round} frightened lasts {frightened_late:.1f}s (floor 2.0s)",
    )

    game.round = 1
    check(
        "frightened duration base",
        abs(game.frightened_duration() - 6.0) < 1e-6,
        "round 1 frightened lasts 6.0s, shrinking 0.5s per round",
    )

    # --- lives ----------------------------------------------------------------------------
    game.new_game()
    skip_ready()
    game.maze.pellets.discard((22, 12))
    eaten_marker = (22, 12) not in game.maze.pellets
    chaser = game.ghosts[0]
    chaser.state = CHASE
    chaser.x, chaser.y = game.player.x, game.player.y
    game._handle_collisions()
    check(
        "ghost collision costs a life",
        game.lives == START_LIVES - 1,
        f"{game.lives} lives left after touching a CHASE ghost",
    )

    game.state_timer = 0.0
    game.update(step)
    check(
        "pellets stay eaten after death",
        eaten_marker and (22, 12) not in game.maze.pellets,
        "an eaten pellet did not come back when the life reset",
    )

    game.lives = 1
    game.state = game_module.PLAYING
    chaser.state = CHASE
    chaser.x, chaser.y = game.player.x, game.player.y
    game._handle_collisions()
    game.state_timer = 0.0
    game.update(step)
    check(
        "final life ends the game",
        game.state == GAME_OVER,
        f"lives {game.lives}, state {game.state}",
    )

    game.new_game()
    fresh = (
        game.score == 0
        and game.lives == START_LIVES
        and game.round == 1
        and game.maze.pellets_remaining > 300
        and game.state == READY
        and game.controls.pending is None
        and not game.extra_life_awarded
        and all(g.state in (HOUSE, SCATTER) for g in game.ghosts)
    )
    check(
        "restart resets everything",
        fresh,
        "score, lives, round, pellets, ghosts, timers and the request buffer all cleared",
    )

    # --- extra life -------------------------------------------------------------------------
    game.new_game()
    game.score = 0
    game.lives = 3
    game.add_score(EXTRA_LIFE_SCORE)
    once = game.lives == 4
    game.add_score(EXTRA_LIFE_SCORE)
    game.add_score(EXTRA_LIFE_SCORE)
    check(
        "extra life at 10,000",
        once and game.lives == 4,
        f"awarded once at {EXTRA_LIFE_SCORE}; score {game.score} still gives {game.lives} lives",
    )

    # --- ghosts ------------------------------------------------------------------------------
    game.new_game()
    skip_ready()
    roles = [g.role for g in game.ghosts]
    check(
        "four ghosts with roles",
        len(game.ghosts) == 4 and roles == ["chaser", "ambusher", "flanker", "drifter"],
        ", ".join(roles),
    )

    chaser, _ambusher, _flanker, drifter = game.ghosts
    game.player.reset_to((22, 13), "left")
    for ghost in game.ghosts:
        ghost.state = CHASE
    targets = {g.role: g.chase_target(game.player, chaser) for g in game.ghosts}
    check(
        "Chaser targets the player",
        targets["chaser"] == game.player.tile,
        f"target {targets['chaser']} equals the player tile",
    )
    expected_ambush = (game.player.tile[0], game.player.tile[1] - 4)
    check(
        "Ambusher aims ahead",
        targets["ambusher"] == expected_ambush,
        f"target {targets['ambusher']}, four tiles ahead of a player facing left",
    )
    check(
        "Flanker differs from Chaser",
        targets["flanker"] != targets["chaser"],
        f"target {targets['flanker']} is the Chaser reflected through a point ahead",
    )
    drifter.x, drifter.y = maze.tile_center(1, 1)
    far_target = drifter.chase_target(game.player, chaser)
    drifter.x, drifter.y = game.player.x, game.player.y
    near_target = drifter.chase_target(game.player, chaser)
    check(
        "Drifter changes with distance",
        far_target == game.player.tile and near_target == drifter.scatter_target,
        f"far -> {far_target}, close -> {near_target} (its own corner)",
    )

    unique_targets = len(set(targets.values()))
    check(
        "roles produce distinct targets",
        unique_targets >= 3,
        f"{unique_targets} distinct targets among the four ghosts",
    )

    # Long simulation: no ghost may ever stand in a wall, and none may freeze.
    game.new_game()
    skip_ready()
    intruders, frozen = [], []
    moved = {g.role: 0 for g in game.ghosts}
    previous_positions = {g.role: g.position for g in game.ghosts}
    for index in range(60 * 45):
        game.request_direction(("left", "up", "right", "down")[(index // 40) % 4])
        game.update(step)
        if game.state in (game_module.DYING, ROUND_CLEAR):
            game.state_timer = 0.0
        for ghost in game.ghosts:
            row, column = ghost.tile
            if not maze.is_walkable(row, column, doors=True, house=True):
                intruders.append((ghost.role, (row, column)))
            if ghost.position != previous_positions[ghost.role]:
                moved[ghost.role] += 1
            previous_positions[ghost.role] = ghost.position
    frozen = [role for role, count in moved.items() if count < 60]
    check(
        "ghosts never enter walls",
        not intruders,
        f"45 simulated seconds, 0 wall intrusions across {len(game.ghosts)} ghosts"
        if not intruders
        else f"{len(intruders)} intrusions: {intruders[:4]}",
    )
    check(
        "no ghost gets stuck",
        not frozen,
        "every ghost kept moving: " + ", ".join(f"{r}:{c}" for r, c in moved.items()),
    )

    # Timed house exits.
    game.new_game()
    skip_ready()
    in_house_at_start = [g.role for g in game.ghosts if g.in_house]
    left_times = {}
    for index in range(60 * 20):
        game.update(step)
        for ghost in game.ghosts:
            if ghost.role not in left_times and not ghost.in_house:
                left_times[ghost.role] = index / 60.0
    staggered = len({round(t, 1) for t in left_times.values()}) >= 3
    check(
        "ghosts leave the house in turn",
        len(left_times) == 4 and staggered,
        "exit times "
        + ", ".join(f"{r} {t:.1f}s" for r, t in sorted(left_times.items(), key=lambda kv: kv[1])),
    )
    check(
        "house starts occupied",
        len(in_house_at_start) == 3,
        f"{in_house_at_start} start inside, chaser starts outside",
    )

    # Scatter/chase schedule and the reversal on a mode switch.
    game.new_game()
    skip_ready()
    check(
        "schedule starts in scatter",
        game.mode == SCATTER,
        f"{len(MODE_SCHEDULE)} phases: "
        + ", ".join(f"{m[:2]}{'' if d is None else int(d)}" for m, d in MODE_SCHEDULE),
    )

    outside = [g for g in game.ghosts if not g.in_house]
    before_dirs = {g.role: g.direction for g in outside}
    game.mode_timer = 0.0
    game.update(step)
    reversed_any = any(g.direction != before_dirs[g.role] for g in outside)
    check(
        "scatter -> chase switch",
        game.mode == CHASE,
        f"mode advanced to {game.mode} when the phase timer expired",
    )
    check(
        "ghosts reverse on mode change",
        reversed_any,
        "at least one ghost turned around when the global mode changed",
    )

    # Frightened -> eaten -> back to the house.
    game.new_game()
    skip_ready()
    ghost = game.ghosts[0]
    ghost.state = CHASE
    ghost.reset_to((11, 13), "left")
    game._start_frightened()
    check(
        "frightened mode entered",
        ghost.state == FRIGHTENED and ghost.current_speed() < ghost.base_speed * TILE,
        f"speed drops to {ghost.current_speed() / TILE:.1f} tiles/s while edible",
    )

    ghost.get_eaten()
    check(
        "eaten ghost speeds up",
        ghost.state == EATEN and ghost.current_speed() > ghost.base_speed * TILE,
        f"eyes travel at {ghost.current_speed() / TILE:.1f} tiles/s",
    )

    returned = False
    for _ in range(60 * 12):
        game.update(step)
        if ghost.state in (HOUSE, SCATTER, CHASE):
            returned = True
            break
    check(
        "eaten ghost returns home",
        returned,
        f"the eyes reached the house and the ghost became {ghost.state}",
    )

    game.new_game()
    skip_ready()
    harmless = game.ghosts[1]
    harmless.state = EATEN
    harmless.x, harmless.y = game.player.x, game.player.y
    lives_before = game.lives
    game._handle_collisions()
    check(
        "eaten ghosts are harmless",
        game.lives == lives_before,
        "walking through a pair of eyes costs nothing",
    )

    # --- the control seam ----------------------------------------------------------------------
    game.new_game()
    skip_ready()
    game.player.reset_to((22, 13), "left")
    game.request_direction("right")
    play_frames(20)
    programmatic = game.player.direction

    game.new_game()
    skip_ready()
    game.player.reset_to((22, 13), "left")
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    game.handle_event(event)
    keyboard_request = game.controls.pending
    play_frames(20)
    keyboard = game.player.direction
    check(
        "keyboard uses the generic seam",
        keyboard_request == "right",
        "a key press only produced a direction request, it did not move anything",
    )
    check(
        "programmatic control is identical",
        programmatic == keyboard == "right",
        f"request_direction and the arrow key both produced direction {keyboard!r}",
    )

    seam_ok = True
    for direction in DELTA:
        fresh_controls = Controls()
        fresh_controls.request_direction(direction)
        if fresh_controls.pending != direction:
            seam_ok = False
    check(
        "all four directions requestable",
        seam_ok,
        "left, right, up and down all pass through Controls unchanged",
    )

    # --- separation from the CNN -----------------------------------------------------------------
    # Substring scanning is the wrong instrument here: a docstring that says "this does not
    # import torch" would trip it. What matters is what the modules actually import.
    import ast

    banned_modules = {
        "src",  # the whole recognition side: game/ must never import from it
        "torch",
        "torchvision",
        "cv2",
        "numpy.distutils",
        "gesture_recognizer",
        "realtime_gesture",
        "data_pipeline",
        "train_model",
        "evaluate_model",
    }
    package = os.path.dirname(os.path.abspath(__file__))
    offenders: list[str] = []
    scanned: list[str] = []
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py"):
            continue
        scanned.append(name)
        with open(os.path.join(package, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                found = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders.extend(
                f"{name} imports {module}" for module in found if module in banned_modules
            )
    check(
        "no CNN or webcam imports",
        not offenders,
        f"{len(scanned)} modules ({', '.join(scanned)}) import none of {sorted(banned_modules)}"
        if not offenders
        else f"{offenders}",
    )

    # Stronger still: nothing the game pulls in may drag the CNN stack into the process.
    loaded = {m for m in ("torch", "torchvision", "cv2") if m in sys.modules}
    check(
        "CNN stack never loaded",
        not loaded,
        "torch, torchvision and cv2 are absent from sys.modules after running the "
        "whole game and its tests"
        if not loaded
        else f"loaded: {sorted(loaded)}",
    )

    pygame.quit()
    print("-" * 86)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)}: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} checks passed)")
    return 0


def _expires(controls: Controls) -> bool:
    """A buffered request must disappear once the grace period has elapsed."""
    controls.request_direction("up")
    controls.tick(controls.grace * 0.5)
    still_there = controls.pending == "up"
    controls.tick(controls.grace)
    return still_there and controls.pending is None


def fps_probe(frames: int) -> int:
    """How fast the loop runs uncapped - headroom against the 60 FPS target."""
    _headless()
    import pygame

    from game.engine import PLAYING, Game

    game = Game(headless=True)
    game.state = PLAYING
    step = 1.0 / 60.0
    started = time.perf_counter()
    for _ in range(frames):
        pygame.event.pump()
        game.update(step)
        game.draw()
        if game.state == "game_over":
            game.new_game()
            game.state = PLAYING
    elapsed = time.perf_counter() - started
    pygame.quit()
    print(
        f"frames {frames} | wall {elapsed:.2f}s | uncapped {frames / elapsed:.0f} FPS "
        f"| {elapsed / frames * 1000:.2f} ms per frame (target 16.67 ms at 60 FPS)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Keyboard Pac-Man-style maze game.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--fps", type=int, metavar="FRAMES", help="headless uncapped frame-rate probe"
    )
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.fps:
        return fps_probe(args.fps)
    return play()


if __name__ == "__main__":
    sys.exit(main())
