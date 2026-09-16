"""Memory footprint and leak check for the two long-running parts: the game loop and the recognizer.

Nothing in the project had ever measured memory, so "it runs for five minutes" was the only
evidence that nothing grew. This measures it directly.

    game        a headless game played by a random-input bot for several simulated minutes,
                restarting on every Game Over, so rounds, deaths, popups, fruit and new games
                are all exercised. Resident memory is sampled throughout, and tracemalloc
                measures Python-heap growth between the end of warm-up and the end of the run.
    recognizer  the frozen CNN on CUDA over thousands of synthetic regions of interest:
                resident memory, and CUDA memory allocated, reserved and at peak.

A leak shows up as steady growth after warm-up, so the verdict is on the slope, not the size.

Usage:
    python -m src.measure_memory --game
    python -m src.measure_memory --recognizer
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Final

import psutil

# Resolved here rather than imported from src.data_pipeline, which imports torch: the game
# profile must measure the game's real footprint, and the keyboard game never loads torch.
PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
REPORT_PATH: Final = PROJECT_ROOT / "reports" / "memory_profile.json"
MB: Final = 1024 * 1024

GAME_FPS: Final = 60
BOT_REQUEST_EVERY: Final = 20  # frames between random direction requests
GAME_WARMUP_SECONDS: Final = 30.0
GAME_SAMPLE_EVERY: Final = 600  # frames, i.e. every 10 simulated seconds

RECOGNIZER_WARMUP: Final = 50
RECOGNIZER_SAMPLE_EVERY: Final = 100
ROI_SIDE: Final = 300

# Growth below these is noise from allocator behaviour, not a leak.
GAME_RSS_LIMIT_MB_PER_MINUTE: Final = 1.0
GAME_HEAP_LIMIT_MB: Final = 1.0
RECOGNIZER_RSS_LIMIT_MB_PER_1K: Final = 1.0


def rss_mb(process: psutil.Process) -> float:
    return float(process.memory_info().rss) / MB


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y against x; 0.0 when there are too few points to fit."""
    if len(points) < 2:
        return 0.0
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if spread == 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / spread


def profile_game(minutes: float, seed: int) -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from game.controls import DIRECTIONS
    from game.engine import GAME_OVER, Game

    process = psutil.Process()
    before_game = rss_mb(process)
    game = Game(headless=True, seed=seed)
    after_game = rss_mb(process)
    bot = random.Random(seed)  # gameplay input, not security

    dt = 1.0 / GAME_FPS
    frames = round(minutes * 60 * GAME_FPS)
    warmup = round(GAME_WARMUP_SECONDS * GAME_FPS)
    samples: list[tuple[float, float]] = []
    games = 1
    heap_at_warmup = 0
    started = time.perf_counter()

    tracemalloc.start()
    for frame in range(frames):
        if frame % BOT_REQUEST_EVERY == 0:
            game.request_direction(bot.choice(DIRECTIONS))
        game.update(dt)
        game.draw()
        if game.state == GAME_OVER:
            game.new_game()
            games += 1
        if frame == warmup:
            gc.collect()
            heap_at_warmup = tracemalloc.get_traced_memory()[0]
        if frame % GAME_SAMPLE_EVERY == 0:
            samples.append((frame * dt / 60.0, rss_mb(process)))
    gc.collect()
    heap_at_end, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - started

    steady = [(t, r) for t, r in samples if t * 60.0 >= GAME_WARMUP_SECONDS]
    rss_growth = slope(steady)
    heap_growth = (heap_at_end - heap_at_warmup) / MB
    return {
        "simulated_minutes": minutes,
        "frames": frames,
        "games_played": games,
        "wall_seconds": round(elapsed, 1),
        "rss_before_game_mb": round(before_game, 1),
        "rss_game_created_mb": round(after_game, 1),
        "rss_end_mb": round(samples[-1][1], 1),
        "rss_growth_mb_per_minute_after_warmup": round(rss_growth, 3),
        "python_heap_growth_after_warmup_mb": round(heap_growth, 3),
        "python_heap_peak_mb": round(heap_peak / MB, 2),
        # Evidence the footprint is the game's own: run standalone, neither should be loaded.
        "cnn_stack_loaded": sorted(m for m in ("torch", "torchvision", "cv2") if m in sys.modules),
        "no_leak": rss_growth < GAME_RSS_LIMIT_MB_PER_MINUTE and heap_growth < GAME_HEAP_LIMIT_MB,
    }


def profile_recognizer(frames: int, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch

    from src.gesture_recognizer import DirectionRecognizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: the recognizer is never profiled on the CPU")

    process = psutil.Process()
    before_model = rss_mb(process)
    recognizer = DirectionRecognizer()
    after_model = rss_mb(process)
    generator = np.random.default_rng(seed)
    pool = [
        generator.integers(0, 256, size=(ROI_SIDE, ROI_SIDE, 3), dtype=np.uint8) for _ in range(16)
    ]

    for index in range(RECOGNIZER_WARMUP):
        recognizer.predict_roi(pool[index % len(pool)])
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    # The first inferences initialise the CUDA context and cuDNN, a large one-off host
    # allocation. Recording the level after warm-up separates that from growth over time.
    after_warmup = rss_mb(process)

    samples: list[tuple[float, float]] = []
    started = time.perf_counter()
    for index in range(frames):
        recognizer.predict_roi(pool[index % len(pool)])
        if index % RECOGNIZER_SAMPLE_EVERY == 0:
            samples.append((index / 1000.0, rss_mb(process)))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    growth = slope(samples)
    return {
        "frames": frames,
        "device": torch.cuda.get_device_name(0),
        "wall_seconds": round(elapsed, 1),
        "rss_before_model_mb": round(before_model, 1),
        "rss_model_loaded_mb": round(after_model, 1),
        "rss_after_warmup_mb": round(after_warmup, 1),
        "rss_end_mb": round(samples[-1][1], 1),
        "rss_growth_mb_per_1k_frames": round(growth, 3),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated() / MB, 1),
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / MB, 1),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / MB, 1),
        "no_leak": growth < RECOGNIZER_RSS_LIMIT_MB_PER_1K,
    }


def write_report(section: str, result: dict[str, Any]) -> None:
    report: dict[str, Any] = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    report[section] = result
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--game", action="store_true", help="profile the headless game loop")
    target.add_argument("--recognizer", action="store_true", help="profile the CUDA recognizer")
    parser.add_argument("--minutes", type=float, default=10.0, help="simulated game minutes")
    parser.add_argument("--frames", type=int, default=5000, help="recognizer frames")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.game:
        section, result = "game", profile_game(args.minutes, args.seed)
    else:
        section, result = "recognizer", profile_recognizer(args.frames, args.seed)
    write_report(section, result)
    for name, value in result.items():
        print(f"  {name:<42} {value}")
    print(f"RESULT: {'PASS' if result['no_leak'] else 'FAIL'} ({section}) -> {REPORT_PATH}")
    return 0 if result["no_leak"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
