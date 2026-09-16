"""The memory profiler: its leak arithmetic, its report handling, and a short real game profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("psutil")

from src import measure_memory
from src.measure_memory import profile_game, slope, write_report


def test_slope_of_a_straight_line() -> None:
    assert slope([(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]) == pytest.approx(2.0)


def test_slope_of_flat_noise_is_near_zero() -> None:
    points = [(float(x), 100.0 + (0.1 if x % 2 else -0.1)) for x in range(20)]
    assert abs(slope(points)) < 0.02


@pytest.mark.parametrize("points", [[], [(1.0, 5.0)], [(2.0, 1.0), (2.0, 9.0)]])
def test_slope_without_enough_spread_is_zero(points: list[tuple[float, float]]) -> None:
    assert slope(points) == 0.0


def test_report_sections_are_merged_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "reports" / "memory_profile.json"
    monkeypatch.setattr(measure_memory, "REPORT_PATH", report)
    write_report("game", {"no_leak": True})
    write_report("recognizer", {"no_leak": False})
    write_report("game", {"no_leak": True, "frames": 10})
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "game": {"no_leak": True, "frames": 10},
        "recognizer": {"no_leak": False},
    }


@pytest.mark.slow
def test_a_short_game_profile_runs_through_game_overs_without_growth() -> None:
    result = profile_game(minutes=1.0, seed=1)
    assert result["frames"] == 3600
    assert result["games_played"] >= 1
    assert result["python_heap_growth_after_warmup_mb"] < measure_memory.GAME_HEAP_LIMIT_MB
    assert result["no_leak"] is True
