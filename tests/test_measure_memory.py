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


@pytest.mark.slow
def test_the_recognizer_profile_runs_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(measure_memory, "RECOGNIZER_WARMUP", 3)
    monkeypatch.setattr(measure_memory, "RECOGNIZER_SAMPLE_EVERY", 5)
    result = measure_memory.profile_recognizer(frames=20, seed=0, device="cpu")
    assert result["device"] == "cpu"
    assert result["frames"] == 20
    assert result["cuda_peak_allocated_mb"] is None
    # Absolute readings only: inside a long test run the collector can shrink the process
    # between two samples, so "after >= before" would be flaky.
    assert result["rss_before_model_mb"] > 0
    assert result["rss_end_mb"] > 0
    assert isinstance(result["no_leak"], bool)


@pytest.mark.parametrize(("no_leak", "status"), [(True, 0), (False, 1)])
def test_main_writes_the_section_and_fails_on_a_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_leak: bool,
    status: int,
) -> None:
    report = tmp_path / "memory.json"
    seen: dict[str, object] = {}

    def fake_profile(frames: int, seed: int, device: str) -> dict[str, object]:
        seen.update(frames=frames, seed=seed, device=device)
        return {"frames": frames, "device": device, "no_leak": no_leak}

    monkeypatch.setattr(measure_memory, "REPORT_PATH", report)
    monkeypatch.setattr(measure_memory, "profile_recognizer", fake_profile)
    monkeypatch.setattr(
        "sys.argv", ["measure_memory", "--recognizer", "--frames", "40", "--device", "cpu"]
    )
    assert measure_memory.main() == status
    assert seen == {"frames": 40, "seed": 0, "device": "cpu"}
    # A CPU profile is kept apart from the GPU one rather than replacing it.
    assert json.loads(report.read_text(encoding="utf-8"))["recognizer_cpu"]["frames"] == 40
    assert ("RESULT: PASS" if no_leak else "RESULT: FAIL") in capsys.readouterr().out


def test_main_profiles_the_game_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(measure_memory, "REPORT_PATH", tmp_path / "memory.json")
    monkeypatch.setattr(
        measure_memory, "profile_game", lambda minutes, seed: {"minutes": minutes, "no_leak": True}
    )
    monkeypatch.setattr("sys.argv", ["measure_memory", "--game", "--minutes", "0.5"])
    assert measure_memory.main() == 0
