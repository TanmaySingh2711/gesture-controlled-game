"""Combining live sessions: grouping by person and room, rates with intervals, the P6 records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")  # analyze_evaluation, which supplies the Wilson interval

from src import live_report as lr


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trial(person: str, room: str, expected: str, correct: str) -> dict[str, Any]:
    return {
        "participant": person,
        "condition": room,
        "expected_class": expected,
        "correct": correct,
    }


def change(person: str, room: str, ms: float, spurious: int) -> dict[str, Any]:
    return {
        "participant": person,
        "condition": room,
        "transition": "left->right",
        "ms_to_stable": ms,
        "spurious_count": spurious,
    }


@pytest.fixture
def sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "live_sessions"
    monkeypatch.setattr(lr, "SESSIONS_DIR", folder)
    monkeypatch.setattr(lr, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(lr, "REPORT_PATH", tmp_path / "live_summary.json")
    return folder


def test_sessions_are_combined_per_person_and_per_room(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_csv(
        sessions / "a_trials.csv",
        [
            trial("asha", "bright", "up", "yes"),
            trial("asha", "bright", "left", "yes"),
            trial("asha", "dim", "down", "no"),
        ],
    )
    write_csv(
        sessions / "b_idle.csv",
        [trial("ravi", "dim", "no_hand", "yes"), trial("ravi", "dim", "idle_hand", "no")],
    )
    write_csv(
        sessions / "c_transitions.csv",
        [change("ravi", "dim", 900, 0), change("ravi", "dim", 2100, 1)],
    )

    assert lr.main([]) == 0

    report = json.loads((sessions.parent / "live_summary.json").read_text("utf-8"))
    assert (report["participants"], report["conditions"]) == (2, 2)
    overall = report["overall"]
    assert overall["held_gestures_correct"]["count"] == 2
    assert overall["held_gestures_correct"]["total"] == 3
    assert overall["false_commands_idle_or_absent"]["count"] == 1
    assert overall["changes_through_a_wrong_command"]["count"] == 1
    assert overall["change_seconds_median"] == pytest.approx(1.5)
    dim = report["by_condition"]["dim"]
    assert dim["held_gestures_correct"]["rate"] == 0.0
    low, high = report["by_participant"]["asha"]["held_gestures_correct"]["wilson_95"]
    assert 0.0 < low < 2 / 3 < high < 1.0, "three trials give a wide interval"
    assert "2 participant(s), 2 condition(s)" in capsys.readouterr().out


def test_the_p6_records_are_included_only_when_asked(
    sessions: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert lr.main([]) == 1
    assert "No live sessions found" in capsys.readouterr().out
    assert lr.main(["--include-p6"]) == 0
    report = json.loads((sessions.parent / "live_summary.json").read_text("utf-8"))
    p6 = report["by_participant"][lr.P6_LABEL]
    assert p6["held_gestures_correct"]["count"] == 80
    assert p6["false_commands_idle_or_absent"]["count"] == 0
    # Both P6 transition runs count: 2 of the 18 changes passed through a wrong command.
    assert (
        p6["changes_through_a_wrong_command"]["count"],
        p6["changes_through_a_wrong_command"]["total"],
    ) == (2, 18)


def test_an_empty_group_reports_no_rate() -> None:
    block = lr.summarise([])
    assert block["held_gestures_correct"] == {
        "count": 0,
        "total": 0,
        "rate": None,
        "wilson_95": None,
    }
    assert block["change_seconds_median"] is None
    assert lr.describe(block).startswith("held -")
