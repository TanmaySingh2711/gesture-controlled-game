"""Combine live recording sessions into one report: per person, per room condition, and overall.

`python src/realtime_gesture.py --trials|--idle|--transitions --participant NAME --condition ROOM`
writes one CSV per session into `reports/live_sessions/`. This reads all of them and answers the
questions the offline numbers cannot:

* **Held gestures** - how often the stable command matched the gesture being held.
* **Idle and absent hands** - how often a command appeared when none should have.
* **Gesture changes** - how often a change passed through a wrong command, and how long it took.

Each rate carries a Wilson 95% interval, so a handful of trials is reported as the uncertain
evidence it is. `--include-p6` adds the original P6 records from `reports/p6_live/`, labelled as such.

Usage:
    python -m src.live_report
    python -m src.live_report --include-p6
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

from src.analyze_evaluation import wilson_interval
from src.paths import shown

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
SESSIONS_DIR: Final = PROJECT_ROOT / "reports" / "live_sessions"
REPORT_PATH: Final = PROJECT_ROOT / "reports" / "live_summary.json"
P6_DIR: Final = PROJECT_ROOT / "reports" / "p6_live"
P6_FILES: Final = (
    P6_DIR / "live_direction_test_baseline.csv",
    P6_DIR / "live_direction_idle_baseline.csv",
    P6_DIR / "live_direction_transitions_firstattempt.csv",
    P6_DIR / "live_direction_transitions.csv",
)
P6_LABEL: Final = "p6-developer (count not recorded)"
NO_COMMAND: Final = ("no_hand", "idle_hand")


def read_rows(paths: list[Path], default_label: str | None = None) -> list[dict[str, str]]:
    """Every row of every file, with participant and condition filled in where missing."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row.setdefault("participant", default_label or "anonymous")
                row.setdefault(
                    "condition", "p6-development-room" if default_label else "unspecified"
                )
                row["source_file"] = path.name
                rows.append(row)
    return rows


def rate(hits: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {"count": 0, "total": 0, "rate": None, "wilson_95": None}
    low, high = wilson_interval(hits, total)
    return {"count": hits, "total": total, "rate": hits / total, "wilson_95": [low, high]}


def summarise(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Held-gesture accuracy, false commands and transition quality for one group of rows."""
    held = [r for r in rows if "expected_class" in r and r["expected_class"] not in NO_COMMAND]
    quiet = [r for r in rows if r.get("expected_class") in NO_COMMAND]
    changes = [r for r in rows if "transition" in r]
    seconds = [float(r["ms_to_stable"]) / 1000.0 for r in changes]
    return {
        "held_gestures_correct": rate(sum(r["correct"] == "yes" for r in held), len(held)),
        "false_commands_idle_or_absent": rate(sum(r["correct"] == "no" for r in quiet), len(quiet)),
        "changes_through_a_wrong_command": rate(
            sum(int(r["spurious_count"]) > 0 for r in changes), len(changes)
        ),
        "change_seconds_median": statistics.median(seconds) if seconds else None,
        "change_seconds_max": max(seconds) if seconds else None,
    }


def build_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_person: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_person[row["participant"]].append(row)
        by_condition[row["condition"]].append(row)
    return {
        "sessions": sorted({row["source_file"] for row in rows}),
        "participants": len(by_person),
        "conditions": len(by_condition),
        "overall": summarise(rows),
        "by_participant": {name: summarise(group) for name, group in sorted(by_person.items())},
        "by_condition": {name: summarise(group) for name, group in sorted(by_condition.items())},
    }


def describe(block: dict[str, Any]) -> str:
    def part(label: str, entry: dict[str, Any]) -> str:
        if not entry["total"]:
            return f"{label} -"
        low, high = entry["wilson_95"]
        return f"{label} {entry['count']}/{entry['total']} [{low:.0%}-{high:.0%}]"

    median = block["change_seconds_median"]
    return "  ".join(
        (
            part("held", block["held_gestures_correct"]),
            part("false cmds", block["false_commands_idle_or_absent"]),
            part("bad changes", block["changes_through_a_wrong_command"]),
            f"change median {median:.1f}s" if median is not None else "change median -",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--include-p6", action="store_true", help="add the original P6 records")
    args = parser.parse_args(argv)

    sessions = sorted(SESSIONS_DIR.glob("*.csv")) if SESSIONS_DIR.is_dir() else []
    rows = read_rows(sessions)
    if args.include_p6:
        rows += read_rows([p for p in P6_FILES if p.exists()], default_label=P6_LABEL)
    if not rows:
        print(
            "No live sessions found. Record one with, for example:\n"
            "    python src/realtime_gesture.py --trials --participant NAME --condition bright-room"
        )
        return 1

    report = build_report(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(
        f"{len(report['sessions'])} session file(s), {report['participants']} participant(s), "
        f"{report['conditions']} condition(s)"
    )
    print(f"overall      {describe(report['overall'])}")
    for title, key in (("participant", "by_participant"), ("condition", "by_condition")):
        print(f"by {title}:")
        for name, block in report[key].items():
            print(f"  {name:<34} {describe(block)}")
    print(f"report: {shown(REPORT_PATH, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
