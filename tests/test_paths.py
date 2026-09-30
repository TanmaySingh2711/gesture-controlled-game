"""Showing paths in messages never fails, whatever folder or drive the path is on."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import paths
from src.paths import PROJECT_ROOT, shown


def test_a_path_inside_the_project_is_shown_relative() -> None:
    assert shown(PROJECT_ROOT / "reports" / "x.json") == str(Path("reports") / "x.json")


def test_a_path_outside_the_root_is_shown_in_full(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "result.csv"
    assert shown(outside, root=tmp_path / "project") == str(outside)


def test_a_path_on_another_drive_is_shown_in_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, os.path.relpath raises for another drive; `shown` must not.

    Simulated without touching any real drive: `relative_to` is made to fail exactly as it does
    across drives.
    """

    def across_drives(self: Path, *_other: object) -> Path:
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(paths.Path, "relative_to", across_drives)
    assert shown("some/output.csv") == "some/output.csv"
