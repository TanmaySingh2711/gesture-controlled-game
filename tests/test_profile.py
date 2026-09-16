"""The persisted profile must never break the game, whatever state its file is in."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from game.profile import PROFILE_ENV, Profile, ProfileStore, default_profile_path


def test_missing_file_loads_defaults(tmp_path: Path) -> None:
    assert ProfileStore(tmp_path / "absent.json").load() == Profile()


def test_round_trip(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "nested" / "profile.json")
    store.save(Profile(high_score=4560, theme="high-contrast", muted=True))
    assert store.load() == Profile(high_score=4560, theme="high-contrast", muted=True)


def test_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path / "profile.json")
    store.save(Profile(high_score=10))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["profile.json"]


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '"a string"', '{"high_score": "lots"}', ""],
    ids=["broken", "list", "string", "bad-number", "empty"],
)
def test_corrupt_file_loads_defaults_and_warns(
    tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="game.profile"):
        assert ProfileStore(path).load() == Profile()
    assert "ignoring unreadable profile" in caplog.text


def test_negative_high_score_is_clamped(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"high_score": -50}), encoding="utf-8")
    assert ProfileStore(path).load().high_score == 0


def test_unwritable_location_is_logged_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    store = ProfileStore(blocker / "profile.json")
    with caplog.at_level(logging.WARNING, logger="game.profile"):
        store.save(Profile(high_score=1))
    assert "could not save profile" in caplog.text


def test_pathless_store_is_purely_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = ProfileStore(None)
    store.save(Profile(high_score=999))
    assert store.load() == Profile()
    assert list(tmp_path.iterdir()) == []


def test_environment_variable_relocates_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROFILE_ENV, str(tmp_path))
    assert default_profile_path() == tmp_path / "profile.json"
