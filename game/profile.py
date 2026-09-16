"""The player's persisted profile: high score, colour theme and mute setting.

Stored as a small JSON file, by default ``~/.gesture_pacman/profile.json``. Set the
``GESTURE_PACMAN_HOME`` environment variable to keep it somewhere else.

Two rules keep this safe:

* **It is never allowed to break the game.** A missing, unreadable or hand-edited file loads
  as a default profile, and a failed save is logged and ignored.
* **Writes are atomic.** The profile is written to a temporary file and renamed over the old
  one, so a crash mid-save cannot leave a truncated file behind.

Only an interactive game (a real window on a real display) persists anything; headless games
and the test suite use a store with no path, which reads defaults and writes nowhere.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

log = logging.getLogger(__name__)

PROFILE_ENV: Final = "GESTURE_PACMAN_HOME"


def default_profile_path() -> Path:
    base = os.environ.get(PROFILE_ENV)
    root = Path(base) if base else Path.home() / ".gesture_pacman"
    return root / "profile.json"


@dataclass
class Profile:
    high_score: int = 0
    theme: str = "classic"
    muted: bool = False


class ProfileStore:
    """Loads and saves a `Profile`. A store with no path is purely in-memory."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> Profile:
        if self.path is None or not self.path.exists():
            return Profile()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Profile(
                high_score=max(0, int(data.get("high_score", 0))),
                theme=str(data.get("theme", "classic")),
                muted=bool(data.get("muted", False)),
            )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            log.warning("ignoring unreadable profile %s (%s)", self.path, error)
            return Profile()

    def save(self, profile: Profile) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(asdict(profile), indent=1), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as error:
            log.warning("could not save profile to %s (%s)", self.path, error)
