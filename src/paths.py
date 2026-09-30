"""How file paths are shown in messages.

Relative to the project when the file is inside it, which keeps output short, and in full
otherwise. `os.path.relpath` cannot be used for this: on Windows it raises when the file is on
another drive than the project (an output folder on D: with the project on C:, or a temporary
folder on C: with the project on D:, as on GitHub's Windows runners).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent


def shown(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> str:
    """`path` relative to `root` (default: the project) when inside it, else `path` unchanged."""
    base = Path(root) if root is not None else PROJECT_ROOT
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)
