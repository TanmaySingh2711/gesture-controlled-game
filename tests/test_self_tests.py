"""The standalone self-test scripts, run exactly as a developer runs them.

Each script asserts that torch and cv2 were never imported, which is only meaningful in a fresh
interpreter - the pytest process has long since imported both - so they run as subprocesses.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "model" / "best_direction_model.pt"
HAS_DATASET = any((ROOT / "dataset").rglob("*.jpg"))

# check_data_pipeline.py is deliberately absent: every run regenerates its augmentation sample
# sheets in the project root, which would dirty the working tree on each test run.
SCRIPTS = [
    pytest.param(["game/main.py", "--selftest"], id="game-rules"),
    pytest.param(["src/check_integration.py"], id="integration"),
    pytest.param(["src/check_ui.py"], id="ui"),
    pytest.param(
        ["src/check_final_application.py"],
        id="final-application",
        marks=pytest.mark.skipif(not CHECKPOINT.exists(), reason="frozen checkpoint not present"),
    ),
    pytest.param(
        ["src/check_dataset.py"],
        id="dataset",
        marks=pytest.mark.skipif(not HAS_DATASET, reason="dataset images are not tracked in git"),
    ),
    pytest.param(
        ["src/environment_check.py", "--skip-webcam"], id="environment", marks=pytest.mark.gpu
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("arguments", SCRIPTS)
def test_self_test_script_passes(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    report = result.stdout + result.stderr
    assert result.returncode == 0, report
    assert "RESULT: PASS" in result.stdout, report
