"""The one-command installer's decisions, without installing anything.

`run` is replaced with a recorder, so each test sees exactly which commands setup would run.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load_installer() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("install", ROOT / "install.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading install.py from the project root must not leave a __pycache__/ folder there.
    writes_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes_bytecode
    return module


@pytest.fixture
def installer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    module: Any = load_installer()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "VENV", tmp_path / "venv")
    return module


@pytest.fixture
def commands(installer: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every command setup would run, as text; nothing is executed."""
    ran: list[str] = []
    monkeypatch.setattr(
        installer, "run", lambda step, command: ran.append(" ".join(str(c) for c in command))
    )
    monkeypatch.setattr(installer, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(installer, "windows_long_paths_enabled", lambda: True)
    return ran


def make_venv(folder: Path, created_at: Path, options: str = "") -> None:
    folder.mkdir(parents=True)
    command = f"C:\\Python312\\python.exe -m venv {options}{created_at}"
    (folder / "pyvenv.cfg").write_text(
        f"home = C:\\Python312\nversion = 3.12.10\ncommand = {command}\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "options", ["", "--clear ", "--upgrade-deps --clear ", '--prompt="my env" --clear ']
)
def test_the_folder_is_read_past_any_venv_options(
    installer: Any, tmp_path: Path, options: str
) -> None:
    """`install.py --fresh` itself records --clear; that must not look like a moved venv."""
    home = tmp_path / "folder with spaces" / "venv"
    make_venv(home, home, options)
    assert installer.venv_origin(home) == home
    assert installer.moved_venv_problem(home) is None


def test_a_machine_without_a_gpu_gets_the_cpu_build(installer: Any, commands: list[str]) -> None:
    assert installer.main([]) == 0
    assert any("-r requirements/cpu.txt" in c for c in commands)
    assert any("-m venv" in c for c in commands)
    assert any("-e . --no-deps" in c for c in commands)
    assert commands[-1].endswith("src/environment_check.py --skip-webcam")
    assert not any("requirements/dev.txt" in c for c in commands)


def test_a_gpu_gets_the_cuda_build_unless_cpu_is_forced(
    installer: Any, commands: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "has_nvidia_gpu", lambda: True)
    assert installer.main(["--dev"]) == 0
    assert any(c.endswith("-r requirements/cuda.txt") for c in commands)
    assert any("-r requirements/dev.txt" in c for c in commands)
    commands.clear()
    assert installer.main(["--cpu"]) == 0
    assert any("-r requirements/cpu.txt" in c for c in commands)


def test_a_venv_moved_from_another_folder_is_caught(
    installer: Any, commands: list[str], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    make_venv(tmp_path / "venv", tmp_path / "old-place" / "venv")
    assert installer.venv_origin() == tmp_path / "old-place" / "venv"
    assert installer.main([]) == 1
    assert "created in another folder" in capsys.readouterr().out
    assert commands == [], "nothing is installed into a broken environment"

    assert installer.main(["--fresh"]) == 0
    assert any("-m venv --clear" in c for c in commands)


def test_a_venv_created_here_is_reused(installer: Any, commands: list[str], tmp_path: Path) -> None:
    make_venv(tmp_path / "venv", tmp_path / "venv")
    assert installer.moved_venv_problem() is None
    assert installer.venv_origin(tmp_path / "missing") is None


def test_a_path_too_long_for_windows_stops_before_installing(
    installer: Any,
    commands: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(installer, "VENV", Path("C:/" + "x" * 150) / "venv")
    monkeypatch.setattr(installer, "windows_long_paths_enabled", lambda: False)
    assert installer.main([]) == 1
    assert "too long" in capsys.readouterr().out
    assert commands == []
    monkeypatch.setattr(installer, "windows_long_paths_enabled", lambda: True)
    assert installer.path_problem() is None


def test_the_wrong_python_is_refused(
    installer: Any, commands: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    assert installer.main([]) == 1
    assert commands == []


@pytest.mark.parametrize(
    ("which", "result", "found"),
    [
        (None, None, False),
        ("nvidia-smi", subprocess.CompletedProcess([], 0, "GPU 0: RTX 3050 Ti", ""), True),
        ("nvidia-smi", subprocess.CompletedProcess([], 9, "", "driver not loaded"), False),
        ("nvidia-smi", OSError("cannot run"), False),
    ],
)
def test_gpu_detection(
    monkeypatch: pytest.MonkeyPatch, which: str | None, result: Any, found: bool
) -> None:
    module: Any = load_installer()
    monkeypatch.setattr(module.shutil, "which", lambda _name: which)

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.has_nvidia_gpu() is found


def test_a_failed_step_stops_setup_with_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module: Any = load_installer()
    monkeypatch.setattr(
        module.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 3)
    )
    with pytest.raises(SystemExit, match="'install the project' failed"):
        module.run("install the project", ["pip", "install", "-e", "."])
