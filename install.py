"""One-command setup: virtual environment, the right PyTorch build, the project, and a check.

    python install.py            # player setup: everything needed to play
    python install.py --dev      # ...plus the test, lint and type-check tools
    python install.py --cpu      # force the CPU build even on a machine with an NVIDIA GPU
    python install.py --fresh    # rebuild venv/ from scratch (after moving the project folder)

What it does, in order:

1. checks this is 64-bit Python 3.12,
2. creates ``venv/`` (or reuses it - unless it was created in another folder, whose command
   shortcuts such as ``venv/Scripts/mypy.exe`` would still point there),
3. installs ``requirements.txt`` (CUDA build) when an NVIDIA GPU is found, otherwise
   ``requirements-cpu.txt`` - gesture control works on either,
4. installs the project itself in editable mode, so ``game`` and ``src`` import from this folder
   wherever it lives (a moved folder is fixed by simply running this again),
5. runs ``src/environment_check.py`` inside the new environment.

Uses only the standard library, so it runs with any Python 3.12 before anything is installed.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
REQUIRED_PYTHON = (3, 12)

# Windows limits paths to 260 characters unless long paths are enabled, and PyTorch installs
# files nested about 140 characters deep inside site-packages. A project folder with a long
# path therefore fails half-way through the install with a confusing pip error.
WINDOWS_MAX_PATH = 260
DEEPEST_PACKAGE_PATH = 140


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def has_nvidia_gpu() -> bool:
    """True when `nvidia-smi` exists and reports at least one GPU."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return False
    try:
        result = subprocess.run(
            [tool, "--list-gpus"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def windows_long_paths_enabled() -> bool:
    """Whether Windows' LongPathsEnabled policy is on (always True off Windows)."""
    if sys.platform != "win32":
        return True
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        return False
    return bool(value)


def path_problem() -> str | None:
    """A message when this folder's path is too long for PyTorch to install, else None."""
    site_packages = VENV / "Lib" / "site-packages"
    if len(str(site_packages)) + DEEPEST_PACKAGE_PATH <= WINDOWS_MAX_PATH:
        return None
    if windows_long_paths_enabled():
        return None
    return (
        f"This folder's path is too long for Windows to install PyTorch into:\n    {ROOT}\n"
        "Move the project to a shorter path (for example C:\\gesture-pacman) and run this "
        "again, or enable Windows long paths: https://pip.pypa.io/warnings/enable-long-paths"
    )


def venv_origin(venv: Path | None = None) -> Path | None:
    """Where `venv` (default: VENV) was created, from the `command` line in pyvenv.cfg."""
    config = (venv or VENV) / "pyvenv.cfg"
    if not config.exists():
        return None
    for line in config.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "command" and " -m venv " in value:
            return Path(value.split(" -m venv ", 1)[1].strip().strip('"'))
    return None


def moved_venv_problem(venv: Path | None = None) -> str | None:
    """A message when `venv` (default: VENV) was created somewhere else, else None.

    Python itself keeps working in a moved venv, but every command shortcut in it (pip.exe,
    mypy.exe, pytest.exe) still points at the old folder and fails without saying why.
    """
    venv = venv or VENV
    origin = venv_origin(venv)
    if origin is None or origin.resolve() == venv.resolve():
        return None
    return (
        f"venv/ was created in another folder:\n    {origin}\n"
        "Its command shortcuts (such as mypy and pytest) still point there and will not run.\n"
        "Run  python install.py --fresh  to rebuild it here."
    )


def run(step: str, command: list[str | Path]) -> None:
    print(f"\n==> {step}")
    print("    " + " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"\nSetup stopped: '{step}' failed (exit code {result.returncode}).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up CNN Gesture Controlled Pac-Man.")
    parser.add_argument("--dev", action="store_true", help="also install the developer tools")
    parser.add_argument("--cpu", action="store_true", help="use the CPU build of PyTorch")
    parser.add_argument("--fresh", action="store_true", help="rebuild venv/ from scratch")
    args = parser.parse_args(argv)

    if sys.version_info[:2] != REQUIRED_PYTHON or struct.calcsize("P") != 8:
        print(
            f"This project needs 64-bit Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}; "
            f"this is {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)."
        )
        return 1

    problem = path_problem() or (None if args.fresh else moved_venv_problem())
    if problem:
        print(problem)
        return 1

    gpu = not args.cpu and has_nvidia_gpu()
    requirements = "requirements.txt" if gpu else "requirements-cpu.txt"
    print(f"NVIDIA GPU: {'found' if gpu else 'not used'} -> installing {requirements}")

    if args.fresh and VENV.exists():
        # --clear empties venv/ and recreates it in place: only the environment is rebuilt.
        run("rebuild the virtual environment", [sys.executable, "-m", "venv", "--clear", VENV])
    elif not venv_python().exists():
        run("create the virtual environment", [sys.executable, "-m", "venv", VENV])
    python = venv_python()
    run("upgrade pip", [python, "-m", "pip", "install", "--upgrade", "pip"])
    run("install the runtime libraries", [python, "-m", "pip", "install", "-r", requirements])
    if args.dev:
        run(
            "install the developer tools",
            [python, "-m", "pip", "install", "-r", "requirements-dev.txt"],
        )
    run("install the project", [python, "-m", "pip", "install", "-e", ".", "--no-deps"])
    run("check the environment", [python, "src/environment_check.py", "--skip-webcam"])

    activate = r"venv\Scripts\activate" if sys.platform == "win32" else "source venv/bin/activate"
    print("\nSetup complete. To play:")
    print(f"    {activate}")
    print("    python src/play_gesture.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
