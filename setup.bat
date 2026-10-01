@echo off
rem One-click setup on Windows: double-click this file, or run  setup.bat  in a terminal.
rem It finds Python 3.12 and runs install.py, which builds venv\, installs the right PyTorch
rem build (NVIDIA GPU or CPU), installs the project and checks that everything works.
rem Options are passed through:  setup.bat --dev   setup.bat --cpu   setup.bat --fresh
setlocal
cd /d "%~dp0"

set "PYTHON="
where py >nul 2>nul && py -3.12 -c "import sys" >nul 2>nul && set "PYTHON=py -3.12"
if not defined PYTHON where python >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
    echo Python was not found. Install 64-bit Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in its installer, then run this again.
    goto :failed
)

%PYTHON% install.py %*
if errorlevel 1 goto :failed

echo.
echo Setup finished. Double-click run_game.bat to play.
if not defined CI pause
exit /b 0

:failed
echo.
echo Setup did not finish. The message above says what went wrong.
if not defined CI pause
exit /b 1
