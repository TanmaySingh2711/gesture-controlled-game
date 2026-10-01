@echo off
rem Start the game on Windows: double-click this file, or run  run_game.bat  in a terminal.
rem Gestures steer Pac-Man and the keyboard keeps working. Options are passed through:
rem     run_game.bat --no-camera     keyboard only, no webcam
rem     run_game.bat --device cpu    run the gesture model on the CPU
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo The game is not set up yet. Double-click setup.bat first.
    if not defined CI pause
    exit /b 1
)

"venv\Scripts\python.exe" src\play_gesture.py %*
if errorlevel 1 (
    echo.
    echo The game stopped with an error. The message above says what went wrong.
    if not defined CI pause
    exit /b 1
)
exit /b 0
