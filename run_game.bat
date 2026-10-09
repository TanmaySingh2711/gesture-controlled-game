@echo off
rem Start the game on Windows: double-click this file, or run  run_game.bat  in a terminal.
rem Gestures steer Pac-Man and the keyboard keeps working. Options are passed through:
rem     run_game.bat --no-camera     keyboard only, no webcam
rem     run_game.bat --device cpu    run the gesture model on the CPU
rem ESC in the game closes the game and this window together.
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo The game is not set up yet. Double-click setup.bat first.
    if not defined CI pause
    exit /b 1
)

rem A window opened by a double-click goes to the middle of the screen. Windows Terminal places
rem its own windows (status 2), so then this reopens once in the classic console window, which
rem can be moved. A command prompt you typed into, and CI, are left where they are.
if defined CI goto play
call :started_by_double_click || goto play
title Gesture Pac-Man
"venv\Scripts\python.exe" -B src\console_window.py
if errorlevel 2 if not defined GESTURE_PACMAN_REOPENED (
    set "GESTURE_PACMAN_REOPENED=1"
    start "Gesture Pac-Man" conhost.exe cmd /c ""%~f0" %*"
    exit /b 0
)

:play
"venv\Scripts\python.exe" src\play_gesture.py %*
if errorlevel 1 (
    echo.
    echo The game stopped with an error. The message above says what went wrong.
    if not defined CI pause
    exit /b 1
)
exit /b 0

:started_by_double_click
rem Explorer starts a batch file as  cmd /c ""...\run_game.bat" " ; a prompt you type into has no /c.
setlocal EnableDelayedExpansion
set "line=!cmdcmdline!"
if /i "!line:/c =!"=="!line!" exit /b 1
exit /b 0
