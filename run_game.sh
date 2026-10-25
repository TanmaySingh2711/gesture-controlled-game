#!/usr/bin/env bash
# Start the game on macOS and Linux:  bash run_game.sh
# Gestures steer Pac-Man and the keyboard keeps working. Options are passed through:
#     bash run_game.sh --no-camera     keyboard only, no webcam
#     bash run_game.sh --device cpu    run the gesture model on the CPU
# ESC in the game closes it, and this script ends with it.
set -euo pipefail
cd "$(dirname "$0")"
# pygame prints a two-line greeting on import; keep the terminal clean.
export PYGAME_HIDE_SUPPORT_PROMPT=1

if [ ! -x venv/bin/python ]; then
    echo "The game is not set up yet. Run  bash setup.sh  first."
    exit 1
fi

status=0
# ${@+"$@"}: with no options, the bash 3.2 that macOS ships fails on a bare "$@" under set -u.
venv/bin/python src/play_gesture.py ${@+"$@"} || status=$?
if [ "$status" -ne 0 ]; then
    echo
    echo "The game stopped with an error. The message above says what went wrong."
fi
exit "$status"
