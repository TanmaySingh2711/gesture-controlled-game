# Manual Test Plan

The checks a person has to make, because they depend on a real hand, a real webcam, real eyes and
real ears. Automated suites cover the logic underneath every row (`pytest`, plus the three
self-test scripts); this plan covers whether it actually works and feels right.

**Setup:** normal indoor lighting, sit about an arm's length from the webcam, launch
`python src/play_gesture.py`. Mark each row **Pass**, **Fail** or **N/A**, and write down anything
odd - a failure with a note is far more useful than a tick without one.

## 1. Start-up

| # | Check | Expected | Result |
|---|---|---|---|
| 1.1 | Launch the application | Window opens on the start screen within a few seconds, no terminal errors | |
| 1.2 | Watch the status line | Moves from `Initializing gesture recognition...` to `Camera ready` | |
| 1.3 | Press `C` on the start screen | Colour theme changes immediately | |
| 1.4 | Press `M` on the start screen | The sound label changes to off / on | |
| 1.5 | Press SPACE | Gameplay starts at `READY!` | |

## 2. Gestures

| # | Check | Expected | Result |
|---|---|---|---|
| 2.1 | Fist in the box | Panel shows `LEFT`, Pac-Man turns left at the next opening | |
| 2.2 | Open palm | `RIGHT`, and Pac-Man follows | |
| 2.3 | Thumbs up | `UP`, and Pac-Man follows | |
| 2.4 | Thumbs down | `DOWN`, and Pac-Man follows | |
| 2.5 | Hold a gesture before reaching a junction | Pac-Man takes the turn when it becomes possible (pre-turning) | |
| 2.6 | Take your hand out of the box | Panel shows `NO COMMAND`; Pac-Man keeps moving, never stops by itself | |
| 2.7 | Rest a relaxed hand in the box | No unwanted turns over about 20 seconds | |
| 2.8 | Switch naturally between gestures | The new direction arrives; no wrong turn in between | |
| 2.9 | Navigate a full lap of the maze using gestures only | Achievable without the keyboard | |

## 3. Keyboard and options

| # | Check | Expected | Result |
|---|---|---|---|
| 3.1 | Arrow keys and WASD | Both move Pac-Man; gestures still work afterwards | |
| 3.2 | `P` during play | Everything freezes with `PAUSED`; `P` again resumes | |
| 3.3 | `H` or `F1` | Help overlay lists gestures and keys; closes with the same key | |
| 3.4 | `C` during play | Cycles classic, high contrast, colour-blind safe; maze stays readable in each | |
| 3.5 | `M` during play | Sound effects stop and start | |

## 4. Gameplay

| # | Check | Expected | Result |
|---|---|---|---|
| 4.1 | Eat pellets | They disappear, score rises by 10, a short chirp plays | |
| 4.2 | Eat a power pellet | Ghosts turn violet/blue with a wavy mouth and flash before recovering | |
| 4.3 | Eat a frightened ghost | Score popup (200, then 400...), the ghost becomes eyes and returns home | |
| 4.4 | Touch a normal ghost | `LIFE LOST` with remaining lives; eaten pellets stay eaten | |
| 4.5 | Bonus fruit | Appears below the ghost house after about 70 pellets; eating it scores and shows a popup | |
| 4.6 | Use the tunnel | Pac-Man wraps to the other side and gestures still steer | |
| 4.7 | Clear a round | `ROUND N CLEARED!`, pellets refill, score and lives carry over | |
| 4.8 | Lose every life | `GAME OVER` with score and round; `R` restarts cleanly | |
| 4.9 | Beat your previous score, then relaunch | `HIGH` shows the new best, and `NEW HIGH SCORE!` appeared at Game Over | |

## 5. Robustness and performance

| # | Check | Expected | Result |
|---|---|---|---|
| 5.1 | Play for several minutes | Motion stays smooth; the panel's `fps` stays near 60 | |
| 5.2 | Unplug the webcam during play (if USB) | Panel shows `RECONNECTING`; plug back in and gestures resume, or `CAMERA ERROR` after the retries - the keyboard keeps working either way | |
| 5.3 | `python src/play_gesture.py --no-camera` | Plays on the keyboard; the panel reads `GESTURE CONTROL: OFF` | |
| 5.4 | Quit with ESC during play | Window closes promptly with no error | |
| 5.5 | Quit with the window close button | Same | |
| 5.6 | Quit from the start screen | Same | |

## Sign-off

| | |
|---|---|
| Tester | |
| Date | |
| Webcam / lighting | |
| Overall | Pass / Fail |
| Notes | |
