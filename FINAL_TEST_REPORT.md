# Final Test Report — CNN Gesture Controlled Pac-Man

**Objective 11 — Test the Complete Application**
Run date: 2026-09-13 · Windows 11 · Python 3.12.10 · PyTorch 2.14.0 + CUDA 13.0

**Status: PASSED — automated, structured and manual acceptance testing all passed. Project COMPLETE.**

---

## 1. Application under test

| | |
|---|---|
| Launch | `python src/play_gesture.py` |
| Keyboard-only | `python src/play_gesture.py --no-camera` |
| Model | `model/best_direction_model.pt` — MobileNetV2, 4 classes, 160×160 |
| Mapping | Fist → LEFT · Open palm → RIGHT · Thumbs up → UP · Thumbs down → DOWN |
| Recognition | threshold 0.90 · smoothing 3-of-5 · 300×300 ROI · mirrored webcam |
| Offline accuracy | 99.00% on the untouched 200-image test split (P5) |

One canonical control path was confirmed by audit and by test:

```
webcam -> RecognitionWorker -> DirectionRecognizer -> stable_command
       -> GestureController -> game.request_direction() -> Pac-Man movement
```

The obsolete endless-runner checkpoint is referenced only by guard tests that assert it is
refused; obsolete game and dataset code live in `*_archive_endless_runner/` and are not imported.

## 2. Automated suites

| Suite | Scope | Result |
|---|---|---|
| `src/realtime_gesture.py --selftest` | P6 recognizer (live webcam + CUDA) | **15/15 PASS** |
| `game/main.py --selftest` | P7 game rules | **58/58 PASS** |
| `src/check_integration.py` | Objective 9 integration | **28/28 PASS** |
| `src/check_ui.py` | Objective 10 interface | **21/21 PASS** |
| `src/check_final_application.py` | Objective 11 cross-subsystem contract | **27/27 PASS** |
| **Total** | | **149/149 PASS** |

The Objective 11 suite drives whole-application flows *through the gesture path*: checkpoint
identity and mapping, frozen settings, startup state, single worker, `None` behaviour, pellets,
power pellet and frightened mode, eating a ghost, life loss with the board preserved, respawn
not steered by stale gestures, Game Over, restart, gestures after restart, round completion via
the real last-pellet code path, ghost difficulty progression, tunnel, extra life once, keyboard
coexistence, keyboard-only mode (verified in a fresh process: torch, torchvision and cv2 never
imported), camera-failure fallback, clean shutdown, checkpoint unchanged.

## 3. Live application runs (real webcam, real CUDA model)

Each scenario ran in a fresh process against the unmodified application; instrumentation only
wrapped existing functions to count model constructions and camera opens/releases.

| Scenario | Ready after | Model loads | Camera open / release | Shutdown | Threads left | Traceback |
|---|---|---|---|---|---|---|
| Exit from start screen (ESC) | 2.30 s | 1 | 1 / 1 | 351 ms | none | none |
| Play 10 s → ESC | 2.17 s | 1 | 1 / 1 | 329 ms | none | none |
| Play 10 s → window close | 1.51 s | 1 | 1 / 1 | 333 ms | none | none |
| Stability, 330 s → ESC | 1.31 s | 1 | 1 / 1 | 352 ms | none | none |

Startup status progressed `loading model → opening camera → ready` every time.

### Stability run — 330 seconds, 30 s windows

| Window end (s) | Game FPS | Camera FPS | CNN mean (ms) | CNN p95 (ms) | Snapshot age mean (ms) | writes − frames |
|---|---|---|---|---|---|---|
| 30 | 60.3 | 30.5 | 17.82 | 26.63 | 16.6 | 3 |
| 60 | 60.1 | 30.5 | 18.12 | 27.54 | 17.0 | 3 |
| 90 | 60.2 | 30.5 | 18.18 | 26.80 | 16.8 | 3 |
| 120 | 60.1 | 30.5 | 18.07 | 27.14 | 17.0 | 3 |
| 150 | 60.2 | 30.5 | 18.57 | 28.19 | 17.1 | 3 |
| 180 | 60.0 | 30.5 | 18.44 | 28.19 | 16.8 | 3 |
| 210 | 60.1 | 30.5 | 18.06 | 26.37 | 17.1 | 3 |
| 240 | 60.0 | 30.5 | 18.42 | 27.88 | 17.0 | 3 |
| 270 | 60.0 | 30.5 | 18.06 | 26.60 | 17.0 | 3 |
| 300 | 60.1 | 30.5 | 18.52 | 28.01 | 17.1 | 3 |
| 330 | 60.0 | 30.5 | 18.59 | 27.36 | 16.9 | 3 |

Whole run: **game 60.1 FPS, frame p95 17.0 ms · camera 30.4 FPS · CNN mean 18.26 ms,
median 17.12, p95 27.40**. Six Game Over → R restarts occurred live without a model reload
or camera reopen.

- **No backlog:** `writes − frames` stayed exactly 3 (the startup status publishes) for the
  whole run — one snapshot slot, never a queue.
- **No lag growth:** the age of the snapshot the game acted on stayed ~17 ms from first window
  to last.
- **No degradation:** game and camera rates were flat across all eleven windows.

## 4. Integrity

| Artifact | MD5 | Size | Modified | After Objective 11 |
|---|---|---|---|---|
| `model/best_direction_model.pt` | `00FF609038E11575F48D8B360F1B6656` | 9,165,963 | 2026-09-13 01:09:49 | unchanged |
| `data_splits.json` | `A2E16B6E73AAF8F9D7B242BD38C7B74B` | 188,758 | 2026-09-13 01:01:09 | unchanged |
| `class_mapping.json` | `C11F96950F87DE43ADC6AC041B7A521C` | 587 | 2026-09-13 01:01:09 | unchanged |
| `model/direction_training_history.json` | `86C007015E03FC6E2AB004D754BE436F` | 4,434 | 2026-09-13 01:09:50 | unchanged |
| `model/direction_evaluation_metrics.json` | `B23FD7EBE54789715E4B01D59D34C9E9` | 4,990 | 2026-09-13 01:20:19 | unchanged |
| `dataset/` | 2,004 files, newest 2026-09-13 00:50:31 | | | unchanged |

No training, fine-tuning, dataset edit or checkpoint replacement occurred.

## 5. Defects

**Application defects found: none.** No application code was changed during Objective 11.

Test-side errors found and fixed while building the final suite (none affected the app):
four checks were constructed wrongly on the first run (a `None` test that drove Pac-Man into a
wall, a ghost-speed check that read `speed` instead of `base_speed`, a hard-coded score the game
had since changed, and a keyboard-only purity check that inspected its own already-loaded
modules); one worker check started a real worker and has been replaced with a device-free stub.

## 6. Limits of this testing, stated plainly

- **Memory was not measured.** The working-set probe in the live harness returned 0; flat frame
  rate and latency over 330 s show no stall symptoms, but that is indirect evidence.
- **One early exit was not explained.** A first stability attempt ended cleanly after ~62 s
  without the harness requesting it. The rerun, which logged external quit events, ran the full
  330 s with none. Most likely a desktop window event; not reproduced.
- **Unattended recognition:** during the 330 s run no one was gesturing, yet 25 of 19,835 game
  frames (0.13%) carried a stable command. What was in the camera's view is unknown, so this is
  recorded as an observation, not attributed.
- **Gesture feel cannot be automated.** Live gesture accuracy, gesture-only navigation, no-hand
  and idle-hand behaviour, natural transitions, and pre-turning by hand require a person.

## 7. Known limitation

The CNN has no reject/background class. A deliberate, well-formed gesture outside the four
commands (a peace sign, three fingers) can be confidently classified as one of them — measured
at P6 as RIGHT at 99.3%. Relaxed and absent hands are correctly rejected by the 0.90 threshold.
This was accepted by design; the start screen instructs "Use only the four gestures shown above."

## 8. Final manual acceptance

Approved by the project owner after live play on 2026-09-13.

| Check | Result |
|---|---|
| Final UI looks correct | PASS |
| All four gestures control Pac-Man correctly | PASS |
| Maze can be navigated by gesture alone | PASS |
| No-hand / idle-hand behaviour acceptable | PASS |
| Natural transitions acceptable | PASS |
| Pre-turning works by hand | PASS |
| Game feels responsive and smooth | PASS |
| Gameplay mechanics work live | PASS |
| Restart works | PASS |
| Shutdown works | PASS |
| Ready for college demonstration | PASS |

## 9. Verdict

**PASS.** 149/149 automated checks, all live scenarios, integrity unchanged, and final manual
acceptance approved. Objective 11 is complete and the project is complete.
