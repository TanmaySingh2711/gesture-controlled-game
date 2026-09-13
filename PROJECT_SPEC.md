# PROJECT SPECIFICATION — Pac-Man Revision (Frozen)

> **Revision notice.** This document replaces the previous endless-runner specification in
> full. The project is now a Pac-Man-style maze game, and the CNN command set is
> LEFT / RIGHT / UP / DOWN with no NEUTRAL class. Everything specific to the old
> obstacle-dodger design — JUMP, NEUTRAL, gravity, ground lanes, cacti — is withdrawn and must
> not be carried forward. Section 18 records exactly what survives the migration.

## 1. Project Title

**CNN-Based Gesture Controlled Gaming Application**

## 2. Problem Statement

Develop an interactive game application that uses a Convolutional Neural Network (CNN) to
recognize hand gestures captured through a webcam and translate them into corresponding game
actions.

## 3. Game Type

**Pac-Man-style maze game**, single player, offline, single screen. Gameplay rules follow the
original arcade game as closely as is practical, using **our own maze layout, graphics, sounds
and code**. No original sprites, sounds, maze artwork, character likenesses or official assets
are copied.

## 4. Core Gameplay Concept

The player guides a character continuously through a walled maze, eating pellets while four
ghosts pursue it. Eating a power pellet briefly turns the ghosts frightened and edible. Clearing
every pellet completes the round. Contact with a normal ghost costs a life; running out of lives
ends the game.

**Gameplay loop (one frame):**

1. Read the latest gesture prediction and, if confident, record it as a *direction request*.
2. Apply the requested direction if maze geometry allows it; otherwise keep the current direction.
3. Move the player continuously along the current direction, handling tunnel wraparound.
4. Update each ghost's state machine and movement.
5. Resolve pellet and power-pellet consumption, scoring and frightened mode.
6. Resolve player-ghost contact: eat the ghost, or lose a life.
7. If every pellet is eaten, advance to the next round; if lives are exhausted, Game Over.

## 5. The Four Gesture Classes

The CNN is a 4-class classifier. These are the only classes:

| Class | Game meaning |
|---|---|
| `LEFT` | Request that the player turn left |
| `RIGHT` | Request that the player turn right |
| `UP` | Request that the player turn up |
| `DOWN` | Request that the player turn down |

**There is no NEUTRAL class.** The previous design needed one because the endless runner had a
meaningful "no command" state in which the character stood still. Pac-Man has no such state: the
character is always travelling in some direction, and the absence of input means *continue as you
are*, not *stop*. A NEUTRAL class would therefore encode a game state that does not exist.

Its role is taken by the confidence threshold: when the top-class confidence falls below the
threshold, the frame issues **no new direction command** and the player keeps moving in its
current direction. This is described fully in section 9.

## 6. Physical Gesture Mapping

| Class | Physical hand gesture | HaGRID source class | Description |
|---|---|---|---|
| LEFT | Closed fist | `fist` | All fingers folded, hand facing the camera |
| RIGHT | Open palm | `palm` | All five fingers extended and spread, palm facing the camera |
| UP | Thumbs up | `like` | Fist with the thumb extended upward |
| DOWN | Thumbs down | `dislike` | Fist with the thumb extended downward |

Rules for the player:

- Use one hand only, the same hand throughout (right hand is the default).
- Keep the whole hand inside the capture region and let it fill most of the box.
- Hold the gesture clearly toward the camera.

**Why this set.** `fist` and `palm` are carried over unchanged from the previous design, where
they were measured at 100% live accuracy over 25 trials each. `like` and `dislike` are genuine
HaGRID classes, are easy to perform, and map intuitively onto up and down.

**Known risk — an orientation-dependent pair.** `like` and `dislike` are largely the same hand
shape rotated about 180°, which makes them the hardest pair in this set, much as
pointing-left/pointing-right was in the original design. Two consequences are binding:

- **Vertical flip must never be used as augmentation**, at any stage: it would turn a thumbs-up
  into a thumbs-down and corrupt the label.
- **Rotation augmentation must stay small.** The current ±10° is safe. Large rotations would blur
  the only feature separating UP from DOWN.

Horizontal flip remains safe: mirroring a thumbs-up leaves it a thumbs-up, and none of the four
gestures encodes a left/right direction in its own shape.

## 7. Webcam Capture Convention

Unchanged from the previous design and still frozen:

- Every webcam frame is **horizontally flipped immediately after capture**, at every stage —
  dataset work, real-time prediction and gameplay — so a player watching themselves sees a
  natural mirror and the model never meets a differently-oriented image than the one on screen.
- Only a **fixed 300x300 region of interest** at x 300-600, y 90-390 of the 640x480 frame is
  sent to the CNN. No hand tracking, segmentation or second model is used.
- A `LEFT` prediction always turns the player left, `RIGHT` right, `UP` up, `DOWN` down. No
  compensating swap exists anywhere in the code.

## 8. Pac-Man Gameplay Rules (Frozen)

Faithful to the original where practical; every deliberate simplification is named.

### 8.1 Maze

- A tile grid of walls, pellets and power pellets, in the classic 28 x 31 tile proportions but
  with **our own layout**, drawn from simple shapes.
- Walls block movement completely.
- **Tunnel wraparound is included**: a corridor on each side of the maze wraps horizontally, so
  a character leaving the left edge re-enters at the right edge on the same row, and vice versa.
- Four power pellets sit near the maze corners.
- The round is cleared when **every pellet and power pellet has been eaten**.

### 8.2 Player Movement

- The player moves **continuously** at a constant speed and never stops voluntarily.
- Movement is grid-aligned: turns are only possible at tile centres where the target tile is not
  a wall.
- A direction command is a **request**, not an impulse. The requested direction is stored and
  applied at the first moment it becomes legal; until then the player continues in its current
  direction. This reproduces the arcade feel of pre-turning into a corner.
- A pending request is held for a short grace period (a few tiles of travel) and then discarded,
  so a turn missed long ago does not fire unexpectedly later.
- Reversing direction is always permitted, since the opposite tile is by definition open.
- If the player runs into a wall head-on it stops at the tile boundary and waits there until a
  legal direction is requested. This is the only circumstance in which it stops.

### 8.3 Ghosts

Four ghosts, each with a distinct role, using our own names and colours:

| Role | Behaviour |
|---|---|
| Chaser | Targets the player's current tile directly |
| Ambusher | Targets a few tiles ahead of the player's current direction |
| Flanker | Targets a tile derived from both the player and the Chaser, so it cuts across |
| Drifter | Chases while far from the player, but retreats to its corner when close |

Ghost state machine, faithful to the original:

- **Scatter** — each ghost heads for its own corner of the maze.
- **Chase** — each ghost pursues its role's target tile.
- The game alternates scatter and chase on a fixed schedule (approximately 7 s scatter / 20 s
  chase for the first phases, settling into permanent chase later in the round).
- **Frightened** — triggered by a power pellet. Ghosts reverse direction, slow down, wander, and
  become edible. The mode ends after a fixed duration, with a visible warning flash near the end.
- **Eaten** — an eaten ghost becomes a pair of eyes that returns to the ghost house at high speed,
  then re-enters play in its normal state.
- Ghosts reverse direction when the global mode changes, as in the original.
- Ghosts leave the ghost house on a simple timed schedule at the start of each life and round.

**Simplifications, stated explicitly:** exact arcade target-tile arithmetic (including the
original's up-direction targeting overflow bug), per-level speed percentage tables, and exact
per-ghost house-exit dot counters are **not** reproduced. Roles, the four-state machine, the
scatter/chase alternation and the frightened/eaten behaviour are.

### 8.4 Scoring

| Event | Points |
|---|---|
| Pellet | 10 |
| Power pellet | 50 |
| Ghosts eaten in one frightened period | 200, 400, 800, 1600 |

The ghost multiplier resets at the start of each frightened period. An extra life is awarded at
10,000 points. **Bonus fruit is optional and out of scope for the first implementation**; if
added later it must be documented here with its values.

### 8.5 Lives and Game Over

- The player starts with **3 lives**.
- Contact with a ghost in its normal or scatter state costs one life.
- If lives remain: the player and ghosts return to their starting positions, ghost modes reset,
  and **pellets already eaten stay eaten**. Play resumes after a brief pause.
- With no lives remaining: **Game Over**, showing the final score, with a key to restart and a
  key to quit. Restart is keyboard-driven; the CNN never controls menus.

### 8.6 Rounds and Difficulty

- Clearing every pellet completes the round; the maze refills and the next round begins with
  positions reset and the score carried over.
- Later rounds are harder in two bounded ways: ghosts move slightly faster, and the frightened
  duration shortens (for example about 6 s in round 1, decreasing toward a floor of about 2 s).
- Both progressions are **capped** so the game stays playable given the ~100 ms gesture
  stabilization latency measured for this recognizer.

## 9. CNN Interaction Contract (Frozen)

This is the contract between the recognizer and the game, and it differs fundamentally from the
old endless-runner control model.

1. The recognizer produces one of `LEFT`, `RIGHT`, `UP`, `DOWN` per frame, with a softmax
   confidence.
2. **If confidence is below the threshold, no command is emitted at all.** The recognizer reports
   "no new command"; it does not invent a neutral state and it does not clear the player's
   current direction.
3. A command is a **direction request**, never a one-frame movement impulse. `LEFT` means
   "please travel left from the next legal opportunity", not "move one step left now".
4. The game holds the most recent request and applies it when maze geometry permits, per
   section 8.2. Until then the player continues in its current direction.
5. **No new command therefore means "keep going"**, which is the correct Pac-Man behaviour.
6. Temporal smoothing still applies: a direction must be predicted consistently across a short
   rolling window before it becomes a request, so a single misclassified frame cannot turn the
   player. When no class holds the majority, the result is "no new command".
7. Repeating the same direction request is harmless — it is idempotent — so a held gesture simply
   keeps the same request current. No edge-triggering is required, unlike the old one-shot JUMP.

## 10. Minimum Required Features

**Game side**

1. Tile-based maze with walls, pellets and power pellets
2. Continuously moving player with request-based turning
3. Four ghosts with scatter / chase / frightened / eaten states
4. Tunnel wraparound
5. Pellet and power-pellet consumption
6. Frightened mode and ghost eating with the 200/400/800/1600 chain
7. Collision handling, lives and life reset
8. Round completion and progression
9. Score display, lives display, Game Over and restart

**Deep learning side**

10. Webcam capture with the fixed ROI
11. CNN directional recognition (4 classes)
12. Confidence threshold with a "no new command" outcome
13. Temporal smoothing
14. On-screen display of the current detected direction and confidence

## 11. Out of Scope

Multiplayer, online play, accounts, databases, cloud services, level editors, story mode, bonus
fruit (first implementation), cut-scenes, original Pac-Man sprites, sounds, fonts or maze
artwork, exact arcade timing tables and target-tile arithmetic, additional gesture classes,
CNN-controlled menus, mobile builds, 3D graphics, voice control, full-body pose recognition, and
external hand-tracking libraries.

## 12. High-Level System Flow

```text
Player performs a hand gesture
            |
            v
Webcam captures frame  (horizontally flipped immediately - section 7)
            |
            v
Fixed 300x300 ROI is cropped and preprocessed
            |
            v
CNN classifies the gesture -> LEFT / RIGHT / UP / DOWN (+ confidence)
            |
            v
Below threshold, or no stable majority?  --> no new command
            |
            v
Direction request is stored
            |
            v
Game applies the request at the first legal tile; otherwise keeps the current direction
            |
            v
Player eats pellets, avoids or eats ghosts, clears the round
```

## 13. Expected Final Output

The player starts the application, sees the maze and a webcam panel, performs one of the four
gestures, sees the detected direction and confidence on screen, and watches the character turn
accordingly. They eat pellets, use power pellets to hunt ghosts, lose lives on contact, clear
rounds, and reach Game Over. The demo makes the chain visible end to end:

**Hand Gesture -> CNN Prediction -> Direction Request -> Maze Movement**

## 14. Technology Direction

- **Python 3.12.10** — main language, in the project-local `venv/`.
- **PyTorch 2.14.0 + torchvision 0.29.0 (CUDA 13.0 build)** — CNN definition, training,
  inference.
- **OpenCV 5.0.0** — webcam capture, flipping, ROI cropping, preprocessing.
- **Pygame 2.6.1** — maze rendering, game loop, timing.
- **NumPy, Matplotlib, scikit-learn, pyarrow** — arrays, plots, metrics, dataset preparation.
- **Dataset** — a balanced subset of [HaGRID](https://github.com/hukenovs/hagrid)
  (Kapitanov et al.), CC-BY-SA-4.0, taken from `cj-mills/hagrid-sample-500k-384p` together with
  the official `ann_train_val` bounding-box annotations bundled in that archive. Images are
  cropped to the annotated hand box with 25% padding and squared off, so training data resembles
  the square webcam ROI.

## 15. GPU Policy (Unchanged)

CNN training and inference run on CUDA, on the NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM,
compute capability 8.6). Deep-learning code uses `device = torch.device("cuda")` explicitly and
**never silently falls back to the CPU**: if CUDA is unavailable it raises a clear error. Because
VRAM is limited, the model stays lightweight (MobileNetV2), batch sizes stay conservative, and
only the current batch is transferred to the GPU. OpenCV capture, Pygame rendering and file I/O
stay on the CPU.

## 16. Preprocessing and Data Pipeline

Carried over unchanged except for the class set. **Implemented and verified in P3.**

- **Model input:** `3 x 160 x 160` RGB.
- **Class mapping (new, frozen):** `left=0`, `right=1`, `up=2`, `down=3`. Never rely on
  alphabetical folder ordering.
- **Split:** 80/10/10, stratified per class, seed 42, persisted to `data_splits.json`.
- **Normalization:** ImageNet `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
- **Training augmentation (training split only):** horizontal flip p=0.5, rotation +/-10 degrees,
  translation up to 5%, scale 0.9-1.1, brightness/contrast jitter 0.2. **No vertical flip and no
  large rotation**, for the reason given in section 6. `check_data_pipeline.py` walks the composed
  transforms and fails if a vertical flip appears or the rotation exceeds 10 degrees, so the rule
  cannot be lost to a later edit.
- **Loaders:** batch 32, train shuffled, validation and test in fixed order, 2 workers,
  pinned memory, persistent workers.
- **Validation and test:** resize, tensor, normalize only — fully deterministic.
- **Inference contract:** webcam code reuses the evaluation transform exactly, via
  `data_pipeline.inference_transform()`.

## 17. Model and Training Method

MobileNetV2 with ImageNet pretrained weights, final classifier replaced by a 4-output layer
following the frozen mapping. Two-stage transfer learning: stage 1 trains the head with the
feature extractor frozen (AdamW, lr 1e-3, 5 epochs); stage 2 fine-tunes `features[14:]` plus the
classifier (AdamW, lr 1e-4, up to 8 epochs, patience 3). CrossEntropyLoss, weight decay 1e-4,
float16 mixed precision, best checkpoint selected on **validation loss** with validation accuracy
as the tie-break. The test split is never loaded during training.

**Trained in P4:** `model/best_direction_model.pt`, 2,228,996 parameters, best validation
accuracy 0.9950 (199/200) at validation loss 0.0362, stage 2 epoch 8. Per class: left 49/50,
right 50/50, up 50/50, down 50/50 — zero `up`/`down` confusion in either direction.

Every checkpoint records its own class mapping, and `load_direction_checkpoint()` refuses one
that disagrees with the active mapping. This matters because the retired endless-runner
checkpoint also has four outputs and would otherwise load without error while meaning
`jump` at index 2. It is archived in `model/archive_endless_runner/`.

## 18. Migration Status

| Component | Status | Notes |
|---|---|---|
| Python / CUDA / PyTorch environment | **Reusable as-is** | Nothing about it depends on the game or class set |
| Project structure, `src/` layout, README/spec workflow | **Reusable as-is** | |
| Webcam ROI, mirror convention, capture geometry | **Reusable as-is** | Independent of which gestures are used |
| HaGRID source, annotation cropping approach | **Reused** (P2) | `like` (27,721 images) and `dislike` (28,537) are present in the same archive and annotation set |
| `dataset/left` (fist), `dataset/right` (palm) | **Reused as-is** (P2) | 500 images each, verified and kept untouched; the mapping is unchanged |
| `dataset/up` (like), `dataset/down` (dislike) | **Built** (P2) | 500 annotation-aligned crops each, from their own genuine HaGRID classes |
| `dataset/jump` (peace), `dataset/neutral` (no_gesture) | **Obsolete** — archived (P2) | Moved to `dataset_archive_endless_runner/`, outside the active dataset |
| `src/crop_hagrid_hands.py` | **Updated** (P2) | New `GESTURE_FOR` mapping, a `--classes` flag, and a duplicate guard against images already in `dataset/` |
| `src/check_dataset.py` | **Updated** (P2) | New class set, exact-count and stray-folder checks, up/down orientation QA sheet |
| `src/prepare_hagrid_dataset.py` | **Retired** (P2) | Built from a geometrically cropped source whose official boxes do not align; now refuses to run |
| `src/collect_dataset.py` | **Updated** (P2) | Optional tool; on-screen gesture hints and class keys only |
| `src/data_pipeline.py` | **Updated** (P3) | `CLASS_TO_INDEX` is now left/right/up/down; transforms, split logic and loader settings unchanged |
| `src/check_data_pipeline.py` | **Updated** (P3) | New mapping, split-regeneration, transform-safety, per-class and loader-config checks; up/down augmentation sheet |
| `data_splits.json`, `class_mapping.json` | **Regenerated** (P3) | Four directions, 1600/200/200, seed 42; old copies archived |
| `src/train_model.py` | **Reusable as-is** | Class names are imported, not hardcoded |
| `src/evaluate_model.py` | **Updated** (P5) | Directional checkpoint and artifact names, metadata/split verification, inference mode, confidence analysis, latency distribution |
| `model/best_gesture_model.pt` and all metrics | **Replaced** (P4) | Archived to `model/archive_endless_runner/`; superseded by `best_direction_model.pt` |
| `src/train_model.py` | **Updated** (P4) | New checkpoint/history/curve paths, validation-loss selection, self-describing checkpoints and a load guard, per-class validation breakdown, training curves |
| `src/gesture_recognizer.py` | **Updated** (P6) | `DirectionRecognizer`; `NEUTRAL` replaced by a `None` no-command sentinel; loads via the P4 guard; smoothing counts only real directions |
| `src/realtime_gesture.py` | **Updated** (P6) | New overlay, directional trial recorder, no-hand/idle recorder, transition stress test with command-latency measurement |
| Live trial results and the 0.90 threshold | **Redone and frozen** (P6) | Re-measured for the four directions: threshold 0.90 and 3-of-5 smoothing retained because live evidence supports them, not by inheritance |
| `game/game.py`, `game/main.py` (endless runner) | **Archived** (P7) | Moved to `game_archive_endless_runner/`; `game/` now holds only the Pac-Man implementation (maze, controls, entity, player, ghost, game, main) |
| `src/game_integration.py` | **New** (P8 / Objective 9) | Recognition worker thread, single-slot shared snapshot, gesture-to-request bridge with a 0.75 s stale timeout |
| `src/play_gesture.py` | **New** (P8 / Objective 9) | The integrated application: game, diagnostic camera panel, `--no-camera` and `--benchmark` modes |
| `src/check_integration.py` | **New** (P8 / Objective 9) | 28 headless integration checks, including the 30/60 FPS cadence, buffered turns and clean shutdown |
| `src/play_gesture.py` | **Reworked** (Objective 10) | Start screen, final gesture panel, pure display functions for every status, pre-rendered static text |
| `game/game.py` HUD and banners | **Reworked** (Objective 10) | Overlapping HUD row fixed, scheduler-mode debug readout removed, `hud_text()` and `banner_lines()` added; no rule, timing or physics change |
| `src/check_final_application.py`, `FINAL_TEST_REPORT.md` | **New** (Objective 11) | 27 end-to-end application checks and the final test record; no application code changed |
| `src/check_ui.py` | **New** (Objective 10) | 21 headless interface checks: every status wording, every banner state, and proof that drawing changes nothing about the game |

**Net effect:** the environment, tooling and method survived intact; the dataset, the trained model,
the live tuning, the game and the integration between them have all been rebuilt for the four
directions and verified.

## 19. Revised Roadmap

Phases are renamed `P1`-`P8` so they cannot be confused with the old objective numbers.

| Phase | Title | Summary |
|---|---|---|
| **P1** | Freeze the Pac-Man revision | This document. Redesign, migration analysis, roadmap. **Complete.** |
| **P2** | Rebuild the dataset for four directions | Kept `left`/`right`; built `up` from HaGRID `like` and `down` from `dislike` with the existing annotation-cropping script; 500 per class, 2000 total, balanced; integrity check and visual inspection passed. **Complete.** |
| **P3** | Preprocess and split | Frozen class mapping updated to left/right/up/down, `data_splits.json` and `class_mapping.json` regenerated (1600/200/200, seed 42, byte-reproducible), tensors, determinism, CUDA transfer and the absence of any vertical flip all verified. **Complete.** |
| **P4** | Retrain MobileNetV2 for four directions | Same two-stage method from ImageNet weights; best validation accuracy 0.9950, loss 0.0362, no up/down confusion; checkpoint, history and curves written; test split untouched. **Complete.** |
| **P5** | Evaluate the directional model | Frozen checkpoint evaluated once on the untouched 200-image test split: accuracy 0.9900 (198/200), loss 0.0292, macro F1 0.9900. Two errors: `right`->`down` at 0.60 confidence and `down`->`up` at 0.90. Batch-1 CNN latency 6.04 ms. **Complete.** |
| **P6** | Real-time directional recognition | Recognizer emits a direction or `None` (no new command). Live trials: 80/80 held gestures, 0/40 idle false commands, 0/6 spurious transitions at natural speed. Threshold 0.90 and 3-of-5 smoothing confirmed by evidence and frozen; command latency 143 ms mean. **Complete.** |
| **P7** | Build the Pac-Man game | Maze (28x31, original layout), pellets, four ghosts with the full state machine, lives, scoring, rounds, tunnel; keyboard-only through the `request_direction` seam; 58 automated checks pass; 346 FPS headless headroom; manual play-test approved, including that the game stays playable at the ~150 ms gesture latency. **Complete.** |
| **P8** | Integrate and test *(renumbered **Objective 9**)* | The recognizer drives the game through the same `request_direction` seam the keyboard uses. Two decoupled loops: game 59.5 FPS on the main thread, one recognition worker at 29.6 FPS owning the webcam and the model; they share a single immutable snapshot behind a lock, never a queue. `None` issues no command and never stops Pac-Man; a 0.75 s stale timeout prevents a sticky direction; camera failure degrades to keyboard. 28 integration checks pass, with P6 (15/15) and P7 (58/58) still green and the checkpoint byte-identical. Manual sanity test approved. **Complete.** |

Each phase ends with a verification step and a stop, as in the previous workflow.

Work continues under the objective numbering that replaced `P8`.

| Objective | Title | Summary |
|---|---|---|
| **10** | Final user interface | One 824x704 window: the maze at its native size with a 264 px gesture panel beside it, never over it. A start screen explains the four gestures and waits for SPACE or ENTER while the model loads behind it, so the recognizer keeps a single lifecycle. The panel shows the ROI the CNN sees, the stable command with its confidence, the four mappings with the live one highlighted, and gesture-control health (`READY` / `WAITING` / `OFF` / `CAMERA ERROR`); the raw per-frame class stays in small diagnostic print. Every pause explains itself (`READY!`, `LIFE LOST`, `ROUND N CLEARED!`, a full `GAME OVER` panel). 21 UI checks pass, 60.1 FPS live, and drawing was proved to leave the game's physics, score and lives bit-identical. Manual review approved. **Complete.** |
| **11** | Test the complete application | Final audit confirmed one canonical path (webcam → recognizer → `stable_command` → controller → `request_direction` → movement). 149/149 automated checks across five suites, including a new 27-check cross-subsystem suite that drives every whole-application flow through the gesture path. Live runs on the real webcam and CUDA model: clean start-screen, ESC and window-close shutdowns (~330–350 ms, one model load, one camera open/release, no threads left), and a 330 s stability run holding game 60 FPS / camera 30 FPS with constant snapshot age and zero backlog. Model, splits, mapping, history, metrics and dataset byte-identical before and after. No application defects found. Final manual acceptance approved. **Complete.** Results in `FINAL_TEST_REPORT.md`. |

### Interface principles fixed by Objective 10

The interface never presents a raw per-frame prediction as the command. Three values exist -
raw, thresholded and stable - and only the stable command steers the game, so only it is
given headline treatment. Confidence is printed only when a command is actually steering:
a percentage beside `NO COMMAND` or `WAITING` would imply the game is acting on a direction
it is deliberately ignoring.

All UI text is plain ASCII. The system fonts available here render hand emoji as
missing-glyph boxes, so the gesture legend uses `FIST`, `OPEN PALM`, `THUMBS UP` and
`THUMBS DOWN` rather than risking a broken-looking demonstration.

## 20. Project Status

**The project is COMPLETE.** Every phase (P1–P7) and objective (9–11) passed its automated
verification and its manual stop point. Final verified numbers: 99.00% held-out test accuracy,
threshold 0.90 with 3-of-5 smoothing, game 60 FPS beside a 30 FPS recognizer, 149/149
automated checks. The known out-of-vocabulary limitation (no reject class) remains documented
and accepted by design.

## 21. Revision Status

The Pac-Man revision is **frozen**. The game type, gameplay rules, gesture mapping, class set,
no-command behaviour, control contract, migration classification and roadmap above are final.
Any change requires an explicit decision to reopen this specification.
