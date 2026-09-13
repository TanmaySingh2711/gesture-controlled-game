# CNN-Based Gesture Controlled Gaming Application

A **Pac-Man-style maze game** controlled by hand gestures recognized in real time by a CNN from a webcam feed.

> **Project revised.** The project was previously a 2D endless runner with LEFT/RIGHT/JUMP/NEUTRAL gestures. It is now a Pac-Man-style maze game with directional control — **LEFT / RIGHT / UP / DOWN**, no NEUTRAL class. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the frozen specification and the migration analysis.

## Current Status

**Project COMPLETE.** The finished application passed final end-to-end testing and manual acceptance — see [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md).

```bash
python src/play_gesture.py              # the application
python src/play_gesture.py --no-camera  # keyboard only
```

| Final verified result | |
|---|---|
| Gestures | Fist → LEFT · Open palm → RIGHT · Thumbs up → UP · Thumbs down → DOWN |
| Held-out test accuracy | **99.00%** (198/200), macro F1 0.9900 |
| Live recognition | threshold 0.90, 3-of-5 smoothing, 300×300 mirrored ROI |
| Performance (330 s run) | game **60 FPS**, camera **30 FPS**, CNN 18 ms mean — no backlog, no lag growth |
| Automated checks | **149/149** pass across five suites |
| Known limitation | no reject class — deliberate unsupported gestures (e.g. peace sign) can be misread; use only the four |

| Phase | Status |
|---|---|
| **P1 — Freeze the Pac-Man revision** | **Complete** — spec, migration analysis and roadmap frozen |
| **P2 — Rebuild dataset for 4 directions** | **Complete** — 4 × 500 = 2000 hand crops; `left`/`right` reused, `up`/`down` built from HaGRID `like`/`dislike` |
| **P3 — Preprocess and split** | **Complete** — mapping `left=0 right=1 up=2 down=3`, deterministic 1600/200/200 split, augmentation verified vertical-flip-free |
| **P4 — Retrain MobileNetV2** | **Complete** — 99.50% validation accuracy (199/200), no up/down confusion |
| **P5 — Evaluate directional model** | **Complete** — 99.00% test accuracy (198/200), macro F1 0.9900 |
| **P6 — Real-time directional recognition** | **Complete** — 80/80 held gestures, 0/40 idle false commands, 143 ms command latency |
| **P7 — Build the Pac-Man game** | **Complete** — 58 automated checks pass, manual play-test approved |
| **Objective 9 — Integrate CNN with the game** | **Complete** — 28 integration checks pass, game 59.5 FPS beside a 29.6 FPS recognizer, manual sanity test approved |
| **Objective 10 — Final user interface** | **Complete** — start screen and gesture panel, 21 UI checks pass, 60.1 FPS, manual review approved |
| **Objective 11 — Test the complete application** | **Complete** — 149/149 checks, live shutdown and 330 s stability runs, artifacts unchanged, final acceptance approved |

## Game

A Pac-Man-style maze game in Pygame, keyboard-controlled, with no CNN or webcam dependency — verified by an AST import scan and by asserting `torch`/`cv2` never enter `sys.modules`.

```bash
python game/main.py              # play
python game/main.py --selftest   # 58 headless rule checks
python game/main.py --fps 3000   # frame-rate headroom probe
```

| | |
|---|---|
| Maze | 28 x 31 tiles, 20 px each, original layout, 330 pellets + 4 power pellets |
| Controls | Arrows / WASD, `R` restart, `ESC` quit |
| Player | 6.2 tiles/s, turns buffered for 0.35 s |
| Ghosts | 5.4 tiles/s base, +0.2 per round, capped 6.0 — always below the player |
| Scoring | pellet 10, power pellet 50, ghost chain 200/400/800/1600, extra life at 10,000 (once) |
| Ghost roles | Chaser, Ambusher, Flanker, Drifter — separate targeting, one shared state machine |
| Ghost states | HOUSE, SCATTER, CHASE, FRIGHTENED, EATEN |
| Schedule | scatter 7 / chase 20 / 7 / 20 / 5 / 20 / 5, then permanent chase |
| Frightened | 6.0 s in round 1, −0.5 s per round, floor 2.0 s |
| Performance | 346 FPS uncapped headless, 2.89 ms per frame against a 16.67 ms budget |

**The control seam** is the whole point of the architecture. Keyboard code never moves the player; it only calls `request_direction`, and P8 will call exactly the same method from `DirectionRecognizer.stable_command`:

```python
game.request_direction("left")     # identical effect to pressing the Left arrow
```

Requests are buffered for 0.35 s, so a turn asked for slightly before an intersection still fires there — sized against the 217 ms p95 command latency measured in P6.

Play-tested and approved: visuals, all four directions, pre-turning, tunnel, ghost behaviour, power pellets, life loss and restart all confirmed working, and the game was judged playable at the ~150 ms gesture latency P8 will introduce.

Carried over intact: Python 3.12.10 + CUDA PyTorch environment · HaGRID annotation-cropping pipeline · the existing `left` and `right` images · 160×160 preprocessing · two-stage MobileNetV2 training method · webcam ROI and mirror convention.

Superseded: the old checkpoint and all its metrics (wrong output classes, archived in `model/archive_endless_runner/`) · the 0.90 live threshold (different gestures, different fallback rule) · the endless-runner game in `game/`.

## Model

MobileNetV2 with ImageNet weights, classifier replaced by `Linear(1280, 4)` — 2,228,996 parameters. Two-stage transfer learning on the RTX 3050 Ti with float16 mixed precision: stage 1 trains the head alone (5 epochs, AdamW, lr 1e-3), stage 2 fine-tunes `features[14:]` plus the classifier (8 epochs, lr 1e-4, patience 3). The best checkpoint is selected on **validation loss** — the test split is never loaded during training.

| Result | Value |
|---|---|
| Best validation accuracy | **0.9950** (199/200) |
| Best validation loss | **0.0362** |
| Best epoch | stage 2, epoch 8 |
| Per-class validation | left 49/50, right 50/50, up 50/50, down 50/50 |
| `up`↔`down` confusion | **0 in each direction** |
| Training time | 63 s, peak 161 MB VRAM |

```bash
python src/train_model.py              # train + verify the checkpoint reload
```

The checkpoint is [model/best_direction_model.pt](model/best_direction_model.pt), history in `model/direction_training_history.json`, curves in `model/direction_training_curves.png`. It stores its own class mapping, and `train_model.load_direction_checkpoint()` refuses any checkpoint whose mapping disagrees with the active one — the obsolete four-output endless-runner model would otherwise load cleanly and silently report `jump` as `up`.

## Evaluation

Final offline evaluation of the frozen checkpoint on the 200-image test split, run once, on CUDA under `torch.inference_mode()` with the deterministic P3 transform.

| Metric | Value |
|---|---|
| Test accuracy | **0.9900** (198/200) |
| Test loss | **0.0292** |
| Macro F1 | **0.9900** |
| Weighted F1 | **0.9900** |
| Per-class recall | left 1.00, right 0.98, up 1.00, down 0.98 |
| Errors | `right`→`down` (conf 0.60), `down`→`up` (conf 0.90) |
| CNN latency, batch 1 | 6.04 ms mean, 6.88 ms p95 (~166 predictions/sec) |

```bash
python src/evaluate_model.py           # test metrics, confusion matrix, predictions CSV
```

Artifacts: `model/direction_evaluation_metrics.json`, `direction_confusion_matrix.png`, `direction_test_predictions.csv`, `direction_misclassified_samples.png`, `direction_low_confidence_correct.png`.

> **This measures held-out HaGRID hand crops, not live webcam reliability.** That was established separately in P6 from webcam trials; the live threshold is **not** derived from these numbers.

## Real-Time Recognition

[src/gesture_recognizer.py](src/gesture_recognizer.py) turns a mirrored webcam frame into a direction request. No Pygame, no game logic — P7/P8 consume it directly.

```python
recognizer = DirectionRecognizer()
result = recognizer.predict(mirrored_frame)
result.stable_command      # "left" / "right" / "up" / "down" / None  <- act on this
result.stable_changed      # True on the frame the command just changed
result.raw_direction, result.raw_confidence, result.thresholded_direction
result.inference_ms
```

`None` means **no new command** — not "stop". Pac-Man keeps travelling the way it already was. It is never a class and never appears in `CLASS_TO_INDEX`.

**Frozen live configuration** (chosen from webcam trials, never from the test set):

| Setting | Value |
|---|---|
| Confidence threshold | **0.90** |
| Temporal smoothing | **3-of-5** frames |
| ROI | 300×300 at x 300–600, y 90–390 of a mirrored 640×480 frame |
| Webcam frame rate | ~28–31 FPS |
| CNN latency, live | 12.5 ms mean, 18.5 ms p95 |
| Command latency | 143 ms mean, 217 ms p95 |

**Live results:** held gestures **80/80** (left 20/20, right 20/20, up 20/20, down 20/20), zero UP↔DOWN confusion. Idle hand and empty box **40/40 correct — zero stable false commands**. Natural-speed transitions **0/6 spurious**.

The threshold works because the two confidence distributions do not overlap: intentional gestures never fell below **0.961**, while idle and no-hand frames never exceeded **0.617**.

```bash
python src/realtime_gesture.py               # live preview
python src/realtime_gesture.py --selftest    # 15 headless checks
python src/realtime_gesture.py --trials      # guided held-gesture recorder
python src/realtime_gesture.py --idle        # no-hand / idle-hand recorder
python src/realtime_gesture.py --transitions # transition stress test + latency
```

> **Known limitation.** The CNN has no reject class, so a *deliberate, well-formed* gesture outside the four commands can be confidently misread — a peace sign reads as RIGHT at 99.3%, which no threshold can catch. Relaxed and absent hands are handled correctly; only unfamiliar deliberate shapes are at risk. A player has no reason to form one mid-game, so this is documented rather than fixed.

## Gesture Mapping

| Command | Gesture | HaGRID class |
|---|---|---|
| **LEFT** | Closed fist | `fist` |
| **RIGHT** | Open palm | `palm` |
| **UP** | Thumbs up | `like` |
| **DOWN** | Thumbs down | `dislike` |

There is **no NEUTRAL class**. Pac-Man is always moving, so "no input" means *keep going*, not *stop*. When confidence is below threshold the recognizer emits **no new command** and the character continues in its current direction. Commands are **direction requests**, applied at the first legal tile — not one-frame impulses.

## Gesture-Controlled Play

The finished application: the CNN drives the maze game from a webcam, with the keyboard still
live alongside it.

```bash
python src/play_gesture.py                 # gesture control; keyboard still works
python src/play_gesture.py --no-camera     # keyboard only, loads no CNN
python src/play_gesture.py --benchmark 900 # timed run with an integration report
python src/check_integration.py            # 28 headless integration checks
python src/check_ui.py                     # 21 headless interface checks
```

**The interface.** One 824 x 704 window: the maze at its native 28 x 31 tiles on the left, a
264 px gesture panel beside it — added next to the maze, never over it. A start screen
explains the four gestures and waits for SPACE or ENTER while the model loads behind it, so
the recognizer has one lifecycle and never reloads.

The panel shows what the CNN actually sees (the 300 x 300 ROI), the **stable command** in
large type with its confidence, the four mappings with the live one highlighted, and the
health of gesture control: `READY`, `WAITING`, `OFF` or `CAMERA ERROR`. The raw per-frame
prediction stays in small diagnostic print — the game acts only on the smoothed command, and
the interface says so. Confidence is shown only when a command is actually steering, never
beside `NO COMMAND` or `WAITING`.

The game area keeps `SCORE`, `ROUND`, `LIVES` and the live command, and each pause explains
itself: `READY!`, `LIFE LOST`, `ROUND N CLEARED!`, and a `GAME OVER` panel with the final
score, round reached and how to restart or quit.

**Two loops that never wait for each other.** The game runs at ~60 FPS on the main thread; a
single `RecognitionWorker` thread owns the webcam and the model and runs at camera speed. The
game never calls `read()` and never runs inference, so a slow camera frame cannot slow the
game down.

They communicate through **one immutable snapshot behind a lock — latest state, never a
queue**. Old camera frames have no gameplay value, so there is nothing that can accumulate.

| Measured live | |
|---|---|
| Game | 59.5 FPS, frame p95 17.0 ms |
| Camera | 29.6 FPS |
| CNN inference | mean 19.1 ms, median 17.8, p95 28.7 |
| Backlog | none — publishes match recognitions exactly |

**Gesture and keyboard are the same input.** Both call `game.request_direction`; neither can
move Pac-Man directly, and the most recent request simply wins. A held gesture re-requests its
direction every frame, so the intent stays armed until the turn becomes legal — which is what
makes pre-turning work at a ~150 ms recognition latency.

`None` means *no new command*, never *stop*: Pac-Man carries on and the game's own 0.35 s
request buffer expires the stale request. A snapshot older than **0.75 s** is ignored entirely,
so a dead worker cannot leave an old direction stuck on. If the webcam fails, the error is
logged once, the panel reads `CAMERA ERROR`, and the keyboard keeps working.

## Python Version

Python **3.12.10** (64-bit), in the project-local `venv/`.

## Main Technologies

| Purpose | Library |
|---|---|
| CNN definition, training, inference | PyTorch 2.14.0 + torchvision 0.29.0 (CUDA 13.0 build) |
| Webcam capture and image preprocessing | OpenCV 5.0.0 |
| Game window, rendering, physics | Pygame 2.6.1 |
| Numerical arrays | NumPy 2.5.3 |
| Training plots | Matplotlib 3.11.2 |
| Data splitting and metrics | scikit-learn 1.9.1 |

## GPU Acceleration

CNN training and inference run on CUDA, on the **NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM, compute capability 8.6)**.

Project rule: deep-learning code uses `device = torch.device("cuda")` explicitly and **never silently falls back to the CPU**. If CUDA is unavailable, training and inference raise a clear error. Because VRAM is limited to 4 GB, the CNN stays lightweight, batch sizes stay conservative, and only the current batch is moved to the GPU. OpenCV capture, Pygame rendering, and file I/O stay on the CPU.

## Setup

Activate the virtual environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# Git Bash
source venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` carries the official PyTorch CUDA index directive, so this installs the GPU build. Installing `torch` from plain PyPI would give a CPU-only build and is not supported here. The equivalent manual command is:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install opencv-python pygame numpy matplotlib scikit-learn
```

## Verify the Environment

```bash
python src/environment_check.py
```

Prints a PASS/FAIL line for the Python version, every required library, CUDA availability, the detected GPU, a real CUDA tensor allocation and matrix multiplication, and webcam availability. Add `--skip-webcam` on a machine with no camera.

## Dataset

Source: a subset of **HaGRID** (HAnd Gesture Recognition Image Dataset, Kapitanov et al.), CC-BY-SA-4.0, taken from [`cj-mills/hagrid-sample-500k-384p`](https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p) on Hugging Face together with the official `ann_train_val` bounding-box annotations bundled in that archive.

Every image is a **hand-region crop**: the official HaGRID box for the required gesture, padded 25% on each side and expanded to a square. This makes the training images resemble the tight square webcam ROI used during gameplay, instead of full room-and-person scenes.

| Command | HaGRID class | Gesture | Images | Origin |
|---|---|---|---|---|
| `left` | `fist` | Closed fist | 500 | reused unchanged |
| `right` | `palm` | Open palm | 500 | reused unchanged |
| `up` | `like` | Thumbs up | 500 | built in P2 |
| `down` | `dislike` | Thumbs down | 500 | built in P2 |

**2000 images, four classes, exactly balanced.** `up` and `down` come from their own genuine HaGRID classes — neither is ever produced by flipping or rotating the other, because they are the same hand rotated roughly 180°.

The old `jump` (peace) and `neutral` (no_gesture) folders were moved to `dataset_archive_endless_runner/` so that no script can mistake them for active classes.

```bash
python src/crop_hagrid_hands.py --promote               # build all four classes
python src/crop_hagrid_hands.py --classes up down       # build only some of them
python src/check_dataset.py --grid --updown-grid        # integrity check + QA sheets
```

`crop_hagrid_hands.py` reads the 13.4 GB source archive over HTTP range requests, so it transfers only the annotations and the ~2000 images it actually uses (~210 MB), never the whole archive. Seed 42 makes the selection reproducible.

Crops are kept raw at their natural size (96–384 px square). Resizing, normalization, augmentation and splitting belong to P3.

Two QA sheets are produced: [dataset_sample_grid.jpg](dataset_sample_grid.jpg) covers all four classes, and [updown_orientation_grid.jpg](updown_orientation_grid.jpg) puts many thumbs-up and thumbs-down samples side by side, since that pair is the one a mis-crop would be hardest to spot in.

`src/prepare_hagrid_dataset.py` built an earlier full-scene version from a source whose images were geometrically cropped, so the official boxes did not align with them. It is retired and now refuses to run; `crop_hagrid_hands.py` is the only supported way to build the dataset.

## Data Pipeline

Preprocessing happens at run time through torchvision transforms — the 2000 images in `dataset/` are never modified or duplicated.

| Setting | Value |
|---|---|
| Model input | `3 × 160 × 160` RGB |
| Class mapping (frozen in P3) | `left=0, right=1, up=2, down=3` |
| Split | 80/10/10 stratified, seed 42 → 1600 / 200 / 200 |
| Per class | 400 train, 50 val, 50 test |
| Normalization | ImageNet `mean [0.485, 0.456, 0.406]`, `std [0.229, 0.224, 0.225]` |
| Batch size | 32 |
| Augmentation | **training split only** — h-flip 0.5, ±10° rotation, 5% translate, 0.9–1.1 scale, 0.2 brightness/contrast. **Never vertical flip, never 90°/180° rotation** — either would turn thumbs-up into thumbs-down. `check_data_pipeline.py` walks the composed transforms and fails if one appears |
| Validation / test | resize + tensor + normalize only, fully deterministic |

```bash
python src/data_pipeline.py            # (re)build data_splits.json + class_mapping.json
python src/check_data_pipeline.py      # split, tensor, determinism and CUDA checks + grid
```

[data_splits.json](data_splits.json) and [class_mapping.json](class_mapping.json) are the canonical split and mapping, regenerated in P3 for the four directions. Rebuilding with seed 42 reproduces the split file byte-for-byte. The stale endless-runner copies were archived to `dataset_archive_endless_runner/pipeline/`.

Horizontal flip is kept deliberately: mirroring changes handedness and viewpoint but a mirrored thumbs-up still points up, and a mirrored fist is still a fist. Two QA sheets record this — [augmentation_sample_grid.jpg](augmentation_sample_grid.jpg) for all four classes and [augmentation_updown_grid.jpg](augmentation_updown_grid.jpg) for the thumbs-up/thumbs-down pair specifically. Both use training-split images only, so the test set stays reserved for P5.

**Inference contract:** webcam code must reuse `data_pipeline.inference_transform()` — flip → 300×300 ROI → BGR→RGB → resize 160 → tensor → ImageNet normalize. Training augmentation is never applied at inference. See §16 of [PROJECT_SPEC.md](PROJECT_SPEC.md).

`src/collect_dataset.py` is an **optional** webcam capture tool kept for custom sanity-checking and as the reference for the gameplay capture geometry (flip, then crop the fixed **300×300 ROI at x 300–600, y 90–390** of the 640×480 frame). It is not part of the training-data workflow.

## Project Structure

```text
gesture-controlled-game/
├── dataset/          # active four-class dataset, 500 images each
│   ├── left/         # fist    - closed fist
│   ├── right/        # palm    - open palm
│   ├── up/           # like    - thumbs up
│   └── down/         # dislike - thumbs down
├── dataset_archive_endless_runner/   # retired jump/ and neutral/, not read by anything
├── model/            # checkpoint, history, metrics, plots
├── game/             # Pac-Man maze game - no CNN, no webcam, playable on its own
│   ├── maze.py       # 28x31 layout, pellets, tunnel, reachability
│   ├── controls.py   # the request_direction seam, 0.35 s turn buffer
│   ├── entity.py     # shared grid movement
│   ├── player.py
│   ├── ghost.py      # four roles, one navigation rule, full state machine
│   ├── game.py       # rules, scoring, rounds, drawing
│   └── main.py       # keyboard launcher + 58 self-tests
├── game_archive_endless_runner/      # retired Dino runner, not read by anything
├── src/              # reusable Python modules
│   ├── environment_check.py
│   ├── prepare_hagrid_dataset.py  # retired, refuses to run
│   ├── crop_hagrid_hands.py
│   ├── check_dataset.py
│   ├── data_pipeline.py
│   ├── check_data_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── gesture_recognizer.py  # reusable recognition class
│   ├── realtime_gesture.py    # webcam demo + trial recorder
│   ├── game_integration.py    # recognition worker, shared state, gesture bridge
│   ├── play_gesture.py        # the integrated application
│   ├── check_integration.py   # 28 headless integration checks
│   ├── check_ui.py            # 21 headless interface checks
│   ├── check_final_application.py  # 27 end-to-end application checks
│   └── collect_dataset.py     # optional webcam tool
├── venv/             # virtual environment (not tracked)
├── data_splits.json  # frozen 80/10/10 split
├── class_mapping.json
├── FINAL_TEST_REPORT.md  # Objective 11 results
├── PROJECT_SPEC.md
├── requirements.txt
└── README.md
```

> All objectives are complete. The CNN Gesture Controlled Pac-Man application passed final testing, and the project is complete.
