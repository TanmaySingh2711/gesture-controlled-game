# Changelog

All notable changes to CNN Gesture Controlled Pac-Man. The project was built in numbered phases,
each verified before the next began; the entries below follow those phases.

## [Unreleased] - quality and rigour round

### Removed
- `pyarrow`, which only the retired parquet-based dataset script used. Removed from
  `requirements.txt`, CI and the mypy config; the full test suite passes without it installed.
- Dead code found by a dead-code scan: `Ghost.body_color()` and its colour constants
  (duplicated by the themed `Game.ghost_body_color()`, which is what actually draws, and is now
  what the test checks, in every theme), nine unused palette aliases in `engine.py`, and the
  unused `Maze.is_door()` / `Maze.is_house()`.
- Empty `game/.gitkeep` and `model/.gitkeep` files (both folders have tracked files), and
  TensorFlow-era patterns (`*.h5`, `*.keras`, `*.tflite`, `*.pb`, `*.ckpt`) from `.gitignore`.
- **Every remaining endless-runner artefact**, now that the project is fully the Pac-Man game:
  `game_archive_endless_runner/`, `dataset_archive_endless_runner/` (1,005 images) and
  `model/archive_endless_runner/` (the retired checkpoint and its README). The checkpoint's only
  remaining role was one narrow regression check (the loader refuses it by class mapping, not
  just by looking different); that guard is still tested, now with a small synthetic checkpoint
  instead of the retired file, so no coverage was lost. Also removed: the dead
  `GestureRecognizer` alias, leftover `model/` files from the old model's training run (fully
  superseded by the `direction_*`-prefixed Pac-Man files), the one-off `--quick` study report,
  and the four QA image sheets that regenerate on every run of the check scripts (now
  gitignored instead of committed).

### Added
- **Accessibility:** high-contrast and colour-blind-safe themes (`C`), checked by tests for WCAG
  contrast and for simulated protanopia, deuteranopia and tritanopia; frightened ghosts now also
  differ in shape, so no state depends on colour alone.
- **Gameplay:** pause (`P`), help overlay (`H` / `F1`), synthesised sound effects with mute (`M`),
  bonus fruit, score popups, a death animation, and a persisted high score.
- **Ghost AI:** the Chaser speeds up as the board empties, capped below Pac-Man's speed.
- **Robustness:** the recognition worker reopens a webcam that stops delivering frames before
  declaring `CAMERA ERROR`; capture-to-request latency is measured and reported.
- **Security:** the model loader verifies a pinned SHA-256, loads with `weights_only=True`, and
  still checks the class mapping. See `SECURITY.md`.
- **ML rigour:** Wilson and bootstrap confidence intervals, calibration (ECE, Brier, NLL) and a
  reliability diagram from the saved test predictions; an architecture and hyperparameter study on
  train and validation only; a HaGRID lineage audit for subject leakage and an unseen-subject set.
- **Engineering:** `pyproject.toml` packaging, ruff, mypy, a pytest suite with markers for GPU and
  webcam tests, CI, pre-commit hooks, `.gitattributes`, and `docs/` (architecture, manual test plan).
- **Model study results:** 42 runs on train and validation only (six backbones and nine MobileNetV2
  recipes, three seeds each), written up in `docs/MODEL_STUDY.md`. No backbone was clearly better
  than MobileNetV2, and the frozen run sits inside its seed range.
- **Memory profiling:** `src/measure_memory.py` and `reports/memory_profile.json`. The game uses
  47 MB and stays flat over 10 simulated minutes. The recognizer peaks at 22 MB of GPU memory, with
  no growth over 5,000 frames.
- **Lineage and leakage audit:** all 2,000 dataset images were traced back to their HaGRID photo
  and person. 41% of test images (82/200) come from people also in training, because the split
  was per image.
- **Unseen-person evaluation:** `src/evaluate_external.py` evaluated the frozen model once on
  2,000 images from 1,727 people with no image in the dataset. It scored 99.6% (95% interval
  99.2%-99.8%), not significantly different from P5 (Newcombe interval on the difference: -0.2
  to +3.2 points). Familiar hands did not inflate the reported accuracy.
- **Tests:** ghosts, player and movement, engine rules, the audio mixer, the data pipeline, the
  recognizer's threshold and vote, the memory profiler and the external evaluation. Test coverage
  rose from 39% to 71% across `game/` and `src/` (365 tests), and every game module is at 97-100%.
- The standalone self-test scripts now run under pytest, each in a fresh interpreter, with their
  coverage measured through coverage's subprocess patch. CI's separate self-test step became
  part of the test run.

### Changed
- **Every module is fully type-annotated**, and mypy now enforces it project-wide
  (`disallow_untyped_defs`), not just in the game and application modules. Doing so surfaced and
  fixed several real gaps: an optimizer or gradient scaler that could be `None` during training,
  a missing sample image that would have crashed the recognizer self-test instead of failing it,
  and check scripts that indexed a banner or controller without handling its absence.
- Pillow is pinned in `requirements.txt` and CI, since the code imports it directly.
- The QA image sheets written by `check_dataset.py` and `check_data_pipeline.py` go to
  `reports/qa/` (gitignored) instead of the project root.
- `.gitignore` rewritten into clear groups, with accurate comments.
- `game/game.py` is now `game/engine.py`, and every module uses package imports instead of
  editing `sys.path`.
- The static maze is pre-rendered with connected wall outlines: a headless frame went from 2.9 ms
  to 0.6 ms.
- Long functions in the training, evaluation, cropping, collection and live-trial scripts were
  split into tested pieces; evaluation and training can now be exercised end to end without
  touching the test split or the frozen checkpoint.
- `prepare_hagrid_dataset.py` moved to `archive/retired_scripts/`.

### Fixed
- `check_data_pipeline.py`'s split-regeneration check **overwrote the frozen `data_splits.json`**
  on every run, and failed on Windows after a fresh checkout because the rebuild wrote CRLF line
  endings where git had checked out LF. The split writer now always writes LF, and the check
  rebuilds into a temporary folder and compares, so the committed files are never touched.
  Generated JSON reports are likewise written with LF and a final newline on every platform.
- Ghosts were seeded with `hash(role)`, which Python randomises per process, so frightened ghosts
  behaved differently on every launch.
- `python game/main.py --fps` crashed after the package rename.
- The "game never imports the CNN" check could no longer see imports from the `src` package.
- Snapshot freshness treated a timestamp of `0.0` as missing.
- The help screen claimed sound was on when no audio device existed.
- A bug introduced during this round's refactor of `evaluate_model.py`: the figure functions named
  their loop variable after their output-path parameter, so saving a figure overwrote the last
  dataset image it displayed. It damaged three images (`left_00135`, `up_00142`, `down_00078`)
  during test runs, never during the frozen P5 evaluation, whose code saved to fixed paths. Each
  image was restored byte-for-byte by replaying the deterministic crop walk; the source was accepted
  only when exactly one candidate lay between the file's intact neighbours. Study runs made while
  the damage was present were discarded, and the evaluation test now fails if any dataset image
  changes.

- The lineage audit's leakage report read the split file's `counts` block (also keyed
  train/val/test) instead of the real splits, and reported zero images everywhere. Found on its
  first real run, fixed, recomputed from the saved lineage, and covered by a regression test.

### Known issues
- **Two gameplay additions from this round go beyond the frozen `PROJECT_SPEC.md`, and a decision
  is pending.** Bonus fruit is listed as out of scope for the first implementation (sections 8.4
  and 11), and any later addition must be recorded in the spec with its values. The Chaser's
  end-of-round speed-up is not described in sections 8.3 or 8.6. Neither was preceded by the
  explicit decision the spec requires before it is reopened. Both are tested and capped (the
  Chaser never outruns Pac-Man), but they stay unapproved until that decision is made: either
  reopen the spec to record them, or remove them. The other additions (pause, help, colour
  themes, synthesised sound, persisted high score) are interface features, not gameplay rules.
  Original synthesised effects are not the "original Pac-Man sounds" section 11 excludes.
- `square_crop` in `crop_hagrid_hands.py` rounds the window's position and size separately, so
  in rare edge cases a crop comes out one pixel narrower than it is tall. A property-based test
  found it. None of the 2,000 dataset images is affected (a test checks every one is exactly
  square). The geometry is deliberately left unchanged, because the dataset, the frozen model and
  the lineage audit's byte-for-byte replay depend on its exact output. The quirk is pinned by a
  regression test, so any change to it has to be deliberate.

## Objective 11 - Final application testing

149 automated checks, live start-up, shutdown and 330-second stability runs on the real webcam and
GPU, integrity of every ML artefact verified, and final manual acceptance.

## Objective 10 - Final user interface

Start screen, gesture panel with the camera region, stable command and confidence, and banners for
every game state.

## Objective 9 (P8) - Gesture integration

A recognition worker thread publishing a single latest-state snapshot; gestures and the keyboard
share one direction-request seam; camera failure falls back to the keyboard.

## P7 - The Pac-Man game

Original 28x31 maze, four ghosts with distinct targeting over one shared state machine, scoring,
lives, rounds and a tunnel, playable on the keyboard with no CNN present.

## P6 - Real-time recognition

Threshold 0.90 and 3-of-5 smoothing, both frozen from live webcam trials: 80/80 held gestures,
0/40 idle false commands, 143 ms mean command latency.

## P5 - Evaluation

99.00% accuracy (198/200) on the untouched test split, evaluated exactly once.

## P4 - Training

MobileNetV2 two-stage transfer learning; 99.5% validation accuracy; checkpoints carry their class
mapping and refuse to load under a different one.

## P3 - Data pipeline

Deterministic 1600/200/200 split and augmentation verified free of vertical flips.

## P2 - Dataset

2,000 annotation-aligned HaGRID hand crops, 500 per direction.

## P1 - Project revision

The endless-runner design was replaced by the Pac-Man direction game; the old code and data were
archived rather than deleted.
