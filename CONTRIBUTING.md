# Contributing

## Setting up

Python 3.12, 64-bit, and an NVIDIA GPU with a CUDA 13 driver for the gesture application.

```bash
python -m venv venv
venv\Scripts\activate                 # Windows  (source venv/bin/activate elsewhere)
pip install -r requirements.txt       # includes the CUDA build of PyTorch
pip install -r requirements-dev.txt   # ruff, mypy, pytest, coverage, pip-audit
pip install -e . --no-deps            # makes `game` and `src` importable everywhere
pre-commit install                    # optional: run the quality gates on every commit
```

## Quality gates

Every change should pass these before it is committed. CI runs the same ones.

```bash
ruff format .                      # formatting
ruff check .                       # lint, including security and bug-prone patterns
mypy game src tests                # static types
pytest                             # every test, including the standalone self-test scripts
pytest --cov=game --cov=src        # the same, with a coverage report
pytest -m "not slow"               # a quick loop while editing (skips training and self-tests)
```

`pytest` runs the self-test scripts (`game/main.py --selftest`, `check_integration.py`,
`check_ui.py`, `check_final_application.py`, `check_dataset.py` and `environment_check.py`)
through `tests/test_self_tests.py`, each in a fresh interpreter, since several of them assert that
torch and cv2 were never imported. Tests needing CUDA, the dataset images or the frozen checkpoint
skip themselves when those are missing, as they are in CI.

On a machine with the GPU and a webcam, also run:

```bash
python src/realtime_gesture.py --selftest     # recognizer checks against the live camera
GESTURE_WEBCAM_TESTS=1 pytest -m webcam        # webcam-marked tests
```

## Project rules that protect the results

These exist because the project's reported numbers depend on them:

* **Never evaluate on the test split again.** P5 used it exactly once. Analyse the saved
  predictions (`python -m src.analyze_evaluation`) rather than rerunning `evaluate_model.py`.
* **Never overwrite `model/best_direction_model.pt` without a deliberate decision.** A retrain
  must update `FROZEN_CHECKPOINT_SHA256` in the same change (see `SECURITY.md`).
* **Keep `game/` free of the CNN.** Nothing under `game/` may import `torch`, `cv2` or anything
  from `src`; `python game/main.py --selftest` fails if it does.
* **Tune live settings from live trials only**, never from the offline test set.

## Commits

Small commits, each one passing the quality gates, with a message that says what changed and
why. A conventional prefix makes the history easy to scan:

```text
feat: add colour-blind-safe theme
fix: keep CAMERA ERROR visible after the worker exits
test: cover the reconnect loop with scripted cameras
docs: record the subject-leakage audit
refactor: split evaluate_model.main into report stages
chore: pin ruff 0.16.7
```

Do not commit generated or bulky local files - `.gitignore` already covers the virtual
environment, caches, dataset images, the unseen-subject test set and study weights.

## Where things live

| Path | Contents |
|---|---|
| `game/` | The Pac-Man game - pure Pygame, no CNN, fully playable on the keyboard |
| `src/` | Dataset tools, training, evaluation, the recognizer and the integrated application |
| `tests/` | The pytest suite |
| `reports/` | Generated analysis: calibration, model study, memory profile, lineage and leakage audit |
| `model/` | The frozen checkpoint and its training and evaluation records |
| `docs/` | Architecture, model card, dataset card, model study and the manual test plan |
