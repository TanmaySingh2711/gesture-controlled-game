# Documentation

Start with the [main README](../README.md). The documents here go deeper.

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the parts fit together: the recognition thread, the game loop and the boundary between them |
| [MODEL_CARD.md](MODEL_CARD.md) | The CNN: what it does, how it was trained, how well it works and where it fails |
| [MODEL_STUDY.md](MODEL_STUDY.md) | MobileNetV2 compared with five other networks and different training settings |
| [DATASET_CARD.md](DATASET_CARD.md) | The 2,000 hand images: where they come from, how they were cropped and split |
| [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md) | The full application tested end to end, with live webcam runs |
| [MANUAL_TEST_PLAN.md](MANUAL_TEST_PLAN.md) | The checks that need a real person, hand and webcam |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setting up for development and the rules that protect the reported results |
| [SECURITY.md](SECURITY.md) | The security risks of a local desktop app and how each one is handled, and how to report a problem |
| [CHANGELOG.md](CHANGELOG.md) | What changed, in order |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | The original design and the build plan, kept as a record |

## Phase labels

Some documents mention build phases such as "P5" or "Objective 9". They follow the order the
project was built in:

| Label | What happened in that phase |
|---|---|
| P1 | The design was written down and frozen |
| P2 | The hand-image dataset was built from HaGRID |
| P3 | The images were split and the preprocessing was fixed |
| P4 | The CNN was trained |
| P5 | The CNN was tested once on the 200 held-out images (results in `reports/test_evaluation/`) |
| P6 | Real-time recognition was built and tried live (records in `reports/live_trials/`) |
| P7 | The Pac-Man game was built |
| Objective 9 | The CNN was connected to the game |
| Objective 10 | The final user interface was built |
| Objective 11 | The complete application was tested |

## Other folders

- [`../reports/`](../reports/) holds every evaluation result as JSON, CSV or images.
- [`../model/`](../model/) holds the trained model and its training history.
- [`images/`](images/) holds the screenshots used in the main README.
