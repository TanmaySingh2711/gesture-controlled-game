# Security

CNN Gesture Controlled Pac-Man is a local desktop application. It has no network service, no
user accounts, no credentials and no telemetry. The realistic risks are therefore a tampered
model file, untrusted data read from disk, and the one-time dataset download. This document
records the threat model and the controls in place for each.

## Threat model

| Asset or input | Threat | Control |
|---|---|---|
| `model/best_direction_model.pt` | A swapped, truncated or tampered checkpoint loads silently and steers the game wrongly | The loader verifies the file's **SHA-256** against `FROZEN_CHECKPOINT_SHA256` in `src/train_model.py` and refuses any other bytes |
| Any `.pt` file | Python pickles can execute arbitrary code while loading | Checkpoints are loaded with **`torch.load(weights_only=True)`**, which restricts unpickling to tensors and plain containers. `tests/test_checkpoint_security.py` proves a malicious pickle never runs |
| Checkpoint meaning | A different model with the same shape (the retired left/right/jump/neutral one) loads cleanly but means different gestures | The stored class mapping must equal the active mapping, checked independently of the digest |
| `~/.gesture_pacman/profile.json` | A corrupted or hand-edited file crashes the game, or a crash mid-save truncates it | Every read is validated and falls back to defaults; writes go to a temporary file and are atomically renamed |
| HaGRID dataset download | A non-HTTPS or redirected URL feeds unexpected content into the dataset | Download URLs are module constants validated as `https://` before any request; downloaded images are decoded and size-checked, and never executed |
| `dataset/` images | A bug or tool silently rewrites training images, invalidating every result built on them (this happened once during testing: see `CHANGELOG.md`) | `tests/test_evaluate_model.py` hashes every dataset image before and after running the evaluation pipeline, and fails if any byte changed. `src/audit_hagrid_lineage.py` independently re-derives each image byte for byte from HaGRID. Damaged images are restored only when exactly one source is proven by that replay |
| Webcam frames | Malformed frames crash the worker | Empty reads trigger bounded reconnection; any worker exception is reported once as `CAMERA ERROR` and the game keeps running on the keyboard |
| Python dependencies | Known-vulnerable packages | Versions are pinned in `requirements.txt` and audited with `pip-audit` (CI runs it on every push) |

## Verifying the model file yourself

```bash
python -c "from src.train_model import file_sha256, FROZEN_CHECKPOINT_SHA256 as pin; print(file_sha256('model/best_direction_model.pt') == pin)"
```

## After a deliberate retrain

A new checkpoint has a new digest, so the loader will refuse it until the pin is updated. Update
`FROZEN_CHECKPOINT_SHA256` in the same change as the new checkpoint, so the history shows the
two moving together.

## Dependency audit

`pip-audit` reported **no known vulnerabilities** in the installed environment. The CUDA builds
of `torch` and `torchvision` come from the PyTorch index rather than PyPI, so `pip-audit` cannot
look them up. As a partial check, the same version numbers were audited against the PyPI
advisory database, also with no known vulnerabilities:

```bash
printf 'torch==2.14.0\ntorchvision==0.29.0\n' > torch_versions.txt
pip-audit --no-deps --disable-pip -r torch_versions.txt
```

That covers advisories filed against those releases, but not anything specific to the CUDA
packaging. Track PyTorch's security advisories for those two directly.

## Reporting a problem

This is a student project with no production deployment. If you find a security issue, open an
issue in the repository describing it, or contact the repository owner directly for anything
you would rather not post publicly.
