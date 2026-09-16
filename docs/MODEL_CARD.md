# Model Card

The CNN that turns a webcam view of a hand into a Pac-Man direction.

## Summary

| | |
|---|---|
| Model | MobileNetV2 (torchvision, ImageNet-pretrained) with a new four-way output layer |
| Parameters | 2,228,996 |
| Input | 3 x 160 x 160 RGB, ImageNet mean and std normalisation |
| Output | Softmax over `left`, `right`, `up`, `down` (index 0-3) |
| File | `model/best_direction_model.pt`, SHA-256 `e57cab3b2fc5ddeba362f7e41586663b02feeed7ca2af49420f7bc7bc255cd3c` |
| Task id | `pacman_direction_v1`, stored in the checkpoint and checked on every load |
| Framework | PyTorch 2.14 with CUDA 13, RTX 3050 Ti Laptop GPU (4 GB); no CPU fallback, by project policy |
| Data | 2,000 HaGRID hand crops, see [DATASET_CARD.md](DATASET_CARD.md) |

## Intended use

**What it is for:** steering this game. A single player holds one of four gestures inside the
on-screen box, at roughly arm's length from a webcam, in ordinary indoor light. The model is only
one part of the recognizer, which also:
1. drops predictions below 0.90 confidence
2. turns a direction into a command only when it holds 3 of the last 5 frames.

**What it is not for:**
- recognising any other gesture
- identifying people
- sign language
- anything where a wrong reading has consequences beyond a game

## Gestures

| Gesture | Class | Game direction |
|---|---|---|
| Closed fist | `left` | LEFT |
| Open palm | `right` | RIGHT |
| Thumbs up | `up` | UP |
| Thumbs down | `down` | DOWN |

## Training (P4)

Two stages on the 1,600 training images, both using the AdamW optimiser with weight decay 1e-4:

1. **Output layer only**, backbone frozen: 5 epochs, learning rate 1e-3.
2. **Fine-tuning** with `features[14:]` unfrozen (the last inverted-residual blocks and the final
   convolution): up to 8 epochs, learning rate 1e-4. Training stops early after 3 epochs without
   improvement.

One "best so far" record runs across both stages. A checkpoint is saved whenever validation loss
beats it, with validation accuracy breaking an exact tie, so the frozen model is the best epoch
of the whole run. Batch size is 32. Training augmentation is horizontal flip, rotation up to 10 degrees, small shifts and
scaling, and brightness and contrast changes. There is never a vertical flip, which would turn
`up` into `down`. Final validation: **99.5% (199/200), loss 0.036**.

A later 42-run study, on train and validation only, found no better backbone and showed this run
is typical of its recipe, not a lucky one. See [MODEL_STUDY.md](MODEL_STUDY.md).

## Evaluation on the held-out test split (P5)

Evaluated exactly once, on 200 images (50 per class) never seen during training or selection.

| Metric | Value |
|---|---|
| Accuracy | **99.0%** (198/200), Wilson 95% interval 96.4%-99.7% |
| Macro-F1 | 0.990, bootstrap 95% interval 0.974-1.000 |
| Calibration | expected calibration error 0.005, maximum 0.232, Brier score 0.016, negative log-likelihood 0.029 |
| At the live 0.90 threshold | 98.0% of images accepted; 99.5% of those correct |
| Errors | `right` read as `down` (confidence 0.60); `down` read as `up` (confidence 0.90) |
| GPU forward pass, batch 1 | 6.0 ms mean |

The second error passes the 0.90 threshold, which is exactly why smoothing exists. A single
confident wrong frame cannot win a 3-of-5 vote.

Calibration details and the reliability diagram are in `reports/evaluation_rigor.json` and
`reports/reliability_diagram.png`.

## Generalisation to people it has never seen

The lineage audit traced every dataset image back to its HaGRID photo and the person who took
it. **41% of the P5 test images (82 of 200) come from people who also appear in training**, so
P5's 99% could have been flattered by familiar hands.

To check, the frozen model was evaluated once on a separate set, `src/evaluate_external.py`:
2,000 HaGRID crops, 500 per class. They come from 1,727 people, none of whom has a single image
anywhere in the dataset. No image is byte-identical to a dataset image, and nothing measured was
used to tune anything.

| Metric | Unseen people | P5 test |
|---|---|---|
| Accuracy | **99.6%** (1,992/2,000), Wilson 95% interval 99.2%-99.8% | 99.0% (198/200), 96.4%-99.7% |
| Macro-F1 | 0.996, bootstrap 95% interval 0.993-0.999 | 0.990, 0.974-1.000 |
| Per-class recall | left 99.4%, right 99.2%, up 99.8%, down 100% | - |
| Calibration | expected calibration error 0.0035, Brier score 0.007 | 0.005, 0.016 |
| At the live 0.90 threshold | 98.4% accepted; 99.95% of those correct (1 error) | 98.0%; 99.5% |

**Finding: familiar hands did not inflate the result.** The difference is +0.6 points, with a
Newcombe 95% interval of -0.2 to +3.2, which is not significant. If anything, unseen people
score slightly higher, and the larger set narrows the interval considerably.

Of the 8 errors, 7 had confidence below 0.90 and would be rejected in the live game. The eighth
(`right` read as `up`, 0.915) would still have to win the 3-of-5 vote. Seven of the eight were
predicted as a thumb gesture (`up` or `down`); only one confused `up` with `down` themselves. Details are in
`reports/external_evaluation.json` and `reports/external_predictions.csv`.

These are still HaGRID photos, not webcam frames, so this measures generalisation across people,
not across cameras.

## Live behaviour (P6, real webcam)

| Measure | Result |
|---|---|
| Held gestures recognised | 80/80 |
| False commands from an idle or absent hand | 0/40 |
| Wrong turns during natural-speed gesture changes | 0/6 |
| Gesture seen to stable command | 143 ms mean, 217 ms 95th percentile |
| Memory | 22 MB peak GPU memory; no growth over 5,000 frames (`reports/memory_profile.json`) |

The threshold and the smoothing window were both chosen from these live trials, never from the
test set.

## Limitations and risks

- **No reject class.** A clear gesture outside the four can be read confidently as one of them:
  a peace sign was read as RIGHT at 99.3%. The start screen tells players to use only the four
  gestures. Relaxed and absent hands are correctly rejected by the threshold.
- **Stray commands with nobody gesturing.** In a 330-second run with no one gesturing, 25 of
  19,835 game frames (0.13%) carried a stable command. What the camera saw was not recorded.
- **Small test set.** At 200 images, the honest claim is "96.4% to 99.7%", not "99%". The
  2,000-image unseen-person set narrows this to 99.2%-99.8%.
- **One dataset, one live setup.** Training images come only from HaGRID. The live trials used the
  development webcam and room; how many people took part was not recorded. Performance in other
  lighting, camera angles or hands is untested beyond that.
- **Demographics not analysed.** The HaGRID sample used carries no age or skin-tone labels, so
  fairness across them could not be measured.
- **Same people in train and test.** 41% of P5 test images share a person with training, because
  HaGRID was not split by person. The unseen-person evaluation above shows this did not inflate
  accuracy, but P5's number alone should not be read as a person-independent result.
- **CUDA required.** Gesture control does not run without an NVIDIA GPU; the keyboard game does.

## Security

The loader refuses any file whose SHA-256 differs from the pinned value, loads with
`torch.load(weights_only=True)` so the file cannot run code, and refuses a checkpoint whose class
mapping differs from the active one. See [SECURITY.md](../SECURITY.md).
