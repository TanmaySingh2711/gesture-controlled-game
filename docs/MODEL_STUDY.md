# Model Study

The frozen model came from a single P4 training run. This study asks three questions that one run
cannot answer:

- **Was MobileNetV2 the right backbone?**
- **Was its fine-tuning recipe a good one?**
- **Was the frozen run a lucky outlier?**

The source is `src/model_study.py`; every number below is in `reports/model_study.json`.

## Rules

- **Train and validation splits only.** The test split is never iterated (`test_split_used: false`),
  so the P5 test result stays a one-time measurement.
- **The frozen checkpoint is never written.** It is not replaced by anything found here.
- **Same recipe as P4.** Every architecture gets the same two-stage fine-tuning, selected on
  validation loss.
- **Three seeds per configuration** (42, 7, 1234). Results are mean ± standard deviation across
  them.
- **42 runs in total:** 6 backbones × 3 seeds, plus 9 MobileNetV2 recipes × 3 seeds.

Keep the scale in mind: the validation split has 200 images, so one image is 0.5% accuracy. A
difference smaller than about one standard deviation is noise.

## Backbones (default recipe: stage-2 learning rate 1e-4, unfreeze from block 14)

| Backbone | Parameters | Val loss | Val accuracy | ECE | GPU latency, batch 1 |
|---|---|---|---|---|---|
| ResNet-18 | 11.18 M | 0.035 ± 0.016 | 98.5% ± 0.9 | 0.013 ± 0.004 | 3.3 ms |
| **MobileNetV2 (frozen model's backbone)** | **2.23 M** | **0.040 ± 0.010** | **98.8% ± 0.6** | **0.013 ± 0.003** | **6.2 ms** |
| MobileNetV3-Large | 4.21 M | 0.041 ± 0.012 | 98.0% ± 0.5 | 0.011 ± 0.002 | 8.9 ms |
| MobileNetV3-Small | 1.52 M | 0.049 ± 0.002 | 97.8% ± 0.3 | 0.016 ± 0.004 | 7.3 ms |
| ShuffleNetV2 ×1.0 | 1.26 M | 0.065 ± 0.003 | 98.3% ± 0.6 | 0.033 ± 0.006 | 7.6 ms |
| EfficientNet-B0 | 4.01 M | 0.069 ± 0.002 | 98.0% ± 0.5 | 0.015 ± 0.006 | 10.6 ms |

**Finding: MobileNetV2 was a sound choice.**

- **Only ResNet-18 comes close, and it isn't clearly better.** Its mean validation loss is lower,
  but by less than one standard deviation, and it has five times the parameters.
- **Its speed edge doesn't matter here.** ResNet-18 is faster on this GPU, but both are far below
  the ~33 ms recognizer frame budget, so latency doesn't separate them.
- **No backbone is clearly more accurate.** Every one lands between 97.8% and 98.8%.
- **ShuffleNetV2 is the worst calibrated** (ECE 0.033).

## Is the frozen model an outlier?

The frozen P4 run had a validation loss of **0.036** and 99.5% accuracy. MobileNetV2's three study
seeds span **0.032-0.051** and **98.5-99.5%**. The frozen model is a typical run of its recipe, not
a lucky one, so its reported results are representative.

## MobileNetV2 fine-tuning recipes

The stage-2 learning rate is the number across the top; "unfreeze from" is the first MobileNetV2
block trained in stage 2. A lower block number means more of the network is fine-tuned.

| Unfreeze from | lr 3e-5 | lr 1e-4 (P4) | lr 3e-4 |
|---|---|---|---|
| block 10 | 0.044 ± 0.003 / 99.0% | 0.028 ± 0.008 / 99.2% | **0.022 ± 0.018 / 99.5%** |
| block 14 (P4) | 0.059 ± 0.004 / 98.3% | 0.040 ± 0.010 / 98.8% | 0.033 ± 0.012 / 98.7% |
| block 17 | 0.172 ± 0.015 / 93.7% | 0.115 ± 0.014 / 96.0% | 0.084 ± 0.022 / 97.5% |

Each cell is validation loss / validation accuracy.

**Finding 1: fine-tuning too little of the network clearly hurts.** Unfreezing only from block 17
is the worst row at every learning rate, by far more than the seed noise. This is the one strong
result in the grid.

**Finding 2: a deeper, faster fine-tune may be slightly better, but the evidence is weak.** Unfreeze
from block 10 at lr 3e-4 is the best cell: validation loss 0.022, 99.5% accuracy on every seed, ECE
0.006. Its loss spread (± 0.018) overlaps the P4 recipe's, and its accuracy gain is about 1.4
validation images. That is not strong enough to justify replacing a frozen, already-tested model.
If the model is ever retrained, it is the recipe to try first, and judging it would need a larger
validation set.

## What this does not show

- **Validation is not deployment.** These are 200 HaGRID validation images, not live webcam frames.
  Live behaviour was measured separately in P6.
- **The same people are in train and validation.** The split is per image, so 81 of the 200
  validation images come from people also in training (`reports/subject_leakage.json`). The
  frozen model's accuracy on 1,727 unseen people (99.6%) shows this did not flatter it, but the
  small validation differences above could be affected, one more reason not to over-read them.

## Reproduce

```bash
python -m src.model_study           # resumes from reports/model_study.json; about 40 minutes on an RTX 3050 Ti
python -m src.model_study --quick   # one short run per backbone, as a smoke test
```
