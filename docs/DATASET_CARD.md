# Dataset Card

The 2,000 hand images the direction model was trained, validated and tested on.

## Summary

| | |
|---|---|
| Images | 2,000 JPEG hand crops, 500 per class |
| Classes | `left` (fist), `right` (open palm), `up` (thumbs up), `down` (thumbs down) |
| Source | [HaGRID](https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p), the 500k-image 384p sample (Kapitanov et al.) |
| Licence | CC-BY-SA-4.0, inherited from HaGRID |
| Split | 1,600 train / 200 validation / 200 test, stratified per class, seed 42 (`data_splits.json`) |
| Built by | `src/crop_hagrid_hands.py --promote` |
| Checked by | `src/check_dataset.py`, `src/check_data_pipeline.py`, `tests/test_data_pipeline.py`, `tests/test_crop_hagrid_hands.py` |
| In git | No. The images are gitignored and rebuilt from HaGRID; the split file and the build code are tracked. |

## How each image was made

Every image is built directly from a HaGRID photo and its official bounding-box annotation:

1. **Pick a hand box.** Take the photo's largest box labelled with the wanted gesture. Boxes
   smaller than 10% of the photo's longer side are skipped.
2. **Pad it.** Add 25% of the box size on every side. That keeps the whole hand and some wrist,
   like the webcam's region of interest.
3. **Make it square.** Expand the shorter side around the centre, then shift the window, never
   shrink it, so it stays inside the photo.
4. **Crop and save.** The crop is encoded as JPEG at quality 95. It is **not resized**, so image
   sizes vary.
5. **Reject bad candidates.** A candidate is skipped when:
   - the photo is missing or unreadable
   - the crop is under 96 px
   - the photo was already used for another class
   - the crop is byte-identical to one already accepted

Candidates are visited in a seeded shuffle (seed 42), so the whole build is deterministic. Running
it again with the same settings reproduces the same bytes.

Thumbs-up and thumbs-down come from HaGRID's own `like` and `dislike` classes. **Neither is made by
rotating the other.** The two classes differ only in vertical orientation, so a rotated or flipped
copy would carry the wrong label.

## What the images look like

| Class | Source gesture | Images | Side length (px) | Average side |
|---|---|---|---|---|
| left | fist | 500 | 96-384 | 137 |
| right | palm | 500 | 96-362 | 152 |
| up | like | 500 | 96-384 | 143 |
| down | dislike | 500 | 96-384 | 146 |

Every image decodes as 3-channel RGB and is exactly square. There are no duplicate files and no
filename collisions between classes.

## The split

`src/data_pipeline.py` shuffles each class's sorted file list with seed 42 and takes 400 / 50 / 50
images for train / validation / test. The result is saved to `data_splits.json`. A test checks
that rebuilding it from the dataset reproduces the committed file exactly, and the evaluation
refuses a test split that is not 200 images, 50 per class, with no overlap with train or
validation.

The test split was used exactly once, in P5. Later analysis reads P5's saved predictions and
never runs the model on it again.

## Where each image came from, and who

No record of each image's source was saved when the dataset was built. `src/audit_hagrid_lineage.py`
recovered it by replaying the deterministic build and matching every crop byte for byte.

**All 2,000 images matched** (`reports/dataset_lineage.json`). The `left` and `right` classes were
built in one run with all four classes; `up` and `down` came from a later run of just those two.

With each image's HaGRID person identifier known, `reports/subject_leakage.json` measures how
people spread across the splits:

| | Train | Validation | Test |
|---|---|---|---|
| Images | 1,600 | 200 | 200 |
| Distinct people | 1,226 | 186 | 190 |
| Most images from one person | 7 | 3 | 2 |

| Splits | People in both | Images from those people |
|---|---|---|
| Train / validation | 68 | 81 of 200 validation images |
| Train / test | 77 | **82 of 200 test images (41%)** |
| Validation / test | 17 | 19 of 200 test images |

The split was made per image, not per person, so a large share of test images come from people
the model trained on. The frozen model was therefore also evaluated on a separate unseen-person
set: 2,000 crops, one image per person per class, from 1,727 people with no image anywhere in
`dataset/`. It scored 99.6%, no worse than on the P5 test split. See
[MODEL_CARD.md](MODEL_CARD.md). That set is rebuilt with
`python -m src.audit_hagrid_lineage --external 500` and is not tracked in git.

## Augmentation (training only)

Horizontal flip, rotation up to 10 degrees, 5% translation, 0.9-1.1 scale, and ±20% brightness
and contrast. Nothing may flip an image vertically or rotate it far, because that would turn
`up` into `down`. `tests/test_data_pipeline.py` checks this directly: an image bright on top must
still be brighter on top after augmentation.

## Known issues

- **A rounding quirk in the crop geometry.** In rare edge cases, the window's position and size
  round in opposite directions, and the crop comes out one pixel narrower than it is tall.
  Property-based testing found it. **None of the 2,000 images is affected**, and a test checks
  that every one is exactly square. The code is deliberately left as it is, because rebuilding
  the dataset byte for byte, and the lineage audit, both depend on its exact output.
- **Three images were damaged during testing and restored.** On 2026-09-13, a bug in the
  evaluation script's figure code overwrote `left_00135.jpg`, `up_00142.jpg` and `down_00078.jpg`
  with chart images during a test run. It never affected the frozen P5 evaluation, whose code
  saved figures elsewhere. Each image was restored byte for byte:
  1. Replay the deterministic build.
  2. Place every intact image at its exact position in that replay.
  3. Accept a source only if it is the one candidate between the damaged file's neighbours that
     matches no intact image.
  
  The lineage audit later matched the restored `left` image to its HaGRID source independently.
  The evaluation test now hashes every dataset image before and after it runs.

## Limitations

- **Four gestures only.** There is no "no gesture" or "other gesture" class, so the model has no
  way to reject a hand shape it was not trained on.
- **Demographics were not analysed.** HaGRID publishes no per-image age, skin tone or
  camera-setup labels in this sample, so balance across those could not be checked.
- **HaGRID photos are not webcam frames.** They are photos taken by many people in many rooms,
  close to but not the same as the live 300 x 300 webcam region. Live behaviour was measured
  separately (P6).
- **The split is per image, not per person.** 41% of test images share a person with training
  (see above). The split is frozen, so this is disclosed, not changed. The unseen-person
  evaluation shows it did not inflate accuracy.
