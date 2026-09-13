OBSOLETE CHECKPOINT - DO NOT LOAD AGAINST THE CURRENT CLASS MAPPING.

best_gesture_model_OBSOLETE.pt was trained for the endless-runner project on:

    0 = left    (fist)
    1 = right   (palm)
    2 = jump    (peace)
    3 = neutral (no_gesture)

The Pac-Man project uses:

    0 = left    (fist)
    1 = right   (palm)
    2 = up      (like / thumbs up)
    3 = down    (dislike / thumbs down)

Both have four output neurons, so this file loads WITHOUT a shape error and then silently
produces wrong meanings for indices 2 and 3. It was archived here in P4 for provenance only.

The active model is model/best_direction_model.pt, whose metadata carries its own class
mapping so a loader can refuse a mismatch.
