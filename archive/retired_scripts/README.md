# Retired scripts

Kept for the project record; nothing in the application imports them.

| Script | Why it was retired |
|---|---|
| `prepare_hagrid_dataset.py` | Built crops from a geometrically cropped HaGRID repack whose official bounding boxes do not line up with its pixels. Replaced by `src/crop_hagrid_hands.py`, which reads the pure-downscale archive where the boxes align. It refuses to run. |
