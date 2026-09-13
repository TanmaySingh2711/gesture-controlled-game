Obsolete gesture classes from the endless-runner version of this project.

    jump    <- HaGRID peace
    neutral <- HaGRID no_gesture

The project was revised in P1 to a Pac-Man-style maze game whose four commands are
LEFT / RIGHT / UP / DOWN, with no NEUTRAL class. These two folders were moved out of
dataset/ in P2 so that no later script can mistake them for active classes.

They are kept only as a record of the earlier dataset. Nothing in the project reads them.

pipeline/  holds the stale data_splits.json and class_mapping.json from the endless-runner
           model (left=0, right=1, jump=2, neutral=3). They were archived in P3 and replaced
           by the four-direction versions in the project root. Nothing reads these copies.
