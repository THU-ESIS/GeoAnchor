# Training data

`spatial_train.jsonl` and `spatial_dev.jsonl` contain the spatial question
supervision. `spatial_train_augmented.jsonl` is the training file used by the
released GeoAnchor adapter.

The `toponym/` directory contains optional pairwise training files for the
candidate-ranking application. Each record stores the question, answer, and
the OSM IRIs used to select entity vectors. The IRIs are not included in the
question text.
