# OSM-SCB

This directory contains the OSM Spatial Cognition Benchmark used to evaluate
structured geospatial question answering.

- `named.jsonl` retains place names in the questions.
- `anonymous.jsonl` replaces place names with entity labels.
- `id_named.jsonl` and `id_anonymous.jsonl` contain the ID subset.
- `ood_named.jsonl` and `ood_anonymous.jsonl` contain the OOD subset.

The named and anonymous tracks each contain 4,048 records representing 1,012
logical questions. Each logical question has four option orders and the
benchmark covers 12 subtasks.
