# GeoAnchor Qwen3-4B release

This directory contains the GeoAnchor adapters and entity table for
`Qwen/Qwen3-4B-Instruct-2507`.

- `entity_embeddings.pt` contains the 443,863 x 256 entity table and IRI index
- `prefix_adapter.pt` maps entity vectors to the language-model prefix space
- `lora/` contains the PEFT adapter and tokenizer files

Use `scripts/predict.py` or `genanchor.model.load_geoanchor` to load these
components with the Qwen3 base model.
