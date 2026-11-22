# GeoAnchor Qwen3-4B LoRA adapter

This directory contains the PEFT LoRA adapter and tokenizer files used by
GeoAnchor. It is not a standalone language model.

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Adapter type: LoRA
- Rank: 4
- Alpha: 4
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Dropout: 0.05

Load it together with `prefix_adapter.pt` and the released entity table through
`genanchor.model.load_geoanchor` or `scripts/predict.py`.
