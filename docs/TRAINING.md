# Training guide

GeoAnchor has two training stages.

1. The spatial entity encoder reads the OSM-derived graph and 38 explicit
   features. A TransE-style objective and auxiliary spatial losses learn a
   218-dimensional relational component. Concatenating the two components
   produces the 256-dimensional entity table.
2. The language-model stage uses the entity table and supervised spatial
   questions. A linear projection maps each entity vector to one continuous
   prefix position, while a parameter-efficient language-model adapter learns
   the task interface.

## Settings of the released adapter

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Entity vector | 38 explicit + 218 relational = 256 dimensions |
| Prefix positions per entity | 1 |
| Entity configuration | one subject entity |
| LoRA rank / alpha | 4 / 4 |
| LoRA dropout | 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimizer | AdamW |
| Adapter learning rate | `2e-4` |
| Prefix projection learning rate | `1e-3` |
| Epochs | 1 |
| Maximum sequence length | 320 tokens |

The commands for both stages are in the top-level README. Restore the released
entity table under `models/geoanchor_qwen3_4b/` with
`python scripts/restore_assets.py` after cloning.

## Entity IDs

The serialized entity table uses zero-based IDs. The prefix adapter reserves
ID 0 for padding and receives entity IDs shifted by one. The helper
`entity_ids_from_iris` performs this conversion. A missing entity is encoded
as a padding row, which lets the language model use the shared task prefix.

## Evaluation

`scripts/evaluate_osm_scb.py` resolves display names in an OSM-SCB record with
`data/source/osm_rdf/name_index.json` and then uses the same model interface as
explicit IRI input. Applications that resolve names upstream can pass IRIs
directly when identity is known.
