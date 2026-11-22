# GeoAnchor model card

## Model summary

GeoAnchor is a parameter-efficient adapter for
`Qwen/Qwen3-4B-Instruct-2507`. It maps a 256-dimensional geospatial entity
vector to a continuous prefix position and combines that prefix with the
question token embeddings. The entity vector contains 38 explicit spatial
features and 218 learned relational dimensions. The Qwen3 base model is not
redistributed in this package.

## Released components

`models/geoanchor_qwen3_4b/` contains the following after running
`python scripts/restore_assets.py`:

- `entity_embeddings.pt`, a 443,863-row entity table with its IRI index
- `prefix_adapter.pt`, the trainable projection from 256 to 2,560 dimensions
- `lora/`, the PEFT adapter and tokenizer files

The LoRA configuration uses rank 4, alpha 4, dropout 0.05, and the attention
projections `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Intended use

The model is intended for research on structured geospatial question
answering, spatial relation classification, administrative hierarchy
reasoning, and candidate ranking after a place-name index has returned
possible entities. An application supplies the question as text. When an
entity is available, its IRI selects the corresponding row in the entity
table. The IRI is an internal key and is not shown to the language model.

When no entity has been resolved, the model can still use the task adapter and
its shared prefix to answer from the question text.

## Out-of-scope use

The checkpoint is not a general geocoder, GIS calculator, or current-data
service. It should not be used as the sole source for safety-critical routing,
land administration, emergency response, or legal decisions. The entity table
is an OSM-derived snapshot and can be incomplete, stale, or inconsistent with
current authoritative geographic data.

## Evaluation context

The associated experiments reported the following reference values on the
project OSM-SCB protocol:

- Qwen3-4B: 31.34 percent average accuracy
- GeoAnchor: 66.20 percent average accuracy
- Entity-free no-KGE comparison: 55.05 percent average accuracy
- Approximately 0.09 percent of Qwen3-4B parameters were updated

The package also includes an open toponym application benchmark. It tests
whether the same entity representation can rank name-index candidates. It is
an application study rather than a claim of complete end-to-end entity
linking.

## Limitations

- Coverage is limited by the OSM-derived entity catalog and its names.
- The representation is learned from a static graph and does not model
  temporal change or continuous spatial fields.
- Multi-entity questions require correct entity order and semantic roles. A
  single-subject prefix cannot replace a GIS engine for arbitrary geometry.
- Evaluation uses Chinese and selected English records. Performance on other
  languages, data sources, models, and open-world entity inventories is not
  established by this release.
- Downstream systems should validate option parsing, units, coordinate
  conventions, and entity identity.

## Responsible use

Report the OSM source and attribution when using the model. For operational
applications, combine the model with an authoritative spatial database,
explicit geometric computation, and an auditable entity-resolution pipeline.
