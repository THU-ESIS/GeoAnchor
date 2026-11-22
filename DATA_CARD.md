# Data card

## Scope

This package contains a derived spatial knowledge snapshot, spatial
adaptation data, and evaluation data used by GeoAnchor. It does not contain
the original OSM dump or the full WikToR corpus.

The spatial snapshot was derived from OpenStreetMap data converted to RDF with
the `osm2rdf` workflow. It covers a China-focused regional sample spanning
approximately 71.19--142.09 degrees east and 8.67--54.00 degrees north. The
bounding box does not imply complete geographic coverage.

## OSM-RDF source data

Files are in `data/source/osm_rdf/`. The three largest TSV files are restored
from compressed parts in `assets/` by running `python scripts/restore_assets.py`.

| File | Contents |
| --- | --- |
| `entities.parquet` | 200,088 selected core OSM objects |
| `geometry.parquet` | Geometry and attribute records for the selected objects |
| `entity_features.npy` | 443,863 x 38 aligned explicit feature matrix |
| `graph_entity_index.json` | Index of 443,863 graph nodes |
| `encoder_entity_index.json` | Mapping from core entity IRIs to graph rows |
| `relation_index.json` | 29 graph relation types |
| `name_index.json` | OSM entity IRI to display-name mapping |
| `triples.tsv` | 8,100,276 derived graph triples |
| `triples_base.tsv` | Base relation triples before derived splits |
| `triples_train.tsv` | Training triples for the entity encoder |
| `triples_valid.tsv` | Validation triples for the entity encoder |
| `triples_test.tsv` | Test triples for the entity encoder |

The 200,088 core objects include 100,088 administrative objects, 40,000
points of interest, 30,000 natural features, and 30,000 transport features.
Administrative objects were retained subject to the source selection rule.
The other categories were sampled across one-degree spatial cells with fixed
category quotas. The remaining graph nodes represent names, object types,
administrative levels, geohash cells, and relation values. Consequently, the
core-object count, graph-node count, and triple count describe different
levels of the data representation.

The 38 explicit features encode coordinates, geometry, administrative level,
object type, local context, and related spatial attributes. They are aligned
to graph-node order in `entity_features.npy`. The entity encoder learns the
remaining 218 dimensions from the relation graph and produces a 256-dimensional
entity table.

## Language-model training data

The spatial adaptation files are in `data/train/`.

| File | Use | Size |
| --- | --- | ---: |
| `spatial_train.jsonl` | Spatial task supervision | 26,579 records |
| `spatial_dev.jsonl` | Development data | 500 records |
| `spatial_train_augmented.jsonl` | Training data used by the released adapter | 69,899 records |

Each record stores the question, answer, and the OSM IRIs used to select
entity vectors. The IRIs are internal keys and are not inserted into the
question text.

The optional toponym adaptation files are under `data/train/toponym/`.
`pairwise/` contains 2,358 training pairs and 564 development pairs.
`candidate_attributes/` has the same pair counts and includes candidate
attribute fields. `candidate_context/` contains 4,455 training pairs and
1,086 development pairs with candidate context fields.

## OSM-SCB benchmark

The OSM-SCB files are in `data/benchmark/osm_scb/`.

- `named.jsonl` retains place names in the questions.
- `anonymous.jsonl` replaces the place names with entity labels.
- `id_named.jsonl` and `id_anonymous.jsonl` contain the ID subset.
- `ood_named.jsonl` and `ood_anonymous.jsonl` contain the OOD subset.

The included named and anonymous tracks each contain 4,048 records. They
represent 1,012 logical questions with four option orders and 12 subtasks.
The ID and OOD files are subsets of the named and anonymous tracks.

## Toponym evaluation

The candidate-ranking files are in `data/benchmark/toponym/`.

- `china_subject.jsonl` contains 2,760 Chinese-context queries.
- `china_all_roles.jsonl` contains 888 Chinese-context queries.
- `wikitor_all.jsonl` contains 78 queries mapped from WikToR.
- `wikitor.jsonl` contains the 75-query evaluation subset retained after the
  candidate mapping review.

The task provides a place mention, its text context, and candidate OSM
entities. The target is candidate selection or rejection. The data therefore
supports an application study of entity disambiguation and does not represent
a complete open-world geocoder.

## Licensing and attribution

OSM-derived files carry the rights and conditions of OpenStreetMap data,
including ODbL attribution and share-alike requirements where applicable.
WikToR-derived files remain subject to the terms of the source corpus. See
`THIRD_PARTY_NOTICES.md` before redistributing the data or a modified version.
