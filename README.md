<div align="center">

# GeoAnchor

**Open-source implementation and release package for GeoAnchor**

Structured geospatial knowledge injection for parameter-efficient language-model adaptation

</div>

GeoAnchor injects structured information about geographic entities into a
large language model. It combines explicit spatial features with learned
relational representations, then exposes the resulting entity vector through
a continuous prefix interface. This repository contains the training code,
derived OSM-RDF inputs, released Qwen3-4B adapters, entity representations,
and evaluation data.

## At a glance

| Component | Contents |
| --- | --- |
| Entity representation | 38 explicit spatial features + 218 learned relational dimensions |
| Released entity table | 443,863 rows with 256 dimensions per row |
| Language model | `Qwen/Qwen3-4B-Instruct-2507` |
| Adaptation | Continuous entity prefix and parameter-efficient task adapter |
| Evaluation | OSM-SCB spatial question benchmark and toponym candidate ranking |

The raw OSM dump and the Qwen3 base model are not redistributed here. Their
provenance and licensing conditions are documented in
[`DATA_CARD.md`](DATA_CARD.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Repository map

```text
GenAnchor/
├── README.md
├── DATA_CARD.md
├── MODEL_CARD.md
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── paths.example.yaml
│   └── qwen3_4b.yaml
├── assets/                 # compressed parts of large model and graph files
├── src/genanchor/
│   ├── data.py
│   ├── entity_encoder.py
│   ├── model.py
│   └── train.py
├── scripts/
│   ├── train_entity_encoder.py
│   ├── train_genanchor.py
│   ├── evaluate_osm_scb.py
│   ├── evaluate_toponym.py
│   ├── predict.py
│   └── restore_assets.py
├── data/
│   ├── source/osm_rdf/
│   ├── train/
│   └── benchmark/
├── models/geoanchor_qwen3_4b/
└── docs/
```

## Quick start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate genanchor
pip install -e .
```

### 2. Download the base model

Download `Qwen/Qwen3-4B-Instruct-2507` separately. The released GeoAnchor
components are compatible with this base model.

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir "$HOME/models/Qwen3-4B-Instruct-2507"
```

### 3. Restore the large files

The entity table and three graph files are stored as compressed Git LFS parts
to keep individual uploads small. Restore their original paths after cloning:

```bash
python scripts/restore_assets.py
```

The command produces `models/geoanchor_qwen3_4b/entity_embeddings.pt` and
`data/source/osm_rdf/triples*.tsv`. It leaves the compressed parts intact.

## Run the released model

### Evaluate OSM-SCB

The following command evaluates a small portion of the released OSM-SCB
files.

```bash
python scripts/evaluate_osm_scb.py \
  --base-model "$HOME/models/Qwen3-4B-Instruct-2507" \
  --checkpoint-dir models/geoanchor_qwen3_4b \
  --entity-table models/geoanchor_qwen3_4b/entity_embeddings.pt \
  --data data/benchmark/osm_scb/named.jsonl \
  --name-map data/source/osm_rdf/name_index.json \
  --max-records 16
```

### Run one spatial question

When an application has identified the relevant OSM entity, pass its IRI to
select the corresponding row in the entity table.

```bash
python scripts/predict.py \
  --base-model "$HOME/models/Qwen3-4B-Instruct-2507" \
  --checkpoint-dir models/geoanchor_qwen3_4b \
  --entity-table models/geoanchor_qwen3_4b/entity_embeddings.pt \
  --question "What is the approximate area of Futian District in Shenzhen?" \
  --iri osmrel:5664191
```

The IRI is an internal lookup key and is not added to the question shown to
the language model. The `--iri` argument can be omitted when no entity has
been resolved. The model can then use the learned task adapter and shared
prefix to answer from the question text.

## Train GeoAnchor

GeoAnchor is trained in two stages. The first stage learns the relational
part of the entity representation from the derived OSM-RDF graph. The second
stage adapts the language model to use the resulting entity table.

### Stage 1: train the entity encoder

The released entity table is available after the restoration step. To train a
new spatial entity encoder from the supplied graph inputs, run:

```bash
python scripts/train_entity_encoder.py \
  --entity-id-path data/source/osm_rdf/graph_entity_index.json \
  --relation-id-path data/source/osm_rdf/relation_index.json \
  --entities-path data/source/osm_rdf/entities.parquet \
  --raw-features-path data/source/osm_rdf/entity_features.npy \
  --triples-train-path data/source/osm_rdf/triples_train.tsv \
  --out /tmp/geoanchor_entity_encoder
```

### Stage 2: adapt the language model

The language-model stage uses the 256-dimensional entity table and the
spatial supervision in `data/train/`.

```bash
python scripts/train_genanchor.py \
  --base-model "$HOME/models/Qwen3-4B-Instruct-2507" \
  --entity-table models/geoanchor_qwen3_4b/entity_embeddings.pt \
  --train-file data/train/spatial_train_augmented.jsonl \
  --dev-file data/train/spatial_dev.jsonl \
  --variant faithful \
  --lora-r 4 \
  --lora-alpha 4 \
  --out-dir /tmp/geoanchor_training
```

The spatial entity representation contains 38 explicit dimensions and 218
learned relational dimensions. The language-model stage maps each entity
vector to one continuous prefix position while learning the task interface.
See [`docs/TRAINING.md`](docs/TRAINING.md) for the two-stage training
description and settings.

## Data and evaluation

| Path | Role |
| --- | --- |
| `data/source/osm_rdf/` | OSM-derived graph inputs and aligned spatial features for the entity encoder |
| `data/train/` | Spatial question supervision and optional toponym adaptation data |
| `data/benchmark/osm_scb/` | Named and anonymized spatial question evaluation files |
| `data/benchmark/toponym/` | Candidate-ranking files for the toponym application study |

### OSM-RDF inputs

`data/source/osm_rdf/` contains the derived graph snapshot used by the
entity encoder, including files restored from `assets/`. It includes 200,088
selected core OSM objects, 443,863 graph nodes, 29 relation types, and
8,100,276 derived graph triples. The selection rule, geographic scope, and
file descriptions are given in
[`DATA_CARD.md`](DATA_CARD.md).

### OSM-SCB

`data/benchmark/osm_scb/` contains named and anonymized spatial question
tracks, together with ID and OOD subsets. The released files represent 1,012
logical questions, four option orders per question, and 12 subtasks.

### Toponym application study

`data/benchmark/toponym/` contains Chinese candidate-ranking files and a
WikToR-derived open-text application set. The evaluation supplies a place
mention, its text context, and candidate OSM entities, and measures candidate
selection or rejection. It evaluates the use of GeoAnchor representations
after a name index has returned possible entities. It is not an end-to-end
mention detector or an open-world place-discovery system.

The evaluation entry points are
[`scripts/evaluate_osm_scb.py`](scripts/evaluate_osm_scb.py) and
[`scripts/evaluate_toponym.py`](scripts/evaluate_toponym.py).

## Citation

Please cite the GeoAnchor paper and the upstream sources listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) when using this package.
