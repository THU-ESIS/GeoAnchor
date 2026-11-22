# Third-party notices

This file records the upstream components that are required to use or interpret
the release. It does not replace the terms supplied by those projects.

## OpenStreetMap

The spatial snapshot and the OSM-SCB records are derived from OpenStreetMap
data. OpenStreetMap data is generally distributed under the Open Data Commons
Open Database License (ODbL) 1.0, with attribution and share-alike obligations.
Consult the current OpenStreetMap copyright page and the exact source snapshot
terms before redistribution or publication of a modified database.

Attribution: OpenStreetMap contributors.

## osm2rdf

The RDF conversion and spatial relation extraction follow the `osm2rdf`
workflow described by Bast et al. in the associated geospatial RDF work cited
by the paper. The conversion software and any downloaded RDF dump have their
own copyright and license terms. The release includes derived files, not the
raw upstream dump.

## Qwen3

The base model `Qwen/Qwen3-4B-Instruct-2507` is not included. Download it from
its official distribution and follow the Qwen license and usage conditions.
The files in `models/.../lora/` are an adapter for that base model and cannot
be used without it.

## WikToR

The open toponym files are derived from the WikToR geographical parsing corpus
described by Gritta, Pilehvar, and Collier. The full source corpus is not
bundled. Users who redistribute the derived files should follow the original
dataset terms and cite the source publication.

## Python dependencies

PyTorch, Transformers, PEFT, NumPy, pandas, SciPy, PyArrow, and Safetensors are
third-party software packages. Their licenses are not changed by this release.
