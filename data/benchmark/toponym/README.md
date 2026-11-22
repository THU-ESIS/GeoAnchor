# Toponym evaluation

These files support the optional application study in which a name index
returns candidate OSM entities and GeoAnchor ranks them using the query
context and entity representations.

- `china_subject.jsonl` contains Chinese contextual queries with the subject
  entity held out from spatial adaptation.
- `china_all_roles.jsonl` contains the corresponding all-role split.
- `wikitor_all.jsonl` contains the mapped WikToR queries.
- `wikitor.jsonl` is the reviewed WikToR evaluation subset.

The task evaluates candidate selection and rejection. It does not cover
mention detection, fuzzy alias recall, or discovery of entities outside the
released catalog.
