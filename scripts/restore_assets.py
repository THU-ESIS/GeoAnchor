"""Restore large model and graph files from the release's compressed parts."""

from pathlib import Path
import shutil
import tempfile

import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FILES = (
    "models/geoanchor_qwen3_4b/entity_embeddings.pt",
    "data/source/osm_rdf/triples.tsv",
    "data/source/osm_rdf/triples_train.tsv",
    "data/source/osm_rdf/triples_base.tsv",
)


def restore(relative_path: str) -> None:
    target = ROOT / relative_path
    if target.exists():
        print(f"Present: {relative_path}")
        return

    parts = sorted(ASSETS.glob(f"{target.name}.zst.part-*"))
    if not parts:
        raise FileNotFoundError(f"Missing compressed parts for {relative_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp_dir:
        compressed = Path(temp_dir) / "asset.zst"
        restored = Path(temp_dir) / target.name
        with compressed.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        with compressed.open("rb") as source, restored.open("wb") as output:
            zstd.ZstdDecompressor().copy_stream(source, output)
        restored.replace(target)
    print(f"Restored: {relative_path}")


if __name__ == "__main__":
    for filename in FILES:
        restore(filename)
