#!/usr/bin/env python3
"""Run one GeoAnchor prediction with an explicit OSM IRI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genanchor.data import chat_prompt
from genanchor.model import entity_ids_from_iris, load_geoanchor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--entity-table", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--iri", action="append", default=[])
    parser.add_argument("--max-entities", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer, entity_to_id = load_geoanchor(
        args.base_model,
        args.checkpoint_dir,
        args.entity_table,
        device=args.device,
    )
    prompt = chat_prompt(tokenizer, args.question)
    encoded = tokenizer(
        [prompt], return_tensors="pt", padding=True, add_special_tokens=False
    )
    device = next(model.parameters()).device
    entity_ids = entity_ids_from_iris(
        args.iri, entity_to_id, max_entities=args.max_entities
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    encoded["entity_ids"] = torch.tensor(
        [entity_ids], dtype=torch.long, device=device
    )
    output = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        entity_ids=encoded["entity_ids"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
