#!/usr/bin/env python3
"""Evaluate a released GeoAnchor checkpoint on an OSM-SCB JSONL file."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genanchor.data import load_jsonl, load_name_index, pad_batch, parse_option, record_iris
from genanchor.model import entity_ids_from_iris, load_geoanchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--entity-table", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--name-map", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-entities", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data)
    if args.max_records > 0:
        records = records[: args.max_records]
    name_to_iris = load_name_index(args.name_map)
    adapter_meta = torch.load(
        Path(args.checkpoint_dir) / "prefix_adapter.pt",
        map_location="cpu",
        weights_only=False,
    )
    max_entities = args.max_entities or (
        1 if adapter_meta.get("variant", "faithful") == "faithful" else 5
    )
    model, tokenizer, entity_to_id = load_geoanchor(
        args.base_model,
        args.checkpoint_dir,
        args.entity_table,
        device=args.device,
    )

    predictions: list[str] = []
    resolved = 0
    model_device = next(model.parameters()).device
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        batch_ids = []
        for record in batch_records:
            iris = record_iris(record, name_to_iris, max_entities=max_entities)
            resolved += int(bool(iris))
            batch_ids.append(entity_ids_from_iris(iris, entity_to_id, max_entities))
        encoded = pad_batch(
            batch_records,
            tokenizer,
            batch_ids,
            max_length=args.max_length,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            entity_ids=encoded["entity_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for output in generated:
            predictions.append(parse_option(tokenizer.decode(output, skip_special_tokens=True)))

    correct = [prediction == record["answer"] for prediction, record in zip(predictions, records)]
    by_subtask: dict[str, list[bool]] = defaultdict(list)
    by_dim: dict[str, list[bool]] = defaultdict(list)
    for record, ok in zip(records, correct):
        by_subtask[str(record.get("subtask", "unknown"))].append(ok)
        by_dim[str(record.get("dim", "unknown"))].append(ok)
    summary = {
        "data": str(args.data),
        "n_records": len(records),
        "accuracy": sum(correct) / max(len(correct), 1),
        "resolved_entity_rows": resolved,
        "max_entities": max_entities,
        "by_dim": {
            key: sum(values) / max(len(values), 1) for key, values in by_dim.items()
        },
        "by_subtask": {
            key: sum(values) / max(len(values), 1)
            for key, values in by_subtask.items()
        },
        "predictions": predictions,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
