#!/usr/bin/env python3
"""Rank candidate entities in the released open toponym files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genanchor.data import chat_prompt, load_jsonl
from genanchor.model import entity_ids_from_iris, load_geoanchor


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"tokenizer produced no id for {text!r}")
    return int(ids[0])


def score_candidate(model, tokenizer, entity_to_id, prompt: str, iri: str, device) -> float:
    rendered = chat_prompt(tokenizer, prompt)
    encoded = tokenizer(
        [rendered], return_tensors="pt", padding=True, add_special_tokens=False
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    encoded["entity_ids"] = torch.tensor(
        [entity_ids_from_iris([iri], entity_to_id, max_entities=1)],
        dtype=torch.long,
        device=device,
    )
    output = model(**encoded)
    last = int(encoded["attention_mask"][0].sum().item()) - 1
    log_probs = F.log_softmax(output.logits[0, last], dim=-1)
    yes = log_probs[token_id(tokenizer, "是")]
    no = log_probs[token_id(tokenizer, "否")]
    return float((yes - no).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--entity-table", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records > 0:
        records = records[: args.max_records]
    model, tokenizer, entity_to_id = load_geoanchor(
        args.base_model,
        args.checkpoint_dir,
        args.entity_table,
        device=args.device,
    )
    device = next(model.parameters()).device
    output_rows = []
    in_catalog = 0
    correct = 0
    for record in records:
        scored = []
        for candidate in (record.get("candidates") or [])[: args.max_candidates]:
            iri = str(candidate["iri"])
            if iri not in entity_to_id:
                continue
            scored.append(
                {
                    "iri": iri,
                    "score_yes_minus_no": score_candidate(
                        model, tokenizer, entity_to_id, record["prompt"], iri, device
                    ),
                }
            )
        ranked = sorted(
            scored, key=lambda item: item["score_yes_minus_no"], reverse=True
        )
        target = record.get("target_iri")
        is_correct = bool(target and ranked and ranked[0]["iri"] == target)
        if target:
            in_catalog += 1
            correct += int(is_correct)
        output_rows.append(
            {
                "query_id": record.get("query_id"),
                "is_nil": bool(record.get("is_nil")),
                "target_iri": target,
                "ranked_candidates": ranked,
                "top1_iri": ranked[0]["iri"] if ranked else None,
                "top1_correct": is_correct if target else None,
            }
        )
    summary = {
        "n_queries": len(records),
        "in_catalog_queries": in_catalog,
        "in_catalog_top1": correct / max(in_catalog, 1),
        "rows": output_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
