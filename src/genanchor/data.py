"""Data loading and prompt utilities for GeoAnchor experiments."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SYSTEM_PROMPT = "你是地理问答助手, 简短给出答案, 不要解释."


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_name_index(path: str | Path) -> dict[str, list[str]]:
    """Build a display-name to IRI index from the released IRI->name map."""
    iri_to_name = json.loads(Path(path).read_text(encoding="utf-8"))
    name_to_iris: dict[str, list[str]] = defaultdict(list)
    for iri, name in iri_to_name.items():
        if name:
            name_to_iris[str(name)].append(str(iri))
    return dict(name_to_iris)


def record_iris(
    record: Mapping[str, Any],
    name_to_iris: Mapping[str, Sequence[str]],
    max_entities: int = 1,
) -> list[str]:
    """Resolve benchmark display names to the first matching catalog IRI.

    The OSM-SCB files retain display names rather than internal IRIs. The
    release resolves them through the supplied name index. Applications with
    an upstream entity linker can pass IRIs directly instead.
    """
    iris: list[str] = []
    for name in (record.get("entity_names") or [])[:max_entities]:
        matches = list(name_to_iris.get(str(name), ()))
        if matches:
            iris.append(matches[0])
    return iris


def chat_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_option(text: str) -> str:
    """Parse a single A/B/C/D answer from deterministic model output."""
    for line in text.strip().splitlines():
        match = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", line)
        if match:
            return match.group(1)
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else "?"


def pad_batch(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    entity_ids: Sequence[Sequence[int]],
    max_length: int = 1024,
) -> dict[str, Any]:
    prompts = [chat_prompt(tokenizer, str(record["input_text"])) for record in records]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    import torch

    encoded["entity_ids"] = torch.tensor(entity_ids, dtype=torch.long)
    return encoded
