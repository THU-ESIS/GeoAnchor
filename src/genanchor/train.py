"""Train the GeoAnchor prefix adapter and the Qwen3 LoRA adapter."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from .model import GeoAnchorCausalLM, PrefixAdapter, load_entity_table


SYSTEM_PROMPT = "你是地理问答助手, 简短给出答案, 不要解释."


class GeoAnchorDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        entity_to_id: dict[str, int],
        variant: str,
        max_length: int,
    ) -> None:
        self.records = [
            json.loads(line)
            for line in Path(path).open(encoding="utf-8")
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.entity_to_id = entity_to_id
        self.max_entities = 1 if variant == "faithful" else 5
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _format(self, question: str, answer: str | None) -> tuple[str, str | None]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer is None:
            return prompt, None
        return prompt, prompt + answer + self.tokenizer.eos_token

    def _entity_ids(self, iris: list[str]) -> list[int]:
        ids = []
        for iri in iris[: self.max_entities]:
            if iri in self.entity_to_id:
                ids.append(int(self.entity_to_id[iri]) + 1)
            else:
                ids.append(0)
        return (ids + [0] * self.max_entities)[: self.max_entities]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prompt, full = self._format(str(record["input_text"]), str(record["answer"]))
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors=None
        )["input_ids"]
        full_ids = self.tokenizer(
            full, add_special_tokens=False, return_tensors=None
        )["input_ids"]

        if len(full_ids) > self.max_length:
            drop = len(full_ids) - self.max_length
            full_ids = full_ids[drop:]
            prompt_ids = (
                prompt_ids[max(0, len(prompt_ids) - drop) :]
                if drop < len(prompt_ids)
                else []
            )

        answer_start = min(len(prompt_ids), len(full_ids))
        labels = [-100] * answer_start + full_ids[answer_start:]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "entity_ids": self._entity_ids(list(record.get("entities") or [])),
        }


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_tokens = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full(
        (batch_size, max_tokens), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    labels = torch.full((batch_size, max_tokens), -100, dtype=torch.long)
    entity_ids = torch.zeros(
        (batch_size, len(batch[0]["entity_ids"])), dtype=torch.long
    )

    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        start = max_tokens - length
        input_ids[row, start:] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[row, start:] = 1
        labels[row, start:] = torch.tensor(item["labels"], dtype=torch.long)
        entity_ids[row] = torch.tensor(item["entity_ids"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "entity_ids": entity_ids,
    }


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loss(
    model: GeoAnchorCausalLM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = _move(batch, device)
        output = model(**batch)
        size = batch["input_ids"].size(0)
        total += float(output.loss.item()) * size
        count += size
    model.train()
    return total / max(count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--entity-table", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", choices=["faithful", "scenario"], default="faithful")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    )
    llm.config.use_cache = False
    if device.type == "cuda":
        llm.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    if hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()

    entity_emb, entity_to_id = load_entity_table(args.entity_table)
    adapter = PrefixAdapter(
        entity_emb,
        llm_dim=int(llm.config.hidden_size),
        num_prefix_per_entity=1,
    )
    model = GeoAnchorCausalLM(llm, adapter).to(device)

    train_dataset = GeoAnchorDataset(
        args.train_file,
        tokenizer,
        entity_to_id,
        args.variant,
        args.max_length,
    )
    dev_dataset = GeoAnchorDataset(
        args.dev_file,
        tokenizer,
        entity_to_id,
        args.variant,
        args.max_length,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": lambda batch: collate(batch, tokenizer.pad_token_id),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_args)

    adapter_params = list(model.prefix_adapter.adapter.parameters())
    lora_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("prefix_adapter.adapter")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": adapter_params, "lr": args.lr * 5.0},
        ],
        weight_decay=0.0,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else steps_per_epoch * args.epochs
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    log_path = out_dir / "train_log.jsonl"
    log_file = log_path.open("w", encoding="utf-8")
    metadata = {"args": vars(args), "device": str(device), "dtype": str(dtype)}
    (out_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    accumulation = 0
    running_loss = 0.0
    start_time = time.time()
    stop = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = _move(batch, device)
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation
            loss.backward()
            running_loss += float(output.loss.item())
            accumulation += 1
            if accumulation < args.gradient_accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            average_loss = running_loss / accumulation
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": average_loss,
                    "lr_lora": scheduler.get_last_lr()[0],
                    "lr_prefix": scheduler.get_last_lr()[1],
                    "elapsed_s": time.time() - start_time,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            running_loss = 0.0
            accumulation = 0

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(model, tokenizer, out_dir / f"step_{step}", args)
            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break
        dev_loss = evaluate_loss(model, dev_loader, device)
        log_file.write(
            json.dumps(
                {"epoch": epoch, "dev_loss": dev_loss}, ensure_ascii=False
            )
            + "\n"
        )
        log_file.flush()
        if stop:
            break

    save_checkpoint(model, tokenizer, out_dir / "final", args)
    log_file.close()


def save_checkpoint(
    model: GeoAnchorCausalLM,
    tokenizer: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(output_dir / "lora")
    tokenizer.save_pretrained(output_dir / "lora")
    torch.save(
        {
            "adapter_state": model.prefix_adapter.adapter.state_dict(),
            "num_prefix": model.prefix_adapter.num_prefix,
            "llm_dim": model.prefix_adapter.llm_dim,
            "pretrain_dim": model.prefix_adapter.pretrain_dim,
            "variant": args.variant,
        },
        output_dir / "prefix_adapter.pt",
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
