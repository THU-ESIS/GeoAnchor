"""GeoAnchor model components and checkpoint loading.

GeoAnchor keeps the language-model backbone frozen and prepends one learned
continuous prefix vector for each supplied entity.  The released checkpoint
uses a frozen 256-dimensional entity table and a LoRA adapter on Qwen3-4B.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class PrefixAdapter(nn.Module):
    """Project frozen entity vectors to continuous language-model prefixes."""

    def __init__(
        self,
        entity_emb: torch.Tensor,
        llm_dim: int,
        num_prefix_per_entity: int = 1,
    ) -> None:
        super().__init__()
        if entity_emb.ndim != 2:
            raise ValueError(f"entity_emb must be 2-D, got {tuple(entity_emb.shape)}")
        if num_prefix_per_entity < 1:
            raise ValueError("num_prefix_per_entity must be positive")
        self.num_prefix = int(num_prefix_per_entity)
        self.llm_dim = int(llm_dim)
        n_entities, embedding_dim = entity_emb.shape

        # Entity id 0 is reserved for a padded slot.  Checkpoint ids are
        # shifted by one when they enter this table.
        padded = torch.zeros(
            (n_entities + 1, embedding_dim),
            dtype=entity_emb.dtype,
            device=entity_emb.device,
        )
        padded[1:] = entity_emb
        self.ent_embeddings = nn.Embedding.from_pretrained(
            padded, freeze=True, padding_idx=0
        )
        self.pretrain_dim = int(embedding_dim)
        self.adapter = nn.Linear(
            self.pretrain_dim, self.num_prefix * self.llm_dim
        )

    def forward(self, entity_ids: torch.LongTensor) -> torch.Tensor:
        if entity_ids.ndim != 2:
            raise ValueError("entity_ids must have shape [batch, entities]")
        batch_size, n_entities = entity_ids.shape
        emb = self.ent_embeddings(entity_ids)
        prefix = self.adapter(emb)
        return prefix.reshape(
            batch_size, n_entities * self.num_prefix, self.llm_dim
        )


class GeoAnchorCausalLM(nn.Module):
    """Causal LM wrapper that accepts entity ids alongside token inputs."""

    def __init__(self, llm: nn.Module, prefix_adapter: PrefixAdapter) -> None:
        super().__init__()
        self.llm = llm
        self.prefix_adapter = prefix_adapter
        self._embed_tokens = llm.get_input_embeddings()

    def get_token_embeds(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self._embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None = None,
        entity_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        if entity_ids is None:
            raise ValueError("entity_ids is required; use an all-zero row for no entity")
        prefix = self.prefix_adapter(entity_ids)
        token_embeds = self.get_token_embeds(input_ids)
        prefix = prefix.to(dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones(
            (input_ids.size(0), prefix.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        full_labels = labels
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), prefix.size(1)),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=-1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        entity_ids: torch.LongTensor,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        prefix = self.prefix_adapter(entity_ids)
        token_embeds = self.get_token_embeds(input_ids)
        prefix = prefix.to(dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones(
            (input_ids.size(0), prefix.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            **generation_kwargs,
        )


def load_entity_table(path: str | Path) -> tuple[torch.Tensor, dict[str, int]]:
    """Load the released entity table and its zero-based IRI index."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"entity_emb", "entity_to_id"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"entity table is missing keys: {sorted(missing)}")
    entity_emb = checkpoint["entity_emb"].float().contiguous()
    entity_to_id = {str(k): int(v) for k, v in checkpoint["entity_to_id"].items()}
    if entity_emb.ndim != 2 or entity_emb.shape[0] != len(entity_to_id):
        raise ValueError("entity_emb and entity_to_id have inconsistent sizes")
    return entity_emb, entity_to_id


def entity_ids_from_iris(
    iris: Sequence[str],
    entity_to_id: Mapping[str, int],
    max_entities: int = 1,
) -> list[int]:
    """Convert zero-based checkpoint ids to padded one-based adapter ids."""
    if max_entities < 1:
        raise ValueError("max_entities must be positive")
    ids = [int(entity_to_id[iri]) + 1 for iri in iris[:max_entities] if iri in entity_to_id]
    return (ids + [0] * max_entities)[:max_entities]


def load_geoanchor(
    base_model: str | Path,
    checkpoint_dir: str | Path,
    entity_table_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> tuple[GeoAnchorCausalLM, Any, dict[str, int]]:
    """Load Qwen3, the LoRA adapter, the prefix adapter and the entity table.

    The base model is intentionally a separate argument because it is not
    redistributed in this repository.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    lora_dir = checkpoint_dir / "lora"
    adapter_state = torch.load(
        checkpoint_dir / "prefix_adapter.pt", map_location="cpu", weights_only=False
    )
    entity_emb, entity_to_id = load_entity_table(entity_table_path)

    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    tokenizer_source = str(lora_dir) if lora_dir.exists() else str(base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm = AutoModelForCausalLM.from_pretrained(
        str(base_model), torch_dtype=dtype, trust_remote_code=True
    )
    llm = PeftModel.from_pretrained(llm, str(lora_dir))
    llm.eval()

    prefix_adapter = PrefixAdapter(
        entity_emb,
        llm_dim=int(adapter_state["llm_dim"]),
        num_prefix_per_entity=int(adapter_state["num_prefix"]),
    )
    prefix_adapter.adapter.load_state_dict(adapter_state["adapter_state"])
    prefix_adapter.eval()

    model = GeoAnchorCausalLM(llm, prefix_adapter).to(device).eval()
    return model, tokenizer, entity_to_id
