"""Chunked generation, VRAM tracking, chat-template inputs for Step 0."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import defaults as _defaults


def format_math_user_content(problem_text: str) -> str:
    """Dataset problem string + fixed HF-recommended directive for math + \\boxed{}."""
    body = problem_text.strip()
    if not body:
        return _defaults.MATH_USER_PROMPT_SUFFIX
    return f"{body}\n\n{_defaults.MATH_USER_PROMPT_SUFFIX}"


def _tensor_ids_from_chat_template(out: Any) -> torch.Tensor:
    """apply_chat_template may return a Tensor or BatchEncoding/BatchFeature (dict-like)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict) and "input_ids" in out:
        return out["input_ids"]
    ids = getattr(out, "input_ids", None)
    if isinstance(ids, torch.Tensor):
        return ids
    raise TypeError(
        f"Unexpected apply_chat_template return type {type(out)}; expected Tensor or object with input_ids"
    )


def build_chat_inputs(
    tokenizer: PreTrainedTokenizer,
    problem_text: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Returns input_ids (1, L) on device and sequence length L."""
    user_content = format_math_user_content(problem_text)
    messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = _tensor_ids_from_chat_template(raw).to(device)
    return input_ids, int(input_ids.shape[1])


def full_sequence_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """All-valid mask for batches with no padding (avoids pad==eos ambiguity in generate)."""
    return torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)


def logits_forward_sanity_check(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, float | bool | list[int]]:
    """
    Single forward pass on the prompt (no generation). Verifies logits exist, are finite,
    and reports shape / coarse stats at the last position. Does not store full logits.
    """
    dev = next(model.parameters()).device
    input_ids = input_ids.to(dev)
    attention_mask = attention_mask.to(dev)
    with torch.inference_mode():
        out = model(input_ids, attention_mask=attention_mask)
        logits = out.logits
    last = logits[:, -1, :].float()
    return {
        "logits_shape": list(logits.shape),
        "vocab_size": int(last.shape[-1]),
        "batch_size": int(last.shape[0]),
        "all_finite": bool(torch.isfinite(last).all().item()),
        "last_pos_logits_min": float(last.min().item()),
        "last_pos_logits_max": float(last.max().item()),
        "last_pos_logits_mean": float(last.mean().item()),
    }


def generate_chunked_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    chunk_size: int,
    temperature: float,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Batched decoding with KV-cache reuse.
    input_ids: (N, L0). Returns (N, L0 + <=max_new_tokens).

    Notes:
    - Unlike the previous implementation, this does not call model.generate on the
      full sequence for every chunk. We prefill once, then decode incrementally
      with past_key_values to avoid repeated full-prefix recomputation.
    - `chunk_size` is kept for API compatibility and optional progress pacing.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D (N, L)")
    if max_new_tokens <= 0:
        return input_ids

    sequences = input_ids
    batch_size = int(input_ids.shape[0])
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    # Track which rows are still active (haven't emitted EOS yet).
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)

    with torch.inference_mode():
        # Prefill once on the full prompt.
        prefill_mask = full_sequence_attention_mask(sequences)
        out = model(sequences, attention_mask=prefill_mask, use_cache=True)
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1, :]

        # Decode incrementally using KV cache.
        for step_idx in range(max_new_tokens):
            if do_sample:
                temp = max(float(temperature), 1e-5)
                probs = torch.softmax(next_logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            # Keep finished rows stable to avoid continuing text growth.
            next_token = torch.where(
                unfinished,
                next_token,
                torch.full_like(next_token, pad_id),
            )

            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)

            if eos_id is not None:
                unfinished = unfinished & (next_token != eos_id)
                if not bool(unfinished.any()):
                    break

            # Optional pacing boundary keeps compatibility with chunk semantics.
            _ = (step_idx + 1) % max(int(chunk_size), 1)

            dec_out = model(
                next_token.unsqueeze(1),
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = dec_out.past_key_values
            next_logits = dec_out.logits[:, -1, :]

    return sequences


def count_generated_tokens_per_row(
    sequences: torch.Tensor, prompt_len: int, eos_id: int | None
) -> list[int]:
    """Number of generated tokens per row (stops at first eos if present)."""
    out: list[int] = []
    for row in sequences:
        gen = row[prompt_len:]
        if eos_id is None:
            out.append(int(gen.numel()))
            continue
        hits = (gen == eos_id).nonzero(as_tuple=True)[0]
        if hits.numel() == 0:
            out.append(int(gen.numel()))
        else:
            out.append(int(hits[0].item()) + 1)
    return out


def cuda_reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def free_hidden_states(obj: Any) -> None:
    if obj is None:
        return
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
