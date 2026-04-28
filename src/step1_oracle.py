"""Step 1 oracle collection helpers (checkpoint snapshots + full token IDs)."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import defaults as D
from src.step0_utils import build_chat_inputs


def _sample_next(next_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-5)
    probs = torch.softmax(next_logits / temp, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _logprob_of_sampled(
    logits: torch.Tensor, sampled_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """log p(x | logits) for each row, matching sampling temperature (float32 stable)."""
    temp = max(float(temperature), 1e-5)
    lp = torch.nn.functional.log_softmax(logits.float() / temp, dim=-1)
    return lp.gather(-1, sampled_ids.unsqueeze(-1).long()).squeeze(-1)


def collect_problem_oracle(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problem_text: str,
    device: torch.device,
    *,
    n_branches: int = D.N_BRANCHES,
    max_new_tokens: int = D.STEP1_MAX_NEW_TOKENS,
    checkpoints: list[int] | None = None,
    temperature: float = D.TEMPERATURE,
    do_sample: bool = True,
) -> dict[str, Any]:
    """Collect checkpoint snapshots and full generated token IDs for one problem."""
    if checkpoints is None:
        checkpoints = D.STEP1_CHECKPOINTS
    checkpoints = sorted(int(x) for x in checkpoints if x > 0 and x <= max_new_tokens)

    single_ids, prompt_len = build_chat_inputs(tokenizer, problem_text, device)
    input_ids = single_ids.repeat(n_branches, 1)

    batch = n_branches
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0

    gen_tokens = torch.full(
        (batch, max_new_tokens),
        fill_value=pad_id,
        dtype=input_ids.dtype,
        device=device,
    )
    valid_lengths = torch.full((batch,), fill_value=max_new_tokens, dtype=torch.long, device=device)
    finished_at = torch.full((batch,), fill_value=-1, dtype=torch.long, device=device)
    unfinished = torch.ones((batch,), dtype=torch.bool, device=device)

    checkpoint_hidden_states: dict[int, torch.Tensor] = {}
    checkpoint_logits: dict[int, torch.Tensor] = {}
    checkpoint_cumulative_logprob: dict[int, torch.Tensor] = {}
    alive_mask: dict[int, torch.Tensor] = {}
    coverage: dict[int, dict[str, Any]] = {}

    per_step_logprob = torch.zeros((batch, max_new_tokens), dtype=torch.float32, device=device)
    cum_logprob = torch.zeros((batch,), dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    cfg = model.config
    hidden_size = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError(
            "model.config must set hidden_size (or n_embd) for Step 1 oracle padding tensors."
        )

    with torch.inference_mode():
        prefill_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        prefill = model(input_ids, attention_mask=prefill_mask, use_cache=True)
        past = prefill.past_key_values
        # Keep only the last-row logits; drop `prefill` so we do not retain the full
        # [batch, prompt_len, vocab] logits tensor for the entire decode (multi-GB).
        next_logits = prefill.logits[:, -1, :].detach().clone()
        del prefill

        for step in range(1, max_new_tokens + 1):
            # Branches that are active at the start of this decoding step.
            # These branches contribute meaningful state/logits for checkpoint `step`,
            # including rows that emit EOS at this very step.
            alive_before_step = unfinished.clone()

            if do_sample:
                sampled = _sample_next(next_logits, temperature)
            else:
                sampled = torch.argmax(next_logits, dim=-1)

            step_logp = _logprob_of_sampled(next_logits, sampled, temperature)
            step_logp_masked = torch.where(alive_before_step, step_logp, torch.zeros_like(step_logp))
            per_step_logprob[:, step - 1] = step_logp_masked
            cum_logprob = cum_logprob + step_logp_masked

            next_token = torch.where(
                unfinished,
                sampled,
                torch.full_like(sampled, pad_id),
            )
            gen_tokens[:, step - 1] = next_token

            if eos_id is not None:
                newly_finished = unfinished & (next_token == eos_id)
                valid_lengths = torch.where(
                    newly_finished,
                    torch.full_like(valid_lengths, step),
                    valid_lengths,
                )
                finished_at = torch.where(
                    newly_finished,
                    torch.full_like(finished_at, step),
                    finished_at,
                )
                unfinished = unfinished & (~newly_finished)

            # Hidden states only needed at checkpoints; output_hidden_states=True costs
            # extra work every step if left always on (independent of Flash Attention).
            need_hidden = step in checkpoints
            dec_out = model(
                next_token.unsqueeze(1),
                use_cache=True,
                past_key_values=past,
                output_hidden_states=need_hidden,
            )
            past = dec_out.past_key_values
            next_logits = dec_out.logits[:, -1, :]
            last_hidden = (
                dec_out.hidden_states[-1][:, -1, :] if need_hidden else None
            )

            if step in checkpoints:
                assert last_hidden is not None
                step_alive = alive_before_step
                hidden_cp = last_hidden.detach().clone()
                logits_cp = next_logits.detach().clone()
                hidden_cp[~step_alive] = 0
                logits_cp[~step_alive] = 0
                checkpoint_hidden_states[step] = hidden_cp.to("cpu")
                checkpoint_logits[step] = logits_cp.to("cpu")
                # Cumulative logprob after tokens x_1..x_step (sum of logprob_1..logprob_step).
                checkpoint_cumulative_logprob[step] = cum_logprob.detach().clone().to("cpu")
                alive_mask[step] = step_alive.to("cpu")
                alive_idx = torch.nonzero(step_alive, as_tuple=True)[0].tolist()
                coverage[step] = {
                    "num_alive": int(step_alive.sum().item()),
                    "alive_indices": [int(i) for i in alive_idx],
                }

            if not bool(unfinished.any()):
                # Fill remaining checkpoint entries as zero/false if all branches finished early.
                # Use explicit shapes: last forward may not have hidden states (non-checkpoint step).
                z_h = torch.zeros(
                    batch,
                    hidden_size,
                    device="cpu",
                    dtype=next_logits.dtype,
                )
                z_logits = torch.zeros_like(next_logits, device="cpu")
                for cp in checkpoints:
                    if cp in checkpoint_hidden_states or cp < step:
                        continue
                    checkpoint_hidden_states[cp] = z_h.clone()
                    checkpoint_logits[cp] = z_logits.clone()
                    # Sequence logprob does not change after EOS; repeat final cumulative (not zeros).
                    checkpoint_cumulative_logprob[cp] = cum_logprob.detach().clone().to("cpu")
                    alive_mask[cp] = torch.zeros((batch,), dtype=torch.bool, device="cpu")
                    coverage[cp] = {"num_alive": 0, "alive_indices": []}
                break

    elapsed = time.perf_counter() - t0
    return {
        "prompt_len": prompt_len,
        "generated_token_ids": gen_tokens.to("cpu"),
        "valid_lengths": valid_lengths.to("cpu"),
        "finished_at": finished_at.to("cpu"),
        "per_step_logprob": per_step_logprob.to("cpu"),
        "checkpoint_hidden_states": checkpoint_hidden_states,
        "checkpoint_logits": checkpoint_logits,
        "checkpoint_cumulative_logprob": checkpoint_cumulative_logprob,
        "alive_mask": alive_mask,
        "coverage": coverage,
        "elapsed_sec": elapsed,
    }
