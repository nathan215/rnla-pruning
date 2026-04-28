"""Load tokenizer and causal LM for Step 0."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import defaults as _defaults

logger = logging.getLogger(__name__)


def _resolve_attn_implementation(device: torch.device) -> str:
    """CUDA: env ATTN_IMPLEMENTATION or defaults.ATTN_IMPLEMENTATION; CPU: eager."""
    if device.type != "cuda":
        return "eager"
    impl = os.environ.get("ATTN_IMPLEMENTATION", _defaults.ATTN_IMPLEMENTATION)
    if impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash_attn not installed; using sdpa instead. "
                "Install: pip install flash-attn (Linux/CUDA; see requirements.txt note)."
            )
            return "sdpa"
    return impl


def resolve_model_id(model_id: str, project_root: Path | None = None) -> str:
    """若已用 scripts/download_assets.py 下載 snapshot，優先使用本機路徑。"""
    project_root = project_root or Path(__file__).resolve().parents[1]
    local = project_root / "models" / "cache" / model_id.replace("/", "--")
    if (local / "config.json").is_file():
        return str(local)
    return model_id


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
):
    dtype = dtype or torch.bfloat16
    model_id = resolve_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    attn_impl = _resolve_attn_implementation(device)
    kwargs: dict = {
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": attn_impl,
    }
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        if device.type == "cuda" and kwargs.get("attn_implementation") == "flash_attention_2":
            logger.warning("flash_attention_2 load failed (%s); retrying with sdpa.", e)
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        elif device.type == "cuda" and kwargs.get("attn_implementation") == "sdpa":
            logger.warning("sdpa load failed (%s); retrying with eager.", e)
            kwargs["attn_implementation"] = "eager"
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            raise
    if kwargs.get("device_map") is None:
        model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and tokenizer.pad_token_id is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer
