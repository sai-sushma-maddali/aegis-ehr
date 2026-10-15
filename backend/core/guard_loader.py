"""Load the Tier-2 binary prompt guard (Qwen + LoRA or optional merged folder).

Primary path (recommended):
  local Qwen base (pulled via zrt) + Hugging Face LoRA adapter
  ``SushmaMaddali/qwen-medical-prompt-guard-lora``

Optional legacy path:
  a fully merged ``qwen_medical_guard_merged`` folder on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Settings, get_settings

logger = logging.getLogger("aegis.guard_loader")

ID2LABEL = {0: "SAFE", 1: "PROMPT_INJECTION"}
LABEL2ID = {
    "SAFE": 0,
    "PROMPT_INJECTION": 1,
    "BENIGN_CLINICAL": 0,
    "ADVERSARIAL_ATTACK": 1,
}


def guard_identity(settings: Settings | None = None) -> str:
    """Stable cache key for the resident weight-cache process."""
    cfg = settings or get_settings()
    merged = cfg.guard_merged_model
    if merged is not None and merged.exists():
        return f"merged:{merged.resolve()}"
    base = cfg.qwen_base_model
    adapter = cfg.peft_adapter_id
    if base is not None and Path(base).exists() and adapter:
        return f"peft:{Path(base).resolve()}|{adapter}"
    return ""


def can_load_neural_guard(settings: Settings | None = None) -> bool:
    return bool(guard_identity(settings))


def _dtype_for(device: str, torch: Any) -> Any:
    if (
        device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float32


def load_guard_classifier(
    *,
    device: str | None = None,
    settings: Settings | None = None,
) -> tuple[Any, Any, str]:
    """Return ``(tokenizer, model, identity)`` ready for ``.eval()`` inference."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = settings or get_settings()
    resolved_device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = _dtype_for(resolved_device, torch)
    identity = guard_identity(cfg)
    if not identity:
        raise FileNotFoundError(
            "No guard weights available. Set paths.qwen_base_model to a local "
            "Qwen2.5-7B-Instruct folder (zrt pull) and keep "
            "models.peft_adapter_id = SushmaMaddali/qwen-medical-prompt-guard-lora, "
            "or set paths.guard_merged_model to a merged classifier folder."
        )

    merged = cfg.guard_merged_model
    if merged is not None and merged.exists():
        logger.info("guard_loader.merged path=%s", merged)
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = _from_pretrained_cls(
            AutoModelForSequenceClassification,
            str(merged),
            dtype=dtype,
            local_files_only=True,
        )
    else:
        base = cfg.qwen_base_model
        assert base is not None
        adapter = cfg.peft_adapter_id
        logger.info(
            "guard_loader.peft base=%s adapter=%s",
            base,
            adapter,
        )
        from peft import PeftModel

        # Base weights should already be on disk (zrt). Adapter may download
        # from Hugging Face on first boot (cached afterward).
        tokenizer = AutoTokenizer.from_pretrained(
            str(base),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = _from_pretrained_cls(
            AutoModelForSequenceClassification,
            str(base),
            dtype=dtype,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter,
            local_files_only=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(resolved_device)
    model.eval()
    return tokenizer, model, identity


def _from_pretrained_cls(
    cls: Any,
    name_or_path: str,
    *,
    dtype: Any,
    local_files_only: bool,
) -> Any:
    kwargs = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    try:
        return cls.from_pretrained(name_or_path, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(
            name_or_path, torch_dtype=dtype, **kwargs
        )
