"""Zero-API-Key Free Open-Weights Local Transformer Runtime for Haroony AI.

Loads a free open-source pretrained Transformer (e.g., Qwen/Qwen2.5-0.5B-Instruct)
combined with our Harony-trained LoRA adapter (`models/harony_lora_adapter/`)
when `HAROONY_LOCAL_LLM=true` or when `models/harony_lora_adapter` exists.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import MODELS_DIR

_LOCK = threading.Lock()
_CACHED_MODEL: Any = None
_CACHED_TOKENIZER: Any = None
_LOAD_FAILED = False


def is_local_llm_enabled() -> bool:
    adapter_dir = MODELS_DIR / "harony_lora_adapter"
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        return True
    return os.environ.get("HAROONY_LOCAL_LLM", "").strip().lower() in ("1", "true", "yes")


def _ensure_loaded() -> bool:
    global _CACHED_MODEL, _CACHED_TOKENIZER, _LOAD_FAILED
    if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
        return True
    if _LOAD_FAILED:
        return False

    with _LOCK:
        if _CACHED_MODEL is not None and _CACHED_TOKENIZER is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            adapter_dir = MODELS_DIR / "harony_lora_adapter"
            base_model = os.environ.get("HAROONY_FREE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip()
            if (adapter_dir / "harony_base_model.txt").exists():
                base_model = (adapter_dir / "harony_base_model.txt").read_text(encoding="utf-8").strip() or base_model

            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else base_model,
                trust_remote_code=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
            if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_dir))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            _CACHED_MODEL = model
            _CACHED_TOKENIZER = tokenizer
            return True
        except Exception:
            _LOAD_FAILED = True
            return False


def _generate_sync(messages: List[Dict[str, str]], max_new_tokens: int = 360) -> Optional[str]:
    if not _ensure_loaded():
        return None
    try:
        import torch

        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if content:
                prompt_parts.append(f"<|{role}|>\n{content}")
        prompt_parts.append("<|assistant|>\n")
        prompt_text = "\n".join(prompt_parts)

        inputs = _CACHED_TOKENIZER(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(_CACHED_MODEL.device)

        with torch.no_grad():
            out_ids = _CACHED_MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_CACHED_TOKENIZER.pad_token_id,
                eos_token_id=_CACHED_TOKENIZER.eos_token_id,
            )
        gen_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        text = _CACHED_TOKENIZER.decode(gen_tokens, skip_special_tokens=True).strip()
        return text or None
    except Exception:
        return None


async def try_local_transformer_generation(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 360,
) -> Optional[str]:
    if not is_local_llm_enabled():
        return None
    return await asyncio.to_thread(_generate_sync, messages, max_new_tokens)
