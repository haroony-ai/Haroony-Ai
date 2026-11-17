#!/usr/bin/env python3
"""Train a 100% Free, Open-Weights Transformer (LoRA Fine-Tuning) on Harony Physics Materials.

ZERO API KEYS REQUIRED.
Downloads an open-source pretrained LLM (default: Qwen/Qwen2.5-0.5B-Instruct or
HuggingFaceTB/SmolLM2-360M-Instruct), trains a LoRA adapter on Harony's 2026/2027
Notes, Concept Atlas & Cambridge 0625 Mark Schemes, and saves weights to:
  models/harony_lora_adapter/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import MODELS_DIR, TRAINING_DIR


def load_training_samples(max_samples: int) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    seen = set()

    for fname in ("gemini_sft_dataset.jsonl", "harony_sft_dataset.jsonl"):
        p = TRAINING_DIR / fname
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
                if "contents" in obj and len(obj["contents"]) == 2:
                    u = obj["contents"][0]["parts"][0]["text"].strip()
                    a = obj["contents"][1]["parts"][0]["text"].strip()
                elif "messages" in obj:
                    msgs = obj["messages"]
                    u = next((m["content"] for m in msgs if m.get("role") == "user"), "").strip()
                    a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "").strip()
                else:
                    continue
                if len(u) >= 10 and len(a) >= 15 and u not in seen:
                    seen.add(u)
                    samples.append({"user": u[:700], "assistant": a[:900]})
                    if len(samples) >= max_samples:
                        return samples
            except Exception:
                continue
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.environ.get("HAROONY_FREE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--max-samples", type=int, default=450)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=384)
    args = ap.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"Missing free-LLM dependencies ({exc}). Install via: pip install -r requirements-llm.txt"
        )

    samples = load_training_samples(args.max_samples)
    if not samples:
        raise SystemExit("No Harony training samples found in data/training/.")

    out_dir = MODELS_DIR / "harony_lora_adapter"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading free open-source model: {args.base_model} (training on {len(samples)} Harony examples)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sys_msg = (
        "You are Haroony AI, the Cambridge IGCSE (0625) & A-Level Physics tutor for Harony Physics Clinic. "
        "Answer ONLY Physics questions accurately with clear formulas and units."
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, ex in enumerate(samples, 1):
            text = (
                f"<|system|>\n{sys_msg}\n<|user|>\n{ex['user']}\n<|assistant|>\n{ex['assistant']}{tokenizer.eos_token}"
            )
            enc = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach().cpu())
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} step {step}/{len(samples)} avg_loss={total_loss/step:.4f}")

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "harony_base_model.txt").write_text(args.base_model, encoding="utf-8")
    print(f"Saved trained LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
