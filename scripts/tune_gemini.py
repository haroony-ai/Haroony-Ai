#!/usr/bin/env python3
"""Launch a Supervised Fine-Tuning job on Google's Gemini API using Harony's full
2026/2027 Physics corpus (`data/training/gemini_sft_dataset.jsonl`).

Usage:
  export GEMINI_API_KEY="your-google-ai-studio-key"
  .venv/bin/python scripts/tune_gemini.py --base-model models/gemini-1.5-flash-001-tuning
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import TRAINING_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="models/gemini-1.5-flash-001-tuning")
    ap.add_argument("--display-name", default="Haroony-Physics-AI-2027")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-examples", type=int, default=1200)
    args = ap.parse_args()

    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS", "").split(",")[0]).strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment or .env before launching Gemini tuning.")

    ds_path = TRAINING_DIR / "gemini_sft_dataset.jsonl"
    if not ds_path.exists():
        raise SystemExit(f"Missing {ds_path}. Run scripts/export_gemini_training.py first.")

    examples = []
    for ln in ds_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        contents = obj.get("contents") or []
        if len(contents) == 2:
            u = contents[0]["parts"][0]["text"][:1800]
            m = contents[1]["parts"][0]["text"][:1800]
            examples.append({"text_input": u, "output": m})
        if len(examples) >= args.max_examples:
            break

    payload = {
        "display_name": args.display_name,
        "base_model": args.base_model,
        "tuning_task": {
            "hyperparameters": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "epoch_count": args.epochs,
            },
            "training_data": {
                "examples": {"examples": examples}
            },
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/tunedModels?key={api_key}"
    resp = httpx.post(url, json=payload, timeout=120.0)
    print("Status:", resp.status_code)
    print(resp.text[:2000])


if __name__ == "__main__":
    main()
