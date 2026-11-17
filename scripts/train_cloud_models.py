#!/usr/bin/env python3
"""Unified Cloud Training Pipeline for Haroony AI:
1. Exports Harony 2026/2027 Notes, Mark Schemes, Concept Atlas & Past-Paper QA into:
   - data/training/gemini_sft_dataset.jsonl
   - data/training/openai_sft_dataset.jsonl
2. Validates every JSONL record.
3. If GEMINI_API_KEY(S) and/or OPENAI_API_KEY(S) are configured in .env, optionally
   launches fine-tuning jobs on BOTH Gemini and OpenAI (`--launch`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import TRAINING_DIR  # loads .env


def validate_jsonl(path: Path, key: str) -> int:
    count = 0
    for idx, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        obj = json.loads(ln)
        if key not in obj or not isinstance(obj[key], list):
            raise ValueError(f"Invalid record in {path.name} line {idx}")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="Actually submit fine-tuning jobs to Gemini & OpenAI using keys in .env")
    args = ap.parse_args()

    py = sys.executable
    subprocess.run([py, str(ROOT_DIR / "scripts/export_gemini_training.py")], check=True)
    subprocess.run([py, str(ROOT_DIR / "scripts/export_openai_training.py")], check=True)

    gem_n = validate_jsonl(TRAINING_DIR / "gemini_sft_dataset.jsonl", "contents")
    oai_n = validate_jsonl(TRAINING_DIR / "openai_sft_dataset.jsonl", "messages")
    print(f"Validated datasets -> Gemini: {gem_n} rows | OpenAI: {oai_n} rows")

    if not args.launch:
        print("Datasets ready. Add keys to .env (or run scripts/configure_keys.py) and re-run with --launch to start cloud fine-tuning.")
        return

    has_gem = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS", "")).strip())
    has_oai = bool((os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS", "")).strip())

    if has_gem:
        print("\n--- Launching Gemini Fine-Tuning ---")
        subprocess.run([py, str(ROOT_DIR / "scripts/tune_gemini.py")], check=False)
    else:
        print("\n[Skip Gemini tuning] Set GEMINI_API_KEYS in .env first.")

    if has_oai:
        print("\n--- Launching OpenAI Fine-Tuning ---")
        subprocess.run([py, str(ROOT_DIR / "scripts/tune_openai.py")], check=False)
    else:
        print("\n[Skip OpenAI tuning] Set OPENAI_API_KEYS in .env first.")


if __name__ == "__main__":
    main()
