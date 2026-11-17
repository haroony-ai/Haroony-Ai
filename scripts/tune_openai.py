#!/usr/bin/env python3
"""Launch or check an OpenAI Supervised Fine-Tuning job using Harony's full
2026/2027 Physics corpus (`data/training/openai_sft_dataset.jsonl`).

Usage:
  .venv/bin/python scripts/tune_openai.py --base-model gpt-4o-mini-2024-07-18
  .venv/bin/python scripts/tune_openai.py --status ftjob-xxxxx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import TRAINING_DIR  # also loads .env into os.environ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--suffix", default="haroony-physics-2027")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--status", default="", help="Check status of an existing fine-tuning job id")
    ap.add_argument("--wait", action="store_true", help="Poll until fine-tuning finishes")
    args = ap.parse_args()

    api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS", "").split(",")[0]).strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY / OPENAI_API_KEYS in .env before running OpenAI fine-tuning.")

    headers = {"Authorization": f"Bearer {api_key}"}

    if args.status:
        r = httpx.get(f"https://api.openai.com/v1/fine_tuning/jobs/{args.status}", headers=headers, timeout=30.0)
        print("Status:", r.status_code)
        print(r.text[:2000])
        return

    ds_path = TRAINING_DIR / "openai_sft_dataset.jsonl"
    if not ds_path.exists():
        raise SystemExit(f"Missing {ds_path}. Run scripts/export_openai_training.py first.")

    # 1. Upload training file
    with ds_path.open("rb") as f:
        up = httpx.post(
            "https://api.openai.com/v1/files",
            headers=headers,
            data={"purpose": "fine-tune"},
            files={"file": (ds_path.name, f, "application/jsonl")},
            timeout=120.0,
        )
    if up.status_code not in (200, 201):
        raise SystemExit(f"OpenAI file upload failed ({up.status_code}): {up.text[:600]}")

    file_id = up.json()["id"]
    print(f"Uploaded {ds_path.name} -> file_id={file_id}")

    # 2. Launch fine-tuning job
    job_payload = {
        "training_file": file_id,
        "model": args.base_model,
        "suffix": args.suffix,
        "hyperparameters": {"n_epochs": args.epochs},
    }
    job = httpx.post(
        "https://api.openai.com/v1/fine_tuning/jobs",
        headers={**headers, "Content-Type": "application/json"},
        json=job_payload,
        timeout=60.0,
    )
    if job.status_code not in (200, 201):
        raise SystemExit(f"OpenAI fine-tuning launch failed ({job.status_code}): {job.text[:600]}")

    jdata = job.json()
    job_id = jdata["id"]
    print(f"Started OpenAI fine-tuning job: {job_id} (status={jdata.get('status')})")

    if args.wait:
        while True:
            time.sleep(20)
            st = httpx.get(f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}", headers=headers, timeout=30.0).json()
            status = st.get("status")
            ft_model = st.get("fine_tuned_model")
            print(f"[{time.strftime('%H:%M:%S')}] job={job_id} status={status} model={ft_model}")
            if status in ("succeeded", "failed", "cancelled"):
                if ft_model:
                    print(f"\nAdd this to your .env:\nOPENAI_MODEL={ft_model}")
                break


if __name__ == "__main__":
    main()
