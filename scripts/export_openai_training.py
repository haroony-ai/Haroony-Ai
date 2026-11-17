#!/usr/bin/env python3
"""Export Harony Physics Clinic's complete 2026/2027 corpus into official
OpenAI Chat Completions Fine-Tuning JSONL (`data/training/openai_sft_dataset.jsonl`).

Format matches OpenAI fine-tuning schema:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import DATA_DIR, TRAINING_DIR
from backend.llm_bridge import HAROONY_SYSTEM_PROMPT


def main() -> None:
    sft_path = TRAINING_DIR / "harony_sft_dataset.jsonl"
    gem_path = TRAINING_DIR / "gemini_sft_dataset.jsonl"
    out_path = TRAINING_DIR / "openai_sft_dataset.jsonl"

    rows_out: List[str] = []
    seen_users = set()

    if sft_path.exists():
        for ln in sft_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
                msgs = obj.get("messages") or []
                user_txt = next((m["content"] for m in msgs if m.get("role") == "user"), "").strip()
                asst_txt = next((m["content"] for m in msgs if m.get("role") == "assistant"), "").strip()
                if len(user_txt) >= 10 and len(asst_txt) >= 15 and user_txt not in seen_users:
                    seen_users.add(user_txt)
                    rows_out.append(json.dumps({
                        "messages": [
                            {"role": "system", "content": HAROONY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_txt},
                            {"role": "assistant", "content": asst_txt},
                        ]
                    }, ensure_ascii=False))
            except Exception:
                continue

    if gem_path.exists():
        for ln in gem_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
                contents = obj.get("contents") or []
                if len(contents) == 2:
                    u = contents[0]["parts"][0]["text"].strip()
                    a = contents[1]["parts"][0]["text"].strip()
                    if len(u) >= 10 and len(a) >= 15 and u not in seen_users:
                        seen_users.add(u)
                        rows_out.append(json.dumps({
                            "messages": [
                                {"role": "system", "content": HAROONY_SYSTEM_PROMPT},
                                {"role": "user", "content": u},
                                {"role": "assistant", "content": a},
                            ]
                        }, ensure_ascii=False))
            except Exception:
                continue

    # Add Physics-only refusal & Concept Atlas exemplars so the tuned OpenAI model respects guardrails
    atlas_path = DATA_DIR / "harony_concept_atlas.json"
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        for entry in atlas:
            topic = entry.get("topic", "")
            unit = entry.get("unit", "")
            q = f"Explain {topic} in IGCSE Physics ({unit})."
            if q in seen_users:
                continue
            seen_users.add(q)
            pts = "\n".join(f"- {p}" for p in entry.get("points", []))
            fams = "\n".join(f"- `{f}`" for f in entry.get("formulas", []))
            ans = f"### {topic} *({unit})*\n\n{entry.get('summary', '')}\n\n{pts}"
            if fams:
                ans += f"\n\n**Key Equations:**\n{fams}"
            if entry.get("examiner_trap"):
                ans += f"\n\n> **Examiner Trap:** {entry['examiner_trap']}"
            rows_out.append(json.dumps({
                "messages": [
                    {"role": "system", "content": HAROONY_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ]
            }, ensure_ascii=False))

    for off_q in (
        "What is the capital of France?",
        "Write me a Python web scraper.",
        "Give me a pasta recipe.",
        "Explain mitosis in biology.",
    ):
        rows_out.append(json.dumps({
            "messages": [
                {"role": "system", "content": HAROONY_SYSTEM_PROMPT},
                {"role": "user", "content": off_q},
                {
                    "role": "assistant",
                    "content": "I am **Haroony AI** (Harony Physics Clinic), trained strictly for **Physics** (IGCSE & A-Level). Please ask me a Physics question instead!",
                },
            ]
        }, ensure_ascii=False))

    out_path.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    print(f"Exported {len(rows_out)} OpenAI SFT training conversations to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
