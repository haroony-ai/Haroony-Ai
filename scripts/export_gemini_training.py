#!/usr/bin/env python3
"""Export Harony Physics Clinic's complete 2026/2027 corpus into official
Google Gemini Supervised Fine-Tuning JSONL (`data/training/gemini_sft_dataset.jsonl`).

Format matches Gemini API / Google AI Studio fine-tuning schema:
{"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import DATA_DIR, TRAINING_DIR


def main() -> None:
    sft_path = TRAINING_DIR / "harony_sft_dataset.jsonl"
    out_path = TRAINING_DIR / "gemini_sft_dataset.jsonl"

    rows_out: List[str] = []
    if sft_path.exists():
        for ln in sft_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
                msgs = obj.get("messages") or []
                user_txt = next((m["content"] for m in msgs if m.get("role") == "user"), "").strip()
                asst_txt = next((m["content"] for m in msgs if m.get("role") == "assistant"), "").strip()
                if len(user_txt) >= 10 and len(asst_txt) >= 15:
                    rows_out.append(json.dumps({
                        "contents": [
                            {"role": "user", "parts": [{"text": user_txt}]},
                            {"role": "model", "parts": [{"text": asst_txt}]},
                        ]
                    }, ensure_ascii=False))
            except Exception:
                continue

    # Also add everyday physics & calculation exemplars grounded in Harony 0625 mark schemes
    exemplars = [
        (
            "The average density of Jupiter is 1300 kg/m3 and its volume is 1.4 x 10^15 km3. Calculate the mass of Jupiter.",
            "### Step-by-Step IGCSE Physics Calculation\n\n"
            "**1. Given Quantities & SI Conversions:**\n"
            "- Density `ρ = 1300 kg/m³`\n"
            "- Volume `V = 1.4 × 10^15 km³ = 1.4 × 10^15 × 10^9 m³ = 1.4 × 10^24 m³`\n\n"
            "**2. Formula [C1]:**\n"
            "- `m = ρ × V`\n\n"
            "**3. Substitution & Working [C1]:**\n"
            "- `m = 1300 × 1.4 × 10^24 = 1.82 × 10^27 kg`\n\n"
            "**4. Final Answer [A1]:**\n"
            "- **`1.8 × 10^27 kg`** *(2 s.f.)*",
        ),
        (
            "Why does a pressure cooker cook food faster than an open pot?",
            "### Why a Pressure Cooker Cooks Food Faster *(Thermal Physics)*\n\n"
            "- A pressure cooker seals in steam so the **gas pressure above the water rises** above normal atmospheric pressure.\n"
            "- Increasing pressure **raises the boiling point of water** above `100 °C` (typically `~120 °C`).\n"
            "- Because the water and steam reach a **higher temperature**, thermal energy transfers into the food faster, cooking it in less time.",
        ),
        (
            "Why do we feel cold when stepping out of a swimming pool on a windy day?",
            "### Why We Feel Cold After Swimming in the Wind *(Evaporation Cooling)*\n\n"
            "- Water on your skin **evaporates**: the **fastest, highest-kinetic-energy molecules** overcome intermolecular forces and escape into the air.\n"
            "- This reduces the **average kinetic energy** of the remaining water molecules (and takes **latent heat of vaporisation** from your skin), causing **cooling**.\n"
            "- **Wind** continuously sweeps away humid air (water vapour) above your skin, **increasing the rate of evaporation** and making you feel colder faster.",
        ),
        (
            "If a car travels 100 m north then 100 m south in 20 s, what is its average speed and average velocity?",
            "### Average Speed vs Average Velocity *(Kinematics)*\n\n"
            "- **Total distance** `= 100 m + 100 m = 200 m` (scalar) → **Average speed** `= 200 m / 20 s =` **`10 m/s`**.\n"
            "- **Net displacement** `= 100 m north − 100 m south = 0 m` (vector) → **Average velocity** `= 0 m / 20 s =` **`0 m/s`**.",
        ),
    ]
    for u, a in exemplars:
        rows_out.append(json.dumps({
            "contents": [
                {"role": "user", "parts": [{"text": u}]},
                {"role": "model", "parts": [{"text": a}]},
            ]
        }, ensure_ascii=False))

    out_path.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    print(f"Exported {len(rows_out)} Gemini SFT training conversations to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
