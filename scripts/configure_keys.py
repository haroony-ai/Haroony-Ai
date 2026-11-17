#!/usr/bin/env python3
"""Interactive Server Key Setup for Haroony AI (writes ONLY to gitignored .env).

Never commits or exposes keys to GitHub or frontend code.
Optionally tests Gemini + OpenAI keys and launches fine-tuning on Harony materials.
"""
from __future__ import annotations

import getpass
import subprocess
import sys
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


def test_gemini_key(key: str) -> bool:
    if not key:
        return False
    try:
        r = httpx.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=15.0)
        return r.status_code == 200
    except Exception:
        return False


def test_openai_key(key: str) -> bool:
    if not key:
        return False
    try:
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    print("=== Haroony AI — Secure Server Key Setup (.env) ===")
    print("Keys are saved ONLY to local gitignored .env (never pushed to GitHub).\n")

    gem_key = getpass.getpass("Paste your Google Gemini API Key(s) (comma-separated, or Enter to skip): ").strip()
    oai_key = getpass.getpass("Paste your OpenAI API Key(s) (comma-separated, or Enter to skip): ").strip()

    if not gem_key and not oai_key:
        print("No keys entered. Nothing changed.")
        return

    if gem_key:
        ok_g = test_gemini_key(gem_key.split(",")[0].strip())
        print(f"Gemini key check: {'OK' if ok_g else 'Could not verify (check key/network)'}")
    if oai_key:
        ok_o = test_openai_key(oai_key.split(",")[0].strip())
        print(f"OpenAI key check: {'OK' if ok_o else 'Could not verify (check key/network)'}")

    existing = {}
    if ENV_PATH.exists():
        for ln in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                existing[k.strip()] = v.strip()

    if gem_key:
        existing["GEMINI_API_KEYS"] = gem_key
        existing.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
    if oai_key:
        existing["OPENAI_API_KEYS"] = oai_key
        existing.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    existing["HAROONY_LLM_MODE"] = "dual"

    lines = [f"{k}={v}" for k, v in existing.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved securely to {ENV_PATH} (gitignored).")

    ans = input("Launch Gemini & OpenAI fine-tuning on Harony materials now? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        subprocess.run([sys.executable, str(ROOT_DIR / "scripts/train_cloud_models.py"), "--launch"], check=False)


if __name__ == "__main__":
    main()
