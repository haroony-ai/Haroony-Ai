"""Configuration & directory paths for Haroony AI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"
NOTES_PAGES_DIR = STATIC_DIR / "notes_pages"
SAMPLES_DIR = STATIC_DIR / "samples"
ASSETS_DIR = STATIC_DIR / "assets"
UPLOADS_TMP_DIR = ROOT_DIR / "uploads_tmp"
SETTINGS_PATH = DATA_DIR / "runtime_settings.json"

for _d in (DATA_DIR, TRAINING_DIR, MODELS_DIR, STATIC_DIR, NOTES_PAGES_DIR, SAMPLES_DIR, ASSETS_DIR, UPLOADS_TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv_into_os() -> None:
    """Load local .env file (gitignored) into os.environ if present, without overriding real env vars."""
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    try:
        for raw_ln in env_file.read_text(encoding="utf-8").splitlines():
            ln = raw_ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


_load_dotenv_into_os()

DEFAULT_SETTINGS: Dict[str, Any] = {
    "teacher_pin": os.environ.get("HAROONY_TEACHER_PIN", "0625"),
    "strict_physics_only": True,
    "curriculum": "Cambridge IGCSE Physics (0625) & A-Level",
    "ollama_enabled": os.environ.get("OLLAMA_ENABLED", "false").lower() == "true",
    "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    "ollama_model": os.environ.get("OLLAMA_MODEL", "qwen2.5-vl:7b"),
    "openai_compat_enabled": os.environ.get("OPENAI_COMPAT_ENABLED", "false").lower() == "true",
    "openai_compat_base_url": os.environ.get("OPENAI_COMPAT_BASE_URL", ""),
    "openai_compat_api_key": os.environ.get("OPENAI_COMPAT_API_KEY", ""),
    "openai_compat_model": os.environ.get("OPENAI_COMPAT_MODEL", ""),
}


def load_settings() -> Dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                settings.update(saved)
        except Exception:
            pass
    if os.environ.get("HAROONY_TEACHER_PIN"):
        settings["teacher_pin"] = os.environ["HAROONY_TEACHER_PIN"]
    return settings


def save_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    current = load_settings()
    allowed_keys = set(DEFAULT_SETTINGS.keys())
    for k, v in updates.items():
        if k in allowed_keys:
            current[k] = v
    SETTINGS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current
