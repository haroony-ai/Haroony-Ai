"""Hybrid Multi-Key / Multi-Model Neural Bridge for Haroony AI.

100% server-side & secure:
- Reads credentials ONLY from server environment variables (.env gitignored / hosting Secrets).
- Supports MULTI-KEY ROUND-ROBIN + AUTOMATIC FAILOVER POOLS (comma/newline-separated)
  so multiple free-tier official API keys scale capacity seamlessly without rate limits:
  - GROQ_API_KEYS (or GROQ_API_KEY)         — fast open-source Llama-3.3-70B / Qwen / Llama-3.1-8B
  - GEMINI_API_KEYS (or GEMINI_API_KEY)     — multimodal vision + reasoning
  - CEREBRAS_API_KEYS (or CEREBRAS_API_KEY) — ultra-fast open-source Llama-3.3-70B / Qwen
  - OPENROUTER_API_KEYS (or OPENROUTER_API_KEY) — free/open-source model pool
  - HF_TOKENS (or HF_TOKEN / HUGGINGFACE_API_KEY) — HuggingFace Inference Router
  - MISTRAL_API_KEYS (or MISTRAL_API_KEY)   — Mistral free-tier pool
  - OPENAI_API_KEYS (or OPENAI_API_KEY)     — optional cloud multimodal
  - OLLAMA_BASE_URL                         — unlimited local open-source Ollama models
- NEVER exposes third-party provider names or API keys to the frontend UI.
"""
from __future__ import annotations

import base64
import os
import re
import threading
from typing import Any, Dict, List, Optional

import httpx

from backend.config import load_settings

HAROONY_SYSTEM_PROMPT = (
    "You are Haroony AI, the official Cambridge IGCSE (0625) & A-Level (9702) Physics AI Assistant "
    "for Mr. Haroony's Harony Physics Clinic. Never mention third-party model names or API providers.\n\n"
    "STRICT RULES:\n"
    "1. PHYSICS ONLY: Answer ONLY Physics questions. Refuse non-physics requests politely.\n"
    "2. REASON FROM FIRST PRINCIPLES FOR ANY QUESTION: Answer the student's EXACT question directly and accurately, "
    "even if it is a novel scenario or 'why/what happens if' question not word-for-word in the context. "
    "NEVER force an unrelated snippet into your reply, and NEVER repeat an earlier turn's answer when the student switches topic.\n"
    "3. ALIGN WITH HARONY NOTES & CAMBRIDGE 0625 MARK SCHEMES: Use exact definitions, formulas, and bold key marking phrases.\n"
    "4. 4-STEP CALCULATION FORMAT: Formula -> Substitution -> Calculation -> Final Answer with Units (2-3 sig figs).\n"
    "5. IMAGE ACCURACY: When an image is attached, read every label, graph axis, circuit symbol, and number carefully.\n"
)

PREFERRED_OLLAMA_MODELS = [
    "qwen2.5-vl:7b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "llama3.1:8b",
    "deepseek-r1:7b",
    "mistral:7b",
    "phi4:latest",
    "gemma2:9b",
]

_POOL_LOCK = threading.Lock()
_POOL_CURSORS: Dict[str, int] = {}


def _get_key_pool(*env_names: str) -> List[str]:
    """Collect unique non-empty API keys from one or more env vars (comma/semicolon/newline-separated)
    and rotate the starting index round-robin so multiple free-tier keys share load evenly."""
    raw_keys: List[str] = []
    seen = set()
    for name in env_names:
        val = os.environ.get(name, "").strip()
        if not val:
            continue
        for token in re.split(r"[,;\s\n]+", val):
            k = token.strip()
            if k and k not in seen:
                seen.add(k)
                raw_keys.append(k)
    if len(raw_keys) <= 1:
        return raw_keys

    pool_id = "|".join(env_names)
    with _POOL_LOCK:
        start = _POOL_CURSORS.get(pool_id, 0) % len(raw_keys)
        _POOL_CURSORS[pool_id] = (start + 1) % len(raw_keys)
    return raw_keys[start:] + raw_keys[:start]


async def _call_gemini(
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("GEMINI_API_KEYS", "GEMINI_API_KEY")
    if not keys:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    system_parts = []
    contents = []
    for m in messages:
        role = m.get("role", "user")
        txt = m.get("content", "")
        if role == "system":
            system_parts.append({"text": txt})
        else:
            gem_role = "model" if role == "assistant" else "user"
            contents.append({"role": gem_role, "parts": [{"text": txt}]})

    if image_bytes and contents:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        contents[-1]["parts"].append({
            "inline_data": {
                "mime_type": image_mime or "image/jpeg",
                "data": b64,
            }
        })

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    for api_key in keys:
        model_path = model if model.startswith(("models/", "tunedModels/")) else f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
        try:
            async with httpx.AsyncClient(timeout=26.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                cands = data.get("candidates") or []
                if not cands:
                    continue
                parts = cands[0].get("content", {}).get("parts") or []
                out = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
                if out:
                    return out
        except Exception:
            continue
    return None


async def _call_openai_style(
    endpoint_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
    supports_vision: bool = True,
) -> Optional[str]:
    if not endpoint_url or not model:
        return None
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    oai_msgs: List[Dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if supports_vision and idx == len(messages) - 1 and image_bytes and m.get("role") == "user":
            b64 = base64.b64encode(image_bytes).decode("ascii")
            oai_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": m.get("content", "")},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
                ],
            })
        else:
            oai_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    try:
        async with httpx.AsyncClient(timeout=24.0) as client:
            resp = await client.post(
                endpoint_url,
                headers=headers,
                json={
                    "model": model,
                    "messages": oai_msgs,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content", "").strip() or None
    except Exception:
        return None
    return None


async def _call_openai_pool(
    endpoint_url: str,
    keys: List[str],
    models: List[str],
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
    max_tokens: int = 900,
    supports_vision: bool = False,
) -> Optional[str]:
    if not keys:
        return None
    for api_key in keys:
        for model in models:
            if not model:
                continue
            res = await _call_openai_style(
                endpoint_url,
                api_key,
                model,
                messages,
                image_bytes,
                image_mime,
                max_tokens=max_tokens,
                supports_vision=supports_vision,
            )
            if res:
                return res
    return None


async def _call_openai(
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("OPENAI_API_KEYS", "OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return await _call_openai_pool(
        "https://api.openai.com/v1/chat/completions",
        keys,
        [model],
        messages,
        image_bytes,
        image_mime,
        max_tokens=max_tokens,
        supports_vision=True,
    )


async def _call_groq_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("GROQ_API_KEYS", "GROQ_API_KEY")
    primary = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    models = list(dict.fromkeys([primary, "llama-3.3-70b-versatile", "qwen-qwq-32b", "llama-3.1-8b-instant"]))
    return await _call_openai_pool(
        "https://api.groq.com/openai/v1/chat/completions",
        keys,
        models,
        messages,
        max_tokens=max_tokens,
        supports_vision=False,
    )


async def _call_cerebras_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("CEREBRAS_API_KEYS", "CEREBRAS_API_KEY")
    primary = os.environ.get("CEREBRAS_MODEL", "").strip() or "llama-3.3-70b"
    models = list(dict.fromkeys([primary, "llama-3.3-70b", "qwen-3-32b", "llama3.1-8b"]))
    return await _call_openai_pool(
        "https://api.cerebras.ai/v1/chat/completions",
        keys,
        models,
        messages,
        max_tokens=max_tokens,
        supports_vision=False,
    )


async def _call_huggingface_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("HF_TOKENS", "HF_TOKEN", "HUGGINGFACE_API_KEY")
    primary = os.environ.get("HF_MODEL", "").strip() or "Qwen/Qwen2.5-72B-Instruct"
    models = list(dict.fromkeys([primary, "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]))
    return await _call_openai_pool(
        "https://router.huggingface.co/v1/chat/completions",
        keys,
        models,
        messages,
        max_tokens=max_tokens,
        supports_vision=False,
    )


async def _call_openrouter_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
    primary = os.environ.get("OPENROUTER_MODEL", "").strip() or "meta-llama/llama-3.3-70b-instruct:free"
    models = list(dict.fromkeys([primary, "meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"]))
    return await _call_openai_pool(
        "https://openrouter.ai/api/v1/chat/completions",
        keys,
        models,
        messages,
        max_tokens=max_tokens,
        supports_vision=False,
    )


async def _call_mistral_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    keys = _get_key_pool("MISTRAL_API_KEYS", "MISTRAL_API_KEY")
    primary = os.environ.get("MISTRAL_MODEL", "").strip() or "mistral-small-latest"
    models = list(dict.fromkeys([primary, "mistral-small-latest", "open-mistral-nemo"]))
    return await _call_openai_pool(
        "https://api.mistral.ai/v1/chat/completions",
        keys,
        models,
        messages,
        max_tokens=max_tokens,
        supports_vision=False,
    )


async def _call_ollama(
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
) -> Optional[str]:
    base = (os.environ.get("OLLAMA_BASE_URL") or settings.get("ollama_base_url") or "http://127.0.0.1:11434").rstrip("/")
    enabled = bool(settings.get("ollama_enabled") or os.environ.get("OLLAMA_ENABLED", "").lower() == "true" or os.environ.get("OLLAMA_BASE_URL"))
    if not enabled:
        return None

    primary_model = (os.environ.get("OLLAMA_MODEL") or settings.get("ollama_model") or "qwen2.5-vl:7b").strip()
    candidate_models = [primary_model]

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            tags_resp = await client.get(f"{base}/api/tags")
            if tags_resp.status_code == 200:
                installed = [m.get("name", "") for m in tags_resp.json().get("models", []) if m.get("name")]
                for pm in PREFERRED_OLLAMA_MODELS:
                    if pm in installed and pm not in candidate_models:
                        candidate_models.append(pm)
                for im in installed:
                    if im not in candidate_models:
                        candidate_models.append(im)
    except Exception:
        return None

    for model in candidate_models[:3]:
        ollama_msgs: List[Dict[str, Any]] = []
        is_vision_model = any(v in model.lower() for v in ("vl", "vision", "llava", "minicpm-v"))
        for idx, m in enumerate(messages):
            item: Dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if is_vision_model and idx == len(messages) - 1 and image_bytes and m.get("role") == "user":
                item["images"] = [base64.b64encode(image_bytes).decode("ascii")]
            ollama_msgs.append(item)

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    f"{base}/api/chat",
                    json={"model": model, "messages": ollama_msgs, "stream": False},
                )
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "").strip()
                    if content:
                        return content
        except Exception:
            continue
    return None


async def _call_openai_compat(
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
) -> Optional[str]:
    if not settings.get("openai_compat_enabled"):
        return None
    base = (settings.get("openai_compat_base_url") or "").rstrip("/")
    model = settings.get("openai_compat_model") or ""
    if not base or not model:
        return None
    api_key = (settings.get("openai_compat_api_key") or "").strip()
    return await _call_openai_style(
        f"{base}/chat/completions",
        api_key,
        model,
        messages,
        image_bytes,
        image_mime,
        max_tokens=max_tokens,
        supports_vision=True,
    )


def _sanitize_provider_mentions(text: str) -> str:
    """Ensure internal model/vendor names never leak into student-facing text."""
    if not text:
        return text
    text = re.sub(
        r"\b(?:Gemini|OpenAI|ChatGPT|GPT-4o?|Claude|Groq|Cerebras|OpenRouter|Llama|Qwen|Mistral|DeepSeek)\b",
        "Haroony AI",
        text,
        flags=re.I,
    )
    return text


async def try_neural_generation(
    user_prompt: str,
    context_block: str,
    history: Optional[List[Dict[str, str]]] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> Optional[str]:
    """Cascade across multi-key server-side free-tier, cloud, and local open-source LLM pools."""
    settings = load_settings()

    messages: List[Dict[str, str]] = [{"role": "system", "content": HAROONY_SYSTEM_PROMPT}]
    if history:
        for h in history[-6:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:1400]})

    prompt_with_ctx = (
        f"HARONY PHYSICS REFERENCE CONTEXT:\n{context_block}\n\n"
        f"STUDENT QUESTION:\n{user_prompt}\n\n"
        f"Respond clearly in Markdown grounded ONLY in Physics & the Harony context above."
    )
    messages.append({"role": "user", "content": prompt_with_ctx})

    for caller in (
        lambda: _call_gemini(messages, image_bytes, image_mime),
        lambda: _call_groq_opensource(messages),
        lambda: _call_cerebras_opensource(messages),
        lambda: _call_openrouter_opensource(messages),
        lambda: _call_huggingface_opensource(messages),
        lambda: _call_mistral_opensource(messages),
        lambda: _call_openai(messages, image_bytes, image_mime),
        lambda: _call_ollama(settings, messages, image_bytes),
        lambda: _call_openai_compat(settings, messages, image_bytes, image_mime),
    ):
        res = await caller()
        if res:
            return _sanitize_provider_mentions(res)

    return None


async def try_neural_vision_ocr(
    image_bytes: bytes,
    image_mime: str = "image/jpeg",
    purpose: str = "physics_question",
) -> Optional[str]:
    """Optional high-accuracy vision transcription for Physics problem screenshots."""
    if not image_bytes:
        return None
    prompt = "Transcribe ALL text, numbers, units, and diagram/graph labels from this Physics problem image cleanly."
    msgs = [{"role": "user", "content": prompt}]
    settings = load_settings()
    for caller in (
        lambda: _call_gemini(msgs, image_bytes, image_mime, max_tokens=700),
        lambda: _call_openai(msgs, image_bytes, image_mime, max_tokens=700),
        lambda: _call_ollama(settings, msgs, image_bytes),
    ):
        res = await caller()
        if res:
            return res.strip()
    return None


async def has_active_server_llm() -> bool:
    """Return True if at least one multi-key server LLM or local Ollama LLM is configured/reachable."""
    if any(
        os.environ.get(k, "").strip()
        for k in (
            "GEMINI_API_KEYS", "GEMINI_API_KEY",
            "GROQ_API_KEYS", "GROQ_API_KEY",
            "CEREBRAS_API_KEYS", "CEREBRAS_API_KEY",
            "OPENROUTER_API_KEYS", "OPENROUTER_API_KEY",
            "HF_TOKENS", "HF_TOKEN", "HUGGINGFACE_API_KEY",
            "MISTRAL_API_KEYS", "MISTRAL_API_KEY",
            "OPENAI_API_KEYS", "OPENAI_API_KEY",
        )
    ):
        return True
    settings = load_settings()
    if settings.get("openai_compat_enabled") and settings.get("openai_compat_base_url") and settings.get("openai_compat_model"):
        return True
    ollama_enabled = bool(settings.get("ollama_enabled") or os.environ.get("OLLAMA_ENABLED", "").lower() == "true" or os.environ.get("OLLAMA_BASE_URL"))
    if ollama_enabled:
        base = (os.environ.get("OLLAMA_BASE_URL") or settings.get("ollama_base_url") or "http://127.0.0.1:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(f"{base}/api/tags")
                return r.status_code == 200
        except Exception:
            return False
    return False
