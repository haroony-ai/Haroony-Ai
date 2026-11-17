"""Dual Gemini + OpenAI Neural Engine for Haroony AI (Grounded in Harony 2026/2027 RAG).

100% server-side & secure:
- Reads credentials ONLY from server environment variables (.env gitignored / hosting Secrets).
- Supports MULTI-KEY ROUND-ROBIN + AUTOMATIC FAILOVER POOLS (comma/newline-separated):
  - GEMINI_API_KEYS / GEMINI_API_KEY   — Google Gemini (supports tunedModels/... + Vision)
  - OPENAI_API_KEYS / OPENAI_API_KEY   — OpenAI (supports ft:... fine-tuned models + Vision)
- Supports `HAROONY_LLM_MODE`:
  - `dual` (default when both Gemini & OpenAI keys are set): runs Gemini & OpenAI concurrently,
    verifies numerical/mark-scheme consistency against Harony RAG context, and fails over seamlessly.
  - `gemini_first`: tries Gemini first, fails over to OpenAI.
  - `openai_first`: tries OpenAI first, fails over to Gemini.
- NEVER exposes third-party provider names or API keys to the frontend UI or /api/status.
"""
from __future__ import annotations

import asyncio
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
    "1. PHYSICS ONLY: Answer ONLY Physics questions. Politely refuse non-physics requests.\n"
    "2. REASON FROM FIRST PRINCIPLES FOR ANY QUESTION: Answer the student's EXACT question directly and accurately, "
    "even if it is a novel everyday scenario or 'why/what happens if' question not word-for-word in the reference context. "
    "NEVER force an unrelated snippet into your reply, and NEVER repeat an earlier turn's answer when the student switches topic.\n"
    "3. ALIGN WITH HARONY NOTES & CAMBRIDGE 0625 MARK SCHEMES: Use exact definitions, formulas, and bold key marking phrases.\n"
    "4. 4-STEP CALCULATION FORMAT: Formula -> Given & SI Unit Conversions -> Substitution -> Final Answer with Units (2-3 sig figs).\n"
    "5. IMAGE ACCURACY: When an image is attached, read every label, graph axis, circuit symbol, and number carefully.\n"
)

_POOL_LOCK = threading.Lock()
_POOL_CURSORS: Dict[str, int] = {}


def _get_key_pool(*env_names: str) -> List[str]:
    """Collect unique non-empty API keys from one or more env vars and rotate round-robin."""
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
    max_tokens: int = 1100,
) -> Optional[str]:
    keys = _get_key_pool("GEMINI_API_KEYS", "GEMINI_API_KEY")
    if not keys:
        return None
    primary_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    candidate_models = list(dict.fromkeys([primary_model, "gemini-2.5-flash", "gemini-2.0-flash"]))

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
        for model in candidate_models:
            model_path = model if model.startswith(("models/", "tunedModels/")) else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
            try:
                async with httpx.AsyncClient(timeout=28.0) as client:
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
    max_tokens: int = 1100,
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
        async with httpx.AsyncClient(timeout=28.0) as client:
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


async def _call_openai(
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 1100,
) -> Optional[str]:
    keys = _get_key_pool("OPENAI_API_KEYS", "OPENAI_API_KEY")
    if not keys:
        return None
    primary_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    candidate_models = list(dict.fromkeys([primary_model, "gpt-4o-mini", "gpt-4o"]))
    for api_key in keys:
        for model in candidate_models:
            res = await _call_openai_style(
                "https://api.openai.com/v1/chat/completions",
                api_key,
                model,
                messages,
                image_bytes,
                image_mime,
                max_tokens=max_tokens,
                supports_vision=True,
            )
            if res:
                return res
    return None


async def _call_openai_compat(
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 1100,
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
    return re.sub(
        r"\b(?:Gemini|OpenAI|ChatGPT|GPT-4o?|GPT-3\.5|Claude|Anthropic|Google\s+AI)\b",
        "Haroony AI",
        text,
        flags=re.I,
    )


def _score_candidate_answer(ans: str, context_block: str) -> float:
    """Prefer answers that preserve verified solver numbers/units and clear Markdown structure."""
    if not ans:
        return -1.0
    score = min(len(ans), 1400) / 400.0
    if "**" in ans:
        score += 1.0
    if "`" in ans:
        score += 0.5
    # Check if verified solver final answer appears in candidate
    m = re.search(r"Final Answer\*?\*?:?\s*`?([0-9.+\-eE×^ ]+)", context_block)
    if m:
        num_match = re.search(r"[-+]?\d+(?:\.\d+)?", m.group(1))
        if num_match and num_match.group(0) in ans:
            score += 3.0
    return score


async def try_neural_generation(
    user_prompt: str,
    context_block: str,
    history: Optional[List[Dict[str, str]]] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> Optional[str]:
    """Run Gemini + OpenAI grounded in Harony 2026/2027 RAG & Cambridge 0625 Mark Schemes."""
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
        f"Respond clearly in Markdown grounded in Physics & the Harony context above. "
        f"If the reference context does not directly answer an everyday Physics 'why' question, "
        f"reason cleanly from first principles using Cambridge 0625 Physics."
    )
    messages.append({"role": "user", "content": prompt_with_ctx})

    has_gemini = bool(_get_key_pool("GEMINI_API_KEYS", "GEMINI_API_KEY"))
    has_openai = bool(_get_key_pool("OPENAI_API_KEYS", "OPENAI_API_KEY"))
    mode = os.environ.get("HAROONY_LLM_MODE", "dual").strip().lower()

    if has_gemini and has_openai and mode == "dual":
        gem_res, oai_res = await asyncio.gather(
            _call_gemini(messages, image_bytes, image_mime),
            _call_openai(messages, image_bytes, image_mime),
        )
        if gem_res and oai_res:
            best = gem_res if _score_candidate_answer(gem_res, context_block) >= _score_candidate_answer(oai_res, context_block) else oai_res
            return _sanitize_provider_mentions(best)
        if gem_res:
            return _sanitize_provider_mentions(gem_res)
        if oai_res:
            return _sanitize_provider_mentions(oai_res)
    elif mode == "openai_first":
        for caller in (
            lambda: _call_openai(messages, image_bytes, image_mime),
            lambda: _call_gemini(messages, image_bytes, image_mime),
        ):
            res = await caller()
            if res:
                return _sanitize_provider_mentions(res)
    else:
        for caller in (
            lambda: _call_gemini(messages, image_bytes, image_mime),
            lambda: _call_openai(messages, image_bytes, image_mime),
        ):
            res = await caller()
            if res:
                return _sanitize_provider_mentions(res)

    compat_res = await _call_openai_compat(settings, messages, image_bytes, image_mime)
    if compat_res:
        return _sanitize_provider_mentions(compat_res)
    return None


async def try_neural_vision_ocr(
    image_bytes: bytes,
    image_mime: str = "image/jpeg",
    purpose: str = "physics_question",
) -> Optional[str]:
    """High-accuracy vision transcription for Physics problem screenshots via Gemini + OpenAI."""
    if not image_bytes:
        return None
    prompt = "Transcribe ALL text, numbers, units, and diagram/graph labels from this Physics problem image cleanly."
    msgs = [{"role": "user", "content": prompt}]
    for caller in (
        lambda: _call_gemini(msgs, image_bytes, image_mime, max_tokens=700),
        lambda: _call_openai(msgs, image_bytes, image_mime, max_tokens=700),
    ):
        res = await caller()
        if res:
            return res.strip()
    return None


async def has_active_server_llm() -> bool:
    """Return True if Gemini or OpenAI API keys are configured on the server."""
    if any(
        os.environ.get(k, "").strip()
        for k in (
            "GEMINI_API_KEYS", "GEMINI_API_KEY",
            "OPENAI_API_KEYS", "OPENAI_API_KEY",
        )
    ):
        return True
    settings = load_settings()
    if settings.get("openai_compat_enabled") and settings.get("openai_compat_base_url") and settings.get("openai_compat_model"):
        return True
    return False
