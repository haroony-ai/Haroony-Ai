"""Hybrid Multi-Tier Neural Bridge for Haroony AI.

100% server-side & secure:
- Reads credentials ONLY from server environment variables (.env gitignored / cloud Secrets).
- NEVER exposes provider names (Gemini / OpenAI / etc.) or API keys to the frontend UI.
- Supports unlimited, zero-rate-limit cascading:
  Tier 1: Multimodal Cloud Engine A (env var GEMINI_API_KEY)
  Tier 2: Multimodal Cloud Engine B (env var OPENAI_API_KEY)
  Tier 3: Unlimited Self-Hosted Open-Source Model (Ollama / OpenAI-compatible vLLM/Qwen)
  Tier 4: Unlimited Built-In Harony Physics Engine (SymPy + 445-chunk Harony RAG + Cambridge MS)
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional

import httpx

from backend.config import load_settings

HAROONY_SYSTEM_PROMPT = (
    "You are Haroony AI, the official Cambridge IGCSE (0625) & A-Level (9702) Physics AI Assistant "
    "for Mr. Haroony's Harony Physics Clinic. Never mention third-party model names or API providers.\n\n"
    "STRICT RULES:\n"
    "1. PHYSICS ONLY: Answer ONLY Physics questions. Refuse non-physics requests politely.\n"
    "2. STAY STRICTLY ON THE STUDENT'S TOPIC: Never drift to another Physics chapter.\n"
    "3. GROUND IN HARONY NOTES & CAMBRIDGE 0625 MARK SCHEMES: Use exact definitions, formulas, and keywords.\n"
    "4. 4-STEP CALCULATION FORMAT: Formula -> Substitution -> Calculation -> Final Answer with Units (2-3 sig figs).\n"
    "5. IMAGE ACCURACY: When an image is attached, read every label, graph axis, circuit symbol, and number carefully.\n"
)


async def _call_gemini(
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    # Build Gemini REST generateContent payload
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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=28.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return None
            data = resp.json()
            cands = data.get("candidates") or []
            if not cands:
                return None
            parts = cands[0].get("content", {}).get("parts") or []
            out = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
            return out or None
    except Exception:
        return None


async def _call_openai(
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
    image_mime: str,
    max_tokens: int = 900,
) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    oai_msgs: List[Dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if idx == len(messages) - 1 and image_bytes and m.get("role") == "user":
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
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
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


async def _call_ollama(
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    image_bytes: Optional[bytes],
) -> Optional[str]:
    if not settings.get("ollama_enabled"):
        return None
    base = (settings.get("ollama_base_url") or "http://127.0.0.1:11434").rstrip("/")
    model = settings.get("ollama_model") or "qwen2.5-vl:7b"

    ollama_msgs: List[Dict[str, Any]] = []
    for idx, m in enumerate(messages):
        item: Dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content", "")}
        if idx == len(messages) - 1 and image_bytes and m.get("role") == "user":
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
                return content or None
    except Exception:
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

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    api_key = (settings.get("openai_compat_api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    compat_msgs: List[Dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if idx == len(messages) - 1 and image_bytes and m.get("role") == "user":
            b64 = base64.b64encode(image_bytes).decode("ascii")
            compat_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": m.get("content", "")},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
                ],
            })
        else:
            compat_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                headers=headers,
                json={"model": model, "messages": compat_msgs, "temperature": 0.2, "max_tokens": max_tokens},
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices") or []
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip() or None
    except Exception:
        return None


def _sanitize_provider_mentions(text: str) -> str:
    """Ensure internal model/vendor names never leak into student-facing text."""
    if not text:
        return text
    import re
    text = re.sub(r"\b(?:Gemini|OpenAI|ChatGPT|GPT-4o?|Claude)\b", "Haroony AI", text, flags=re.I)
    return text


async def try_neural_generation(
    user_prompt: str,
    context_block: str,
    history: Optional[List[Dict[str, str]]] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> Optional[str]:
    """Cascade across configured server-side neural backends with automatic unlimited fallback."""
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

    # Cascade: Cloud A -> Cloud B -> Open-Source Ollama -> OpenAI-Compat -> None (local engine handles it!)
    for caller in (
        lambda: _call_gemini(messages, image_bytes, image_mime),
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
    purpose: str = "physics_answer",
) -> Optional[str]:
    """Optional high-accuracy vision transcription for handwritten student exam papers."""
    if not image_bytes:
        return None
    prompt = (
        "Transcribe ALL text, equations, numbers, units, and handwritten working from this Physics exam image "
        "verbatim as plain text. Do not invent words the student did not write. Keep question labels like (a), (b), (i) intact."
    )
    if purpose == "physics_question":
        prompt = (
            "Transcribe ALL text, numbers, units, and diagram/graph labels from this Physics problem image cleanly."
        )
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


async def try_neural_examiner_feedback(
    student_text: str,
    ms_title: str,
    ms_summary: str,
    awarded: int,
    max_marks: int,
) -> Optional[str]:
    """Generate concise, honest Cambridge examiner coaching based strictly on real MS marks awarded."""
    if not student_text.strip():
        return None
    prompt = (
        f"Mark Scheme: {ms_title}\n"
        f"Score already awarded by deterministic Cambridge rules: {awarded}/{max_marks}\n"
        f"Official Marking Points:\n{ms_summary[:1800]}\n\n"
        f"Student's actual answer:\n{student_text[:1500]}\n\n"
        "Give 2-3 crisp sentences of Cambridge 0625 examiner feedback explaining ONLY what the student got right and what specific keyword/unit/step they missed. Do NOT change the score."
    )
    msgs = [
        {"role": "system", "content": HAROONY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    settings = load_settings()
    for caller in (
        lambda: _call_gemini(msgs, None, "image/jpeg", max_tokens=300),
        lambda: _call_openai(msgs, None, "image/jpeg", max_tokens=300),
        lambda: _call_ollama(settings, msgs, None),
    ):
        res = await caller()
        if res:
            return _sanitize_provider_mentions(res)
    return None
