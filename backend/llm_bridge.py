"""Hybrid Multi-Model Neural Bridge for Haroony AI.

100% server-side & secure:
- Reads credentials ONLY from server environment variables (.env gitignored / hosting Secrets).
- NEVER exposes third-party provider names or API keys to the frontend UI.
- Supports unlimited, zero-rate-limit cascading across Cloud + Open-Source LLMs:
  1. Cloud Multimodal Engine A (GEMINI_API_KEY)
  2. Cloud Multimodal Engine B (OPENAI_API_KEY)
  3. Open-Source Groq Engine (GROQ_API_KEY: llama-3.3-70b-versatile / qwen-qwq-32b)
  4. Open-Source HuggingFace Router (HF_TOKEN: Qwen/Qwen2.5-72B-Instruct / Llama-3.3-70B)
  5. Open-Source OpenRouter Pool (OPENROUTER_API_KEY: open-source free/fast models)
  6. Unlimited Local Ollama Multi-Model Pool (OLLAMA_BASE_URL: auto-discovers qwen2.5-vl, qwen2.5, llama3.2, deepseek-r1, mistral, phi4)
  7. Custom OpenAI-Compatible Endpoint (OPENAI_COMPAT_BASE_URL: vLLM / LocalAI / Together)
  8. Unlimited Built-In Harony Physics Engine (SymPy + 445-chunk RAG + 19-module Concept Atlas + MS)
"""
from __future__ import annotations

import base64
import os
import re
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
    "NEVER force an unrelated snippet into your reply.\n"
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
        async with httpx.AsyncClient(timeout=26.0) as client:
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
    max_tokens: int = 900,
) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    return await _call_openai_style(
        "https://api.openai.com/v1/chat/completions",
        api_key,
        model,
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
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    for model in (
        os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
        "qwen-qwq-32b",
        "llama-3.1-8b-instant",
    ):
        res = await _call_openai_style(
            "https://api.groq.com/openai/v1/chat/completions",
            api_key,
            model,
            messages,
            None,
            "image/jpeg",
            max_tokens=max_tokens,
            supports_vision=False,
        )
        if res:
            return res
    return None


async def _call_huggingface_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or "").strip()
    if not hf_token:
        return None
    for model in (
        os.environ.get("HF_MODEL", "").strip() or "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
    ):
        res = await _call_openai_style(
            "https://router.huggingface.co/v1/chat/completions",
            hf_token,
            model,
            messages,
            None,
            "image/jpeg",
            max_tokens=max_tokens,
            supports_vision=False,
        )
        if res:
            return res
    return None


async def _call_openrouter_opensource(
    messages: List[Dict[str, str]],
    max_tokens: int = 900,
) -> Optional[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    for model in (
        os.environ.get("OPENROUTER_MODEL", "").strip() or "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ):
        res = await _call_openai_style(
            "https://openrouter.ai/api/v1/chat/completions",
            api_key,
            model,
            messages,
            None,
            "image/jpeg",
            max_tokens=max_tokens,
            supports_vision=False,
        )
        if res:
            return res
    return None


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

    # Auto-discover installed open-source models on the local/remote Ollama instance
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
        r"\b(?:Gemini|OpenAI|ChatGPT|GPT-4o?|Claude|Groq|OpenRouter|Llama|Qwen|Mistral|DeepSeek)\b",
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
    """Cascade across configured server-side cloud + open-source LLM backends."""
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
        lambda: _call_openai(messages, image_bytes, image_mime),
        lambda: _call_groq_opensource(messages),
        lambda: _call_huggingface_opensource(messages),
        lambda: _call_openrouter_opensource(messages),
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


async def has_active_server_llm() -> bool:
    """Return True if at least one real cloud or local Ollama LLM is configured/reachable."""
    if any(
        os.environ.get(k, "").strip()
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "OPENROUTER_API_KEY")
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


async def try_neural_grade_unmatched(
    student_text: str,
    ms_title: str = "",
    ms_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Grade any freeform/unmatched Physics question + student answer using a real server LLM."""
    if not student_text.strip():
        return None
    import json
    sys_msg = (
        "You are Haroony AI, a strict Cambridge IGCSE (0625) & A-Level (9702) Physics Examiner.\n"
        "Evaluate ONLY the student's actual answer against Physics laws (and any Mark Scheme provided).\n"
        "Return ONLY valid JSON matching:\n"
        '{"awarded_marks": int, "max_marks": int, "examiner_feedback": "2-3 sentences", "model_answer": "concise ideal answer", "points": [{"code":"B1","marks_awarded":1,"max_marks":1,"status":"awarded","comment":"reason"}]}'
    )
    user_msg = (
        (f"Mark Scheme ({ms_title}):\n{ms_text[:1500]}\n\n" if (ms_title or ms_text) else "")
        + f"Student Submission:\n{student_text[:2000]}\n\n"
        + "Be 100% honest: if the student's answer is wrong, missing, or says 'I don't know', award 0 for those marks."
    )
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    settings = load_settings()
    raw = None
    for caller in (
        lambda: _call_gemini(msgs, None, "image/jpeg", max_tokens=500),
        lambda: _call_openai(msgs, None, "image/jpeg", max_tokens=500),
        lambda: _call_groq_opensource(msgs, max_tokens=500),
        lambda: _call_huggingface_opensource(msgs, max_tokens=500),
        lambda: _call_openrouter_opensource(msgs, max_tokens=500),
        lambda: _call_ollama(settings, msgs, None),
        lambda: _call_openai_compat(settings, msgs, None, "image/jpeg", max_tokens=500),
    ):
        raw = await caller()
        if raw:
            break
    if not raw:
        return None
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        mx = max(1, int(parsed.get("max_marks", 1)))
        aw = max(0, min(mx, int(parsed.get("awarded_marks", 0))))
        pct = round(100.0 * aw / mx, 1)
        grade_letter = "A*" if pct >= 90 else "A" if pct >= 80 else "B" if pct >= 70 else "C" if pct >= 60 else "D" if pct >= 50 else "E" if pct >= 40 else "U"
        return {
            "ok": True,
            "mode": "neural_examiner",
            "mark_scheme_title": ms_title or "Haroony AI Physics Examiner",
            "awarded_marks": aw,
            "max_marks": mx,
            "percentage": pct,
            "grade": grade_letter,
            "examiner_feedback": _sanitize_provider_mentions(str(parsed.get("examiner_feedback", ""))),
            "questions": [
                {
                    "question_id": "AI Examiner Check",
                    "title": "Physics Reasoning & Accuracy",
                    "awarded_marks": aw,
                    "max_marks": mx,
                    "points": parsed.get("points") if isinstance(parsed.get("points"), list) else [],
                    "model_answer": _sanitize_provider_mentions(str(parsed.get("model_answer", ""))),
                }
            ],
            "extracted_student_text": student_text,
        }
    except Exception:
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
        lambda: _call_groq_opensource(msgs, max_tokens=300),
        lambda: _call_huggingface_opensource(msgs, max_tokens=300),
        lambda: _call_ollama(settings, msgs, None),
    ):
        res = await caller()
        if res:
            return _sanitize_provider_mentions(res)
    return None
