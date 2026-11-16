"""Optional Neural LLM / Vision-Language Bridge for Haroony AI.

Allows Mr. Haroony to optionally connect:
1. Local / Self-Hosted Ollama (`qwen2.5-vl`, `llama3.2-vision`, or custom `haroony-physics` GGUF) — 100% Free, Zero Rate Limits!
2. Any OpenAI-compatible self-hosted or free-tier endpoint (vLLM, llama.cpp server, Hugging Face TGI, etc.).

Always grounded in Harony Physics Clinic Notes + Cambridge Mark Schemes + Strict Physics-Only System Prompt,
and seamlessly falls back to the built-in local Haroony Physics Engine when no external server is active.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import httpx

from backend.config import load_settings

HAROONY_SYSTEM_PROMPT = """You are Haroony AI, the official Physics AI tutor and Cambridge IGCSE (0625) / A-Level examiner created for Harony Physics Clinic.

STRICT RULES:
1. PHYSICS ONLY: You ONLY answer questions about Physics, physical quantities, physics math/calculations, physics experiments, and Physics Mark Schemes. If a user asks about ANY non-physics topic (history, coding, essays, poems, pure biology, pure organic chemistry, sports, cooking, games, politics, general chat), politely refuse and remind them you only teach Physics.
2. HARONY NOTES GROUNDING: Base your explanations on the provided Harony Physics Clinic Notes excerpts and Cambridge 0625 Mark Scheme guidance.
3. EXAM TECHNIQUE: For calculations, always state: (1) Given & SI conversions, (2) Formula in symbols/words [C1], (3) Substitution [C1], (4) Final answer to 2–3 significant figures with SI unit [A1]. Use g = 9.8 N/kg (or 9.8 m/s²) unless stated otherwise.
4. CLARITY: Be encouraging, concise, structured with clear bullet points, and highlight key Cambridge mark-scheme terms in **bold**."""


async def try_neural_generation(
    user_prompt: str,
    context_block: str,
    history: Optional[List[Dict[str, str]]] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: str = "image/jpeg",
) -> Optional[str]:
    settings = load_settings()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": HAROONY_SYSTEM_PROMPT}]
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:1500]})

    grounded_user = (
        f"Reference Context from Harony Physics Clinic Notes & Mark Schemes:\n{context_block}\n\n"
        f"Student Request:\n{user_prompt}"
    )

    # 1. Try Self-Hosted Ollama if enabled
    if settings.get("ollama_enabled") and settings.get("ollama_base_url"):
        base_url = settings["ollama_base_url"].rstrip("/")
        model = settings.get("ollama_model") or "qwen2.5-vl:7b"
        ollama_msgs = list(messages)
        user_entry: Dict[str, Any] = {"role": "user", "content": grounded_user}
        if image_bytes:
            user_entry["images"] = [base64.b64encode(image_bytes).decode("ascii")]
        ollama_msgs.append(user_entry)
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json={"model": model, "messages": ollama_msgs, "stream": False},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    txt = (data.get("message", {}).get("content") or "").strip()
                    if txt:
                        return txt
        except Exception:
            pass

    # 2. Try OpenAI-Compatible Endpoint if enabled
    if settings.get("openai_compat_enabled") and settings.get("openai_compat_base_url"):
        base_url = settings["openai_compat_base_url"].rstrip("/")
        model = settings.get("openai_compat_model") or "gpt-4o-mini"
        api_key = settings.get("openai_compat_api_key") or "EMPTY"
        oai_msgs = list(messages)
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            oai_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": grounded_user},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}},
                ],
            })
        else:
            oai_msgs.append({"role": "user", "content": grounded_user})
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": oai_msgs, "temperature": 0.2},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices:
                        txt = (choices[0].get("message", {}).get("content") or "").strip()
                        if txt:
                            return txt
        except Exception:
            pass

    return None
