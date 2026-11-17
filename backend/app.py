"""FastAPI Web Server for Haroony AI (Harony Physics Clinic).

Serves both the simple student/teacher website and all REST APIs from a single origin
(relative URLs, 0.0.0.0 binding, permissive CORS & iframe-safe headers for previews & Hugging Face Spaces).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pymupdf
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import MODELS_DIR, ROOT_DIR, STATIC_DIR, load_settings, save_settings
from backend.continuous_learner import ContinuousPhysicsLearner
from backend.guardrail import PhysicsGuardrail
from backend.knowledge_base import HaronyKnowledgeBase
from backend.llm_bridge import has_active_server_llm, try_neural_vision_ocr
from backend.ms_grader import grade_submission
from backend.ms_parser import parse_cambridge_ms_pdf, parse_freeform_ms_text
from backend.ocr_engine import extract_from_file_bytes
from backend.tutor_engine import HaroonyTutorEngine
from scripts.train_haroony_ai import train_all_models

app = FastAPI(
    title="Haroony AI — Harony Physics Clinic",
    description="100% Free, Zero-Rate-Limit IGCSE & A-Level Physics AI Tutor & Mark Scheme Examiner",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

guardrail = PhysicsGuardrail()
kb = HaronyKnowledgeBase()
learner = ContinuousPhysicsLearner()
tutor = HaroonyTutorEngine(guardrail=guardrail, kb=kb, learner=learner)


def check_teacher_pin(pin: Optional[str]) -> None:
    expected = str(load_settings().get("teacher_pin", "0625"))
    if not pin or pin.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid Teacher PIN.")


@app.get("/")
async def index_page() -> FileResponse:
    root_index = ROOT_DIR / "index.html"
    if root_index.exists():
        return FileResponse(str(root_index))
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon_ico() -> FileResponse:
    fav = ROOT_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav), media_type="image/x-icon")
    return FileResponse(str(STATIC_DIR / "assets" / "favicon-64.png"), media_type="image/png")


@app.get("/api/status")
async def api_status() -> Dict[str, Any]:
    metrics = {}
    metrics_path = MODELS_DIR / "model_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    settings = load_settings()
    return {
        "ok": True,
        "mode": "server",
        "name": "Haroony AI",
        "clinic": "Harony Physics Clinic",
        "free_unlimited": True,
        "physics_only_guardrail": True,
        "notes_chunks_count": len(kb.chunks),
        "video_solutions_count": len(kb.videos),
        "mark_schemes": [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "paper_code": m.get("paper_code"),
                "session": m.get("session"),
                "total_marks": m.get("total_marks"),
                "question_count": m.get("question_count"),
            }
            for m in kb.mark_schemes
        ],
        "curriculum": kb.curriculum_overview(),
        "metrics": metrics,
        "learning_stats": learner.get_stats(),
        "server_llm_active": await has_active_server_llm(),
    }


@app.post("/api/feedback")
async def api_feedback(
    interaction_id: str = Form(""),
    helpful: str = Form("true"),
    query: str = Form(""),
    unit: str = Form(""),
) -> Dict[str, Any]:
    is_helpful = str(helpful).strip().lower() in ("true", "1", "yes", "up")
    stats = learner.record_feedback(
        interaction_id=interaction_id.strip(),
        helpful=is_helpful,
        query_fallback=query,
        unit_fallback=unit,
    )
    return {"ok": True, "helpful": is_helpful, "learning_stats": stats}


@app.post("/api/chat")
async def api_chat(
    message: str = Form(""),
    history: str = Form("[]"),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    try:
        hist_list = json.loads(history) if history else []
        if not isinstance(hist_list, list):
            hist_list = []
    except Exception:
        hist_list = []

    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None
    file_mime: str = "image/jpeg"
    if file is not None and file.filename:
        file_bytes = await file.read()
        file_name = file.filename
        file_mime = file.content_type or "image/jpeg"

    if not message.strip() and not file_bytes:
        raise HTTPException(status_code=400, detail="Please enter a Physics question or upload a photo/PDF.")

    return await tutor.answer_student(
        message=message,
        history=hist_list,
        attachment_bytes=file_bytes,
        attachment_name=file_name,
        attachment_mime=file_mime,
    )


@app.get("/api/mark-schemes")
async def list_mark_schemes() -> Dict[str, Any]:
    return {
        "ok": True,
        "mark_schemes": [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "paper_code": m.get("paper_code"),
                "session": m.get("session"),
                "total_marks": m.get("total_marks"),
                "question_count": m.get("question_count"),
                "question_ids": [q.get("question_id") for q in m.get("questions", [])],
            }
            for m in kb.mark_schemes
        ],
    }


@app.get("/api/mark-schemes/{ms_id}")
async def get_mark_scheme_detail(ms_id: str) -> Dict[str, Any]:
    ms = kb.get_mark_scheme(ms_id)
    if not ms:
        raise HTTPException(status_code=404, detail="Mark Scheme not found.")
    return {"ok": True, "mark_scheme": ms}


@app.post("/api/grade")
async def api_grade(
    student_text: str = Form(""),
    ms_id: str = Form(""),
    ms_text: str = Form(""),
    question_id: str = Form(""),
    use_sample_demo: str = Form(""),
    student_files: List[UploadFile] = File(default=[]),
    ms_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    ocr_segments: List[str] = []
    ocr_meta: List[Dict[str, Any]] = []

    # Support 1-click built-in Harony ClassWork 2027 Unit 1 (Page 8) sample demo!
    if use_sample_demo == "cw_u1_density":
        sample_work = (
            "2023 March P4 V2 Q02 a - Density:\n"
            "The ship is not solid steel because there are air spaces inside the ship, "
            "so the overall average density of the ship is less than the density of the water."
        )
        if not student_text.strip():
            ocr_segments.append(sample_work)
        if not ms_id:
            ms_id = "harony-cw-2027-u1"
        if not question_id:
            question_id = "2023-March-P4-V2-Q02-a"

    # Extract text from uploaded student photos / PDFs
    for sf in student_files:
        if sf and sf.filename:
            raw = await sf.read()
            if raw:
                res = extract_from_file_bytes(raw, sf.filename, max_pages=10)
                extracted = (res.get("text") or "").strip()
                is_img = res.get("kind") == "image"
                if is_img and (len(extracted) < 45 or float(res.get("avg_confidence") or 0.0) < 0.76):
                    vtxt = await try_neural_vision_ocr(
                        raw,
                        sf.content_type or "image/jpeg",
                        purpose="physics_answer",
                    )
                    if vtxt and len(vtxt) >= max(len(extracted), 12):
                        extracted = vtxt
                if extracted:
                    ocr_segments.append(extracted)
                ocr_meta.append({
                    "filename": sf.filename,
                    "kind": res.get("kind"),
                    "pages": res.get("pages_processed"),
                    "confidence": res.get("avg_confidence", 0.0),
                })

    combined_student = "\n\n".join(
        part for part in ([student_text.strip()] + ocr_segments) if part
    ).strip()

    if not combined_student:
        raise HTTPException(
            status_code=400,
            detail="Please upload a photo/PDF of your Physics answer or type your answer first.",
        )

    # Strict Physics-Only Guardrail check on the student submission!
    # (Skip off-topic refusal when an explicit Physics Mark Scheme is selected/uploaded unless clearly an injection/off-topic essay)
    has_explicit_ms = bool(ms_id.strip() or ms_text.strip() or (ms_file and ms_file.filename))
    g_check = guardrail.check(combined_student, has_attachment=False, has_prior_physics_context=has_explicit_ms)
    if not g_check.allowed and (g_check.category == "jailbreak" or not has_explicit_ms):
        return {
            "ok": False,
            "refused": True,
            "error": g_check.refusal_message,
        }

    # Resolve Mark Scheme (1: uploaded ms_file, 2: pasted ms_text, 3: saved ms_id)
    ms_obj: Optional[Dict[str, Any]] = None
    if ms_file is not None and ms_file.filename:
        ms_bytes = await ms_file.read()
        if ms_bytes:
            fname_low = ms_file.filename.lower()
            if fname_low.endswith(".pdf") or ms_bytes[:4] == b"%PDF":
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(ms_bytes)
                    tmp.flush()
                    ms_obj = parse_cambridge_ms_pdf(tmp.name, title_override=ms_file.filename)
                # If scanned MS PDF yielded 0 questions, OCR it and parse freeform
                if not ms_obj.get("questions"):
                    ocr_ms = extract_from_file_bytes(ms_bytes, ms_file.filename, max_pages=10)
                    ms_obj = parse_freeform_ms_text(ocr_ms.get("text", ""), title=ms_file.filename)
            else:
                ocr_ms = extract_from_file_bytes(ms_bytes, ms_file.filename, max_pages=5)
                ms_obj = parse_freeform_ms_text(ocr_ms.get("text", ""), title=ms_file.filename)
    elif ms_text.strip():
        ms_obj = parse_freeform_ms_text(ms_text.strip(), title="Pasted Mark Scheme")
    elif ms_id.strip():
        ms_obj = kb.get_mark_scheme(ms_id.strip())

    result = grade_submission(
        student_text=combined_student,
        kb=kb,
        ms_obj=ms_obj,
        target_question_id=question_id.strip() or None,
    )
    result["ocr_files"] = ocr_meta
    return result


@app.get("/api/notes")
async def search_or_browse_notes(
    q: str = Query("", description="Search query"),
    unit: str = Query("", description="Filter by unit"),
    topic: str = Query("", description="Filter by topic"),
    page: Optional[int] = Query(None, description="Exact page number (1..243)"),
) -> Dict[str, Any]:
    if page is not None:
        matches = [c for c in kb.chunks if c.get("page") == page]
        return {"ok": True, "results": matches[:5]}
    if q.strip():
        hits = kb.search(q.strip(), top_k=8)
        return {"ok": True, "results": hits}
    filtered = kb.chunks
    if unit.strip():
        filtered = [c for c in filtered if c.get("unit", "").lower() == unit.strip().lower()]
    if topic.strip():
        filtered = [c for c in filtered if c.get("topic", "").lower() == topic.strip().lower()]
    return {"ok": True, "results": filtered[:24]}


@app.post("/api/teacher/verify")
async def verify_teacher(pin: str = Form("")) -> Dict[str, Any]:
    check_teacher_pin(pin)
    s = load_settings()
    safe_settings = {k: v for k, v in s.items() if k not in ("teacher_pin", "openai_compat_api_key")}
    safe_settings["has_openai_key"] = bool(s.get("openai_compat_api_key"))
    return {"ok": True, "settings": safe_settings}


@app.post("/api/teacher/upload-material")
async def teacher_upload_material(
    pin: str = Form(""),
    title: str = Form(""),
    unit: str = Form("Custom Physics Notes"),
    topic: str = Form(""),
    text_content: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    check_teacher_pin(pin)
    blocks: List[str] = []

    if file is not None and file.filename:
        raw = await file.read()
        fname = file.filename
        if not title.strip():
            title = Path(fname).stem
        if fname.lower().endswith(".pdf") or raw[:4] == b"%PDF":
            doc = pymupdf.open(stream=raw, filetype="pdf")
            for page in doc:
                t = page.get_text("text").strip()
                if len(t) >= 25:
                    blocks.append(t)
            if not blocks:
                ocr_res = extract_from_file_bytes(raw, fname, max_pages=15)
                if ocr_res.get("text"):
                    blocks.extend(ocr_res["text"].split("\n\n"))
        else:
            decoded = raw.decode("utf-8", errors="ignore")
            blocks.extend([p.strip() for p in decoded.split("\n\n") if len(p.strip()) >= 25])

    if text_content.strip():
        blocks.extend([p.strip() for p in text_content.strip().split("\n\n") if len(p.strip()) >= 25])

    if not blocks:
        raise HTTPException(status_code=400, detail="No readable Physics text blocks found in upload.")

    added = kb.add_custom_material(
        title=title.strip() or "Teacher Physics Notes",
        unit=unit.strip() or "Custom Physics Notes",
        topic=topic.strip() or title.strip() or "General Physics",
        text_blocks=blocks,
    )
    metrics = train_all_models()
    guardrail.load()
    kb.reload()
    return {"ok": True, "chunks_added": added, "metrics": metrics}


@app.post("/api/teacher/upload-ms")
async def teacher_upload_ms(
    pin: str = Form(""),
    title: str = Form(""),
    ms_text: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    check_teacher_pin(pin)
    parsed: Optional[Dict[str, Any]] = None

    if file is not None and file.filename:
        raw = await file.read()
        if file.filename.lower().endswith(".pdf") or raw[:4] == b"%PDF":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                parsed = parse_cambridge_ms_pdf(tmp.name, title_override=title.strip() or None)
        else:
            ocr_res = extract_from_file_bytes(raw, file.filename, max_pages=10)
            parsed = parse_freeform_ms_text(ocr_res.get("text", ""), title=title.strip() or file.filename)
    elif ms_text.strip():
        parsed = parse_freeform_ms_text(ms_text.strip(), title=title.strip() or "Teacher Mark Scheme")

    if not parsed or not parsed.get("questions"):
        raise HTTPException(status_code=400, detail="Could not parse marking points from the provided Mark Scheme.")

    kb.save_mark_scheme(parsed)
    metrics = train_all_models()
    guardrail.load()
    kb.reload()
    return {"ok": True, "mark_scheme": parsed, "metrics": metrics}


@app.post("/api/teacher/retrain")
async def teacher_retrain(pin: str = Form("")) -> Dict[str, Any]:
    check_teacher_pin(pin)
    metrics = train_all_models()
    guardrail.load()
    kb.reload()
    return {"ok": True, "metrics": metrics}


@app.post("/api/teacher/settings")
async def update_teacher_settings(
    pin: str = Form(""),
    ollama_enabled: str = Form("false"),
    ollama_base_url: str = Form("http://127.0.0.1:11434"),
    ollama_model: str = Form("qwen2.5-vl:7b"),
    openai_compat_enabled: str = Form("false"),
    openai_compat_base_url: str = Form(""),
    openai_compat_model: str = Form(""),
    openai_compat_api_key: str = Form(""),
) -> Dict[str, Any]:
    check_teacher_pin(pin)
    updates: Dict[str, Any] = {
        "ollama_enabled": ollama_enabled.lower() == "true",
        "ollama_base_url": ollama_base_url.strip(),
        "ollama_model": ollama_model.strip(),
        "openai_compat_enabled": openai_compat_enabled.lower() == "true",
        "openai_compat_base_url": openai_compat_base_url.strip(),
        "openai_compat_model": openai_compat_model.strip(),
    }
    if openai_compat_api_key.strip():
        updates["openai_compat_api_key"] = openai_compat_api_key.strip()
    saved = save_settings(updates)
    return {
        "ok": True,
        "ollama_enabled": saved["ollama_enabled"],
        "openai_compat_enabled": saved["openai_compat_enabled"],
    }


def _write_env_updates(updates: Dict[str, str]) -> None:
    env_path = ROOT_DIR / ".env"
    existing: Dict[str, str] = {}
    if env_path.exists():
        for raw_ln in env_path.read_text(encoding="utf-8").splitlines():
            ln = raw_ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                existing[k.strip()] = v.strip()
    for k, v in updates.items():
        if v:
            existing[k] = v
            os.environ[k] = v
    env_path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n", encoding="utf-8")


@app.post("/api/teacher/cloud-setup")
async def teacher_cloud_setup(
    pin: str = Form(""),
    primary_key: str = Form(""),
    secondary_key: str = Form(""),
    primary_model: str = Form(""),
    secondary_model: str = Form(""),
) -> Dict[str, Any]:
    check_teacher_pin(pin)
    pk = primary_key.strip()
    sk = secondary_key.strip()
    updates: Dict[str, str] = {"HAROONY_LLM_MODE": "dual"}
    if pk:
        updates["GEMINI_API_KEYS"] = pk
        updates.setdefault("GEMINI_MODEL", primary_model.strip() or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    elif primary_model.strip():
        updates["GEMINI_MODEL"] = primary_model.strip()
    if sk:
        updates["OPENAI_API_KEYS"] = sk
        updates.setdefault("OPENAI_MODEL", secondary_model.strip() or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    elif secondary_model.strip():
        updates["OPENAI_MODEL"] = secondary_model.strip()

    _write_env_updates(updates)

    # Verify live connectivity for whichever keys exist in server env
    gem_active = (os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")).split(",")[0].strip()
    oai_active = (os.environ.get("OPENAI_API_KEYS") or os.environ.get("OPENAI_API_KEY", "")).split(",")[0].strip()
    gem_ok = False
    oai_ok = False
    async with httpx.AsyncClient(timeout=15.0) as client:
        if gem_active:
            try:
                r = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gem_active}")
                gem_ok = r.status_code == 200
            except Exception:
                gem_ok = False
        if oai_active:
            try:
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {oai_active}"},
                )
                oai_ok = r.status_code == 200
            except Exception:
                oai_ok = False

    return {
        "ok": True,
        "primary_configured": bool(gem_active),
        "primary_verified": gem_ok,
        "secondary_configured": bool(oai_active),
        "secondary_verified": oai_ok,
        "server_llm_active": await has_active_server_llm(),
    }


@app.post("/api/teacher/train-cloud")
async def teacher_train_cloud(pin: str = Form("")) -> Dict[str, Any]:
    check_teacher_pin(pin)
    from backend.config import TRAINING_DIR

    gem_key = (os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")).split(",")[0].strip()
    oai_key = (os.environ.get("OPENAI_API_KEYS") or os.environ.get("OPENAI_API_KEY", "")).split(",")[0].strip()
    out: Dict[str, Any] = {"ok": True, "primary": {"launched": False}, "secondary": {"launched": False}}

    async with httpx.AsyncClient(timeout=120.0) as client:
        gem_ds = TRAINING_DIR / "gemini_sft_dataset.jsonl"
        if gem_key and gem_ds.exists():
            try:
                examples = []
                for ln in gem_ds.read_text(encoding="utf-8").splitlines()[:1200]:
                    if not ln.strip():
                        continue
                    obj = json.loads(ln)
                    c = obj.get("contents") or []
                    if len(c) == 2:
                        examples.append({
                            "text_input": str(c[0]["parts"][0]["text"])[:1800],
                            "output": str(c[1]["parts"][0]["text"])[:1800],
                        })
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/tunedModels?key={gem_key}",
                    json={
                        "display_name": "Haroony-Physics-AI-2027",
                        "base_model": "models/gemini-1.5-flash-001-tuning",
                        "tuning_task": {
                            "hyperparameters": {"batch_size": 4, "learning_rate": 0.001, "epoch_count": 3},
                            "training_data": {"examples": {"examples": examples}},
                        },
                    },
                )
                out["primary"] = {"launched": resp.status_code in (200, 201), "status": resp.status_code, "detail": resp.json()}
            except Exception as exc:
                out["primary"] = {"launched": False, "error": str(exc)}
        else:
            out["primary"] = {"launched": False, "reason": "Missing Primary API key or dataset"}

        oai_ds = TRAINING_DIR / "openai_sft_dataset.jsonl"
        if oai_key and oai_ds.exists():
            try:
                up = await client.post(
                    "https://api.openai.com/v1/files",
                    headers={"Authorization": f"Bearer {oai_key}"},
                    data={"purpose": "fine-tune"},
                    files={"file": (oai_ds.name, oai_ds.read_bytes(), "application/jsonl")},
                )
                up_j = up.json()
                if up.status_code not in (200, 201) or not up_j.get("id"):
                    out["secondary"] = {"launched": False, "status": up.status_code, "detail": up_j}
                else:
                    job = await client.post(
                        "https://api.openai.com/v1/fine_tuning/jobs",
                        headers={"Authorization": f"Bearer {oai_key}", "Content-Type": "application/json"},
                        json={
                            "training_file": up_j["id"],
                            "model": "gpt-4o-mini-2024-07-18",
                            "suffix": "haroony-physics-2027",
                            "hyperparameters": {"n_epochs": 3},
                        },
                    )
                    out["secondary"] = {"launched": job.status_code in (200, 201), "status": job.status_code, "detail": job.json()}
            except Exception as exc:
                out["secondary"] = {"launched": False, "error": str(exc)}
        else:
            out["secondary"] = {"launched": False, "reason": "Missing Secondary API key or dataset"}

    return out


@app.get("/setup", response_class=HTMLResponse)
async def setup_popup_page() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Haroony AI — Secure Server Key & Training Setup</title>
  <style>
    body{font-family:system-ui,-apple-system,sans-serif;background:#0b1220;color:#e2e8f0;margin:0;padding:24px;display:flex;justify-content:center;align-items:center;min-height:100vh}
    .card{background:#111a2e;border:1px solid #243452;border-radius:16px;padding:28px;max-width:520px;width:100%;box-shadow:0 20px 50px rgba(0,0,0,.45)}
    h1{margin:0 0 8px;font-size:1.35rem}
    p{color:#94a3b8;font-size:.92rem;line-height:1.5}
    label{display:block;margin-top:14px;font-size:.85rem;font-weight:600;color:#cbd5e1}
    input{width:100%;box-sizing:border-box;margin-top:6px;padding:11px 12px;border-radius:10px;border:1px solid #334155;background:#090f1c;color:#f8fafc;font-size:.95rem}
    .row{display:flex;gap:10px;margin-top:18px;flex-wrap:wrap}
    button{flex:1;padding:12px 14px;border:none;border-radius:10px;font-weight:700;cursor:pointer;font-size:.92rem}
    .btn-save{background:#3b82f6;color:#fff}
    .btn-train{background:#10b981;color:#052e16}
    .out{margin-top:16px;padding:12px;border-radius:10px;background:#090f1c;border:1px solid #1e293b;font-size:.84rem;white-space:pre-wrap;display:none}
  </style>
</head>
<body>
  <div class="card">
    <h1>Haroony AI — Secure Key &amp; Training Setup</h1>
    <p>Keys entered here are saved ONLY on your private server (<code>.env</code>), never pushed to GitHub or visible to students.</p>
    <label>Teacher PIN (default: 0625)</label>
    <input id="pin" type="password" value="0625" />
    <label>Primary AI Key (Google AI Studio)</label>
    <input id="pk" type="password" placeholder="Paste your primary API key..." autocomplete="off" />
    <label>Secondary AI Key (OpenAI Platform)</label>
    <input id="sk" type="password" placeholder="Paste your secondary API key..." autocomplete="off" />
    <label>Optional Tuned Primary Model ID (e.g. tunedModels/...)</label>
    <input id="pm" type="text" placeholder="Leave blank until training finishes" />
    <label>Optional Tuned Secondary Model ID (e.g. ft:...)</label>
    <input id="sm" type="text" placeholder="Leave blank until training finishes" />
    <div class="row">
      <button class="btn-save" onclick="saveKeys()">1. Save &amp; Test Keys</button>
      <button class="btn-train" onclick="trainBoth()">2. Train Both on Harony Material</button>
    </div>
    <div id="out" class="out"></div>
  </div>
  <script>
    async function saveKeys(){
      const out = document.getElementById('out'); out.style.display='block'; out.textContent='Saving & verifying keys...';
      const fd = new FormData();
      fd.append('pin', document.getElementById('pin').value);
      fd.append('primary_key', document.getElementById('pk').value);
      fd.append('secondary_key', document.getElementById('sk').value);
      fd.append('primary_model', document.getElementById('pm').value);
      fd.append('secondary_model', document.getElementById('sm').value);
      const r = await fetch('/api/teacher/cloud-setup', {method:'POST', body:fd});
      out.textContent = JSON.stringify(await r.json(), null, 2);
    }
    async function trainBoth(){
      const out = document.getElementById('out'); out.style.display='block'; out.textContent='Uploading Harony 2026/2027 datasets & starting training jobs...';
      const fd = new FormData();
      fd.append('pin', document.getElementById('pin').value);
      const r = await fetch('/api/teacher/train-cloud', {method:'POST', body:fd});
      out.textContent = JSON.stringify(await r.json(), null, 2);
    }
  </script>
</body>
</html>"""
    )
