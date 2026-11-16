"""Haroony AI Physics Tutor Orchestrator.

Combines:
1. Strict Physics-Only Guardrail (blocks all non-physics prompts & non-physics image uploads)
2. Local Vision OCR (for uploaded physics problem photos / PDFs)
3. Symbolic IGCSE Physics Calculation Solver (SymPy-backed 4-step Cambridge working)
4. Harony Physics Clinic 243-Page Notes Retriever + Illustrated Page Cards
5. Cambridge 0625 Mark Scheme Examiner Keywords Boost
6. Optional Ollama / OpenAI-compatible Neural VLM enhancement
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from backend.guardrail import PhysicsGuardrail
from backend.knowledge_base import HaronyKnowledgeBase
from backend.llm_bridge import try_neural_generation
from backend.ms_grader import STOPWORDS
from backend.ocr_engine import extract_from_file_bytes
from backend.physics_solver import try_solve_physics_problem


def format_note_text_cleanly(raw_text: str, topic: str) -> List[str]:
    """Clean raw PDF page text from Harony Notes into crisp, readable study bullet points."""
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    cleaned: List[str] = []
    for idx, ln in enumerate(lines):
        # Skip standalone page/topic numbers at top or bottom of page
        if re.match(r"^\d{1,3}$", ln):
            continue
        if idx <= 2 and len(ln) < 40 and ln.lower() in topic.lower():
            continue
        ln = re.sub(r"^[•o\-*]\s*", "", ln).strip()
        if len(ln) >= 4:
            cleaned.append(ln)

    # Merge broken PDF wrap lines where a line ends mid-sentence (lowercase/preposition) and next line starts lowercase
    merged: List[str] = []
    for ln in cleaned:
        if (
            merged
            and not merged[-1].endswith((".", ":", "?", "!", ")"))
            and (ln[0].islower() or merged[-1].endswith(("of", "to", "in", "the", "a", "an", "and", "or", "is", "are", "with", "by", "from", "that", "as")))
        ):
            merged[-1] = f"{merged[-1]} {ln}"
        else:
            merged.append(ln)
    return merged


def find_related_ms_tips(query: str, kb: HaronyKnowledgeBase) -> List[Dict[str, Any]]:
    """Find matching Cambridge 0625 Mark Scheme points to show students exact examiner keywords."""
    q_words = {
        w for w in re.findall(r"[a-z]{4,}", query.lower())
        if w not in STOPWORDS and w not in {"explain", "describe", "what", "does", "work", "define", "state", "physics"}
    }
    if not q_words:
        return []

    scored: List[tuple[int, Dict[str, Any]]] = []
    for ms in kb.mark_schemes:
        for q in ms.get("questions", []):
            # Focus on conceptual B-mark questions for keyword tips
            b_pts = [p for p in q.get("points", []) if p.get("mark_type") == "B" and len(p.get("answer", "")) >= 8]
            if not b_pts:
                continue
            combined_txt = " ".join(p["answer"] + " " + p.get("guidance", "") for p in b_pts).lower()
            hits = sum(1 for w in q_words if w in combined_txt)
            if hits >= 2 or (len(q_words) == 1 and hits == 1):
                scored.append((hits, {"ms_title": ms["title"], "question": q}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:2]]


class HaroonyTutorEngine:
    def __init__(self, guardrail: PhysicsGuardrail, kb: HaronyKnowledgeBase) -> None:
        self.guardrail = guardrail
        self.kb = kb

    async def answer_student(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        attachment_bytes: Optional[bytes] = None,
        attachment_name: Optional[str] = None,
        attachment_mime: str = "image/jpeg",
    ) -> Dict[str, Any]:
        user_msg = (message or "").strip()
        has_file = attachment_bytes is not None and len(attachment_bytes) > 0
        has_prior_physics = bool(history and any(m.get("role") == "assistant" for m in history))

        # 1. Pre-OCR Guardrail Check
        decision = self.guardrail.check(
            user_msg,
            has_attachment=has_file,
            has_prior_physics_context=has_prior_physics,
        )
        if not decision.allowed:
            return {
                "ok": True,
                "refused": True,
                "category": decision.category,
                "reply": decision.refusal_message,
                "notes_cards": [],
                "ocr_text": None,
                "engine": "Haroony Physics Guardrail",
            }

        if decision.category == "greeting" and not has_file:
            greeting_reply = (
                "Hello! Welcome to **Haroony AI** (**Harony Physics Clinic**).\n\n"
                "I am trained on **Mr. Haroony's 2026 IGCSE Physics Notes (243 pages)**, **Harony ClassWork 2027 (674 pages)**, "
                "and **Harony P2 Classified 2027 (656 pages)** — including **1,546 official Mark Scheme questions** and **2,085 video solutions**, "
                "running **100% free with zero rate limits**.\n\n"
                "Here is how I can help you right now:\n"
                "- **Ask any Physics concept, past paper code, or calculation** (Mechanics, Thermal, Waves, Light, Electricity, Electromagnetism, Nuclear, or Astronomy).\n"
                "- **Upload a photo or PDF of a Physics question** and I will read it and solve it step-by-step.\n"
                "- Switch to **Correct Answer & MS** above to upload your handwritten work + select from Mr. Haroony's 2027 ClassWork / P2 Mark Schemes (or upload any MS)!"
            )
            return {
                "ok": True,
                "refused": False,
                "category": "greeting",
                "reply": greeting_reply,
                "notes_cards": [],
                "video_links": [],
                "ocr_text": None,
                "engine": "Haroony Physics AI",
            }

        # 2. If student attached a photo or PDF, run local Vision OCR!
        ocr_text = None
        if has_file and attachment_bytes is not None:
            ocr_info = extract_from_file_bytes(attachment_bytes, attachment_name or "upload.jpg", max_pages=4)
            ocr_text = (ocr_info.get("text") or "").strip()
            combined_query = f"{user_msg}\n{ocr_text}".strip()
            # Re-verify Physics Guardrail after reading the photo so non-physics images are also refused!
            post_ocr_check = self.guardrail.check(combined_query, has_attachment=False, has_prior_physics_context=False)
            if not post_ocr_check.allowed:
                return {
                    "ok": True,
                    "refused": True,
                    "category": post_ocr_check.category,
                    "reply": post_ocr_check.refusal_message,
                    "notes_cards": [],
                    "ocr_text": ocr_text,
                    "engine": "Haroony Physics Guardrail (Vision Check)",
                }
        else:
            combined_query = user_msg

        # Incorporate last user turn if short follow-up like "give me an example" or "show the formula"
        search_query = combined_query
        if len(combined_query.split()) <= 6 and history:
            prev_users = [m.get("content", "") for m in history if m.get("role") == "user"]
            if prev_users:
                search_query = prev_users[-1] + " " + combined_query

        # 3. Check if student wants a practice / quiz question!
        if re.search(r"\b(?:quiz\s+me|test\s+me|practice\s+question|give\s+me\s+a\s+question)\b", combined_query, re.I):
            return self._generate_practice_question(search_query)

        # 4. Run Symbolic Physics Calculation Solver + Harony Notes Semantic Retrieval
        solved = try_solve_physics_problem(combined_query)
        note_hits = self.kb.search(search_query, top_k=3)
        ms_tips = find_related_ms_tips(search_query, self.kb)
        vid_hits = self.kb.search_videos(search_query, top_k=2)

        # Build context block for optional neural LLM enhancement
        ctx_parts = []
        if solved:
            ctx_parts.append(solved.to_markdown())
        for nh in note_hits:
            ctx_parts.append(f"[Harony Notes p.{nh.get('page')} - {nh.get('unit')} / {nh.get('topic')}]:\n{nh.get('text')}")
        for mt in ms_tips:
            qobj = mt["question"]
            ctx_parts.append(f"[{mt['ms_title']} Q{qobj['question_id']}]: {qobj['summary_answer']}")

        neural_reply = await try_neural_generation(
            user_prompt=combined_query,
            context_block="\n\n".join(ctx_parts),
            history=history,
            image_bytes=attachment_bytes if (has_file and not (attachment_name or "").lower().endswith(".pdf")) else None,
            image_mime=attachment_mime,
        )

        notes_cards = [
            {
                "id": nh.get("id"),
                "page": nh.get("page"),
                "unit": nh.get("unit"),
                "topic": nh.get("topic"),
                "source": nh.get("source"),
                "formulas": nh.get("formulas", []),
                "image_url": nh.get("image_url"),
                "snippet": (nh.get("text") or "")[:280],
            }
            for nh in note_hits
        ]

        if neural_reply:
            return {
                "ok": True,
                "refused": False,
                "category": "physics",
                "reply": neural_reply,
                "notes_cards": notes_cards,
                "video_links": vid_hits,
                "ocr_text": ocr_text,
                "engine": "Haroony AI (Neural + Harony 2026/2027 RAG)",
            }

        # 5. Compose rich, exam-grade local response from Symbolic Solver + Harony Notes + Cambridge MS
        sections: List[str] = []

        if ocr_text and has_file:
            preview = ocr_text[:350] + ("..." if len(ocr_text) > 350 else "")
            sections.append(f"**📷 Read from your uploaded image/PDF:**\n> {preview.replace(chr(10), '  ' + chr(10) + '> ')}")

        if solved:
            sections.append(solved.to_markdown())

        if note_hits:
            primary = note_hits[0]
            p_topic = primary.get("topic", "IGCSE Physics")
            p_unit = primary.get("unit", "")
            p_page = primary.get("page")
            header_cite = f"Harony Notes p.{p_page}" if p_page else primary.get("source", "Harony Notes")

            bullets = format_note_text_cleanly(primary.get("text", ""), p_topic)
            # Also blend complementary points from 2nd matching page if same topic
            if len(note_hits) >= 2 and note_hits[1].get("topic") == p_topic and not solved:
                extra_bullets = format_note_text_cleanly(note_hits[1].get("text", ""), p_topic)
                for eb in extra_bullets[:5]:
                    if eb not in bullets:
                        bullets.append(eb)

            max_bullets = 5 if solved else 9
            bullet_md = "\n".join(f"- {b}" for b in bullets[:max_bullets])

            title_prefix = "Related Concept Notes" if solved else f"{p_topic}"
            sections.append(
                f"### 📘 {title_prefix} *({p_unit} — {header_cite})*\n\n{bullet_md}"
            )

            all_formulas = []
            for nh in note_hits[:2]:
                for fm in nh.get("formulas", []):
                    if fm not in all_formulas:
                        all_formulas.append(fm)
            if all_formulas and not solved:
                fm_md = "\n".join(f"- `{fm}`" for fm in all_formulas[:4])
                sections.append(f"**Key Equations & Relations:**\n{fm_md}")

        if ms_tips and not solved:
            tip = ms_tips[0]
            qobj = tip["question"]
            pts_lines = []
            for p in qobj.get("points", [])[:3]:
                if p.get("mark_type") in ("B", "M", "A"):
                    g_note = ""
                    fl = p.get("flags", {})
                    if fl.get("too_vague"):
                        g_note += f" *(Avoid Too Vague: `{', '.join(fl['too_vague'][:2])}`)*"
                    if fl.get("contradiction"):
                        g_note += f" *(Contradiction penalty: `{', '.join(fl['contradiction'][:2])}`)*"
                    pts_lines.append(f"- **[{p['code']}]** {p['answer']}{g_note}")
            if pts_lines:
                sections.append(
                    f"### 🎯 Cambridge Examiner Mark-Scheme Phrase Guide *({tip['ms_title']}, Q{qobj['question_id']})*\n"
                    + "\n".join(pts_lines)
                )

        if not sections:
            sections.append(
                "### Physics Concept Guidance\n"
                "Could you share a few more details (such as the specific quantities, diagram details, or IGCSE topic)? "
                "You can ask about any topic in **Mechanics, Thermal Physics, Waves, Light, Electricity, Electromagnetism, Nuclear Physics, or Astronomy**, "
                "or upload a photo of the question!"
            )

        return {
            "ok": True,
            "refused": False,
            "category": "physics",
            "reply": "\n\n---\n\n".join(sections),
            "notes_cards": notes_cards,
            "video_links": vid_hits,
            "ocr_text": ocr_text,
            "engine": "Haroony Physics AI (Harony 2026/2027 Books)",
        }

    def _generate_practice_question(self, query: str) -> Dict[str, Any]:
        hits = self.kb.search(query, top_k=3)
        ms_list = self.kb.mark_schemes
        chosen_q = None
        if ms_list:
            qs = ms_list[0].get("questions", [])
            # Try matching topic keywords to MS questions
            q_tokens = set(re.findall(r"[a-z]{4,}", query.lower())) - STOPWORDS - {"quiz", "test", "practice", "question", "give"}
            if q_tokens:
                best = [
                    q for q in qs
                    if any(t in q.get("summary_answer", "").lower() for t in q_tokens)
                ]
                if best:
                    chosen_q = random.choice(best)
            if chosen_q is None and qs:
                chosen_q = random.choice(qs)

        topic_label = hits[0]["topic"] if hits else "IGCSE Physics"
        notes_cards = [
            {
                "id": nh.get("id"),
                "page": nh.get("page"),
                "unit": nh.get("unit"),
                "topic": nh.get("topic"),
                "source": nh.get("source"),
                "formulas": nh.get("formulas", []),
                "image_url": nh.get("image_url"),
                "snippet": (nh.get("text") or "")[:280],
            }
            for nh in hits[:2]
        ]

        practice_bank = [
            ("Kinematics & Kinetic Energy", "A small aircraft of mass **820 kg** travels at a constant speed of **60 m/s**. Calculate the energy in the kinetic store of the aircraft. **[2 marks]**"),
            ("Moments & Equilibrium", "State the formula for the **moment of a force** about a pivot, and state the **two conditions** required for an object to be in equilibrium. **[3 marks]**"),
            ("Thermal Physics", "200 g of water (specific heat capacity = `4200 J/(kg °C)`) is heated from **22 °C** to **80 °C**. Calculate the thermal energy transferred to the water. **[2 marks]**"),
            ("Waves & Diffraction", "State **two changes** you could make in a ripple tank experiment to increase the **diffraction** of water waves passing through a gap. **[2 marks]**"),
            ("Electromagnetism & Transformers", "A transformer has **360 turns** on its primary coil connected to **240 V a.c.**, and **40 turns** on its secondary coil. (i) Calculate the secondary output voltage **[2 marks]**. (ii) Explain why a transformer does **not** work with d.c. **[2 marks]**."),
            ("Nuclear Physics", "Explain what happens during **nuclear fusion** in the Sun, naming the nuclei involved. **[3 marks]**"),
        ]
        # Pick matching practice question
        picked_title, picked_q = practice_bank[0]
        for pt_title, pt_q in practice_bank:
            if any(w in query.lower() for w in pt_title.lower().split()):
                picked_title, picked_q = pt_title, pt_q
                break
        else:
            picked_title, picked_q = random.choice(practice_bank)

        reply = (
            f"### 📝 IGCSE Physics Practice Question — {picked_title}\n\n"
            f"{picked_q}\n\n"
            f"*Reply with your working & answer (or write it on paper and upload a photo in **Correct My Answer**) and I will mark it strictly using Cambridge 0625 rules!*"
        )
        return {
            "ok": True,
            "refused": False,
            "category": "physics",
            "reply": reply,
            "notes_cards": notes_cards,
            "ocr_text": None,
            "engine": "Haroony Physics AI Examiner",
        }
