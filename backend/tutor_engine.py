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
from backend.llm_bridge import try_neural_generation, try_neural_vision_ocr
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
        if w not in STOPWORDS and w not in {
            "explain", "describe", "what", "does", "work", "works", "define",
            "state", "physics", "laws", "effect", "rule", "first", "second", "third", "fluid", "fluids",
        }
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
                "Hello! Welcome to **Haroony AI**.\n\n"
                "Ask me any **IGCSE or A-Level Physics** question, request **YouTube video solutions** for any topic, "
                "or attach/paste (`Ctrl+V`) a photo or PDF of a Physics problem!"
            )
            return {
                "ok": True,
                "refused": False,
                "category": "greeting",
                "reply": greeting_reply,
                "notes_cards": [],
                "video_links": [],
                "ocr_text": None,
                "engine": "Haroony AI",
            }

        # 2. If student attached a photo or PDF, run enhanced local Vision OCR (+ optional server vision boost)!
        ocr_text = None
        is_image_upload = bool(has_file and not (attachment_name or "").lower().endswith(".pdf"))
        if has_file and attachment_bytes is not None:
            ocr_info = extract_from_file_bytes(attachment_bytes, attachment_name or "upload.jpg", max_pages=4)
            ocr_text = (ocr_info.get("text") or "").strip()
            if is_image_upload and (len(ocr_text) < 45 or float(ocr_info.get("avg_confidence") or 0.0) < 0.75):
                vision_txt = await try_neural_vision_ocr(attachment_bytes, attachment_mime, purpose="physics_question")
                if vision_txt and len(vision_txt) > len(ocr_text):
                    ocr_text = vision_txt
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

        # 3. Check if student wants YouTube video solutions for a topic or question!
        if re.search(r"\b(?:youtube|youtu\.be|videos?|video\s+solution|watch)\b", combined_query, re.I):
            note_hits = self.kb.search(search_query, top_k=3)
            pref_unit = note_hits[0].get("unit") if note_hits and note_hits[0].get("score", 0.0) >= 0.25 else None
            pref_topic = note_hits[0].get("topic") if note_hits else None
            if pref_unit:
                note_hits = [nh for nh in note_hits if nh.get("unit") == pref_unit][:2]
            vid_hits = self.kb.search_videos(search_query, top_k=8, preferred_unit=pref_unit, preferred_topic=pref_topic)
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
                    "score": nh.get("score", 0.0),
                }
                for nh in note_hits
                if nh.get("score", 0.0) >= 0.20
            ]
            if vid_hits:
                vid_lines = []
                for idx, v in enumerate(vid_hits, 1):
                    loc = f"{v.get('book', 'Harony 2027')} • {v.get('chapter') or v.get('unit', '')}"
                    if v.get("page"):
                        loc += f" (p.{v['page']})"
                    vid_lines.append(f"{idx}. **{v.get('label', 'Video Solution')}** *({loc})* — {v['video_url']}")
                reply_md = (
                    "### 🎥 Harony Physics YouTube Video Solutions\n\n"
                    "Here are the matching Mr. Haroony video walkthroughs from **Harony ClassWork 2027**, **P2 Classified 2027**, and **P4 Classified 2027**:\n\n"
                    + "\n".join(vid_lines)
                )
            else:
                reply_md = (
                    "### 🎥 YouTube Video Solutions\n\n"
                    "Tell me the specific IGCSE Physics topic (for example: *density, moments, thermal, waves, diffraction, electricity, transformers, half-life, astronomy*) "
                    "or paper code (e.g. *2023 March P4 V2 Q02*) and I will send you Mr. Haroony's exact YouTube walkthrough links!"
                )
            return {
                "ok": True,
                "refused": False,
                "category": "physics",
                "reply": reply_md,
                "notes_cards": notes_cards,
                "video_links": vid_hits,
                "ocr_text": ocr_text,
                "rag_context": "",
                "solved_markdown": None,
                "neural_generated": False,
                "engine": "Haroony Video Library (2,793 Solutions)",
            }

        # 3b. Check if student wants a practice / quiz question!
        if re.search(r"\b(?:quiz\s+me|test\s+me|practice\s+question|give\s+me\s+a\s+question)\b", combined_query, re.I):
            return self._generate_practice_question(search_query)

        # 4. Run Symbolic Physics Calculation Solver + Trained Unit/Intent Classifiers + Harony RAG
        solved = try_solve_physics_problem(combined_query)
        predicted_unit = self.kb.predict_unit(search_query)
        predicted_intent = self.kb.predict_intent(search_query)
        raw_note_hits = self.kb.search(search_query, top_k=5)
        primary_unit = (
            raw_note_hits[0].get("unit")
            if raw_note_hits and raw_note_hits[0].get("score", 0.0) >= 0.24
            else predicted_unit
        )
        primary_topic = raw_note_hits[0].get("topic") if raw_note_hits else None
        note_hits = [nh for nh in raw_note_hits if not primary_unit or nh.get("unit") == primary_unit][:3]
        if not note_hits and raw_note_hits:
            note_hits = raw_note_hits[:3]
        concept_entry = self.kb.match_concept_atlas(search_query, preferred_unit=primary_unit)
        if not concept_entry:
            concept_entry = self.kb.match_concept_atlas(search_query)
        ms_tips = find_related_ms_tips(search_query, self.kb)
        vid_hits = self.kb.search_videos(search_query, top_k=3, preferred_unit=primary_unit, preferred_topic=primary_topic)

        # Build rich context block for neural LLM generation
        ctx_parts = []
        if solved:
            ctx_parts.append(solved.to_markdown())
        if concept_entry:
            ctx_parts.append(
                f"[Harony Concept Guide — {concept_entry['topic']} ({concept_entry['unit']})]:\n"
                f"{concept_entry['summary']}\n"
                + "\n".join(concept_entry.get("points", []))
                + (f"\nFormulas: {', '.join(concept_entry.get('formulas', []))}" if concept_entry.get("formulas") else "")
                + (f"\nExaminer Trap: {concept_entry['examiner_trap']}" if concept_entry.get("examiner_trap") else "")
            )
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
                "score": nh.get("score", 0.0),
            }
            for nh in note_hits
            if nh.get("score", 0.0) >= 0.22
        ]

        rag_context_str = "\n\n".join(ctx_parts)
        solved_md_str = solved.to_markdown() if solved else None

        if neural_reply:
            return {
                "ok": True,
                "refused": False,
                "category": "physics",
                "reply": neural_reply,
                "notes_cards": notes_cards,
                "video_links": vid_hits,
                "ocr_text": ocr_text,
                "rag_context": rag_context_str,
                "solved_markdown": solved_md_str,
                "neural_generated": True,
                "engine": "Haroony AI (Neural + Harony 2026/2027 RAG)",
            }

        # 5. Compose rich, exam-grade local response from Symbolic Solver + Concept Atlas + Harony Notes + Cambridge MS
        sections: List[str] = []

        if ocr_text and has_file:
            preview = ocr_text[:350] + ("..." if len(ocr_text) > 350 else "")
            sections.append(f"**📷 Read from your uploaded image/PDF:**\n> {preview.replace(chr(10), '  ' + chr(10) + '> ')}")

        if solved:
            sections.append(solved.to_markdown())

        if concept_entry and not solved:
            c_pts = "\n".join(f"- {p}" for p in concept_entry.get("points", []))
            c_md = f"### ⚡ {concept_entry['topic']} *({concept_entry['unit']})*\n\n{concept_entry['summary']}\n\n{c_pts}"
            if concept_entry.get("formulas"):
                c_md += "\n\n**Key Equations:**\n" + "\n".join(f"- `{fm}`" for fm in concept_entry["formulas"])
            if concept_entry.get("examiner_trap"):
                c_md += f"\n\n> **Examiner Trap:** {concept_entry['examiner_trap']}"
            sections.append(c_md)

        if note_hits:
            primary = note_hits[0]
            p_topic = primary.get("topic", "IGCSE Physics")
            p_unit = primary.get("unit", "")
            p_page = primary.get("page")
            header_cite = f"Harony Notes p.{p_page}" if p_page else primary.get("source", "Harony Notes")

            bullets = format_note_text_cleanly(primary.get("text", ""), p_topic)
            if len(note_hits) >= 2 and note_hits[1].get("topic") == p_topic and not solved:
                extra_bullets = format_note_text_cleanly(note_hits[1].get("text", ""), p_topic)
                for eb in extra_bullets[:5]:
                    if eb not in bullets:
                        bullets.append(eb)

            max_bullets = 4 if (solved or concept_entry) else 9
            bullet_md = "\n".join(f"- {b}" for b in bullets[:max_bullets])

            title_prefix = "Harony Notes Key Points" if (solved or concept_entry) else f"{p_topic}"
            sections.append(
                f"### 📘 {title_prefix} *({p_unit} — {header_cite})*\n\n{bullet_md}"
            )

            all_formulas = []
            for nh in note_hits[:2]:
                for fm in nh.get("formulas", []):
                    if fm not in all_formulas:
                        all_formulas.append(fm)
            if all_formulas and not solved and not concept_entry:
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
            "rag_context": rag_context_str,
            "solved_markdown": solved_md_str,
            "neural_generated": False,
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
