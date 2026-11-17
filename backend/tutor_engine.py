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
from backend.physics_reasoner import try_reason_physics_variation
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


def rank_relevant_bullets(bullets: List[str], query: str, limit: int = 8) -> List[str]:
    """Prioritize sentences that directly address the student's specific question terms."""
    q_words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in STOPWORDS}
    if not q_words or len(bullets) <= 3:
        return bullets[:limit]
    scored = []
    for idx, b in enumerate(bullets):
        b_low = b.lower()
        hits = sum(1 for w in q_words if w in b_low)
        scored.append((hits, -idx, b))
    scored.sort(reverse=True)
    # Keep original reading order among top matches if they tie at 0
    if scored[0][0] == 0:
        return bullets[:limit]
    top = [b for _, _, b in scored[:limit]]
    return top


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
            pts = [p for p in q.get("points", []) if len(p.get("answer", "")) >= 6]
            if not pts:
                continue
            combined_txt = (
                str(q.get("title", "")) + " "
                + str(q.get("chapter", "")) + " "
                + " ".join(p["answer"] + " " + p.get("guidance", "") for p in pts)
            ).lower()
            hits = sum(1 for w in q_words if w in combined_txt)
            if hits >= 2 or (len(q_words) == 1 and hits == 1 and q_words.issubset(set(re.findall(r"[a-z]{4,}", str(q.get("title", "")).lower())))):
                scored.append((hits, {"ms_title": ms["title"], "question": q}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:2]]


class HaroonyTutorEngine:
    def __init__(self, guardrail: PhysicsGuardrail, kb: HaronyKnowledgeBase, learner: Optional[Any] = None) -> None:
        self.guardrail = guardrail
        self.kb = kb
        self.learner = learner

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

        # Only inherit previous turn if the message is a genuine anaphoric follow-up (never on a new topic!)
        PURE_FOLLOWUP_RE = re.compile(
            r"^\s*(?:why\??|how\??|explain\s+more|tell\s+me\s+more|simpler|simplify|"
            r"give\s+(?:me\s+)?(?:an?\s+)?example|another\s+example|show\s+(?:the\s+)?(?:formula|equation|working|steps?)|"
            r"what(?:'s|\s+is)\s+(?:the|its)\s+unit\??|(?:youtube\s+)?videos?\s+for\s+(?:this|that|it)\??)\s*[?!.]*\s*$",
            re.I,
        )
        search_query = combined_query
        if history and PURE_FOLLOWUP_RE.match(combined_query):
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

        # 4. Run Symbolic Physics Calculation Solver + First-Principles Reasoner + Trained & Online ML Classifiers + Harony RAG
        solved = try_solve_physics_problem(combined_query)
        reasoned_md = None if solved else try_reason_physics_variation(combined_query)
        predicted_unit = self.kb.predict_unit(search_query)
        online_pred = self.learner.predict_unit_online(search_query) if self.learner else None
        learned_hint = self.learner.find_learned_guidance(search_query) if self.learner else None
        predicted_intent = self.kb.predict_intent(search_query)
        raw_note_hits = self.kb.search(search_query, top_k=5)
        if raw_note_hits and raw_note_hits[0].get("score", 0.0) >= 0.24:
            primary_unit = raw_note_hits[0].get("unit")
        elif online_pred and online_pred.get("confidence", 0.0) >= 0.62:
            primary_unit = online_pred["unit"]
        else:
            primary_unit = predicted_unit
        primary_topic = raw_note_hits[0].get("topic") if raw_note_hits else None
        note_hits = [nh for nh in raw_note_hits if not primary_unit or nh.get("unit") == primary_unit][:3]
        if not note_hits and raw_note_hits:
            note_hits = raw_note_hits[:3]
        concept_entry = self.kb.match_concept_atlas(search_query, preferred_unit=primary_unit)
        if not concept_entry:
            concept_entry = self.kb.match_concept_atlas(search_query)
        # Only use Question Bank when exact paper code or very strong match
        qb_hits = self.kb.search_question_bank(search_query, top_k=2, preferred_unit=primary_unit)
        is_paper_code_q = bool(re.search(r"\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+", combined_query, re.I))
        ms_tips = find_related_ms_tips(search_query, self.kb) if is_paper_code_q else []
        vid_hits = self.kb.search_videos(search_query, top_k=3, preferred_unit=primary_unit, preferred_topic=primary_topic)
        # Supplement videos from matched ClassWork / P4 / P2 question pages if needed
        existing_vurls = {v.get("video_url") for v in vid_hits if v.get("video_url")}
        for qb in qb_hits:
            if qb.get("score", 0.0) < 0.45:
                continue
            for yt_u in qb.get("youtube_links", []):
                if yt_u and yt_u not in existing_vurls and len(vid_hits) < 3:
                    existing_vurls.add(yt_u)
                    vid_hits.append({
                        "book": qb.get("book", "Harony 2027"),
                        "unit": qb.get("unit", ""),
                        "chapter": qb.get("chapter", ""),
                        "page": qb.get("page"),
                        "label": (qb.get("paper_codes") or [qb.get("chapter") or "Past Paper Walkthrough"])[0],
                        "video_url": yt_u,
                    })

        # Build rich context block for neural LLM generation
        ctx_parts = []
        if solved:
            ctx_parts.append(solved.to_markdown())
        if learned_hint and learned_hint.get("helpful_votes", 0) >= 2:
            ctx_parts.append(
                f"[Student-Verified Learned Pattern (sim={learned_hint['similarity']})]:\n"
                f"Similar Q: {learned_hint['query']}\nKey points: {learned_hint['reply_excerpt']}"
            )
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
        for qb in qb_hits:
            ms_txt = " | ".join(f"{m.get('title')}: {m.get('answer')}" for m in qb.get("ms_answers", []) if m.get("answer"))
            ctx_parts.append(
                f"[Matched Past Paper ({qb.get('book')} p.{qb.get('page')} - {qb.get('chapter')})]:\n"
                f"Question: {qb.get('question_text', '')[:600]}"
                + (f"\nOfficial Mark Scheme: {ms_txt}" if ms_txt else "")
            )
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

        resolved_topic = (
            (concept_entry or {}).get("topic")
            or primary_topic
            or (learned_hint or {}).get("topic")
            or "Physics"
        )
        resolved_unit = primary_unit or (online_pred or {}).get("unit") or ""
        top_conf = float(raw_note_hits[0].get("score", 0.0)) if raw_note_hits else 0.0
        learn_conf = max(top_conf, float((online_pred or {}).get("confidence", 0.0)), 0.85 if (solved or concept_entry) else 0.5)

        if neural_reply:
            interaction_id = (
                self.learner.record_interaction(
                    query=combined_query,
                    unit=resolved_unit,
                    topic=resolved_topic,
                    reply=neural_reply,
                    confidence=max(learn_conf, 0.85),
                    neural_generated=True,
                )
                if self.learner
                else ""
            )
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
                "interaction_id": interaction_id,
                "unit": resolved_unit,
                "topic": resolved_topic,
                "learning_stats": self.learner.get_stats() if self.learner else None,
                "engine": "Haroony AI (Neural + Harony 2026/2027 RAG)",
            }

        # 5. Compose rich, exam-grade local response from Symbolic Solver + Concept Atlas + Harony Notes + Cambridge MS
        sections: List[str] = []

        if ocr_text and has_file:
            preview = ocr_text[:350] + ("..." if len(ocr_text) > 350 else "")
            sections.append(f"**📷 Read from your uploaded image/PDF:**\n> {preview.replace(chr(10), '  ' + chr(10) + '> ')}")

        if solved:
            sections.append(solved.to_markdown())
        elif reasoned_md:
            sections.append(reasoned_md)

        if concept_entry and not solved and not reasoned_md:
            c_pts = "\n".join(f"- {p}" for p in concept_entry.get("points", []))
            c_md = f"### ⚡ {concept_entry['topic']} *({concept_entry['unit']})*\n\n{concept_entry['summary']}\n\n{c_pts}"
            if concept_entry.get("formulas"):
                c_md += "\n\n**Key Equations:**\n" + "\n".join(f"- `{fm}`" for fm in concept_entry["formulas"])
            if concept_entry.get("examiner_trap"):
                c_md += f"\n\n> **Examiner Trap:** {concept_entry['examiner_trap']}"
            sections.append(c_md)

        if note_hits and not (solved or reasoned_md or concept_entry):
            # Pick the page whose individual sentences actually match the question best
            q_words = {w for w in re.findall(r"[a-z0-9]{3,}", combined_query.lower()) if w not in STOPWORDS}
            scored_pages = []
            for idx_nh, nh in enumerate(note_hits):
                nh_bullets = format_note_text_cleanly(nh.get("text", ""), nh.get("topic", ""))
                best_sent_hits = max((sum(1 for w in q_words if w in b.lower()) for b in nh_bullets), default=0)
                total_hits = sum(1 for w in q_words if w in (nh.get("text") or "").lower())
                scored_pages.append((best_sent_hits, total_hits, -idx_nh, nh, nh_bullets))
            scored_pages.sort(reverse=True)

            if scored_pages and scored_pages[0][0] >= 1:
                primary = scored_pages[0][3]
                p_topic = primary.get("topic", "IGCSE Physics")
                p_unit = primary.get("unit", "")
                p_page = primary.get("page")
                header_cite = f"Harony Notes p.{p_page}" if p_page else primary.get("source", "Harony Notes")

                bullets = list(scored_pages[0][4])
                for _, _, _, nh2, b2 in scored_pages[1:]:
                    if nh2.get("unit") == p_unit:
                        for eb in b2:
                            if eb not in bullets:
                                bullets.append(eb)

                ranked_bullets = rank_relevant_bullets(bullets, combined_query, limit=6)
                if ranked_bullets:
                    bullet_md = "\n".join(f"- {b}" for b in ranked_bullets)
                    sections.append(
                        f"### 📘 {p_topic} *({p_unit} — {header_cite})*\n\n{bullet_md}"
                    )

                all_formulas = []
                for nh in note_hits[:2]:
                    for fm in nh.get("formulas", []):
                        if fm not in all_formulas:
                            all_formulas.append(fm)
                if all_formulas:
                    fm_md = "\n".join(f"- `{fm}`" for fm in all_formulas[:4])
                    sections.append(f"**Key Equations & Relations:**\n{fm_md}")

        if qb_hits and (has_file or is_paper_code_q or qb_hits[0].get("score", 0.0) >= 0.62) and not solved:
            qb0 = qb_hits[0]
            ms_list = [m for m in qb0.get("ms_answers", []) if m.get("answer")]
            if ms_list:
                ms_lines = [f"- **{m.get('title') or m.get('question_id')}** *({m.get('max_marks', 0)} marks)*: {m['answer']}" for m in ms_list[:2]]
                sections.append(
                    f"### 📝 Matched Harony 2027 Past-Paper & Official Mark Scheme *({qb0.get('book')} — {qb0.get('chapter')})*\n"
                    + "\n".join(ms_lines)
                )

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

        final_local_reply = "\n\n---\n\n".join(sections)
        interaction_id = (
            self.learner.record_interaction(
                query=combined_query,
                unit=resolved_unit,
                topic=resolved_topic,
                reply=final_local_reply,
                confidence=learn_conf,
                neural_generated=False,
            )
            if self.learner
            else ""
        )

        return {
            "ok": True,
            "refused": False,
            "category": "physics",
            "reply": final_local_reply,
            "notes_cards": notes_cards,
            "video_links": vid_hits,
            "ocr_text": ocr_text,
            "rag_context": rag_context_str,
            "solved_markdown": solved_md_str,
            "neural_generated": False,
            "interaction_id": interaction_id,
            "unit": resolved_unit,
            "topic": resolved_topic,
            "learning_stats": self.learner.get_stats() if self.learner else None,
            "engine": "Haroony Physics AI (Harony 2026/2027 Books)",
        }

    def _generate_practice_question(self, query: str) -> Dict[str, Any]:
        hits = self.kb.search(query, top_k=3)
        qb_candidates = [
            qb for qb in self.kb.search_question_bank(query, top_k=5)
            if qb.get("ms_answers") and len(qb.get("question_text", "")) >= 60
        ]
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

        if qb_candidates:
            picked = random.choice(qb_candidates)
            q_preview = re.sub(r"https?://\S+", "", picked["question_text"]).strip()[:480]
            pc = (picked.get("paper_codes") or [picked.get("chapter") or "Cambridge 0625"])[0]
            vids = [
                {
                    "book": picked.get("book", "Harony 2027"),
                    "unit": picked.get("unit", ""),
                    "chapter": picked.get("chapter", ""),
                    "page": picked.get("page"),
                    "label": pc,
                    "video_url": u,
                }
                for u in picked.get("youtube_links", [])[:2]
            ]
            reply = (
                f"### 📝 IGCSE Physics Past-Paper Practice — {picked.get('chapter') or topic_label}\n"
                f"**Source:** `{picked.get('book')}` *(p.{picked.get('page')} — {pc})*\n\n"
                f"> {q_preview}\n\n"
                f"*Reply here with your working and answer and I will walk you through the official Cambridge 0625 mark scheme!*"
            )
            return {
                "ok": True,
                "refused": False,
                "category": "physics",
                "reply": reply,
                "notes_cards": notes_cards,
                "video_links": vids,
                "ocr_text": None,
                "engine": "Haroony Physics AI Examiner (Full 2027 Bank)",
            }

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
            f"*Reply here with your working and answer and I will walk you through the Cambridge 0625 marking points!*"
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
