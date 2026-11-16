"""Cambridge IGCSE / A-Level & General Physics Mark Scheme (MS) Parser.

Parses:
1. Official Cambridge / Edexcel Mark Scheme PDFs using coordinate-aware column extraction
   (Question | Answer | Marks | Guidance), accurately distinguishing additive marks (B, M, A)
   from compensatory marks (C) and extracting Acceptable (A), Ignore (I), Too Vague (TV),
   Contradiction (CON), and Special Case (SC) examiner notes.
2. Plain-text or OCR-extracted Mark Schemes (when a student pastes an MS or uploads a photo of an MS).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

# Normalize Private-Use-Area symbol glyphs common in Cambridge PDFs (Symbol font)
PUA_CHAR_MAP = {
    "\uf0b4": "×",
    "\uf0b8": "÷",
    "\uf044": "Δ",
    "\uf071": "θ",
    "\uf06c": "λ",
    "\uf072": "ρ",
    "\uf061": "α",
    "\uf062": "β",
    "\uf067": "γ",
    "\uf02d": "-",
    "\uf03d": "=",
    "\uf0b0": "°",
    "\u0472": "θ",
    "\u0473": "θ",
    "\u2009": " ",
    "\u00a0": " ",
}

MARK_CODE_RE = re.compile(r"\b([BMCA])(\d+)\b")
QUESTION_ID_RE = re.compile(r"^(\d+(?:\([a-z]+\))?(?:\([ivx]+\))?)$", re.IGNORECASE)
INLINE_Q_RE = re.compile(r"(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+\s*(?:\([a-z]\))?\s*(?:\([ivx]+\))?)\s*[:\-–]?\s*", re.IGNORECASE)


def clean_ms_text(text: str) -> str:
    if not text:
        return ""
    for k, v in PUA_CHAR_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_guidance_flags(guidance: str) -> Dict[str, List[str]]:
    """Split Cambridge guidance strings into structured accept / ignore / tv / con / sc lists."""
    flags: Dict[str, List[str]] = {
        "accept": [],
        "ignore": [],
        "too_vague": [],
        "contradiction": [],
        "special_case": [],
        "notes": [],
    }
    if not guidance:
        return flags

    lines = [ln.strip(" •\t") for ln in clean_ms_text(guidance).splitlines() if ln.strip(" •\t")]
    current_bucket = "notes"
    for ln in lines:
        m = re.match(r"^(A|Allow|Accept|I|Ignore|TV|CON|SC)\b\s*(.*)$", ln, flags=re.IGNORECASE)
        if m:
            tag = m.group(1).upper()
            body = m.group(2).strip(" :.-")
            if tag in ("A", "ALLOW", "ACCEPT"):
                current_bucket = "accept"
            elif tag in ("I", "IGNORE"):
                current_bucket = "ignore"
            elif tag == "TV":
                current_bucket = "too_vague"
            elif tag == "CON":
                current_bucket = "contradiction"
            elif tag == "SC":
                current_bucket = "special_case"
            if body:
                flags[current_bucket].append(body)
        else:
            if flags[current_bucket] and current_bucket != "notes":
                flags[current_bucket][-1] = f"{flags[current_bucket][-1]} {ln}".strip()
            else:
                flags["notes"].append(ln)
    return flags


def compute_part_max_marks(points: List[Dict[str, Any]]) -> int:
    """In Cambridge Physics Mark Schemes:
    B, M, and A marks sum to the question part's maximum score.
    C marks are compensatory marks awarded ONLY if the preceding/parent A mark is not scored.
    """
    additive = 0
    c_only = 0
    for pt in points:
        mtype = pt.get("mark_type", "B")
        mval = int(pt.get("marks", 1))
        if mtype in ("B", "M", "A"):
            additive += mval
        elif mtype == "C":
            c_only += mval
    return additive if additive > 0 else max(c_only, 1)


def parse_cambridge_ms_pdf(pdf_path: str | Path, title_override: Optional[str] = None) -> Dict[str, Any]:
    """Parse an official Cambridge IGCSE/A-Level Mark Scheme PDF into structured question parts."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))

    first_page_text = clean_ms_text(doc[0].get_text("text")) if len(doc) > 0 else ""
    syllabus_match = re.search(r"\b(\d{4}/\d{2})\b", first_page_text)
    session_match = re.search(r"((?:May/June|Oct/Nov|Feb/March|March|June|November)\s+20\d{2})", first_page_text, re.I)
    max_mark_match = re.search(r"Maximum Mark:\s*(\d+)", first_page_text, re.I)

    paper_code = syllabus_match.group(1) if syllabus_match else "Physics MS"
    session = session_match.group(1) if session_match else ""
    default_title = f"Cambridge IGCSE Physics {paper_code}" + (f" ({session})" if session else "")
    title = title_override or default_title

    questions_map: Dict[str, Dict[str, Any]] = {}
    question_order: List[str] = []

    current_qid: Optional[str] = None

    for pno in range(len(doc)):
        page = doc[pno]
        words = page.get_text("words")
        if not words:
            continue

        # Detect if this page has the Question | Answer | Marks | Guidance table header
        header_ys = [
            w[1] for w in words
            if w[4] == "Question" and w[0] < 110
        ]
        if not header_ys:
            continue

        min_table_y = min(header_ys) + 10.0
        # Filter out top running header and bottom copyright footer
        page_height = page.rect.height
        body_words = [
            w for w in words
            if w[1] >= min_table_y and w[1] < (page_height - 52)
            and not (w[4] in ("Question", "Answer", "Marks", "Guidance") and any(abs(w[1] - hy) < 6 for hy in header_ys))
        ]
        if not body_words:
            continue

        # Determine column boundaries dynamically from headers if present, with safe defaults
        ans_xs = [w[0] for w in words if w[4] == "Answer"]
        marks_xs = [w[0] for w in words if w[4] == "Marks"]
        guid_xs = [w[0] for w in words if w[4] == "Guidance"]

        q_col_max = 118.0
        marks_col_min = (marks_xs[0] - 12.0) if marks_xs else 428.0
        guid_col_min = (guid_xs[0] - 130.0) if guid_xs else 472.0
        if marks_col_min >= guid_col_min:
            guid_col_min = marks_col_min + 38.0

        # Group words into visual lines (y tolerance ~3.5pt) within each column
        def group_lines(col_words: List[Tuple]) -> List[Tuple[float, str]]:
            if not col_words:
                return []
            col_words = sorted(col_words, key=lambda x: (round(x[1] / 3.0), x[0]))
            lines: List[Tuple[float, List[Tuple]]] = []
            for w in col_words:
                wy = w[1]
                if lines and abs(wy - lines[-1][0]) <= 4.2:
                    lines[-1][1].append(w)
                else:
                    lines.append((wy, [w]))
            out = []
            for ly, lw in lines:
                lw_sorted = sorted(lw, key=lambda x: x[0])
                txt = clean_ms_text(" ".join(x[4] for x in lw_sorted))
                if txt:
                    out.append((ly, txt))
            return out

        q_lines = group_lines([w for w in body_words if w[0] < q_col_max])
        ans_lines = group_lines([w for w in body_words if q_col_max <= w[0] < marks_col_min])
        mark_lines = group_lines([w for w in body_words if marks_col_min <= w[0] < guid_col_min])
        guid_lines = group_lines([w for w in body_words if w[0] >= guid_col_min])

        # Build sorted question anchors by y
        q_anchors: List[Tuple[float, str]] = []
        for y, txt in q_lines:
            compact = txt.replace(" ", "")
            if QUESTION_ID_RE.match(compact):
                q_anchors.append((y, compact))

        # Check if repeated table header resets section mid-page
        section_breaks = sorted(header_ys)

        def get_qid_for_y(y: float) -> Optional[str]:
            nonlocal current_qid
            chosen = current_qid
            for qy, qid in q_anchors:
                if qy <= y + 6.0:
                    chosen = qid
            return chosen

        # Each mark_line (e.g., B1, A3, C1, M1) anchors a marking point band!
        parsed_marks: List[Tuple[float, str, int]] = []
        for my, mtxt in mark_lines:
            m = MARK_CODE_RE.search(mtxt)
            if m:
                parsed_marks.append((my, m.group(1).upper(), int(m.group(2))))

        if not parsed_marks:
            continue

        for idx, (my, mtype, mval) in enumerate(parsed_marks):
            qid = get_qid_for_y(my)
            if not qid:
                continue
            current_qid = qid

            # In Cambridge Mark Schemes, each mark code sits at the top y0 of its cell row
            band_top = my - 3.8 if idx > 0 else min(min_table_y, my - 3.8)

            if idx + 1 < len(parsed_marks):
                next_my = parsed_marks[idx + 1][0]
                next_qs = [qy for qy, _ in q_anchors if my + 4.0 < qy <= next_my + 4.0]
                next_hdrs = [hy for hy in section_breaks if my + 4.0 < hy <= next_my]
                stop_candidates = [next_my - 3.9] + [qy - 3.9 for qy in next_qs] + [hy - 3.9 for hy in next_hdrs]
                band_bottom = min(stop_candidates)
            else:
                band_bottom = page_height - 52.0

            ans_chunk = "\n".join(txt for ay, txt in ans_lines if band_top <= ay <= band_bottom)
            guid_chunk = "\n".join(txt for gy, txt in guid_lines if band_top <= gy <= band_bottom)

            if qid not in questions_map:
                questions_map[qid] = {
                    "question_id": qid,
                    "page": pno + 1,
                    "points": [],
                }
                question_order.append(qid)

            questions_map[qid]["points"].append({
                "mark_type": mtype,
                "marks": mval,
                "code": f"{mtype}{mval}",
                "answer": clean_ms_text(ans_chunk),
                "guidance": clean_ms_text(guid_chunk),
                "flags": parse_guidance_flags(guid_chunk),
            })

    # Finalize max_marks and combined summary per question part
    questions_list: List[Dict[str, Any]] = []
    computed_total = 0
    for qid in question_order:
        qobj = questions_map[qid]
        qmax = compute_part_max_marks(qobj["points"])
        qobj["max_marks"] = qmax
        computed_total += qmax
        qobj["summary_answer"] = " ; ".join(
            f"[{p['code']}] {p['answer']}" for p in qobj["points"] if p["answer"]
        )
        questions_list.append(qobj)

    total_marks = int(max_mark_match.group(1)) if max_mark_match else computed_total

    # Fall back to plain-text parser if PDF wasn't a standard coordinate-layout Cambridge MS
    if not questions_list:
        raw_all = "\n".join(clean_ms_text(p.get_text("text")) for p in doc)
        return parse_freeform_ms_text(raw_all, title=title_override or pdf_path.stem)

    return {
        "id": re.sub(r"[^a-z0-9]+", "-", (paper_code + "-" + session).lower()).strip("-") or pdf_path.stem.lower(),
        "title": title,
        "paper_code": paper_code,
        "session": session,
        "source_file": pdf_path.name,
        "total_marks": total_marks,
        "computed_marks": computed_total,
        "question_count": len(questions_list),
        "questions": questions_list,
    }


def parse_freeform_ms_text(ms_text: str, title: str = "Uploaded Mark Scheme") -> Dict[str, Any]:
    """Parse pasted text or OCR-extracted Mark Scheme into structured marking points."""
    cleaned = clean_ms_text(ms_text)
    if not cleaned:
        return {
            "id": "custom-ms",
            "title": title,
            "paper_code": "Custom",
            "session": "",
            "total_marks": 0,
            "computed_marks": 0,
            "question_count": 0,
            "questions": [],
        }

    # Try splitting by question labels like 1(a)(i), 2(b), Q1, etc., if present
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    questions_map: Dict[str, List[str]] = {}
    q_order: List[str] = []
    cur_q = "Q1"

    q_start_re = re.compile(
        r"^(?:Q(?:uestion)?\s*)?(\d+(?:\s*\([a-z]\))?(?:\s*\([ivx]+\))?)\b[\s:\-–.)]*(.*)$",
        re.IGNORECASE,
    )
    for ln in lines:
        # Avoid mistaking equations like "200 x 4.2 = 840" or "3 half-lives" for question numbers
        m = q_start_re.match(ln)
        is_q_header = False
        if m:
            cand_q = re.sub(r"\s+", "", m.group(1))
            rest = m.group(2).strip()
            if "(" in cand_q or ln.lower().startswith("q"):
                is_q_header = True
            elif len(cand_q) <= 2 and rest and not re.match(r"^[×x*+/=\d.\s%-]", rest) and not rest.lower().startswith(("half-li", "mark", "point", "reason")):
                # Only treat bare integer as question number if line has a mark code or looks like a Q label
                if MARK_CODE_RE.search(ln) or re.search(r"\[\d+\]", ln):
                    is_q_header = True
        if is_q_header and m:
            cur_q = re.sub(r"\s+", "", m.group(1))
            rest = m.group(2).strip()
            if cur_q not in questions_map:
                questions_map[cur_q] = []
                q_order.append(cur_q)
            if rest:
                questions_map[cur_q].append(rest)
        else:
            if cur_q not in questions_map:
                questions_map[cur_q] = []
                q_order.append(cur_q)
            questions_map[cur_q].append(ln)

    questions_list: List[Dict[str, Any]] = []
    computed_total = 0

    for qid in q_order:
        qlines = questions_map[qid]
        points: List[Dict[str, Any]] = []
        for ln in qlines:
            # Check for explicit Cambridge mark codes (B1, M1, C1, A2) or [1], (1 mark)
            codes = list(MARK_CODE_RE.finditer(ln))
            bracket_m = re.search(r"\[(\d+)\s*(?:marks?)?\]|\((\d+)\s*marks?\)", ln, re.I)
            if codes:
                mc = codes[-1]
                mtype, mval = mc.group(1).upper(), int(mc.group(2))
                ans_part = (ln[:mc.start()] + " " + ln[mc.end():]).strip(" -–:;|")
                points.append({
                    "mark_type": mtype,
                    "marks": mval,
                    "code": f"{mtype}{mval}",
                    "answer": ans_part or ln,
                    "guidance": "",
                    "flags": parse_guidance_flags(""),
                })
            elif bracket_m:
                mval = int(bracket_m.group(1) or bracket_m.group(2))
                ans_part = re.sub(r"\[\d+\s*(?:marks?)?\]|\(\d+\s*marks?\)", "", ln, flags=re.I).strip(" -–:;|")
                points.append({
                    "mark_type": "B",
                    "marks": mval,
                    "code": f"B{mval}",
                    "answer": ans_part or ln,
                    "guidance": "",
                    "flags": parse_guidance_flags(""),
                })
            else:
                # Plain bullet/criterion line = 1 mark unless attached to previous
                if ln.upper().startswith(("A ", "ALLOW ", "ACCEPT ", "I ", "IGNORE ", "TV ", "CON ")) and points:
                    points[-1]["guidance"] = (points[-1]["guidance"] + "\n" + ln).strip()
                    points[-1]["flags"] = parse_guidance_flags(points[-1]["guidance"])
                else:
                    points.append({
                        "mark_type": "B",
                        "marks": 1,
                        "code": "B1",
                        "answer": ln.lstrip("•-* "),
                        "guidance": "",
                        "flags": parse_guidance_flags(""),
                    })

        if points:
            qmax = compute_part_max_marks(points)
            computed_total += qmax
            questions_list.append({
                "question_id": qid,
                "page": 1,
                "max_marks": qmax,
                "points": points,
                "summary_answer": " ; ".join(f"[{p['code']}] {p['answer']}" for p in points),
            })

    return {
        "id": "custom-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
        "title": title,
        "paper_code": "Custom",
        "session": "",
        "total_marks": computed_total,
        "computed_marks": computed_total,
        "question_count": len(questions_list),
        "questions": questions_list,
    }
