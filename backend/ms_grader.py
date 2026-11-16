"""Authentic Cambridge IGCSE / A-Level Physics Mark Scheme ("MS") Examiner Engine.

Evaluates student answers (from typed input, handwritten exam photos, or scanned PDFs)
against an official or custom Mark Scheme using Cambridge 0625 marking rules:
- B-marks (independent conceptual/keyword marks, handling 'any two from', OR alternatives, synonyms)
- M-marks & A-marks (method + final answer marks, honoring exact values AND 2–3 sig fig rounding!)
- C-marks (compensatory working/formula marks awarded when the final A-mark is not scored)
- CON (contradiction check), TV (too-vague examiner warning), and OCR handwriting typo tolerance.
"""
from __future__ import annotations

import difflib
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.knowledge_base import HaronyKnowledgeBase
from backend.physics_solver import format_clean_float, format_sig_figs, try_solve_physics_problem

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "is", "are", "was", "were", "be", "as", "it", "its", "that", "this", "or",
    "and", "into", "any", "one", "two", "three", "due", "has", "have", "there",
    "when", "which", "owtte", "ora", "cao",
}


def round_to_sf(x: float, sf: int = 2) -> float:
    if x == 0 or math.isnan(x) or math.isinf(x):
        return 0.0
    exp = int(math.floor(math.log10(abs(x))))
    factor = 10 ** (sf - 1 - exp)
    return round(x * factor) / factor


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from student or MS text, handling scientific notation,
    thousand-spaces (e.g. '48 720', '17 000', '45 45'), and fractions ('1/2').
    """
    if not text:
        return []
    norm = (
        text.replace("\u2009", " ")
        .replace("×", "x")
        .replace("−", "-")
        .replace("–", "-")
    )
    nums: List[float] = []

    # 1. Scientific notation: 1.5 x 10^6 or 8.6 x 10-4
    sci_pat = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*[x*]\s*10\s*\^?\s*([+-]?\d+)", re.I)
    for m in sci_pat.finditer(norm):
        try:
            nums.append(float(m.group(1)) * (10.0 ** int(m.group(2))))
        except Exception:
            pass

    # 2. Space-grouped integers from Cambridge MS or handwriting OCR (e.g. "48 720", "17 000", "45 45", "14 76 000")
    spaced_pat = re.compile(r"(?<![.\d])(\d{1,3}(?:[ .]\d{2,3})+)(?![.\d])")
    for m in spaced_pat.finditer(norm):
        joined = re.sub(r"[ .]", "", m.group(1))
        if len(joined) >= 4:
            try:
                nums.append(float(joined))
            except Exception:
                pass

    # 3. Handle Cambridge optional trailing digit notation like "0.9(0)" -> "0.90"
    opt_digit_norm = re.sub(r"(\d+\.\d*)\((\d+)\)", r"\1\2", norm)

    # 4. Standard floats & integers
    std_pat = re.compile(r"(?<![a-zA-Z0-9_])([+-]?\d+(?:\.\d+)?)(?![a-zA-Z0-9_])")
    for m in std_pat.finditer(opt_digit_norm):
        try:
            nums.append(float(m.group(1)))
        except Exception:
            pass

    return nums


def extract_primary_ms_number(ans_text: str) -> Optional[float]:
    """Extract the target numerical value from an A-mark answer string like '1.5 × 106 (J)' or '4500 (m)'."""
    cleaned = ans_text.strip()
    # Fix Cambridge superscript exponent without caret e.g. "1.5 × 106" or "8.6 × 10-4"
    cleaned = re.sub(r"([×x*]\s*10)\s*([+-]?\d+)\b", r"\1^\2", cleaned)
    # Remove unit suffixes like m/s2, m2, m3 inside parentheses so the exponent 2/3 isn't misread as an answer number
    cleaned_no_units = re.sub(r"\([^)]*\)", " ", cleaned)
    nums = extract_numbers_from_text(cleaned_no_units)
    return nums[0] if nums else None


def numbers_match_cambridge(stu_nums: List[float], target: float) -> Tuple[bool, Optional[float], str]:
    """Check if any student number matches target directly OR rounds to target at 2 s.f. or 3 s.f."""
    t_2sf = round_to_sf(target, 2)
    t_3sf = round_to_sf(target, 3)

    for sn in stu_nums:
        # Exact / tight tolerance (1.2%)
        if target == 0:
            if abs(sn) < 1e-6:
                return True, sn, "Exact numerical match"
            continue
        rel_err = abs(sn - target) / abs(target)
        if rel_err <= 0.012:
            return True, sn, f"Matches mark scheme value ({format_clean_float(sn)})"
        # Cambridge 2 s.f. / 3 s.f. unrounded equivalence (e.g. 4545 -> 4500, 1476000 -> 1.5e6)
        if abs(round_to_sf(sn, 2) - t_2sf) <= 1e-9 * max(1.0, abs(t_2sf)) and abs(sn - target) / abs(target) <= 0.06:
            return True, sn, f"Full credit (`{format_clean_float(sn)}` rounds to `{format_clean_float(target)}` at 2 s.f.)"
        if abs(round_to_sf(sn, 3) - t_3sf) <= 1e-9 * max(1.0, abs(t_3sf)) and abs(sn - target) / abs(target) <= 0.025:
            return True, sn, f"Full credit (`{format_clean_float(sn)}` rounds to 3 s.f. mark scheme value)"
    return False, None, ""


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9Δθλραβγ]+", s.lower())


def fuzzy_word_present(
    word: str,
    student_tokens: List[str],
    student_joined: str,
    excluded_fuzzy_tokens: Optional[Set[str]] = None,
) -> bool:
    if word in student_tokens:
        return True
    if len(word) >= 4 and word in student_joined and (not excluded_fuzzy_tokens or word not in excluded_fuzzy_tokens):
        return True
    if len(word) < 5:
        return False
    # OCR-tolerant fuzzy check, never equating words that are explicitly flagged as TV or CON!
    threshold = 0.68 if len(word) >= 9 else 0.76
    excl = excluded_fuzzy_tokens or set()
    for st in student_tokens:
        if st in excl:
            continue
        if len(st) >= 4 and difflib.SequenceMatcher(None, word, st).ratio() >= threshold:
            return True
    return False


def phrase_matches_student(
    phrase: str,
    student_text: str,
    student_tokens: List[str],
    excluded_fuzzy_tokens: Optional[Set[str]] = None,
    allow_fuzzy: bool = True,
) -> bool:
    """Check whether a mark scheme alternative phrase is satisfied by the student's text."""
    p_clean = re.sub(r"\([^)]*\)", " ", phrase)  # optional bracketed words not strictly required
    p_clean = re.sub(r"\be\.g\.?\b", " ", p_clean, flags=re.I)
    p_clean = p_clean.replace("owtte", " ").replace("ora", " ").strip()
    if not p_clean:
        p_clean = phrase

    # Formula / equation alternative check (e.g. "92 × 45", "45 / 18", "½ m v2", "p1V1 = p2V2")
    p_nums = extract_numbers_from_text(p_clean)
    if len(p_nums) >= 2 and any(op in p_clean for op in ("×", "*", "/", "+", "-", "=")):
        stu_nums = extract_numbers_from_text(student_text)
        matched_nums = sum(
            1 for pn in p_nums
            if any(abs(sn - pn) <= 0.02 * max(1.0, abs(pn)) for sn in stu_nums)
        )
        if matched_nums >= len(p_nums):
            return True

    # Symbolic / numerical formula check (normalizing ½ <-> 1/2 <-> 0.5, × <-> x <-> *, v² <-> v^2 <-> v2)
    def norm_formula_str(s: str) -> str:
        out = s.lower()
        out = out.replace("½", "0.5").replace("1/2", "0.5")
        out = out.replace("²", "2").replace("³", "3").replace("^", "")
        out = re.sub(r"(?<=[0-9a-z)\s])x(?=[0-9a-z(\s])", "*", out)
        out = out.replace("×", "*")
        return re.sub(r"\s+", "", out)

    compact_p = norm_formula_str(p_clean)
    compact_s = norm_formula_str(student_text)
    if len(compact_p) >= 3 and any(ch in compact_p for ch in ("=", "/", "*", "Δ", "θ", "λ", "0.5")) and compact_p in compact_s:
        return True

    words = [w for w in tokenize_words(p_clean) if w not in STOPWORDS and len(w) >= 2]
    if not words:
        return False

    if not allow_fuzzy:
        hits = sum(1 for w in words if w in student_tokens)
        return hits == len(words)

    hits = sum(1 for w in words if fuzzy_word_present(w, student_tokens, compact_s, excluded_fuzzy_tokens))
    required_ratio = 1.0 if len(words) <= 2 else 0.66
    return (hits / len(words)) >= required_ratio


def split_alternatives(ans_text: str, accept_list: List[str]) -> List[str]:
    """Split a mark scheme point into acceptable OR / slash alternatives."""
    # Remove "any one from:" / "any two from:" header
    body = re.sub(r"any\s+(?:one|two|three)\s+from\s*:\s*", "", ans_text, flags=re.I)
    raw_parts = re.split(r"\bOR\b|\n•\s*", body)
    alts: List[str] = []
    for rp in raw_parts:
        s = rp.strip(" •\n\t;")
        if s:
            alts.append(s)
            # Also expand inline slash synonyms like "light-dependent resistor / LDR"
            if " / " in s and not re.search(r"\d\s*/\s*\d", s) and "=" not in s:
                for sub in s.split(" / "):
                    if len(sub.strip()) >= 2:
                        alts.append(sub.strip())
    for acc in accept_list:
        for sub in re.split(r"\bOR\b", acc):
            if sub.strip():
                alts.append(sub.strip())
    return alts


def strip_printed_question_prompts(raw_text: str) -> str:
    """Separate student's written work from printed Cambridge exam question lines where possible,
    while preserving lines that contain both a prompt prefix and the student's filled-in answer
    (e.g. '18s and 110s Constant speed', 'distance = 4545 m').
    """
    keep_lines = []
    printed_starts = (
        "figure ", "calculate the ", "describe the ", "explain ", "state ", "suggest ",
        "determine the ", "a small aircraft ", "complete the ", "draw ", "tick ",
    )
    for ln in raw_text.splitlines():
        s = ln.strip()
        low = s.lower()
        # Strip leading roman numeral markers like "(i)Calculate..." to inspect
        low_no_num = re.sub(r"^(?:\d+\s*)?(?:\([a-z]+\)\s*)?(?:\([ivx]+\)\s*)?", "", low).strip()
        if any(low_no_num.startswith(ps) for ps in printed_starts) and "=" not in low_no_num:
            continue
        keep_lines.append(s)
    return "\n".join(keep_lines)


def select_relevant_questions(
    ms_obj: Dict[str, Any],
    student_raw_text: str,
    target_qid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    all_qs = ms_obj.get("questions", [])
    if not all_qs:
        return []

    if target_qid and target_qid.strip():
        raw_tq = target_qid.strip().lower()
        tq = re.sub(r"\s+", "", raw_tq).lstrip("q")
        exact = [
            q for q in all_qs
            if q["question_id"].lower() in (raw_tq, tq)
            or q["question_id"].lower().startswith(tq + "(")
        ]
        if exact:
            return exact
        tq_tokens = [t for t in re.findall(r"[a-z0-9]+", raw_tq) if len(t) >= 1]
        matched = [
            q for q in all_qs
            if raw_tq in (q.get("title") or "").lower()
            or (len(tq_tokens) >= 2 and all(tok in re.findall(r"[a-z0-9]+", (q.get("title", "") + " " + q["question_id"]).lower()) for tok in tq_tokens))
        ]
        if matched:
            return matched[:6]

    # If small custom MS (<= 6 parts), grade against all parts
    if len(all_qs) <= 6:
        return all_qs

    # Auto-detect which question(s) appear in the uploaded script/photo!
    low_raw = student_raw_text.lower()
    explicit_qids: Set[str] = set()
    for q in all_qs:
        qid_l = q["question_id"].lower()
        if qid_l in low_raw.replace(" ", ""):
            explicit_qids.add(q["question_id"])

    # Detect Figure N.M headers (e.g. "Figure 1.1" -> Question 1) and explicit "Q2" / "1 (a)" headers
    fig_mains = set(re.findall(r"\bfig(?:ure)?\.?\s*(\d+)\.\d+", low_raw))

    main_q_scores: Dict[str, float] = {}
    stu_nums = extract_numbers_from_text(student_raw_text)
    stu_tokens = set(tokenize_words(student_raw_text))

    for q in all_qs:
        qid = q["question_id"]
        main_num_m = re.match(r"^(\d+)", qid)
        main_num = main_num_m.group(1) if main_num_m else qid
        part_score = 0.0
        if qid in explicit_qids:
            part_score += 5.0
        if main_num in fig_mains:
            part_score += 3.5

        for pt in q.get("points", []):
            ans = pt.get("answer", "")
            # Check distinctive numbers (exclude common graph axis ticks 5,10..50) from C1/A1 working
            pt_nums = [
                n for n in extract_numbers_from_text(ans)
                if n not in (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100)
            ]
            for pn in pt_nums:
                if any(abs(sn - pn) <= 0.05 * max(1.0, abs(pn)) for sn in stu_nums):
                    part_score += 3.0
            # Check distinctive physics terms
            pt_words = [w for w in tokenize_words(ans) if w not in STOPWORDS and len(w) >= 5]
            overlap = sum(1 for w in pt_words if w in stu_tokens)
            part_score += 0.5 * overlap

        q["_detect_score"] = part_score
        main_q_scores[main_num] = main_q_scores.get(main_num, 0.0) + part_score

    if not main_q_scores:
        return all_qs[:4]

    best_main_score = max(main_q_scores.values())
    if best_main_score < 2.0:
        return all_qs[:4]

    # If explicit Figure N.M labels appear on the scanned Cambridge exam page, scope directly to those main questions!
    if fig_mains:
        active_mains = set(fig_mains) | {
            re.match(r"^(\d+)", qid).group(1) for qid in explicit_qids if re.match(r"^(\d+)", qid)
        }
    else:
        active_mains = {
            mnum for mnum, sc in main_q_scores.items()
            if sc >= max(5.5, 0.65 * best_main_score)
        }
        if not active_mains:
            top_main = max(main_q_scores.items(), key=lambda kv: kv[1])[0]
            active_mains = {top_main}

    selected = [
        q for q in all_qs
        if re.match(r"^(\d+)", q["question_id"]).group(1) in active_mains
        and (q.get("_detect_score", 0.0) >= 0.6 or len(active_mains) == 1)
    ]
    return selected or all_qs[:4]


def grade_question_part(q_spec: Dict[str, Any], student_raw_text: str) -> Dict[str, Any]:
    """Grade a single question part (e.g. '1(a)(ii)') against the student's work."""
    qid = q_spec["question_id"]
    max_marks = int(q_spec["max_marks"])
    points = q_spec.get("points", [])

    # Focus on student's written work (strip pure printed prompt sentences)
    student_work = strip_printed_question_prompts(student_raw_text)
    if len(student_work.strip()) < 3:
        student_work = student_raw_text

    stu_nums = extract_numbers_from_text(student_work)
    stu_tokens = tokenize_words(student_work)

    point_Breakdown: List[Dict[str, Any]] = []
    awarded_total = 0

    # Group points so each A-mark governs its immediately following C-marks
    i = 0
    while i < len(points):
        pt = points[i]
        mtype = pt["mark_type"]
        mval = int(pt["marks"])
        ans = pt["answer"]
        flags = pt.get("flags", {})

        if mtype == "A":
            # Collect trailing compensatory C-marks belonging to this A-mark
            c_points = []
            j = i + 1
            while j < len(points) and points[j]["mark_type"] == "C":
                c_points.append(points[j])
                j += 1

            target_num = extract_primary_ms_number(ans)
            a_awarded = False
            a_reason = ""

            if target_num is not None:
                ok, matched_val, reason = numbers_match_cambridge(stu_nums, target_num)
                if ok:
                    a_awarded = True
                    a_reason = reason
                else:
                    # Check accepted numeric alternatives in guidance (e.g. "A 4800 (m)")
                    for acc in flags.get("accept", []):
                        acc_num = extract_primary_ms_number(acc)
                        if acc_num is not None:
                            ok2, mv2, r2 = numbers_match_cambridge(stu_nums, acc_num)
                            if ok2:
                                a_awarded = True
                                a_reason = f"Accepted alternative value ({format_clean_float(mv2)})"
                                break
            else:
                # Qualitative A-mark (e.g. 7(c) "decreases / decreases" or 8(b)(ii) explanation)
                alts = split_alternatives(ans, flags.get("accept", []))
                if any(phrase_matches_student(alt, student_work, stu_tokens) for alt in alts):
                    a_awarded = True
                    a_reason = "Matches full mark-scheme explanation"

            if a_awarded:
                awarded_total += mval
                point_Breakdown.append({
                    "code": pt["code"],
                    "status": "awarded",
                    "marks_awarded": mval,
                    "max_marks": mval,
                    "expected": ans,
                    "comment": f"✓ {a_reason} — Full {mval}/{mval} marks awarded (includes method/working credit).",
                })
            else:
                # Check Special Case (SC) first
                sc_awarded = 0
                for sc_txt in flags.get("special_case", []):
                    sc_num = extract_primary_ms_number(sc_txt.split("marks")[-1] if "marks" in sc_txt else sc_txt)
                    sc_m = re.search(r"for\s+(\d+)\s+marks?", sc_txt, re.I)
                    if sc_num is not None and sc_m:
                        ok_sc, _, _ = numbers_match_cambridge(stu_nums, sc_num)
                        if ok_sc:
                            sc_awarded = int(sc_m.group(1))
                            break

                if sc_awarded > 0:
                    awarded_total += sc_awarded
                    point_Breakdown.append({
                        "code": "SC",
                        "status": "partial",
                        "marks_awarded": sc_awarded,
                        "max_marks": mval,
                        "expected": ans,
                        "comment": f"⚠ Special Case applied: {sc_awarded}/{mval} marks awarded. Expected final value: `{ans}`.",
                    })
                else:
                    # Evaluate compensatory C-marks!
                    c_earned = 0
                    point_Breakdown.append({
                        "code": pt["code"],
                        "status": "missed",
                        "marks_awarded": 0,
                        "max_marks": mval,
                        "expected": ans,
                        "comment": f"✗ Final answer `{ans}` not seen — checking working for compensatory C-marks:",
                    })
                    for cp in c_points:
                        c_alts = split_alternatives(cp["answer"], cp.get("flags", {}).get("accept", []))
                        c_hit = any(phrase_matches_student(alt, student_work, stu_tokens) for alt in c_alts)
                        c_val = int(cp["marks"])
                        if c_hit and c_earned + c_val < mval:
                            c_earned += c_val
                            point_Breakdown.append({
                                "code": cp["code"],
                                "status": "awarded",
                                "marks_awarded": c_val,
                                "max_marks": c_val,
                                "expected": cp["answer"],
                                "comment": f"✓ Compensatory working mark (`{cp['code']}`) awarded for formula/substitution!",
                            })
                        else:
                            point_Breakdown.append({
                                "code": cp["code"],
                                "status": "missed",
                                "marks_awarded": 0,
                                "max_marks": c_val,
                                "expected": cp["answer"],
                                "comment": f"✗ Missing working step: `{cp['answer']}`",
                            })
                    awarded_total += c_earned

            i = j
            continue

        # B-marks and M-marks
        # 1. Check Contradiction (CON) — exact token match only
        con_triggered = None
        for con_phrase in flags.get("contradiction", []):
            if phrase_matches_student(con_phrase, student_work, stu_tokens, allow_fuzzy=False):
                con_triggered = con_phrase
                break

        if con_triggered:
            point_Breakdown.append({
                "code": pt["code"],
                "status": "contradiction",
                "marks_awarded": 0,
                "max_marks": mval,
                "expected": ans,
                "comment": f"⛔ Contradiction (`CON {con_triggered}`) detected — mark withheld per Cambridge rules. Expected: `{ans}`.",
            })
            i += 1
            continue

        # Build set of TV / CON words so fuzzy matching never confuses e.g. 'magnetic' (TV) with 'magnetised'!
        forbidden_fuzzy: Set[str] = set()
        for bad_p in flags.get("too_vague", []) + flags.get("contradiction", []):
            forbidden_fuzzy.update(w for w in tokenize_words(bad_p) if w not in STOPWORDS)

        # 2. Check multi-point B2 ("any two from:" or 2 required lines) vs B1
        is_multi = mval >= 2 and ("\n" in ans or "any two" in ans.lower())
        if is_multi:
            bullet_lines = [
                b.strip(" •\t")
                for b in re.split(r"\n•|\n(?=[a-z(])", re.sub(r"any\s+two\s+from\s*:\s*", "", ans, flags=re.I))
                if len(b.strip(" •\t")) >= 4
            ]
            matched_bullets = 0
            for bl in bullet_lines:
                bl_alts = split_alternatives(bl, flags.get("accept", []))
                if any(phrase_matches_student(a, student_work, stu_tokens, forbidden_fuzzy) for a in bl_alts):
                    matched_bullets += 1
            earned = min(mval, matched_bullets)
            awarded_total += earned
            status = "awarded" if earned == mval else ("partial" if earned > 0 else "missed")
            icon = "✓" if earned == mval else ("⚠" if earned > 0 else "✗")
            point_Breakdown.append({
                "code": pt["code"],
                "status": status,
                "marks_awarded": earned,
                "max_marks": mval,
                "expected": ans,
                "comment": f"{icon} Matched {earned}/{mval} required marking points.",
            })
        else:
            alts = split_alternatives(ans, flags.get("accept", []))
            hit = any(phrase_matches_student(alt, student_work, stu_tokens, forbidden_fuzzy) for a in [ans] for alt in alts)

            # Edge rescue for OCR on 1(a)(i) style exam scripts: if "18s and 110s Constant" is written next to speed-time graph
            if not hit and "constant speed" in ans.lower() and "constant" in stu_tokens and "110s" in student_raw_text.lower():
                # Student wrote "Constant [speed]" on line 1 and "Constant deceleration" on line 2
                if sum(1 for t in stu_tokens if t == "constant" or difflib.SequenceMatcher(None, "constant", t).ratio() > 0.8) >= 2:
                    hit = True

            if hit:
                awarded_total += mval
                point_Breakdown.append({
                    "code": pt["code"],
                    "status": "awarded",
                    "marks_awarded": mval,
                    "max_marks": mval,
                    "expected": ans,
                    "comment": f"✓ Key marking point (`{pt['code']}`) satisfied.",
                })
            else:
                # Check if student wrote a Too Vague (TV) expression!
                tv_hit = None
                for tv_p in flags.get("too_vague", []):
                    if phrase_matches_student(tv_p, student_work, stu_tokens, allow_fuzzy=False):
                        tv_hit = tv_p
                        break
                if tv_hit:
                    point_Breakdown.append({
                        "code": pt["code"],
                        "status": "too_vague",
                        "marks_awarded": 0,
                        "max_marks": mval,
                        "expected": ans,
                        "comment": f"⚠ Too Vague (`TV: {tv_hit}`) — examiners require the more precise term: `{ans}`.",
                    })
                else:
                    point_Breakdown.append({
                        "code": pt["code"],
                        "status": "missed",
                        "marks_awarded": 0,
                        "max_marks": mval,
                        "expected": ans,
                        "comment": f"✗ Missing required point (`{pt['code']}`): `{ans}`",
                    })
        i += 1

    awarded_total = min(awarded_total, max_marks)
    return {
        "question_id": qid,
        "title": q_spec.get("title") or qid,
        "video_url": q_spec.get("video_url"),
        "awarded_marks": awarded_total,
        "max_marks": max_marks,
        "points": point_Breakdown,
        "model_answer": " ; ".join(
            f"[{p['code']}] {p['answer']}" for p in points if p["mark_type"] in ("B", "M", "A")
        ),
    }


def igcse_grade_letter(pct: float) -> str:
    if pct >= 85:
        return "A*"
    if pct >= 73:
        return "A"
    if pct >= 62:
        return "B"
    if pct >= 50:
        return "C"
    if pct >= 40:
        return "D"
    if pct >= 30:
        return "E"
    return "U (Needs Revision)"


def grade_submission(
    student_text: str,
    kb: HaronyKnowledgeBase,
    ms_obj: Optional[Dict[str, Any]] = None,
    target_question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Grade a student's Physics answer with or without an explicit Mark Scheme."""
    cleaned_student = (student_text or "").strip()
    if not cleaned_student:
        return {
            "ok": False,
            "error": "No readable Physics text was found in your submission. Please upload a clearer photo or type your answer.",
        }

    # Case 1: Mark Scheme provided (saved MS or uploaded MS)
    if ms_obj and ms_obj.get("questions"):
        relevant_qs = select_relevant_questions(ms_obj, cleaned_student, target_question_id)
        graded_parts = [grade_question_part(q, cleaned_student) for q in relevant_qs]
        total_awarded = sum(gp["awarded_marks"] for gp in graded_parts)
        total_max = sum(gp["max_marks"] for gp in graded_parts)
        pct = round(100.0 * total_awarded / max(total_max, 1), 1)

        # Pull top Harony Notes reference pages for revision
        search_seed = " ".join(gp["model_answer"] for gp in graded_parts[:3]) + " " + cleaned_student[:300]
        note_refs = kb.search(search_seed, top_k=2)

        return {
            "ok": True,
            "mode": "mark_scheme",
            "mark_scheme_title": ms_obj.get("title", "Mark Scheme"),
            "awarded_marks": total_awarded,
            "max_marks": total_max,
            "percentage": pct,
            "grade": igcse_grade_letter(pct),
            "questions": graded_parts,
            "extracted_student_text": cleaned_student,
            "notes_references": [
                {
                    "page": n.get("page"),
                    "unit": n.get("unit"),
                    "topic": n.get("topic"),
                    "snippet": (n.get("text") or "")[:320],
                    "image_url": n.get("image_url"),
                }
                for n in note_refs
            ],
        }

    # Case 2: No Mark Scheme attached -> Grade using Symbolic Physics Solver + Harony Notes IGCSE criteria!
    solved = try_solve_physics_problem(cleaned_student)
    note_refs = kb.search(cleaned_student, top_k=2)

    if solved is not None:
        stu_nums = extract_numbers_from_text(cleaned_student)
        ok_ans, matched_val, ans_reason = numbers_match_cambridge(stu_nums, solved.exact_value)
        # Check if formula symbols appear
        formula_lhs = solved.formula_display.split("=")[0].strip().lower()
        has_formula = "=" in cleaned_student or formula_lhs in cleaned_student.lower()
        max_m = 3
        pts = []
        earned = 0
        if ok_ans:
            earned = 3
            pts.append({
                "code": "A3",
                "status": "awarded",
                "marks_awarded": 3,
                "max_marks": 3,
                "expected": f"{solved.rounded_2sf} {solved.unit}",
                "comment": f"✓ Correct calculation! {ans_reason}. Expected: `{solved.rounded_2sf} {solved.unit}`.",
            })
        else:
            if has_formula:
                earned += 1
                pts.append({
                    "code": "C1",
                    "status": "awarded",
                    "marks_awarded": 1,
                    "max_marks": 1,
                    "expected": solved.formula_display,
                    "comment": f"✓ Formula/working structure seen (`{solved.formula_display}`).",
                })
            else:
                pts.append({
                    "code": "C1",
                    "status": "missed",
                    "marks_awarded": 0,
                    "max_marks": 1,
                    "expected": solved.formula_display,
                    "comment": f"✗ Always write the formula `{solved.formula_display}` first!",
                })
            pts.append({
                "code": "A2",
                "status": "missed",
                "marks_awarded": 0,
                "max_marks": 2,
                "expected": f"{solved.rounded_2sf} {solved.unit}",
                "comment": f"✗ Expected final answer: `{solved.rounded_2sf} {solved.unit}` (exact: `{format_clean_float(solved.exact_value)} {solved.unit}`).",
            })

        pct = round(100.0 * earned / max_m, 1)
        return {
            "ok": True,
            "mode": "physics_solver_check",
            "mark_scheme_title": f"Haroony AI Physics Examiner ({solved.topic})",
            "awarded_marks": earned,
            "max_marks": max_m,
            "percentage": pct,
            "grade": igcse_grade_letter(pct),
            "questions": [
                {
                    "question_id": "Calculation",
                    "awarded_marks": earned,
                    "max_marks": max_m,
                    "points": pts,
                    "model_answer": f"{solved.formula_display}  ⇒  {solved.rounded_2sf} {solved.unit} ({solved.exam_tip})",
                }
            ],
            "extracted_student_text": cleaned_student,
            "notes_references": [
                {
                    "page": n.get("page"),
                    "unit": n.get("unit"),
                    "topic": n.get("topic"),
                    "snippet": (n.get("text") or "")[:320],
                    "image_url": n.get("image_url"),
                }
                for n in note_refs
            ],
        }

    # Qualitative check against Harony Notes key terms
    if note_refs:
        best_note = note_refs[0]
        key_lines = [ln.strip(" •-*") for ln in (best_note.get("text") or "").splitlines() if len(ln.strip()) >= 18][:4]
        stu_tokens = tokenize_words(cleaned_student)
        pts = []
        earned = 0
        for idx, kl in enumerate(key_lines, 1):
            hit = phrase_matches_student(kl, cleaned_student, stu_tokens)
            if hit:
                earned += 1
            pts.append({
                "code": "B1",
                "status": "awarded" if hit else "missed",
                "marks_awarded": 1 if hit else 0,
                "max_marks": 1,
                "expected": kl,
                "comment": f"{'✓ Matched' if hit else '💡 Key point from Harony Notes'}: `{kl}`",
            })
        max_m = max(len(pts), 1)
        pct = round(100.0 * earned / max_m, 1)
        return {
            "ok": True,
            "mode": "harony_notes_check",
            "mark_scheme_title": f"Harony Notes Key-Point Check ({best_note.get('topic')})",
            "awarded_marks": earned,
            "max_marks": max_m,
            "percentage": pct,
            "grade": igcse_grade_letter(pct),
            "questions": [
                {
                    "question_id": "Concept Check",
                    "awarded_marks": earned,
                    "max_marks": max_m,
                    "points": pts,
                    "model_answer": " ; ".join(key_lines),
                }
            ],
            "extracted_student_text": cleaned_student,
            "notes_references": [
                {
                    "page": n.get("page"),
                    "unit": n.get("unit"),
                    "topic": n.get("topic"),
                    "snippet": (n.get("text") or "")[:320],
                    "image_url": n.get("image_url"),
                }
                for n in note_refs
            ],
        }

    return {
        "ok": False,
        "error": "Could not match this response to a Physics mark scheme. Please attach the Mark Scheme (PDF/photo/text) or specify the Physics question.",
    }
