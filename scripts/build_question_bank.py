#!/usr/bin/env python3
"""Build a unified, unit-verified Question Bank (`data/harony_question_bank.json`)
by pairing all 1,750 OCR'd question pages (`data/extracted_questions.jsonl`)
with official Cambridge Mark Schemes (`data/mark_schemes.json`) and Mr. Haroony's
YouTube walkthroughs (`data/video_solutions.json`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import DATA_DIR, STATIC_DIR
from scripts.ingest_google_drive import normalize_paper_key

YT_ID_RE = re.compile(r"(?:youtu\.be/|studio\.youtube\.com/video/)([A-Za-z0-9_\-]{6,})", re.I)
PAPER_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(\d{1,3}\s*[.)]\s*(?:20\d{2}\s*(?:March|May|Jun|June|Nov)[^\n]{0,70}))",
    re.I,
)
PAPER_CODE_ANY_RE = re.compile(
    r"(20\d{2}\s*(?:March|May|Jun|June|Nov)\s*P[24]\s*(?:V\d)?\s*(?:[Qq]\d+)?)",
    re.I,
)


def clean_question_text(raw: str) -> str:
    lines = []
    for ln in (raw or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(("page ", "question paper", "http://", "https://")):
            continue
        if re.match(r"^u[1-7]\s*ch\d+", low):
            continue
        if re.match(r"^\d{1,3}$", s):
            continue
        lines.append(s)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return text


def main() -> None:
    eq_path = DATA_DIR / "extracted_questions.jsonl"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path}")

    ms_data = json.loads((DATA_DIR / "mark_schemes.json").read_text(encoding="utf-8"))
    videos = json.loads((DATA_DIR / "video_solutions.json").read_text(encoding="utf-8"))

    # Index MS questions by video_url and norm_key
    ms_by_vid: Dict[str, Dict[str, Any]] = {}
    ms_by_nkey: Dict[str, Dict[str, Any]] = {}
    for book in ms_data:
        b_title = book.get("title", "")
        for q in book.get("questions", []):
            rec = {
                "book": b_title,
                "question_id": q.get("question_id", ""),
                "title": q.get("title", ""),
                "chapter": q.get("chapter", ""),
                "unit": q.get("unit", book.get("unit", "")),
                "max_marks": q.get("max_marks", 0),
                "summary_answer": (q.get("summary_answer") or "").strip(),
                "video_url": q.get("video_url"),
            }
            vurl = rec["video_url"]
            if vurl:
                ym = YT_ID_RE.search(vurl)
                if ym:
                    ms_by_vid[ym.group(1)] = rec
            nk = normalize_paper_key(rec["title"] or rec["question_id"])
            if nk and nk != "UNKNOWN":
                ms_by_nkey[nk] = rec

    vid_by_id: Dict[str, Dict[str, Any]] = {}
    for v in videos:
        ym = YT_ID_RE.search(v.get("video_url") or "")
        if ym:
            vid_by_id[ym.group(1)] = v

    bank: List[Dict[str, Any]] = []
    seen_sigs = set()
    matched_ms_count = 0

    for ln in eq_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        raw_txt = row.get("text", "")
        clean_txt = clean_question_text(raw_txt)
        if len(clean_txt) < 35:
            continue

        # Deduplicate near-identical OCR pages across ClassWork vs P4/P2 splits
        sig = re.sub(r"[^a-z0-9]", "", clean_txt.lower())[:140]
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        yt_links = row.get("youtube_links") or []
        ms_matches: List[Dict[str, Any]] = []
        for u in yt_links:
            ym = YT_ID_RE.search(u)
            if ym and ym.group(1) in ms_by_vid:
                ms_matches.append(ms_by_vid[ym.group(1)])

        # Also check paper codes in text if no video match
        if not ms_matches:
            for pc in PAPER_CODE_ANY_RE.findall(raw_txt):
                nk = normalize_paper_key(pc)
                if nk in ms_by_nkey:
                    ms_matches.append(ms_by_nkey[nk])

        # Only keep MS matches whose answer/question keywords actually overlap this page's text
        page_words = set(re.findall(r"[a-z0-9]{4,}", clean_txt.lower()))
        uniq_ms = []
        seen_qid = set()
        for m in ms_matches:
            qid = m["question_id"] + "::" + (m.get("summary_answer") or "")[:40]
            if qid in seen_qid:
                continue
            ms_words = set(re.findall(r"[a-z0-9]{4,}", ((m.get("title") or "") + " " + (m.get("summary_answer") or "")).lower()))
            overlap = len(page_words & ms_words)
            if overlap >= 2 or len(ms_matches) == 1:
                seen_qid.add(qid)
                uniq_ms.append(m)

        if uniq_ms:
            matched_ms_count += 1

        paper_codes = list(dict.fromkeys(PAPER_CODE_ANY_RE.findall(raw_txt)))
        bank.append({
            "id": f"qb-{len(bank)+1:04d}",
            "book": row.get("book", ""),
            "page": row.get("page"),
            "unit": row.get("unit", "Unit 1: Mechanics"),
            "chapter": row.get("chapter", ""),
            "paper_codes": paper_codes[:4],
            "question_text": clean_txt[:1500],
            "youtube_links": yt_links[:4],
            "ms_answers": [
                {
                    "question_id": m["question_id"],
                    "title": m["title"],
                    "max_marks": m["max_marks"],
                    "answer": m["summary_answer"][:500],
                    "video_url": m["video_url"],
                }
                for m in uniq_ms[:3]
            ],
        })

    out_path = DATA_DIR / "harony_question_bank.json"
    out_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact browser QA bundle (smaller keys, up to 750 chars question text)
    compact = [
        {
            "id": b["id"],
            "bk": b["book"].replace("Harony ", "").replace(" Deep Final", ""),
            "u": b["unit"],
            "ch": b["chapter"],
            "pc": b["paper_codes"],
            "q": b["question_text"][:700],
            "yt": b["youtube_links"][:2],
            "ms": [
                {"id": m["question_id"], "mk": m["max_marks"], "a": m["answer"][:280]}
                for m in b["ms_answers"][:2]
            ],
        }
        for b in bank
    ]
    b_path = STATIC_DIR / "browser_qa.json"
    b_path.write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Built Question Bank: {len(bank)} unique question pages ({matched_ms_count} paired with official Mark Schemes)")
    print(f"  -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  -> {b_path} ({b_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
