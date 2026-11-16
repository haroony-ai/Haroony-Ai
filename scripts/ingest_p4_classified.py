#!/usr/bin/env python3
"""Ingest Harony P4 CLS 2027 Deep final New (1-400 + selected) PDFs into Harony AI:
- Extracts all 730+ YouTube video solution links (including studio.youtube.com -> youtu.be normalization & PDF link annotations)
- Parallel-OCRs all 153 Classified Mark Scheme table pages into structured Unit 1–7 P4 Mark Scheme books
- Merges cleanly with existing Harony Notes 2026 + ClassWork 2027 + P2 Classified 2027 datasets
- Rebuilds SFT JSONL & retrains Guardrail + Browser KB bundle!
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import DATA_DIR
from backend.ms_parser import compute_part_max_marks
from scripts.ingest_google_drive import (
    GDRIVE_DIR,
    UNIT_LABELS,
    _init_ocr_worker,
    _ocr_classwork_ms_page_worker,
    build_sft_dataset,
    infer_unit_from_header,
    normalize_paper_key,
    parse_ocr_rows_to_points,
)
from scripts.train_haroony_ai import train_all_models

P4_PART1 = GDRIVE_DIR / "Harony P4 CLS 2027 Deep final New-1-400.pdf"
P4_PART2 = GDRIVE_DIR / "Harony P4 CLS 2027 Deep final New-selected.pdf"

YT_ANY_RE = re.compile(
    r"https?://(?:(?:www\.)?youtu\.be/|studio\.youtube\.com/video/)([A-Za-z0-9_\-]{6,})",
    re.I,
)
Q_HDR_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")


def normalize_yt_url(raw_url: str) -> Optional[str]:
    if not raw_url:
        return None
    m = YT_ANY_RE.search(raw_url)
    if m:
        return f"https://youtu.be/{m.group(1)}"
    return None


def parse_p4_pdf(pdf_path: Path, part_label: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"Processing {pdf_path.name} ({part_label})...")
    t0 = time.time()
    doc = pymupdf.open(str(pdf_path))

    # Map (chapter_slug, q_num_int) AND norm_key -> video info
    video_by_chap_q: Dict[Tuple[str, int], Dict[str, Any]] = {}
    video_by_norm: Dict[str, Dict[str, Any]] = {}
    video_entries: List[Dict[str, Any]] = []
    seen_vid_urls = set()

    ocr_tasks: List[Tuple[int, List[Tuple[float, str]], List[Tuple[float, bytes]]]] = []
    current_unit = "Unit 1: Mechanics"
    current_chapter = "U1 Ch0: Measurements"
    last_qp_header: Optional[Tuple[int, str, str]] = None  # (q_num, label, chapter)

    for pno in range(len(doc)):
        pg = doc[pno]
        txt = pg.get_text("text")
        low = txt.lower()

        # Skip TOC/intro pages in Part 1
        if pdf_path == P4_PART1 and pno < 6:
            continue

        ch_m = re.search(r"(U[1-7]\s*(?:Ch|Topic)\s*\d+\s*:?\s*[^\n]+)", txt, re.I)
        if ch_m:
            current_chapter = re.sub(r"\s+", " ", ch_m.group(1)).strip()
            current_unit = infer_unit_from_header(current_chapter)

        is_ms_page = "classified mark scheme" in low or ("mark scheme" in low and "question paper" not in low)

        if not is_ms_page:
            # Collect typed question headers + y positions and all YouTube links (text + annotations)
            items_y: List[Tuple[float, str, str]] = []  # (y0, kind, value)
            for b in sorted(pg.get_text("blocks"), key=lambda x: x[1]):
                y0 = float(b[1])
                if y0 > 755:
                    continue  # skip footer
                for raw_ln in (b[4] or "").splitlines():
                    ln = re.sub(r"\s+", " ", raw_ln).strip()
                    if not ln or ln == current_chapter:
                        continue
                    yt = normalize_yt_url(ln)
                    if yt:
                        items_y.append((y0, "yt", yt))
                        continue
                    qm = Q_HDR_RE.match(ln)
                    if qm and len(qm.group(2).strip()) >= 5 and ("paper" in ln.lower() or "p4" in ln.lower() or "p2" in ln.lower() or "20" in ln):
                        items_y.append((y0, "hdr", ln))

            # Also include PDF link annotations with their y-coords
            for lk in pg.get_links():
                yt = normalize_yt_url(lk.get("uri") or "")
                if yt:
                    y0 = float(lk["from"].y0) if lk.get("from") else 500.0
                    if not any(k == "yt" and v == yt and abs(yy - y0) < 25 for yy, k, v in items_y):
                        items_y.append((y0, "yt", yt))

            items_y.sort(key=lambda x: x[0])
            for _, kind, val in items_y:
                if kind == "hdr":
                    qm = Q_HDR_RE.match(val)
                    if qm:
                        q_num = int(qm.group(1))
                        clean_lbl = re.sub(r"[.…]{2,}", " ", val).strip()
                        last_qp_header = (q_num, clean_lbl, current_chapter)
                elif kind == "yt":
                    if last_qp_header is not None:
                        q_num, clean_lbl, chap = last_qp_header
                    else:
                        q_num, clean_lbl, chap = 0, current_chapter, current_chapter
                    nkey = normalize_paper_key(clean_lbl)
                    entry = {
                        "book": "Harony P4 CLS 2027",
                        "unit": current_unit,
                        "chapter": chap,
                        "page": pno + 1,
                        "label": clean_lbl,
                        "norm_key": nkey,
                        "video_url": val,
                    }
                    if q_num > 0:
                        video_by_chap_q[(chap.lower(), q_num)] = entry
                    video_by_norm[nkey] = entry
                    if val not in seen_vid_urls:
                        seen_vid_urls.add(val)
                        video_entries.append(entry)
        else:
            imgs = pg.get_images(full=True)
            if not imgs:
                continue
            blocks = pg.get_text("blocks")
            headers_on_page: List[Tuple[float, str]] = []
            for b in sorted(blocks, key=lambda x: x[1]):
                y0 = float(b[1])
                if y0 > 755:
                    continue
                btxt = re.sub(r"\s+", " ", b[4] or "").strip()
                # Match both "1. ...... March 2017 Paper 4" and "12. 2023 March P4 V2 Q02 a -Density"
                if re.match(r"^\d+\.\s+(?:\.{2,}|…+|20\d{2})\b", btxt):
                    headers_on_page.append((y0, f"{current_chapter} | {btxt}"))

            strips: List[Tuple[float, bytes]] = []
            seen_xrefs = set()
            for info in imgs:
                xref = info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                rects = pg.get_image_rects(xref)
                y0 = float(rects[0].y0) if rects else 200.0
                base = doc.extract_image(xref)
                if base and base.get("image"):
                    strips.append((y0, base["image"]))
            strips.sort(key=lambda x: x[0])
            if strips:
                ocr_tasks.append((pno, headers_on_page, strips))

    print(f"   -> {len(video_entries)} YouTube links, {len(ocr_tasks)} MS image pages. Running 2-worker OCR...")
    max_workers = min(2, os.cpu_count() or 1)
    ocr_results_by_page: Dict[int, List[Dict[str, Any]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_ocr_worker) as pool:
        futures = [pool.submit(_ocr_classwork_ms_page_worker, task) for task in ocr_tasks]
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futures):
            pno, items = fut.result()
            ocr_results_by_page[pno] = items
            done_cnt += 1
            if done_cnt % 25 == 0 or done_cnt == len(ocr_tasks):
                print(f"      OCR progress ({part_label}): {done_cnt}/{len(ocr_tasks)} ({time.time() - t0:.1f}s)")

    p4_questions_by_unit: Dict[str, List[Dict[str, Any]]] = {u: [] for u in UNIT_LABELS.values()}
    kb_chunks_by_chapter: Dict[str, List[str]] = {}
    kb_meta_by_chapter: Dict[str, Dict[str, Any]] = {}

    for pno in sorted(ocr_results_by_page.keys()):
        for item in ocr_results_by_page[pno]:
            hdr = item["header"]
            if "|" in hdr:
                chap_part, q_part = [x.strip() for x in hdr.split("|", 1)]
            else:
                chap_part, q_part = "Harony P4 CLS 2027", hdr
            unit_name = infer_unit_from_header(chap_part)

            qm = Q_HDR_RE.match(q_part)
            q_num = int(qm.group(1)) if qm else 0
            clean_q = re.sub(r"[.…]{2,}", " ", q_part).strip()
            nkey = normalize_paper_key(clean_q)
            v_info = video_by_norm.get(nkey) or (video_by_chap_q.get((chap_part.lower(), q_num)) if q_num else None)
            vid_url = v_info["video_url"] if v_info else None

            pts = parse_ocr_rows_to_points(item["rows"])
            if not pts:
                continue
            qmax = compute_part_max_marks(pts)

            short_qid = re.sub(r"^\d+\.\s*", "", clean_q)
            short_qid = short_qid.split("-")[0].strip() if "-" in short_qid else short_qid[:36]
            chap_slug = re.sub(r"[^a-zA-Z0-9]+", "", chap_part)[:8]
            q_slug = re.sub(r"\s+", "-", short_qid.strip())
            short_qid_slug = f"{chap_slug}-{q_slug}"

            existing = next((q for q in p4_questions_by_unit[unit_name] if q["question_id"] == short_qid_slug and q["chapter"] == chap_part), None)
            if existing:
                existing["points"].extend(pts)
                existing["max_marks"] = compute_part_max_marks(existing["points"])
                existing["summary_answer"] = " ; ".join(
                    f"[{p['code']}] {p['answer']}" for p in existing["points"] if p["answer"]
                )
            else:
                p4_questions_by_unit[unit_name].append({
                    "question_id": short_qid_slug,
                    "title": clean_q,
                    "chapter": chap_part,
                    "page": item["page"],
                    "max_marks": qmax,
                    "video_url": vid_url,
                    "points": pts,
                    "summary_answer": " ; ".join(f"[{p['code']}] {p['answer']}" for p in pts if p["answer"]),
                })

            kb_chunks_by_chapter.setdefault(chap_part, [])
            kb_meta_by_chapter.setdefault(chap_part, {"unit": unit_name, "page": item["page"], "videos": []})
            if vid_url and vid_url not in kb_meta_by_chapter[chap_part]["videos"]:
                kb_meta_by_chapter[chap_part]["videos"].append(vid_url)
            kb_chunks_by_chapter[chap_part].append(
                f"{clean_q}: " + " | ".join(f"[{p['code']}] {p['answer']}" for p in pts[:4])
                + (f" (Video: {vid_url})" if vid_url else "")
            )

    p4_chunks: List[Dict[str, Any]] = []
    c_idx = 1
    for chap, lines in kb_chunks_by_chapter.items():
        meta = kb_meta_by_chapter[chap]
        for start in range(0, len(lines), 8):
            batch = lines[start : start + 8]
            p4_chunks.append({
                "id": f"harony-p4cls-{part_label}-{c_idx:03d}",
                "source": "Harony P4 CLS 2027 Deep Final",
                "page": meta["page"],
                "unit": meta["unit"],
                "topic": f"{chap} — P4 Classified Mark Schemes & Video Solutions",
                "text": "\n".join(batch),
                "formulas": [],
                "image_url": None,
                "video_links": meta["videos"][:5],
            })
            c_idx += 1

    print(f"   -> Finished {part_label} in {time.time() - t0:.1f}s: {sum(len(v) for v in p4_questions_by_unit.values())} MS questions, {len(p4_chunks)} chunks.")
    return p4_questions_by_unit, p4_chunks, video_entries


def main() -> None:
    q_by_unit_1, chunks_1, vids_1 = parse_p4_pdf(P4_PART1, "part1")
    q_by_unit_2, chunks_2, vids_2 = parse_p4_pdf(P4_PART2, "part2")

    # Merge unit questions from Part 1 + Part 2
    merged_by_unit: Dict[str, List[Dict[str, Any]]] = {u: [] for u in UNIT_LABELS.values()}
    for u in UNIT_LABELS.values():
        merged_by_unit[u].extend(q_by_unit_1.get(u, []))
        merged_by_unit[u].extend(q_by_unit_2.get(u, []))

    p4_ms_books: List[Dict[str, Any]] = []
    for u_code, u_name in UNIT_LABELS.items():
        qs = merged_by_unit.get(u_name, [])
        if not qs:
            continue
        tot_marks = sum(q["max_marks"] for q in qs)
        p4_ms_books.append({
            "id": f"harony-p4-cls-2027-{u_code.lower()}",
            "title": f"Harony P4 Classified 2027 — {u_name}",
            "paper_code": f"0625/P4 Classified ({u_code})",
            "session": "2016–2025 Paper 4",
            "source_file": "Harony P4 CLS 2027 Deep final New.pdf",
            "is_builtin": True,
            "total_marks": tot_marks,
            "computed_marks": tot_marks,
            "question_count": len(qs),
            "questions": qs,
        })

    # Merge idempotently with existing data/
    existing_chunks = json.loads((DATA_DIR / "harony_notes_chunks.json").read_text(encoding="utf-8"))
    existing_chunks = [c for c in existing_chunks if c.get("source") != "Harony P4 CLS 2027 Deep Final"]
    all_chunks = existing_chunks + chunks_1 + chunks_2
    (DATA_DIR / "harony_notes_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing_ms = json.loads((DATA_DIR / "mark_schemes.json").read_text(encoding="utf-8"))
    existing_ms = [m for m in existing_ms if not str(m.get("id", "")).startswith("harony-p4-cls-2027")]
    all_ms = existing_ms + p4_ms_books
    (DATA_DIR / "mark_schemes.json").write_text(
        json.dumps(all_ms, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing_vids = json.loads((DATA_DIR / "video_solutions.json").read_text(encoding="utf-8"))
    existing_vids = [v for v in existing_vids if v.get("book") != "Harony P4 CLS 2027"]
    all_vids = existing_vids + vids_1 + vids_2
    (DATA_DIR / "video_solutions.json").write_text(
        json.dumps(all_vids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    build_sft_dataset(all_chunks, all_ms)
    metrics = train_all_models()
    print("Retrained models & browser_kb.json!", metrics)
    print(f"Totals -> Chunks: {len(all_chunks)}, MS Books: {len(all_ms)}, MS Questions: {sum(m['question_count'] for m in all_ms)}, Videos: {len(all_vids)}")


if __name__ == "__main__":
    main()
