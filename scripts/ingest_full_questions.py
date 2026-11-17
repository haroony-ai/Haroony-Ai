#!/usr/bin/env python3
"""Parallel OCR Extractor for Full Physics Questions across Harony 2027 Books.

Extracts the actual exam question text, units, chapters, paper codes, and YouTube links
from the non-MS question pages of:
  - Harony ClassWork 2027 Deep Final.pdf
  - Harony P4 CLS 2027 Deep final New-1-400.pdf
  - Harony P4 CLS 2027 Deep final New-selected.pdf
  - Harony P2 CLS 2027 Deep final New.pdf

Writes incrementally to data/extracted_questions.jsonl so every batch persists safely.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import pymupdf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import DATA_DIR
from scripts.ingest_google_drive import GDRIVE_DIR, infer_unit_from_header

OUT_PATH = DATA_DIR / "extracted_questions.jsonl"

BOOK_PATHS = {
    "cw": ("Harony ClassWork 2027 Deep Final", GDRIVE_DIR / "Harony ClassWork 2027 Deep Final.pdf"),
    "p4_1": ("Harony P4 CLS 2027 Deep Final (Part 1)", GDRIVE_DIR / "Harony P4 CLS 2027 Deep final New-1-400.pdf"),
    "p4_2": ("Harony P4 CLS 2027 Deep Final (Part 2)", GDRIVE_DIR / "Harony P4 CLS 2027 Deep final New-selected.pdf"),
    "p2": ("Harony P2 CLS 2027 Deep Final", GDRIVE_DIR / "Harony P2 CLS 2027 Deep final New.pdf"),
}

YT_RE = re.compile(r"https?://(?:(?:www\.)?youtu\.be/|studio\.youtube\.com/video/)([A-Za-z0-9_\-]{6,})", re.I)

_WORKER_OCR = None


def _init_worker() -> None:
    global _WORKER_OCR
    from rapidocr_onnxruntime import RapidOCR
    _WORKER_OCR = RapidOCR()


def _ocr_page_worker(item: Tuple[int, bytes, str, str, str, List[str]]) -> Dict[str, Any]:
    pno, png_bytes, book_name, unit, chapter, yt_links = item
    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    res, _ = _WORKER_OCR(arr)
    lines: List[str] = []
    if res:
        for _, txt, conf in res:
            t = str(txt or "").strip()
            if t and float(conf) >= 0.45:
                lines.append(t)
    raw_text = "\n".join(lines)
    raw_text = (
        raw_text.replace("×", "x")
        .replace("−", "-")
        .replace("\u2009", " ")
        .strip()
    )
    return {
        "book": book_name,
        "page": pno + 1,
        "unit": unit,
        "chapter": chapter,
        "youtube_links": yt_links,
        "text": raw_text,
    }


def load_done_keys() -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if OUT_PATH.exists():
        for ln in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
                done.add((row["book"], int(row["page"])))
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, choices=list(BOOK_PATHS.keys()))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99999)
    ap.add_argument("--dpi", type=int, default=88)
    args = ap.parse_args()

    book_name, pdf_path = BOOK_PATHS[args.book]
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    done = load_done_keys()
    doc = pymupdf.open(str(pdf_path))
    end_p = min(len(doc), args.end)

    tasks: List[Tuple[int, bytes, str, str, str, List[str]]] = []
    in_ms = False
    cur_unit = "Unit 1: Mechanics"
    cur_chap = "U1 Ch0: Measurements"

    for pno in range(len(doc)):
        pg = doc[pno]
        txt = pg.get_text("text")
        low = txt.lower()

        ch_m = re.search(r"(U[1-7]\s*(?:Ch|Topic)\s*\d+\s*:?\s*[^\n]+)", txt, re.I)
        if ch_m:
            cur_chap = re.sub(r"\s+", " ", ch_m.group(1)).strip()
            cur_unit = infer_unit_from_header(cur_chap)

        if args.book == "cw":
            if "mark scheme" in low:
                in_ms = True
            elif "youtu.be" in low or "page 1 of" in low or ("unit " in low and "chapter " in low and len(txt) < 350):
                in_ms = False
            is_ms = in_ms
        else:
            is_ms = "classified mark scheme" in low or ("mark scheme" in low and "question paper" not in low)

        if pno < args.start or pno >= end_p:
            continue
        if is_ms:
            continue
        if args.book in ("cw", "p4_1", "p2") and pno < 6:
            continue
        if (book_name, pno + 1) in done:
            continue

        # Must contain images (embedded question scans) or question text
        if len(pg.get_images()) == 0 and len(txt) < 120:
            continue

        yt_links = []
        for m in YT_RE.findall(txt):
            u = f"https://youtu.be/{m}"
            if u not in yt_links:
                yt_links.append(u)
        for lk in pg.get_links():
            uri = lk.get("uri") or ""
            ym = YT_RE.search(uri)
            if ym:
                u = f"https://youtu.be/{ym.group(1)}"
                if u not in yt_links:
                    yt_links.append(u)

        png_bytes = pg.get_pixmap(dpi=args.dpi).tobytes("png")
        tasks.append((pno, png_bytes, book_name, cur_unit, cur_chap, yt_links))

    print(f"[{args.book}] Queued {len(tasks)} question pages (start={args.start}, end={end_p}, dpi={args.dpi})...")
    if not tasks:
        return

    t0 = time.time()
    written = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, initializer=_init_worker) as ex:
        with OUT_PATH.open("a", encoding="utf-8") as fh:
            for row in ex.map(_ocr_page_worker, tasks, chunksize=4):
                if len(row.get("text", "")) >= 30:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                if written % 25 == 0 and written > 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  [{args.book}] {written}/{len(tasks)} pages OCR'd ({el:.1f}s elapsed)")
    print(f"[{args.book}] Done! Wrote {written} question pages in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
