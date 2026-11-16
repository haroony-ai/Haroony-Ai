"""Local CPU Vision OCR Engine for Haroony AI (100% Free, Zero API Keys, Zero Rate Limits).

Extracts text from:
1. Uploaded smartphone photos / screenshots (JPG, PNG, WebP, BMP) of handwritten or printed Physics work & Mark Schemes.
2. Digital PDFs (fast direct text layer via PyMuPDF) AND Scanned/Photographed PDFs (automatic fallback to RapidOCR ONNX).
Includes exam-paper margin/barcode noise filtering and contrast/resolution optimization for handwriting.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf
from PIL import Image, ImageOps

_OCR_INSTANCE = None

EXAM_NOISE_PATTERNS = [
    re.compile(r"DO\s*NOT\s*WRITE\s*IN\s*THIS\s*(?:MARGIN)?", re.I),
    re.compile(r"^\*?\d{10,}\*?$"),
    re.compile(r"^DFD$", re.I),
    re.compile(r"Cambridge\s*University\s*Press\s*&\s*Assessment", re.I),
    re.compile(r"\[?\s*Turn\s+over\s*\]?", re.I),
]


def get_ocr():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE


def is_exam_margin_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for pat in EXAM_NOISE_PATTERNS:
        if pat.search(s):
            return True
    return False


def preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    """Normalize orientation, convert to RGB, gently boost contrast, and cap dimensions for fast accurate OCR."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    max_side = max(w, h)
    # Upscale small snippets, cap huge 4000px camera photos around 1850px for speed + handwriting clarity
    if max_side < 900:
        scale = 900.0 / max(max_side, 1)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_side > 1900:
        scale = 1900.0 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    rgb = Image.merge("RGB", (gray, gray, gray))
    return np.array(rgb)


def ocr_pil_image(img: Image.Image) -> Dict[str, Any]:
    ocr = get_ocr()
    arr = preprocess_for_ocr(img)
    # Use relaxed box_thresh/unclip_ratio so light cursive handwriting words (like "speed") aren't clipped
    result, _ = ocr(arr, box_thresh=0.35, unclip_ratio=1.8)
    if not result:
        return {"text": "", "lines": [], "avg_confidence": 0.0}

    # Sort blocks top-to-bottom, left-to-right, merging boxes on the same horizontal line
    items: List[Tuple[float, float, float, str, float]] = []
    for box, txt, conf in result:
        if is_exam_margin_noise(txt):
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        y_mid = (min(ys) + max(ys)) / 2.0
        box_h = max(10.0, max(ys) - min(ys))
        x_left = min(xs)
        items.append((y_mid, x_left, box_h, txt.strip(), float(conf)))

    if not items:
        return {"text": "", "lines": [], "avg_confidence": 0.0}

    items.sort(key=lambda it: (it[0], it[1]))
    merged_rows: List[List[Tuple[float, float, float, str, float]]] = []
    for it in items:
        if not merged_rows:
            merged_rows.append([it])
            continue
        prev_row = merged_rows[-1]
        avg_y = sum(x[0] for x in prev_row) / len(prev_row)
        avg_h = sum(x[2] for x in prev_row) / len(prev_row)
        if abs(it[0] - avg_y) <= 0.55 * avg_h:
            prev_row.append(it)
        else:
            merged_rows.append([it])

    lines_out: List[str] = []
    confs: List[float] = []
    for row in merged_rows:
        row.sort(key=lambda x: x[1])
        line_str = " ".join(x[3] for x in row if x[3]).strip()
        if line_str and not is_exam_margin_noise(line_str):
            lines_out.append(line_str)
            confs.extend(x[4] for x in row)

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {
        "text": "\n".join(lines_out),
        "lines": lines_out,
        "avg_confidence": round(avg_conf, 3),
    }


def extract_from_file_bytes(file_bytes: bytes, filename: str, max_pages: int = 8) -> Dict[str, Any]:
    """Extract text from an uploaded image or PDF (digital or scanned)."""
    lower_name = (filename or "").lower()
    if lower_name.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_to_read = min(total_pages, max_pages)
        page_texts: List[str] = []
        used_ocr = False
        confs: List[float] = []

        for pno in range(pages_to_read):
            page = doc[pno]
            direct = page.get_text("text").strip()
            cleaned_lines = [ln.strip() for ln in direct.splitlines() if not is_exam_margin_noise(ln)]
            direct_clean = "\n".join(cleaned_lines).strip()
            if len(direct_clean) >= 35:
                page_texts.append(f"[Page {pno + 1}]\n{direct_clean}")
                confs.append(0.99)
            else:
                used_ocr = True
                pix = page.get_pixmap(dpi=160)
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_res = ocr_pil_image(pil_img)
                if ocr_res["text"]:
                    page_texts.append(f"[Page {pno + 1}]\n{ocr_res['text']}")
                    confs.append(ocr_res["avg_confidence"])

        return {
            "filename": filename,
            "kind": "pdf_ocr" if used_ocr else "pdf_text",
            "pages_processed": pages_to_read,
            "total_pages": total_pages,
            "text": "\n\n".join(page_texts).strip(),
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        }

    # Image file (JPG, PNG, WebP, etc.)
    pil_img = Image.open(io.BytesIO(file_bytes))
    ocr_res = ocr_pil_image(pil_img)
    return {
        "filename": filename,
        "kind": "image_ocr",
        "pages_processed": 1,
        "total_pages": 1,
        "text": ocr_res["text"],
        "avg_confidence": ocr_res["avg_confidence"],
    }
