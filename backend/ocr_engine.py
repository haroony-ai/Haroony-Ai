"""Enhanced Local Vision & Document OCR Engine for Haroony AI.

Upgrades over basic OCR:
1. EXIF transpose (fixes sideways/upside-down smartphone photos automatically).
2. OpenCV CLAHE adaptive local contrast + unsharp-mask denoising + smart upscaling
   (dramatically improves pencil/blue-pen handwriting on lined exam paper).
3. Auto 90°/270° orientation check when initial pass has low confidence/short output.
4. Spatial top-to-bottom / left-to-right line grouping so formulas stay together.
5. PyMuPDF vector-text first for PDFs, falling back to high-DPI page OCR for scans.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps

_OCR_INSTANCE = None


def get_ocr():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE


def _clean_ocr_physics_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\u2009", " ").replace("×", "x").replace("−", "-")
    # Fix common physics OCR glitches
    s = re.sub(r"\b(\d+)\s*[oO]hms?\b", r"\1 ohms", s)
    s = re.sub(r"\bm\s*/\s*s\s*2\b", "m/s2", s)
    s = re.sub(r"\bkg\s*/\s*m\s*3\b", "kg/m3", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()


def _preprocess_pil_for_ocr(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return (color_upscaled_rgb, clahe_enhanced_rgb) numpy arrays."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 1400:
        scale = min(2.6, 1500.0 / max(1, max_dim))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_dim > 2600:
        scale = 2400.0 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE adaptive contrast helps faint pencil & uneven phone shadow lighting
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light unsharp mask for crisp edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
    return rgb, sharp_rgb


def _sort_and_merge_ocr_boxes(result: Any) -> Tuple[str, float]:
    if not result:
        return "", 0.0
    items: List[Tuple[float, float, str, float]] = []
    confs: List[float] = []
    for box, txt, conf in result:
        t = str(txt or "").strip()
        if not t:
            continue
        c = float(conf)
        confs.append(c)
        # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        cy = sum(ys) / 4.0
        lx = min(xs)
        items.append((cy, lx, t, c))

    if not items:
        return "", 0.0

    # Group boxes on roughly the same horizontal line (within ~18px Y tolerance)
    items.sort(key=lambda it: (it[0], it[1]))
    lines: List[List[Tuple[float, float, str, float]]] = []
    for it in items:
        if not lines or abs(it[0] - lines[-1][0][0]) > 20.0:
            lines.append([it])
        else:
            lines[-1].append(it)

    out_lines: List[str] = []
    for ln in lines:
        ln.sort(key=lambda it: it[1])
        out_lines.append(" ".join(it[2] for it in ln))

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return _clean_ocr_physics_text("\n".join(out_lines)), round(avg_conf, 3)


def extract_text_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Run enhanced multi-pass RapidOCR on an uploaded photo/screenshot."""
    img = Image.open(io.BytesIO(image_bytes))
    orig_rgb, enh_rgb = _preprocess_pil_for_ocr(img)

    ocr = get_ocr()
    res_enh, _ = ocr(enh_rgb, box_thresh=0.35, unclip_ratio=1.75)
    txt_enh, conf_enh = _sort_and_merge_ocr_boxes(res_enh)

    best_txt, best_conf = txt_enh, conf_enh

    # Compare against original upscaled RGB if enhanced pass looks short or low-confidence
    if len(best_txt) < 40 or best_conf < 0.72:
        res_orig, _ = ocr(orig_rgb, box_thresh=0.35, unclip_ratio=1.75)
        txt_orig, conf_orig = _sort_and_merge_ocr_boxes(res_orig)
        if (len(txt_orig) * max(0.3, conf_orig)) > (len(best_txt) * max(0.3, best_conf)) * 1.15:
            best_txt, best_conf = txt_orig, conf_orig

    # Sideways photo fallback (90° / 270°) when almost nothing legible was read
    if len(best_txt) < 25 or best_conf < 0.62:
        for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rot_arr = cv2.rotate(enh_rgb, rot_code)
            res_rot, _ = ocr(rot_arr, box_thresh=0.35, unclip_ratio=1.75)
            txt_rot, conf_rot = _sort_and_merge_ocr_boxes(res_rot)
            if len(txt_rot) > len(best_txt) * 1.35 and conf_rot >= best_conf - 0.05:
                best_txt, best_conf = txt_rot, conf_rot

    return {
        "text": best_txt,
        "lines": [ln for ln in best_txt.splitlines() if ln.strip()],
        "avg_confidence": best_conf,
    }


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 8) -> Dict[str, Any]:
    """Extract embedded text from PDF pages; OCR scanned image-only pages automatically."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    all_pages_text: List[str] = []
    ocr_pages = 0
    total_pages = min(len(doc), max_pages)

    for idx in range(total_pages):
        page = doc[idx]
        raw_text = _clean_ocr_physics_text(page.get_text("text") or "")
        if len(raw_text) >= 40:
            all_pages_text.append(f"[Page {idx + 1}]\n{raw_text}")
        else:
            pix = page.get_pixmap(dpi=210)
            ocr_res = extract_text_from_image_bytes(pix.tobytes("png"))
            ocr_pages += 1
            if ocr_res["text"]:
                all_pages_text.append(f"[Page {idx + 1} (OCR)]\n{ocr_res['text']}")

    return {
        "text": "\n\n".join(all_pages_text).strip(),
        "pages_processed": total_pages,
        "ocr_pages": ocr_pages,
        "total_pdf_pages": len(doc),
    }


def extract_from_file_bytes(file_bytes: bytes, filename: Optional[str] = None, max_pages: int = 8) -> Dict[str, Any]:
    """Unified extractor for PNG/JPG/WEBP photos or PDF documents."""
    fname = (filename or "").lower()
    if fname.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        res = extract_text_from_pdf_bytes(file_bytes, max_pages=max_pages)
        res["kind"] = "pdf"
        return res
    res = extract_text_from_image_bytes(file_bytes)
    res["kind"] = "image"
    res["pages_processed"] = 1
    return res
