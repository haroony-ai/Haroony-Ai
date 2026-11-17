"""High-Accuracy Multi-Pass Vision & Document OCR Engine for Haroony AI.

Engineered for real student phone photos, dark-mode screenshots, tilted worksheets, and scanned PDFs:
1. EXIF orientation fix + smart LANCZOS upscaling (up to 2400px) with white padding border.
2. Automatic dark-mode screenshot detection & inversion (white-on-dark -> crisp black-on-white).
3. Illumination / shadow normalization (background division) + CLAHE local contrast + unsharp sharpening.
4. Automatic skew/tilt correction (-18 deg to +18 deg) so slanted phone photos stay on horizontal lines.
5. Multi-pass evaluation: Shadow-Normalized Sharp + Adaptive Threshold Binary + Upscaled RGB,
   plus 90/180/270-degree orientation fallback.
6. Box-height-adaptive spatial line merging + IGCSE Physics notation normalization.
7. PyMuPDF embedded text extraction first for PDFs, falling back to 260-DPI OCR on scanned pages.
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
    s = (
        text.replace("\u2009", " ")
        .replace("\u00a0", " ")
        .replace("×", "x")
        .replace("−", "-")
        .replace("–", "-")
        .replace("Ω", " ohms")
        .replace("µ", "u")
        .replace("μ", "u")
    )
    # Fix common physics OCR glitches
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*[oO]hms?\b", r"\1 ohms", s)
    s = re.sub(r"\bm\s*/\s*s\s*2\b", "m/s2", s)
    s = re.sub(r"\bm\s*s\s*-?\s*2\b", "m/s2", s)
    s = re.sub(r"\bkg\s*/\s*m\s*3\b", "kg/m3", s)
    s = re.sub(r"\bg\s*/\s*cm\s*3\b", "g/cm3", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*[°o0]\s*C\b", r"\1 °C", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*[xX*]\s*10\s*\^?\s*([+-]?\d+)", r"\1 x 10^\2", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _normalize_dark_mode(rgb: np.ndarray) -> np.ndarray:
    """Invert dark-mode screenshots (light text on dark background) into black-on-white."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(np.median(gray)) < 115.0:
        return 255 - rgb
    return rgb


def _deskew_rgb(rgb: np.ndarray) -> np.ndarray:
    """Correct small camera tilt (-18..+18 deg) using foreground text orientation."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Find dark text pixels against lighter background
    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Connect horizontal characters into lines so minAreaRect tracks text-line angle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) < 200:
        return rgb
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = float(rect[-1])
    if angle < -45.0:
        angle = 90.0 + angle
    elif angle > 45.0:
        angle = angle - 90.0
    if abs(angle) < 1.2 or abs(angle) > 18.0:
        return rgb
    h, w = rgb.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        rgb,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _preprocess_pil_for_ocr(img: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (upscaled_rgb, shadow_norm_sharp_rgb, adaptive_binary_rgb)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 1600:
        scale = min(3.0, 1700.0 / max(1, max_dim))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif max_dim > 2800:
        scale = 2600.0 / max_dim
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    rgb = _normalize_dark_mode(np.array(img))
    rgb = _deskew_rgb(rgb)

    # Add small white margin so edge text is never clipped by detector boxes
    rgb = cv2.copyMakeBorder(rgb, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Illumination / shadow normalization (divides out uneven lighting & warm paper tint)
    bg_k = max(31, (min(gray.shape[:2]) // 18) | 1)
    bg = cv2.GaussianBlur(gray, (bg_k, bg_k), 0)
    shadow_norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)

    # CLAHE + mild unsharp sharpening
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(shadow_norm)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.1)
    sharp = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)

    # Adaptive binary pass for faint pencil or heavy shadow creases
    block_sz = max(21, (min(gray.shape[:2]) // 32) | 1)
    denoised = cv2.fastNlMeansDenoising(shadow_norm, None, h=8, templateWindowSize=7, searchWindowSize=15)
    bin_gray = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_sz, 11
    )
    bin_rgb = cv2.cvtColor(bin_gray, cv2.COLOR_GRAY2RGB)

    return rgb, sharp_rgb, bin_rgb


def _sort_and_merge_ocr_boxes(result: Any) -> Tuple[str, float]:
    if not result:
        return "", 0.0
    items: List[Tuple[float, float, float, str, float]] = []
    confs: List[float] = []
    heights: List[float] = []
    for box, txt, conf in result:
        t = str(txt or "").strip()
        if not t:
            continue
        c = float(conf)
        confs.append(c)
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        cy = sum(ys) / 4.0
        lx = min(xs)
        bh = max(8.0, max(ys) - min(ys))
        heights.append(bh)
        items.append((cy, lx, bh, t, c))

    if not items:
        return "", 0.0

    median_h = float(np.median(heights)) if heights else 24.0
    y_tol = max(14.0, min(42.0, median_h * 0.6))

    items.sort(key=lambda it: (it[0], it[1]))
    lines: List[List[Tuple[float, float, float, str, float]]] = []
    for it in items:
        if not lines:
            lines.append([it])
            continue
        avg_line_y = sum(x[0] for x in lines[-1]) / len(lines[-1])
        if abs(it[0] - avg_line_y) > y_tol:
            lines.append([it])
        else:
            lines[-1].append(it)

    out_lines: List[str] = []
    for ln in lines:
        ln.sort(key=lambda it: it[1])
        out_lines.append(" ".join(it[3] for it in ln))

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return _clean_ocr_physics_text("\n".join(out_lines)), round(avg_conf, 3)


def _score_ocr_candidate(txt: str, conf: float) -> float:
    alnum = sum(ch.isalnum() for ch in (txt or ""))
    return alnum * (0.45 + max(0.0, conf))


def extract_text_from_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Run enhanced multi-pass RapidOCR on an uploaded photo/screenshot."""
    img = Image.open(io.BytesIO(image_bytes))
    orig_rgb, sharp_rgb, bin_rgb = _preprocess_pil_for_ocr(img)

    ocr = get_ocr()
    res_sharp, _ = ocr(sharp_rgb, box_thresh=0.30, unclip_ratio=1.85)
    best_txt, best_conf = _sort_and_merge_ocr_boxes(res_sharp)
    best_score = _score_ocr_candidate(best_txt, best_conf)

    # Try original upscaled RGB & adaptive-binary passes if needed
    if len(best_txt) < 65 or best_conf < 0.78:
        for cand_arr in (orig_rgb, bin_rgb):
            res_c, _ = ocr(cand_arr, box_thresh=0.30, unclip_ratio=1.85)
            txt_c, conf_c = _sort_and_merge_ocr_boxes(res_c)
            sc_c = _score_ocr_candidate(txt_c, conf_c)
            if sc_c > best_score * 1.08:
                best_txt, best_conf, best_score = txt_c, conf_c, sc_c

    # Full orientation fallback (90°, 180°, 270°) when photo was rotated/upside-down
    if len(best_txt) < 28 or best_conf < 0.65:
        for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            rot_arr = cv2.rotate(sharp_rgb, rot_code)
            res_rot, _ = ocr(rot_arr, box_thresh=0.30, unclip_ratio=1.85)
            txt_rot, conf_rot = _sort_and_merge_ocr_boxes(res_rot)
            sc_rot = _score_ocr_candidate(txt_rot, conf_rot)
            if sc_rot > best_score * 1.25 and conf_rot >= best_conf - 0.05:
                best_txt, best_conf, best_score = txt_rot, conf_rot, sc_rot

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
            pix = page.get_pixmap(dpi=260)
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
