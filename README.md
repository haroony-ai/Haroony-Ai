# Haroony AI — Harony Physics Clinic ⚡📘

**Haroony AI** is a **100% free, zero-rate-limit IGCSE & A-Level Physics AI Tutor and Cambridge Mark Scheme ("MS") Examiner** built exclusively on **Mr. Haroony's 2026 & 2027 Physics Books** for **Harony Physics Clinic** students.

---

## 📚 Built-In Harony Physics Clinic Corpus

1. **`Harony Notes 2026 Final.pdf` (243 Illustrated Pages)** — Full IGCSE Physics (0625) syllabus across Units 1–7, rendered as zoomable WebP slide cards and indexed in a hybrid TF-IDF + SVD semantic retriever.
2. **`Harony ClassWork 2027 Deep Final.pdf` (674 Pages)** — **520 Official Paper 4 & Paper 2 Mark Scheme questions** across Units 1–7 (extracted via coordinate-aligned ONNX OCR) paired with **772 Mr. Haroony YouTube video explanations**.
3. **`Harony P2 CLS 2027 Deep final New.pdf` (656 Pages)** — **1,026 Classified Paper 2 MCQ Answer Keys** (2016–Nov 2025) paired with **1,313 Mr. Haroony YouTube video explanations**.
4. **Teacher Studio Upload Support** — Upload additional books (including **`Harony P4 CLS 2027 Deep final New.pdf`**) or new Mark Schemes directly from the web UI and re-train in seconds.

---

## ✨ Key Features

- **💬 Strict Physics-Only AI Tutor (`100.0%` Validation F1 Guardrail)** — Answers Physics concepts, past-paper codes, and calculations step-by-step with visual slide citations and **🎥 Mr. Haroony YouTube solution links**, while strictly refusing any non-Physics prompt.
- **✅ Photo + Mark Scheme ("MS") Examiner** — Grade handwritten exam photos, PDFs, or typed answers against all **8 built-in Harony 2027 Mark Scheme books (1,546 questions)**, uploaded MS PDFs/screenshots, or pasted criteria using authentic Cambridge `B`, `M`, `C`, `A`, `CON`, and `TV` rules.
- **📘 Interactive 243-Page Harony Notes Explorer** — Browse and search all 7 Units of *Harony Notes 2026 Final*.
- **🎓 Teacher Training Studio (PIN `0625`)** — Upload PDFs/notes/Mark Schemes, re-train models, or connect local **Ollama** (`qwen2.5-vl:7b`).

---

## 🌐 Dual-Mode Free Deployment (GitHub Pages + Docker / Hugging Face Spaces)

### Option A: GitHub Pages (Instant $0 Static Hosting — Zero Server Needed!)
This repository includes a root `index.html`, `.nojekyll`, `static/browser_engine.js`, `static/browser_kb.json`, and `.github/workflows/deploy-pages.yml`:
1. In your GitHub repo (`toolautomation2-droid/Haroony-Ai`), go to **Settings → General** (make repository **Public** if on GitHub Free).
2. Go to **Settings → Pages** → under **Build and deployment**, select **GitHub Actions** (or **Deploy from a branch → `main` / `/ (root)`**).
3. Your live student app publishes at: **`https://toolautomation2-droid.github.io/Haroony-Ai/`**

### Option B: Full Python + Vision OCR Server (Local / Hugging Face Spaces Free Docker Tier)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train / refresh local ML models & browser bundle
python3 scripts/train_haroony_ai.py

# Start the website on http://0.0.0.0:8000
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
To host the full Python + `RapidOCR` server 24/7 for `$0/month`, connect this repository to a free **Hugging Face Docker Space** (uses the included `Dockerfile` on port `7860`).
