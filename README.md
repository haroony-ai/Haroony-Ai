# Haroony AI — Harony Physics Clinic (Cambridge IGCSE 0625)

**Haroony AI** is a clean, unified single-chat **Cambridge IGCSE Physics (0625 — Core & Extended)** AI Tutor and Mark Scheme Grader built on Mr. Haroony's 2026 & 2027 Physics books, classified papers, mark schemes, and YouTube video solutions.

---

## Repository Structure (Organized for Google AI Studio & Self-Hosting)

```text
Haroony-Ai/
├── index.html                  # Single-screen Haroony AI student chat UI (root for GitHub Pages)
├── favicon.ico / .nojekyll     # Site icon & static hosting marker
├── backend/                    # FastAPI server, 0625 Physics solver/reasoner, guardrail, OCR & MS grader
├── static/                     # Frontend CSS/JS, Harony slide pages (243 WebP), browser KB & QA bundles
├── data/
│   ├── harony_concept_atlas.json   # 35 structured Cambridge 0625 Concept Atlas modules
│   ├── harony_notes_chunks.json    # 445 indexed Harony 2026/2027 Notes chunks + page citations
│   ├── harony_question_bank.json   # 1,731 indexed Question Bank pages
│   ├── mark_schemes.json           # 15 Mark Scheme books (2,103 question parts)
│   ├── video_solutions.json        # 2,793 indexed YouTube video solution links
│   └── training/
│       ├── gemini_sft_dataset.jsonl    # 2,915 formatted Q&A instruction pairs for Google AI Studio / Gemini
│       ├── harony_sft_dataset.jsonl    # Full Harony SFT corpus
│       ├── haroony_sft_dataset.jsonl   # Student-tutor conversational pairs
│       └── openai_sft_dataset.jsonl    # Chat-format SFT export
├── models/                     # Pretrained local ML models (<100 MB each)
│   ├── neural_subtopic_mlp.joblib      # 2-layer Neural Network (35-topic classifier)
│   ├── physics_guardrail.joblib        # Strict 0625 Physics-only guardrail (14,603 samples, 1.0 F1)
│   ├── topic_classifier.joblib         # Cambridge Unit classifier (98.31% acc)
│   ├── question_bank_retriever.joblib  # Past-paper retriever
│   ├── qa_intent_classifier.joblib     # Intent router
│   └── model_metrics.json              # Verified training metrics
├── scripts/                    # Dataset builders, OCR ingestion, local & cloud fine-tuning scripts
├── training/                   # Colab fine-tuning notebook & Modelfile
└── tests/                      # 17 end-to-end integration test suites (`test_haroony_ai.py`)
```

---

## Importing & Training in Google AI Studio

1. **Import Repository / Files:** Connect this repository (or upload `data/training/gemini_sft_dataset.jsonl` plus the JSON files in `data/`) into Google AI Studio.
2. **Fine-Tune / Ground on 0625 Materials:** Use `data/training/gemini_sft_dataset.jsonl` (or run `scripts/tune_gemini.py`) and attach your Harony 2026/2027 PDFs so the model grounds every answer in Harony's Cambridge 0625 notes, mark schemes, and video solutions.
3. **Keep Secrets Server-Side:** Store API keys in `.env` (see `.env.example`) or hosting environment variables — never commit keys or expose third-party provider names in student-facing HTML/JS.

---

## Running Locally or Via Docker

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Retrain local models & rebuild browser KB
python3 scripts/train_haroony_ai.py

# Start server
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Run integration tests
python3 tests/test_haroony_ai.py
```

Or double-click `START_HAROONY_AI.bat` (Windows) / `START_HAROONY_AI.command` (macOS).
