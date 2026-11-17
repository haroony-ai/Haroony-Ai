#!/usr/bin/env python3
"""Train Haroony AI's complete local machine-learning & instruction-tuning suite
on Mr. Haroony's 2026/2027 books (all 5 PDFs + 243 OCR-enriched pages + 2,103 MS parts + 2,793 videos):

1. Physics-Only Domain Guardrail Classifier (`models/physics_guardrail.joblib`)
2. Multi-Class Physics Unit Classifier (`models/topic_classifier.joblib`) — prevents cross-topic drift!
3. Student Query Intent Classifier (`models/qa_intent_classifier.joblib`) — tailors answer format!
4. Dual Word+Char TF-IDF + 160-dim LSA Semantic Retriever (`models/harony_retriever.joblib`)
5. Complete Open-Source LLM Fine-Tuning / SFT Dataset (`data/training/harony_sft_dataset.jsonl`)
6. Browser-ready knowledge & model bundle (`static/browser_kb.json`) for 100% serverless GitHub Pages!
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import DATA_DIR, MODELS_DIR, STATIC_DIR, TRAINING_DIR

POSITIVE_PHYSICS_TEMPLATES = [
    "What is the difference between scalar and vector quantities?",
    "State Newton's second law of motion and give its formula.",
    "Calculate the kinetic energy of a 820 kg car moving at 60 m/s.",
    "How do you avoid parallax error when reading a measuring cylinder?",
    "Define moment of a force about a pivot and state the conditions for equilibrium.",
    "Explain why gas exerts pressure on the walls of a container using particle theory.",
    "Calculate the acceleration from a speed-time graph and find the distance travelled.",
    "Why does temperature remain constant during melting and boiling?",
    "Compare conduction, convection, and thermal radiation.",
    "Why are shiny white surfaces poor emitters and poor absorbers of infrared radiation?",
    "State Snell's law and explain total internal reflection in optical fibres.",
    "Draw a ray diagram for a converging lens forming a virtual magnified image.",
    "Difference between transverse waves and longitudinal waves with examples.",
    "How does ultrasound measure the depth of the sea using echo sounding?",
    "Calculate total resistance of two resistors in parallel and current from a 12V battery.",
    "How does a light-dependent resistor LDR or thermistor work in a potential divider?",
    "Explain why a fuse is connected to the live wire and how an earth wire protects users.",
    "Explain how a split-ring commutator keeps a D.C. motor rotating in one direction.",
    "Why does a transformer require alternating current a.c. and not direct current d.c.?",
    "Use Fleming's left-hand rule to find the direction of force on a wire in a magnetic field.",
    "Compare alpha, beta, and gamma radiation in terms of ionising effect and penetrating power.",
    "Write the beta minus decay nuclear equation of carbon-14 into nitrogen-14.",
    "A radioactive sample has a half-life of 5700 years. How long until 1/8 remains?",
    "Explain nuclear fusion in stars and how hydrogen nuclei combine to form helium.",
    "What is galactic redshift and how does Hubble's constant estimate the age of the universe?",
    "Calculate the orbital speed of a satellite orbiting at radius r with period T.",
    "Mark my answer using the Harony ClassWork 2027 mark scheme: ship is not solid steel and has lower average density.",
    "Why did I lose the A1 mark when I forgot the unit Joules?",
    "Find the specific heat capacity when 48720 J heats 0.2 kg of water by 58 degrees.",
    "What happens to wavelength, frequency, and speed when water waves enter shallower water?",
    "State Hooke's law and identify the limit of proportionality on a force-extension graph.",
    "How does a step-up transformer reduce energy loss in high-voltage transmission cables?",
    "Describe Rutherford's alpha particle scattering experiment with gold foil.",
    "Calculate density of an irregular stone using a displacement can and balance.",
    "Find gravitational potential energy gained when 2.5 kg rises by 4900 m.",
    "Explain motion graphs distance-time and speed-time.",
    "Explain heat lost and heat gained when rate of supplying heat is less than losing heat.",
    "Explain A.C. generator 1st half cycle and 2nd half cycle.",
]

NEGATIVE_NON_PHYSICS_EXAMPLES = [
    "Write a short poem about autumn leaves falling in a forest.",
    "Summarize the plot of Romeo and Juliet by William Shakespeare.",
    "Write a 500-word persuasive essay on why school uniforms should be optional.",
    "Give me a bedtime story about a dragon and a brave princess.",
    "Translate this paragraph into French and Spanish for my language homework.",
    "Correct the grammar and punctuation in my cover letter for a marketing job.",
    "Who wrote Pride and Prejudice and what are its main themes?",
    "Compose a birthday message for my best friend.",
    "What caused the fall of the Roman Empire in 476 AD?",
    "Who was the prime minister of Britain during World War II?",
    "What is the capital city of Australia and what is its population?",
    "List the countries in South America by land area.",
    "Explain the causes of the French Revolution in 1789.",
    "Who won the FIFA World Cup final and scored a hat-trick?",
    "Which actor won the Oscar for Best Actor last year?",
    "Tell me the rules of chess and the Sicilian Defense opening.",
    "Write a Python script to scrape product prices from Amazon using BeautifulSoup.",
    "How do I center a div in CSS using flexbox and grid?",
    "Build a REST API in Node.js and Express with JWT authentication.",
    "Fix this SQL query: SELECT * FROM orders JOIN customers ON id.",
    "Explain recursion and binary search trees in Java.",
    "How to install Arch Linux and configure Neovim plugins?",
    "Write a Discord bot in JavaScript that plays music.",
    "Explain the stages of mitosis: prophase, metaphase, anaphase, and telophase.",
    "How does photosynthesis convert carbon dioxide and water into glucose in chloroplasts?",
    "Describe the role of amylase, protease, and lipase in the human digestive system.",
    "Compare arteries, veins, and capillaries in the circulatory system.",
    "How do white blood cells produce antibodies against pathogens?",
    "Explain transcription and translation of mRNA at the ribosome.",
    "How does auxin control phototropism in plant shoots?",
    "Describe the structure of a nephron and ultrafiltration in the kidney.",
    "Explain fractional distillation of crude oil and cracking of long-chain alkanes.",
    "Describe how to perform an acid-base titration using a burette and methyl orange.",
    "What are the conditions used in the Haber process to manufacture ammonia?",
    "Balance this chemical equation: Fe2O3 + CO -> Fe + CO2.",
    "Explain ionic bonding in sodium chloride and giant covalent structure of diamond.",
    "How do you test for sulfate, halide, and carbonate ions in aqueous solution?",
    "Describe addition polymerisation of ethene to form poly(ethene).",
    "Give me an easy recipe for creamy garlic parmesan pasta.",
    "How long should I bake chocolate chip cookies at 180 degrees?",
    "What is the best workout routine to build bicep and chest muscles?",
    "Should I invest my savings in Bitcoin, Ethereum, or S&P 500 index funds?",
    "How do I treat a sore throat and seasonal allergies naturally?",
    "Best enchanting table setup and diamond mining level in Minecraft.",
    "Recommend 5 thriller movies on Netflix to watch tonight.",
    "Tell me a funny joke about cats and dogs.",
    "What is my horoscope for Scorpio today?",
    "Plan a 5-day tourist itinerary for Tokyo and Kyoto.",
    "Factorise the quadratic x^2 + 7x + 12 = 0 and find the roots.",
    "Calculate compound interest on $5000 at 6% APR over 10 years.",
    "How are you feeling today? Are you bored?",
    "What is your favorite color and favorite food?",
    "Write a resignation email to my manager.",
    "Give me 10 startup ideas for a mobile app.",
]

INTENT_TRAINING_SAMPLES = [
    ("Calculate the kinetic energy of a 820 kg car moving at 60 m/s", "calculation"),
    ("Find the specific heat capacity when 48720 J heats 0.2 kg by 58 C", "calculation"),
    ("A transformer has 360 turns primary 240 V and 40 turns secondary find Vs", "calculation"),
    ("Calculate resistance when voltage is 12 V and current is 3 A", "calculation"),
    ("How long until 1/8 of sample remains if half-life is 5700 years", "calculation"),
    ("Find orbital speed at radius 4.2e7 m and period 86400 s", "calculation"),
    ("Explain convection in fluids", "concept_explain"),
    ("What is Ohm's law?", "concept_explain"),
    ("Define moment of a force and state conditions for equilibrium", "concept_explain"),
    ("Why does temperature stay constant during melting and boiling?", "concept_explain"),
    ("Explain how a split-ring commutator works in a D.C. motor", "concept_explain"),
    ("What is galactic redshift and cosmic microwave background radiation?", "concept_explain"),
    ("Difference between transverse and longitudinal waves", "comparison"),
    ("Compare conduction, convection, and radiation", "comparison"),
    ("Compare nuclear fission and nuclear fusion", "comparison"),
    ("Difference between scalar and vector quantities with examples", "comparison"),
    ("Compare alpha, beta, and gamma penetration and ionising power", "comparison"),
    ("Explain distance-time and speed-time motion graphs", "graph_diagram"),
    ("Draw I-V graph for filament lamp and diode", "graph_diagram"),
    ("Explain A.C. generator 1st half cycle and 2nd half cycle coil positions", "graph_diagram"),
    ("What does gradient and area under a velocity-time graph represent?", "graph_diagram"),
    ("Describe how to measure density of an irregular stone using displacement can", "experiment_method"),
    ("How do you find centre of gravity of an irregular lamina using plumb line?", "experiment_method"),
    ("Describe ripple tank experiment for diffraction of water waves", "experiment_method"),
    ("How do you measure periodic time of a simple pendulum accurately?", "experiment_method"),
    ("Give me YouTube videos for waves and diffraction", "video_request"),
    ("YouTube video solutions for density", "video_request"),
    ("Send videos for transformers and electromagnetic induction", "video_request"),
    ("Quiz me on thermal physics", "practice_quiz"),
    ("Give me a practice question on moments", "practice_quiz"),
    ("2023 March P4 V2 Q02 mark scheme", "paper_lookup"),
    ("Show 2025 Nov P4 V1 Q02 bii", "paper_lookup"),
]


def load_concept_atlas() -> List[Dict[str, Any]]:
    atlas_path = DATA_DIR / "harony_concept_atlas.json"
    if atlas_path.exists():
        try:
            return json.loads(atlas_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def gather_training_pairs() -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    notes_path = DATA_DIR / "harony_notes_chunks.json"
    if notes_path.exists():
        chunks = json.loads(notes_path.read_text(encoding="utf-8"))
        for ch in chunks:
            topic = ch.get("topic", "")
            raw = ch.get("text", "")
            if topic:
                texts.append(f"Explain {topic} in IGCSE Physics.")
                labels.append(1)
            paras = [p.strip() for p in re.split(r"\n{1,2}", raw) if len(p.strip()) >= 30]
            for p in paras[:4]:
                texts.append(p)
                labels.append(1)

    for entry in load_concept_atlas():
        texts.append(f"Explain {entry['topic']}: {entry['summary']}")
        labels.append(1)
        for pt in entry.get("points", []):
            texts.append(pt)
            labels.append(1)

    ms_path = DATA_DIR / "mark_schemes.json"
    if ms_path.exists():
        ms_list = json.loads(ms_path.read_text(encoding="utf-8"))
        for ms in ms_list:
            if ms.get("id") == "harony-p2-cls-2027":
                continue
            for q in ms.get("questions", []):
                for pt in q.get("points", []):
                    ans = (pt.get("answer") or "").strip()
                    if len(ans) >= 12:
                        texts.append(ans)
                        labels.append(1)

    custom_path = DATA_DIR / "custom_materials.json"
    if custom_path.exists():
        custom_list = json.loads(custom_path.read_text(encoding="utf-8"))
        for ch in custom_list:
            txt = (ch.get("text") or "").strip()
            if len(txt) >= 25:
                texts.append(txt)
                labels.append(1)

    for pos in POSITIVE_PHYSICS_TEMPLATES * 3:
        texts.append(pos)
        labels.append(1)

    pos_count = sum(labels)
    neg_repeats = max(3, int(round(pos_count / max(len(NEGATIVE_NON_PHYSICS_EXAMPLES), 1))))
    for neg in NEGATIVE_NON_PHYSICS_EXAMPLES * neg_repeats:
        texts.append(neg)
        labels.append(0)

    return texts, labels


def train_unit_classifier(
    chunks: List[Dict[str, Any]],
    atlas: List[Dict[str, Any]],
    videos: List[Dict[str, Any]],
    ms_data: List[Dict[str, Any]],
) -> Tuple[Pipeline, float]:
    """Train a dedicated 7-Unit Physics Classifier to keep answers & videos locked on the right topic."""
    unit_texts: List[str] = []
    unit_labels: List[str] = []

    valid_units = {
        "Unit 1: General Physics",
        "Unit 2: Thermal Physics",
        "Unit 3: Electricity",
        "Unit 4: Electromagnetism",
        "Unit 5: Nuclear Physics",
        "Unit 6: Waves",
        "Unit 7: Astronomy",
    }

    for ch in chunks:
        u = ch.get("unit", "")
        if u in valid_units:
            txt = f"{ch.get('topic', '')} {ch.get('text', '')[:900]}".strip()
            if len(txt) >= 15:
                unit_texts.append(txt)
                unit_labels.append(u)

    for entry in atlas:
        u = entry.get("unit", "")
        if u in valid_units:
            kws = " ".join(entry.get("keywords", []))
            pts = " ".join(entry.get("points", []))
            for _ in range(4):
                unit_texts.append(f"{entry['topic']} {kws} {entry['summary']} {pts}")
                unit_labels.append(u)

    for v in videos:
        u = v.get("unit", "")
        if u in valid_units:
            lbl = f"{v.get('chapter', '')} {v.get('label', '')}".strip()
            if len(lbl) >= 6:
                unit_texts.append(lbl)
                unit_labels.append(u)

    for ms in ms_data:
        for q in ms.get("questions", []):
            u = q.get("unit", "")
            if u in valid_units:
                ans = f"{q.get('topic', '')} {q.get('summary_answer', '')}".strip()
                if len(ans) >= 15:
                    unit_texts.append(ans)
                    unit_labels.append(u)

    X_tr, X_te, y_tr, y_te = train_test_split(
        unit_texts, unit_labels, test_size=0.15, random_state=42, stratify=unit_labels
    )
    pipe = Pipeline([
        (
            "features",
            FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=18000)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=15000)),
            ]),
        ),
        ("clf", LogisticRegression(C=3.5, max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_tr, y_tr)
    u_acc = float(accuracy_score(y_te, pipe.predict(X_te)))

    pipe.fit(unit_texts, unit_labels)
    joblib.dump(pipe, MODELS_DIR / "topic_classifier.joblib")
    return pipe, u_acc


def train_intent_classifier() -> float:
    texts = [t for t, _ in INTENT_TRAINING_SAMPLES] * 4
    labels = [l for _, l in INTENT_TRAINING_SAMPLES] * 4
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=600)),
    ])
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODELS_DIR / "qa_intent_classifier.joblib")
    preds = pipe.predict([t for t, _ in INTENT_TRAINING_SAMPLES])
    return float(accuracy_score([l for _, l in INTENT_TRAINING_SAMPLES], preds))


def export_sft_dataset(
    chunks: List[Dict[str, Any]],
    atlas: List[Dict[str, Any]],
    ms_data: List[Dict[str, Any]],
) -> int:
    """Build complete instruction-tuning JSONL dataset across all 5 Harony books + Concept Atlas."""
    rows: List[str] = []
    sys_msg = (
        "You are Haroony AI, the official Cambridge IGCSE (0625) & A-Level (9702) Physics assistant "
        "grounded in Mr. Haroony's 2026/2027 books. Answer strictly on Physics using exact mark-scheme wording."
    )

    for entry in atlas:
        ans = (
            f"### {entry['topic']} ({entry['unit']})\n\n"
            f"{entry['summary']}\n\n"
            + "\n".join(f"- {p}" for p in entry.get("points", []))
        )
        if entry.get("formulas"):
            ans += "\n\n**Key Formulas:** " + ", ".join(f"`{f}`" for f in entry["formulas"])
        if entry.get("examiner_trap"):
            ans += f"\n\n> **Examiner Tip:** {entry['examiner_trap']}"
        rows.append(json.dumps({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Explain {entry['topic']} for Cambridge IGCSE Physics."},
                {"role": "assistant", "content": ans},
            ]
        }, ensure_ascii=False))

    for ch in chunks:
        txt = (ch.get("text") or "").strip()
        if len(txt) < 40:
            continue
        pg = f" (Harony Notes p.{ch['page']})" if ch.get("page") else ""
        rows.append(json.dumps({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"Summarize {ch.get('topic', 'this topic')} from {ch.get('source', 'Harony Notes')}{pg}."},
                {"role": "assistant", "content": txt[:1600]},
            ]
        }, ensure_ascii=False))

    for ms in ms_data:
        if ms.get("id") == "harony-p2-cls-2027":
            continue
        for q in ms.get("questions", []):
            ans = (q.get("summary_answer") or "").strip()
            if len(ans) < 15:
                continue
            rows.append(json.dumps({
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"What is the official Cambridge mark scheme answer for {ms.get('title')} {q.get('question_id')} ({q.get('topic', 'Physics')})?"},
                    {"role": "assistant", "content": ans[:1200]},
                ]
            }, ensure_ascii=False))

    out_path = TRAINING_DIR / "harony_sft_dataset.jsonl"
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def export_browser_bundle(
    chunks: List[Dict[str, Any]],
    ms_data: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    texts: List[str],
    labels: List[int],
    atlas: List[Dict[str, Any]],
    videos: List[Dict[str, Any]],
) -> None:
    """Export lightweight JSON bundle (`static/browser_kb.json`) so GitHub Pages runs 100% serverless."""
    js_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=3500, token_pattern=r"[a-z0-9]{2,}")
    X_js = js_vec.fit_transform(texts)
    js_clf = LogisticRegression(C=2.5, max_iter=800, class_weight="balanced")
    js_clf.fit(X_js, labels)

    vocab = js_vec.vocabulary_
    idf = js_vec.idf_
    coefs = js_clf.coef_[0]
    word_weights: Dict[str, float] = {}
    for tok, idx in vocab.items():
        w = float(coefs[idx] * idf[idx])
        if abs(w) >= 0.12:
            word_weights[tok] = round(w, 3)

    units_map: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        u = c.get("unit", "General")
        if u in ("Overview", "Exam Technique"):
            continue
        t = c.get("topic", "General")
        if u not in units_map:
            units_map[u] = {"unit": u, "topics": []}
        if not any(x["topic"] == t for x in units_map[u]["topics"]):
            units_map[u]["topics"].append({"topic": t, "start_page": c.get("page")})

    bundle = {
        "metrics": metrics,
        "curriculum": list(units_map.values()),
        "guardrail_model": {
            "intercept": round(float(js_clf.intercept_[0]), 4),
            "weights": word_weights,
        },
        "concept_atlas": atlas,
        "chunks": chunks,
        "mark_schemes": ms_data,
        "videos": videos,
    }
    (STATIC_DIR / "browser_kb.json").write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def train_all_models() -> Dict[str, Any]:
    atlas = load_concept_atlas()
    texts, labels = gather_training_pairs()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.18, random_state=42, stratify=labels
    )

    guardrail_pipe = Pipeline([
        (
            "features",
            FeatureUnion([
                ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=22000)),
                ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=22000)),
            ]),
        ),
        ("clf", LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced")),
    ])
    guardrail_pipe.fit(X_train, y_train)
    preds = guardrail_pipe.predict(X_val)

    acc = float(accuracy_score(y_val, preds))
    prec = float(precision_score(y_val, preds, zero_division=0))
    rec = float(recall_score(y_val, preds, zero_division=0))
    f1 = float(f1_score(y_val, preds, zero_division=0))

    guardrail_pipe.fit(texts, labels)
    joblib.dump(guardrail_pipe, MODELS_DIR / "physics_guardrail.joblib")

    chunks = []
    for p in (DATA_DIR / "harony_notes_chunks.json", DATA_DIR / "custom_materials.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks.extend([c for c in data if len((c.get("text") or "").strip()) >= 25])
            except Exception:
                pass

    corpus_docs = [
        f"{c.get('unit', '')} {c.get('topic', '')} {c.get('topic', '')} {' '.join(c.get('formulas', []))} {c.get('text', '')}"
        for c in chunks
    ]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=30000)
    tfidf_matrix = tfidf.fit_transform(corpus_docs)
    n_comp = min(160, max(8, tfidf_matrix.shape[0] - 2), tfidf_matrix.shape[1] - 2)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd_matrix = normalize(svd.fit_transform(tfidf_matrix))

    joblib.dump(
        {
            "chunk_count": len(chunks),
            "tfidf": tfidf,
            "svd": svd,
            "tfidf_matrix": tfidf_matrix,
            "svd_matrix": svd_matrix,
        },
        MODELS_DIR / "harony_retriever.joblib",
    )

    ms_data: List[Dict[str, Any]] = []
    if (DATA_DIR / "mark_schemes.json").exists():
        ms_data = json.loads((DATA_DIR / "mark_schemes.json").read_text(encoding="utf-8"))
    ms_count = len(ms_data)
    ms_questions = sum(m.get("question_count", 0) for m in ms_data)

    videos: List[Dict[str, Any]] = []
    if (DATA_DIR / "video_solutions.json").exists():
        videos = json.loads((DATA_DIR / "video_solutions.json").read_text(encoding="utf-8"))

    _, unit_acc = train_unit_classifier(chunks, atlas, videos, ms_data)
    intent_acc = train_intent_classifier()
    sft_rows = export_sft_dataset(chunks, atlas, ms_data)

    metrics = {
        "model_name": "Haroony-Physics-AI-v3 (Multi-Model Open-Source + Harony 2026/2027 Books)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "guardrail_samples": len(texts),
        "validation_accuracy": round(acc, 4),
        "validation_precision": round(prec, 4),
        "validation_recall": round(rec, 4),
        "validation_f1": round(f1, 4),
        "unit_classifier_accuracy": round(unit_acc, 4),
        "intent_classifier_accuracy": round(intent_acc, 4),
        "sft_instruction_pairs": sft_rows,
        "concept_atlas_topics": len(atlas),
        "indexed_notes_chunks": len(chunks),
        "semantic_dimensions": n_comp,
        "mark_schemes_loaded": ms_count,
        "mark_scheme_question_parts": ms_questions,
        "video_solutions_indexed": len(videos),
        "rate_limits": "None (Unlimited — 100% Free Self-Hosted & GitHub Pages)",
    }
    (MODELS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_browser_bundle(chunks, ms_data, metrics, texts, labels, atlas, videos)
    return metrics


if __name__ == "__main__":
    m = train_all_models()
    print(json.dumps(m, indent=2))
