#!/usr/bin/env python3
"""Train Haroony AI's local machine-learning models on Mr. Haroony's 2026/2027 books:
1. Physics-Only Domain Guardrail Classifier (`models/physics_guardrail.joblib`)
2. Harony Physics Clinic Semantic Retriever (`models/harony_retriever.joblib`)
3. Evaluation metrics (`models/model_metrics.json`)
4. Browser-ready knowledge & model bundle (`static/browser_kb.json`) for 100% serverless GitHub Pages hosting!
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

from backend.config import DATA_DIR, MODELS_DIR, STATIC_DIR

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
]

NEGATIVE_NON_PHYSICS_EXAMPLES = [
    # Creative writing, literature, humanities
    "Write a short poem about autumn leaves falling in a forest.",
    "Summarize the plot of Romeo and Juliet by William Shakespeare.",
    "Write a 500-word persuasive essay on why school uniforms should be optional.",
    "Give me a bedtime story about a dragon and a brave princess.",
    "Translate this paragraph into French and Spanish for my language homework.",
    "Correct the grammar and punctuation in my cover letter for a marketing job.",
    "Who wrote Pride and Prejudice and what are its main themes?",
    "Compose a birthday message for my best friend.",
    # History, geography, politics, general trivia
    "What caused the fall of the Roman Empire in 476 AD?",
    "Who was the prime minister of Britain during World War II?",
    "What is the capital city of Australia and what is its population?",
    "List the countries in South America by land area.",
    "Explain the causes of the French Revolution in 1789.",
    "Who won the FIFA World Cup final and scored a hat-trick?",
    "Which actor won the Oscar for Best Actor last year?",
    "Tell me the rules of chess and the Sicilian Defense opening.",
    # Programming, IT, web dev
    "Write a Python script to scrape product prices from Amazon using BeautifulSoup.",
    "How do I center a div in CSS using flexbox and grid?",
    "Build a REST API in Node.js and Express with JWT authentication.",
    "Fix this SQL query: SELECT * FROM orders JOIN customers ON id.",
    "Explain recursion and binary search trees in Java.",
    "How to install Arch Linux and configure Neovim plugins?",
    "Write a Discord bot in JavaScript that plays music.",
    # Pure Biology (non-physics)
    "Explain the stages of mitosis: prophase, metaphase, anaphase, and telophase.",
    "How does photosynthesis convert carbon dioxide and water into glucose in chloroplasts?",
    "Describe the role of amylase, protease, and lipase in the human digestive system.",
    "Compare arteries, veins, and capillaries in the circulatory system.",
    "How do white blood cells produce antibodies against pathogens?",
    "Explain transcription and translation of mRNA at the ribosome.",
    "How does auxin control phototropism in plant shoots?",
    "Describe the structure of a nephron and ultrafiltration in the kidney.",
    # Pure Chemistry (non-physics)
    "Explain fractional distillation of crude oil and cracking of long-chain alkanes.",
    "Describe how to perform an acid-base titration using a burette and methyl orange.",
    "What are the conditions used in the Haber process to manufacture ammonia?",
    "Balance this chemical equation: Fe2O3 + CO -> Fe + CO2.",
    "Explain ionic bonding in sodium chloride and giant covalent structure of diamond.",
    "How do you test for sulfate, halide, and carbonate ions in aqueous solution?",
    "Describe addition polymerisation of ethene to form poly(ethene).",
    # Cooking, lifestyle, finance, health, games
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
    # Pure abstract non-physics math / business / casual chat
    "Factorise the quadratic x^2 + 7x + 12 = 0 and find the roots.",
    "Calculate compound interest on $5000 at 6% APR over 10 years.",
    "How are you feeling today? Are you bored?",
    "What is your favorite color and favorite food?",
    "Write a resignation email to my manager.",
    "Give me 10 startup ideas for a mobile app.",
]


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


def export_browser_bundle(
    chunks: List[Dict[str, Any]],
    ms_data: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    texts: List[str],
    labels: List[int],
) -> None:
    """Export lightweight JSON bundle (`static/browser_kb.json`) so GitHub Pages runs 100% serverless."""
    # Train compact word-unigram LogisticRegression weights for fast client-side JS guardrail scoring
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

    # Curriculum overview
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

    videos = []
    vid_path = DATA_DIR / "video_solutions.json"
    if vid_path.exists():
        try:
            videos = json.loads(vid_path.read_text(encoding="utf-8"))
        except Exception:
            videos = []

    bundle = {
        "metrics": metrics,
        "curriculum": list(units_map.values()),
        "guardrail_model": {
            "intercept": round(float(js_clf.intercept_[0]), 4),
            "weights": word_weights,
        },
        "chunks": chunks,
        "mark_schemes": ms_data,
        "videos": videos[:800],
    }
    (STATIC_DIR / "browser_kb.json").write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def train_all_models() -> Dict[str, Any]:
    texts, labels = gather_training_pairs()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.18, random_state=42, stratify=labels
    )

    guardrail_pipe = Pipeline([
        (
            "features",
            FeatureUnion([
                ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)),
                ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=20000)),
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
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=25000)
    tfidf_matrix = tfidf.fit_transform(corpus_docs)
    n_comp = min(96, max(8, tfidf_matrix.shape[0] - 2), tfidf_matrix.shape[1] - 2)
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

    vid_count = 0
    if (DATA_DIR / "video_solutions.json").exists():
        vid_count = len(json.loads((DATA_DIR / "video_solutions.json").read_text(encoding="utf-8")))

    metrics = {
        "model_name": "Haroony-Physics-AI-v2 (Harony 2026/2027 Books)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "guardrail_samples": len(texts),
        "validation_accuracy": round(acc, 4),
        "validation_precision": round(prec, 4),
        "validation_recall": round(rec, 4),
        "validation_f1": round(f1, 4),
        "indexed_notes_chunks": len(chunks),
        "semantic_dimensions": n_comp,
        "mark_schemes_loaded": ms_count,
        "mark_scheme_question_parts": ms_questions,
        "video_solutions_indexed": vid_count,
        "rate_limits": "None (Unlimited — 100% Free Self-Hosted & GitHub Pages)",
    }
    (MODELS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    export_browser_bundle(chunks, ms_data, metrics, texts, labels)
    return metrics


if __name__ == "__main__":
    m = train_all_models()
    print(json.dumps(m, indent=2))
