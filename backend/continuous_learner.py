"""Continuous Online Machine Learning Engine for Haroony AI.

Learns continuously and safely from real student Physics questions & feedback:
1. Anti-Poisoning Gate: Only valid Physics questions (already passed PhysicsGuardrail,
   >= 3 words, no injections) enter the learning pipeline.
2. Real-Time Incremental ML Classifier (`HashingVectorizer` + `SGDClassifier.partial_fit`):
   Warm-started on Harony 2026/2027 materials and updated incrementally on every new
   student question and positive feedback signal.
3. Verified Student Q&A Memory (TF-IDF + cosine similarity):
   Remembers effective explanations & phrasing patterns from past student interactions,
   automatically promoting 👍-rated answers and demoting 👎-rated ones.
4. Thread-safe persistence to `data/learning/` and `models/online_student_learner.joblib`.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import DATA_DIR, MODELS_DIR

LEARNING_DIR = DATA_DIR / "learning"
INTERACTIONS_PATH = LEARNING_DIR / "student_interactions.jsonl"
MEMORY_PATH = LEARNING_DIR / "learned_qa_memory.json"
ONLINE_MODEL_PATH = MODELS_DIR / "online_student_learner.joblib"

HARONY_UNITS = [
    "Unit 1: Mechanics",
    "Unit 2: Thermal Physics",
    "Unit 3: Electricity",
    "Unit 4: Electromagnetism",
    "Unit 5: Nuclear Physics",
    "Unit 6: Waves",
    "Unit 7: Astronomy",
]

_LOCK = threading.RLock()


class ContinuousPhysicsLearner:
    def __init__(self) -> None:
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.vectorizer = HashingVectorizer(
            ngram_range=(1, 2),
            n_features=2**15,
            alternate_sign=False,
            norm="l2",
        )
        self.clf: Optional[SGDClassifier] = None
        self.memory_entries: List[Dict[str, Any]] = []
        self._mem_tfidf: Optional[TfidfVectorizer] = None
        self._mem_matrix = None
        self.total_learned_questions = 0
        self.positive_feedback_count = 0
        self._pending_saves = 0

        self._load_or_init()

    def _load_or_init(self) -> None:
        with _LOCK:
            if ONLINE_MODEL_PATH.exists():
                try:
                    bundle = joblib.load(ONLINE_MODEL_PATH)
                    self.clf = bundle.get("clf")
                    self.total_learned_questions = int(bundle.get("total_learned_questions", 0))
                    self.positive_feedback_count = int(bundle.get("positive_feedback_count", 0))
                except Exception:
                    self.clf = None

            if self.clf is None:
                self.clf = SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=5,
                    random_state=42,
                )
                self._bootstrap_from_harony_corpus()

            if MEMORY_PATH.exists():
                try:
                    raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        self.memory_entries = raw[-800:]
                except Exception:
                    self.memory_entries = []
            self._rebuild_memory_index()

    def _bootstrap_from_harony_corpus(self) -> None:
        """Warm-start the online SGDClassifier on Harony 2026/2027 notes + SFT dataset + past logs."""
        texts: List[str] = []
        labels: List[str] = []

        chunks_path = DATA_DIR / "harony_notes_chunks.json"
        if chunks_path.exists():
            try:
                for ch in json.loads(chunks_path.read_text(encoding="utf-8")):
                    u = ch.get("unit", "")
                    t = f"{ch.get('topic', '')} {ch.get('text', '')}".strip()
                    if u in HARONY_UNITS and len(t) >= 20:
                        texts.append(t[:1200])
                        labels.append(u)
            except Exception:
                pass

        sft_path = DATA_DIR / "training" / "harony_sft_dataset.jsonl"
        if sft_path.exists():
            try:
                for ln in sft_path.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    row = json.loads(ln)
                    u = row.get("unit", "")
                    q = row.get("instruction", "")
                    if u in HARONY_UNITS and len(q) >= 8:
                        texts.append(q)
                        labels.append(u)
            except Exception:
                pass

        if INTERACTIONS_PATH.exists():
            try:
                for ln in INTERACTIONS_PATH.read_text(encoding="utf-8").splitlines()[-1000:]:
                    if not ln.strip():
                        continue
                    row = json.loads(ln)
                    u = row.get("unit", "")
                    q = row.get("query", "")
                    if u in HARONY_UNITS and len(q) >= 8 and row.get("feedback", 0) >= 0:
                        texts.append(q)
                        labels.append(u)
                        self.total_learned_questions += 1
            except Exception:
                pass

        if texts and self.clf is not None:
            X = self.vectorizer.transform(texts)
            self.clf.partial_fit(X, labels, classes=HARONY_UNITS)
            self._save_model()

    def _save_model(self) -> None:
        try:
            joblib.dump(
                {
                    "clf": self.clf,
                    "total_learned_questions": self.total_learned_questions,
                    "positive_feedback_count": self.positive_feedback_count,
                    "updated_at": int(time.time()),
                },
                ONLINE_MODEL_PATH,
                compress=3,
            )
        except Exception:
            pass

    def _save_memory(self) -> None:
        try:
            MEMORY_PATH.write_text(
                json.dumps(self.memory_entries[-800:], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _rebuild_memory_index(self) -> None:
        active = [m for m in self.memory_entries if m.get("score", 0) >= 0 and len(m.get("query", "")) >= 8]
        if len(active) < 1:
            self._mem_tfidf = None
            self._mem_matrix = None
            return
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            mat = vec.fit_transform([m["query"] for m in active])
            self._mem_tfidf = vec
            self._mem_matrix = mat
            self._active_mem = active
        except Exception:
            self._mem_tfidf = None
            self._mem_matrix = None

    def predict_unit_online(self, query: str) -> Optional[Dict[str, Any]]:
        """Predict curriculum unit + confidence from the online student-trained classifier."""
        if not query or len(query.strip()) < 4 or self.clf is None:
            return None
        try:
            with _LOCK:
                X = self.vectorizer.transform([query])
                probs = self.clf.predict_proba(X)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                return {"unit": HARONY_UNITS[best_idx], "confidence": round(conf, 3)}
        except Exception:
            return None

    def find_learned_guidance(self, query: str, min_sim: float = 0.48) -> Optional[Dict[str, Any]]:
        """Retrieve a past student-verified answer/insight when a new phrasing closely matches."""
        if not query or self._mem_tfidf is None or self._mem_matrix is None:
            return None
        try:
            with _LOCK:
                qv = self._mem_tfidf.transform([query])
                sims = cosine_similarity(qv, self._mem_matrix).ravel()
                if not len(sims):
                    return None
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim < min_sim:
                    return None
                entry = self._active_mem[best_idx]
                return {
                    "similarity": round(best_sim, 3),
                    "unit": entry.get("unit", ""),
                    "topic": entry.get("topic", ""),
                    "query": entry.get("query", ""),
                    "reply_excerpt": entry.get("reply_excerpt", ""),
                    "helpful_votes": int(entry.get("score", 0)),
                }
        except Exception:
            return None

    def record_interaction(
        self,
        query: str,
        unit: str,
        topic: str,
        reply: str,
        confidence: float = 0.7,
        neural_generated: bool = False,
    ) -> str:
        """Log a verified Physics question and incrementally train the online ML classifier."""
        clean_q = re.sub(r"\s+", " ", (query or "").strip())
        if len(clean_q.split()) < 3 or len(clean_q) > 800:
            return ""
        interaction_id = uuid.uuid4().hex[:12]
        valid_unit = unit if unit in HARONY_UNITS else None

        with _LOCK:
            self.total_learned_questions += 1

            # 1. Online incremental ML update (partial_fit) when unit confidence is reliable
            if valid_unit and confidence >= 0.45 and self.clf is not None:
                try:
                    X = self.vectorizer.transform([clean_q])
                    self.clf.partial_fit(X, [valid_unit], classes=HARONY_UNITS)
                except Exception:
                    pass

            # 2. Persist log row for ongoing audit & warm-restart
            row = {
                "id": interaction_id,
                "ts": int(time.time()),
                "query": clean_q,
                "unit": valid_unit or "",
                "topic": topic or "",
                "confidence": round(float(confidence), 3),
                "neural": bool(neural_generated),
                "feedback": 0,
            }
            try:
                with INTERACTIONS_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass

            # 3. Store compact Q&A memory entry (updated/promoted further when student clicks 👍)
            excerpt = re.sub(r"\s+", " ", (reply or "").strip())[:520]
            if excerpt:
                # Deduplicate nearly identical queries in memory
                existing = next((m for m in self.memory_entries if m.get("query", "").lower() == clean_q.lower()), None)
                if existing:
                    existing["id"] = interaction_id
                    existing["hits"] = int(existing.get("hits", 1)) + 1
                    if neural_generated or len(excerpt) > len(existing.get("reply_excerpt", "")):
                        existing["reply_excerpt"] = excerpt
                else:
                    self.memory_entries.append({
                        "id": interaction_id,
                        "query": clean_q,
                        "unit": valid_unit or "",
                        "topic": topic or "",
                        "reply_excerpt": excerpt,
                        "score": 1 if (neural_generated or confidence >= 0.72) else 0,
                        "hits": 1,
                        "ts": int(time.time()),
                    })
                self._rebuild_memory_index()

            self._pending_saves += 1
            if self._pending_saves >= 3:
                self._save_model()
                self._save_memory()
                self._pending_saves = 0

        return interaction_id

    def record_feedback(self, interaction_id: str, helpful: bool, query_fallback: str = "", unit_fallback: str = "") -> Dict[str, Any]:
        """Reinforce (👍) or demote (👎) an answer in the online ML learner & Q&A memory."""
        with _LOCK:
            matched_entry = next((m for m in self.memory_entries if m.get("id") == interaction_id), None)
            q_text = (matched_entry.get("query") if matched_entry else query_fallback).strip()
            u_text = (matched_entry.get("unit") if matched_entry else unit_fallback).strip()

            if matched_entry:
                matched_entry["score"] = int(matched_entry.get("score", 0)) + (2 if helpful else -3)

            if helpful:
                self.positive_feedback_count += 1
                # Extra supervised incremental reinforcement step on verified 👍 feedback
                if q_text and u_text in HARONY_UNITS and self.clf is not None:
                    try:
                        X = self.vectorizer.transform([q_text])
                        for _ in range(2):
                            self.clf.partial_fit(X, [u_text], classes=HARONY_UNITS)
                    except Exception:
                        pass

            self._rebuild_memory_index()
            self._save_model()
            self._save_memory()
            return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        with _LOCK:
            return {
                "online_ml_active": self.clf is not None,
                "total_learned_questions": self.total_learned_questions,
                "positive_feedback_count": self.positive_feedback_count,
                "memory_patterns_count": len([m for m in self.memory_entries if m.get("score", 0) >= 0]),
            }
