"""Harony Physics Clinic Knowledge Base & Hybrid Semantic Retriever.

Indexes all 243 pages of Mr. Haroony's 2026 IGCSE Physics Notes,
custom teacher-uploaded notes, and Cambridge Mark Schemes using a trained
hybrid TF-IDF + Latent Semantic Analysis (SVD) dense vector space + topic/keyword ranker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import DATA_DIR, MODELS_DIR

RETRIEVER_MODEL_PATH = MODELS_DIR / "harony_retriever.joblib"
QB_RETRIEVER_MODEL_PATH = MODELS_DIR / "question_bank_retriever.joblib"
TOPIC_MODEL_PATH = MODELS_DIR / "topic_classifier.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "qa_intent_classifier.joblib"
CONCEPT_ATLAS_PATH = DATA_DIR / "harony_concept_atlas.json"
NOTES_CHUNKS_PATH = DATA_DIR / "harony_notes_chunks.json"
CUSTOM_MATERIALS_PATH = DATA_DIR / "custom_materials.json"
MARK_SCHEMES_PATH = DATA_DIR / "mark_schemes.json"
VIDEO_SOLUTIONS_PATH = DATA_DIR / "video_solutions.json"
QUESTION_BANK_PATH = DATA_DIR / "harony_question_bank.json"

QUERY_SYNONYM_EXPANSIONS = {
    "commutator": "split ring commutator d.c. motor current reverses half turn",
    "generator": "a.c. generator slip rings electromagnetic induction alternating current",
    "motor": "electric d.c. motor fleming left hand rule coil magnetic field",
    "transformer": "step-up step-down primary secondary coil vp vs np ns soft iron core",
    "ohm": "ohms law resistance voltage current potential difference v = ir electric circuits",
    "newton": "newtons first second third law inertia resultant force acceleration f = ma",
    "convection": "convection heat transmission thermal physics density warm fluid rises cool sinks",
    "conduction": "conduction heat transmission thermal physics lattice vibration free electrons",
    "brownian": "brownian motion kinetic particle theory smoke cells air molecules random",
    "diffraction": "diffraction water waves gap wavelength ripple tank spread",
    "ldr": "light dependent resistor brightness resistance decreases circuit components",
    "thermistor": "temperature increases resistance decreases circuit components",
    "meniscus": "measuring cylinder volume parallax bottom of meniscus",
    "parallax": "eye perpendicular vertically above ruler scale",
    "pendulum": "periodic time 20 oscillations stopwatch simple pendulum",
    "hooke": "spring extension force load limit of proportionality f = kx",
    "moment": "force perpendicular distance from pivot clockwise anticlockwise equilibrium",
    "boyle": "pressure volume gas p1v1 = p2v2 constant temperature",
    "snell": "refraction refractive index sin i sin r critical angle total internal reflection",
    "tir": "total internal reflection critical angle optical fibre dense to rare",
    "fusion": "nuclear fusion hydrogen nuclei join helium stars sun",
    "fission": "nuclear fission uranium heavy nucleus splits neutrons chain reaction",
    "redshift": "red-shift wavelength galaxies receding hubble big bang universe",
    "cmbr": "cosmic microwave background radiation big bang universe",
}


class HaronyKnowledgeBase:
    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.mark_schemes: List[Dict[str, Any]] = []
        self.videos: List[Dict[str, Any]] = []
        self.concept_atlas: List[Dict[str, Any]] = []
        self.question_bank: List[Dict[str, Any]] = []
        self.tfidf = None
        self.svd = None
        self.tfidf_matrix = None
        self.svd_matrix = None
        self.qb_tfidf = None
        self.qb_matrix = None
        self.topic_clf = None
        self.intent_clf = None
        self.reload()

    def reload(self) -> None:
        base_chunks: List[Dict[str, Any]] = []
        if NOTES_CHUNKS_PATH.exists():
            try:
                base_chunks.extend(json.loads(NOTES_CHUNKS_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass
        if CUSTOM_MATERIALS_PATH.exists():
            try:
                custom = json.loads(CUSTOM_MATERIALS_PATH.read_text(encoding="utf-8"))
                if isinstance(custom, list):
                    base_chunks.extend(custom)
            except Exception:
                pass
        # Filter out nearly empty cover/divider pages for search quality while keeping substantive pages
        self.chunks = [c for c in base_chunks if len((c.get("text") or "").strip()) >= 25]

        if MARK_SCHEMES_PATH.exists():
            try:
                self.mark_schemes = json.loads(MARK_SCHEMES_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.mark_schemes = []

        if VIDEO_SOLUTIONS_PATH.exists():
            try:
                self.videos = json.loads(VIDEO_SOLUTIONS_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.videos = []

        if CONCEPT_ATLAS_PATH.exists():
            try:
                self.concept_atlas = json.loads(CONCEPT_ATLAS_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.concept_atlas = []

        if QUESTION_BANK_PATH.exists():
            try:
                self.question_bank = json.loads(QUESTION_BANK_PATH.read_text(encoding="utf-8"))
            except Exception:
                self.question_bank = []

        if self.question_bank and QB_RETRIEVER_MODEL_PATH.exists():
            try:
                qbundle = joblib.load(QB_RETRIEVER_MODEL_PATH)
                if qbundle.get("qb_count") == len(self.question_bank):
                    self.qb_tfidf = qbundle["tfidf"]
                    self.qb_matrix = qbundle["matrix"]
            except Exception:
                self.qb_tfidf = None
                self.qb_matrix = None
        if self.question_bank and self.qb_tfidf is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                qb_docs = [
                    f"{q.get('unit', '')} {q.get('chapter', '')} {' '.join(q.get('paper_codes', []))} {q.get('question_text', '')} "
                    + " ".join(m.get("answer", "") for m in q.get("ms_answers", []))
                    for q in self.question_bank
                ]
                self.qb_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=30000)
                self.qb_matrix = self.qb_tfidf.fit_transform(qb_docs)
            except Exception:
                pass

        if TOPIC_MODEL_PATH.exists():
            try:
                self.topic_clf = joblib.load(TOPIC_MODEL_PATH)
            except Exception:
                self.topic_clf = None

        if INTENT_MODEL_PATH.exists():
            try:
                self.intent_clf = joblib.load(INTENT_MODEL_PATH)
            except Exception:
                self.intent_clf = None

        if RETRIEVER_MODEL_PATH.exists():
            try:
                bundle = joblib.load(RETRIEVER_MODEL_PATH)
                if bundle.get("chunk_count") == len(self.chunks):
                    self.tfidf = bundle["tfidf"]
                    self.svd = bundle["svd"]
                    self.tfidf_matrix = bundle["tfidf_matrix"]
                    self.svd_matrix = bundle["svd_matrix"]
                    return
            except Exception:
                pass

        # Build in-memory fallback if model hasn't been trained or chunk count changed
        self._fit_in_memory()

    def _corpus_texts(self) -> List[str]:
        docs = []
        for c in self.chunks:
            unit = c.get("unit", "")
            topic = c.get("topic", "")
            formulas = " ".join(c.get("formulas", []))
            txt = c.get("text", "")
            # Weight topic & formulas higher by repeating
            docs.append(f"{unit} {topic} {topic} {formulas} {txt}")
        return docs

    def _fit_in_memory(self) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if not self.chunks:
            return
        docs = self._corpus_texts()
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=25000,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(docs)
        n_comp = min(96, max(8, self.tfidf_matrix.shape[0] - 2), self.tfidf_matrix.shape[1] - 2)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd_matrix = normalize(self.svd.fit_transform(self.tfidf_matrix))

    def expand_query(self, query: str) -> str:
        low = query.lower()
        extras = [exp for key, exp in QUERY_SYNONYM_EXPANSIONS.items() if key in low]
        if extras:
            return query + " " + " ".join(extras)
        return query

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.chunks or self.tfidf is None or self.tfidf_matrix is None:
            return []

        expanded = self.expand_query(query)
        # Strip generic question verbs so words like "explain" or "calculate" never match unrelated slides
        generic_verbs = {
            "what", "how", "why", "the", "and", "for", "with", "from", "that", "this",
            "explain", "calculate", "define", "state", "describe", "find", "show",
            "give", "tell", "does", "when", "where", "which", "would", "could", "please",
            "law", "laws", "effect", "effects", "rule", "rules", "principle", "theory",
            "work", "works", "second", "first", "third", "fluids", "fluid",
        }
        q_tokens = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in generic_verbs}
        cleaned_expanded = " ".join(w for w in expanded.split() if re.sub(r"[^a-z0-9]", "", w.lower()) not in generic_verbs).strip()
        if not cleaned_expanded:
            cleaned_expanded = expanded

        q_tfidf = self.tfidf.transform([cleaned_expanded])
        lex_scores = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]

        if self.svd is not None and self.svd_matrix is not None:
            q_svd = self.svd.transform(q_tfidf)
            norm = np.linalg.norm(q_svd)
            if norm > 1e-9:
                q_svd = q_svd / norm
            sem_scores = (self.svd_matrix @ q_svd[0])
            # Only allow semantic SVD boost when there is at least some lexical signal
            combined = 0.70 * lex_scores + 0.30 * np.where(lex_scores > 0.025, np.clip(sem_scores, 0.0, 1.0), 0.0)
        else:
            combined = lex_scores

        # Topic exact-word bonus
        for i, ch in enumerate(self.chunks):
            topic_low = (ch.get("topic", "") + " " + ch.get("unit", "")).lower()
            overlap = sum(1 for tok in q_tokens if tok in topic_low)
            if overlap:
                combined[i] += 0.09 * min(overlap, 3)

        # Prefer Harony Notes 2026 slides slightly over dense MS lists ONLY when already relevant
        is_paper_code_query = bool(re.search(r"\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+", query.lower()))
        for i, ch in enumerate(self.chunks):
            src = ch.get("source", "")
            if not is_paper_code_query and src == "Harony Notes 2026 Final" and combined[i] >= 0.12:
                combined[i] += 0.04

        ranked_idx = np.argsort(combined)[::-1][: top_k * 3]
        results: List[Dict[str, Any]] = []
        seen_pages = set()
        for idx in ranked_idx:
            score = float(combined[idx])
            if score < 0.16:
                continue
            ch = dict(self.chunks[int(idx)])
            pg_key = (ch.get("source"), ch.get("page"), ch.get("id"))
            if pg_key in seen_pages:
                continue
            seen_pages.add(pg_key)
            ch["score"] = round(score, 4)
            results.append(ch)
            if len(results) >= top_k:
                break
        return results

    def search_videos(
        self,
        query: str,
        top_k: int = 3,
        preferred_unit: Optional[str] = None,
        preferred_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find strictly relevant, unit-locked Mr. Haroony YouTube video explanations."""
        if not self.videos:
            return []
        ignore_words = {
            "what", "how", "why", "the", "and", "for", "in", "on", "of", "to", "is", "it",
            "at", "by", "as", "or", "an", "be", "do", "no", "so", "if", "up", "vs",
            "explain", "calculate", "define", "state", "describe", "compare", "difference",
            "between", "into", "onto", "than", "then", "physics", "question", "questions",
            "youtube", "video", "videos", "link", "links", "watch", "solve", "solving",
            "solution", "solutions", "give", "send", "show", "find", "please", "any",
            "topic", "topics", "about", "with", "from", "can", "you", "get", "me", "want",
            "need", "work", "works", "does", "make", "law", "laws", "effect", "rule",
            "first", "second", "third", "fluid", "fluids", "step", "steps", "body",
        }
        raw_tokens = [
            t for t in re.findall(r"[a-z0-9]{2,}", query.lower())
            if t not in ignore_words
        ]
        # Expand core physics aliases to exact Harony video chapter keywords
        vid_synonyms = {
            "convection": ["heat", "transmission"],
            "conduction": ["heat", "transmission"],
            "radiation": ["heat", "transmission"],
            "transverse": ["waves", "water"],
            "longitudinal": ["waves", "water", "sound"],
            "transformer": ["transformers"],
            "step-up": ["transformers"],
            "step-down": ["transformers"],
            "ohm": ["electric", "circuits", "resistance"],
            "ohms": ["electric", "circuits", "resistance"],
            "newton": ["forces", "kinematics"],
            "inertia": ["forces"],
            "redshift": ["universe", "redshift", "hubble"],
            "bang": ["universe"],
            "optical": ["refraction", "total", "internal"],
            "fibres": ["refraction", "total", "internal"],
            "fibre": ["refraction", "total", "internal"],
            "tir": ["total", "internal", "reflection"],
            "echo": ["sound"],
            "echoes": ["sound"],
            "ultrasound": ["sound"],
            "half": ["half", "life"],
            "halflife": ["half", "life"],
            "fission": ["alpha", "beta", "gamma", "nuclear", "fission"],
            "fusion": ["nuclear", "fusion", "stars"],
        }
        norm_tokens: List[str] = []
        for t in raw_tokens:
            nt = "jun" if t == "may" else t
            if nt not in norm_tokens:
                norm_tokens.append(nt)
            for syn in vid_synonyms.get(nt, []):
                if syn not in norm_tokens:
                    norm_tokens.append(syn)
        if not norm_tokens and preferred_topic:
            norm_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", preferred_topic.lower()) if t not in ignore_words]
        if not norm_tokens:
            return []

        is_paper_code = bool(re.search(r"\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+", query.lower()))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for v in self.videos:
            v_unit = v.get("unit", "")
            # Strict Unit-Lock when we already identified the Physics Unit from Harony Notes!
            if preferred_unit and not is_paper_code and v_unit and v_unit != preferred_unit:
                continue

            label_words = set(re.findall(r"[a-z0-9]+", (v.get("label") or "").lower()))
            nk_words = set(re.findall(r"[a-z0-9]+", (v.get("norm_key") or "").lower()))
            chap_words = set(re.findall(r"[a-z0-9]+", (v.get("chapter") or "").lower()))
            unit_words = set(re.findall(r"[a-z0-9]+", v_unit.lower()))

            hits = 0.0
            matched_distinct = 0
            for tok in norm_tokens:
                if tok in label_words or tok in nk_words:
                    hits += 3.0
                    matched_distinct += 1
                elif tok in chap_words:
                    hits += 2.2
                    matched_distinct += 1
                elif tok in unit_words and len(tok) >= 4:
                    hits += 1.0
            if hits >= 2.2 and (matched_distinct >= 1):
                scored.append((hits, v))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        seen_urls = set()
        for _, v in scored:
            u = v.get("video_url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                out.append(v)
                if len(out) >= top_k:
                    break
        return out

    def predict_unit(self, query: str) -> Optional[str]:
        """Predict the primary Physics Unit (Unit 1..7) using the trained multi-class classifier."""
        if not self.topic_clf or not query.strip():
            return None
        try:
            probs = self.topic_clf.predict_proba([query])[0]
            classes = self.topic_clf.classes_
            best_idx = int(np.argmax(probs))
            if float(probs[best_idx]) >= 0.42:
                return str(classes[best_idx])
        except Exception:
            pass
        return None

    def predict_intent(self, query: str) -> str:
        if not self.intent_clf or not query.strip():
            return "concept_explain"
        try:
            return str(self.intent_clf.predict([query])[0])
        except Exception:
            return "concept_explain"

    def match_concept_atlas(self, query: str, preferred_unit: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.concept_atlas or not query.strip():
            return None
        low = query.lower()
        q_words = set(re.findall(r"[a-z0-9]{3,}", low))
        best_entry = None
        best_score = 0.0
        for entry in self.concept_atlas:
            if preferred_unit and entry.get("unit") != preferred_unit:
                continue
            sc = 0.0
            for kw in entry.get("keywords", []):
                if kw in low:
                    sc += 3.0 if " " in kw else 2.0
            topic_words = set(re.findall(r"[a-z0-9]{3,}", entry.get("topic", "").lower()))
            sc += 1.2 * len(q_words & topic_words)
            if sc > best_score:
                best_score = sc
                best_entry = entry
        return best_entry if best_score >= 2.0 else None

    def search_question_bank(
        self,
        query: str,
        top_k: int = 2,
        preferred_unit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Match student queries / uploaded exam problems against 1,731 real Harony ClassWork/P4/P2 pages."""
        if not self.question_bank or self.qb_tfidf is None or self.qb_matrix is None or not query.strip():
            return []
        expanded = self.expand_query(query)
        q_vec = self.qb_tfidf.transform([expanded])
        scores = cosine_similarity(q_vec, self.qb_matrix)[0]
        is_paper_code = bool(re.search(r"\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+", query.lower()))
        ranked_idx = np.argsort(scores)[::-1][: top_k * 6]
        out: List[Dict[str, Any]] = []
        for idx in ranked_idx:
            sc = float(scores[idx])
            if sc < 0.22:
                break
            item = dict(self.question_bank[int(idx)])
            if preferred_unit and not is_paper_code and item.get("unit") and item.get("unit") != preferred_unit:
                continue
            item["score"] = round(sc, 4)
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def get_mark_scheme(self, ms_id: str) -> Optional[Dict[str, Any]]:
        for ms in self.mark_schemes:
            if ms.get("id") == ms_id:
                return ms
        return None

    def save_mark_scheme(self, ms_obj: Dict[str, Any]) -> Dict[str, Any]:
        existing = [m for m in self.mark_schemes if m.get("id") != ms_obj.get("id")]
        existing.append(ms_obj)
        self.mark_schemes = existing
        MARK_SCHEMES_PATH.write_text(json.dumps(self.mark_schemes, ensure_ascii=False, indent=2), encoding="utf-8")
        return ms_obj

    def add_custom_material(self, title: str, unit: str, topic: str, text_blocks: List[str]) -> int:
        custom: List[Dict[str, Any]] = []
        if CUSTOM_MATERIALS_PATH.exists():
            try:
                custom = json.loads(CUSTOM_MATERIALS_PATH.read_text(encoding="utf-8"))
            except Exception:
                custom = []
        added = 0
        base_idx = len(custom) + 1
        for i, blk in enumerate(text_blocks):
            clean_blk = blk.strip()
            if len(clean_blk) < 25:
                continue
            custom.append({
                "id": f"custom-{base_idx + i:04d}",
                "source": title,
                "page": i + 1,
                "unit": unit or "Teacher Uploaded Material",
                "topic": topic or title,
                "text": clean_blk,
                "formulas": [ln for ln in clean_blk.splitlines() if "=" in ln and len(ln) < 90][:5],
                "image_url": None,
            })
            added += 1
        CUSTOM_MATERIALS_PATH.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")
        self.reload()
        return added

    def curriculum_overview(self) -> List[Dict[str, Any]]:
        units: Dict[str, Dict[str, Any]] = {}
        for c in self.chunks:
            u = c.get("unit", "General")
            if u == "Overview":
                continue
            t = c.get("topic", "General")
            pg = c.get("page")
            if u not in units:
                units[u] = {"unit": u, "topics": {}}
            if t not in units[u]["topics"]:
                units[u]["topics"][t] = {
                    "topic": t,
                    "start_page": pg,
                    "end_page": pg,
                    "page_count": 0,
                    "sample_image": c.get("image_url"),
                }
            tobj = units[u]["topics"][t]
            tobj["page_count"] += 1
            if pg is not None:
                if tobj["start_page"] is None or pg < tobj["start_page"]:
                    tobj["start_page"] = pg
                if tobj["end_page"] is None or pg > tobj["end_page"]:
                    tobj["end_page"] = pg
        return [
            {"unit": u, "topics": list(uobj["topics"].values())}
            for u, uobj in units.items()
        ]
