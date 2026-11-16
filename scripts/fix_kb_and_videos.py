"""Clean stuck-together PDF/OCR words, fix typos, and fix mislabeled Units/Chapters
across Harony Notes, Mark Schemes, and Video Solutions, then retrain models
and rebuild static/browser_kb.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import DATA_DIR
from scripts.train_haroony_ai import train_all_models

NOTES_PATH = DATA_DIR / "harony_notes_chunks.json"
MS_PATH = DATA_DIR / "mark_schemes.json"
VIDEOS_PATH = DATA_DIR / "video_solutions.json"

# Comprehensive IGCSE / A-Level Physics vocabulary + common words that may be missing or rare
PHYSICS_VOCAB = {
    "physics", "mechanics", "kinematics", "velocity", "acceleration", "deceleration",
    "displacement", "momentum", "impulse", "equilibrium", "perpendicular", "clockwise",
    "anticlockwise", "proportional", "proportionality", "extension", "compression",
    "centripetal", "friction", "air", "resistance", "resultant", "terminal", "gravity",
    "gravitational", "weight", "mass", "density", "volume", "pressure", "barometer",
    "manometer", "hydraulic", "pascal", "joule", "joules", "watt", "watts", "newton",
    "newtons", "efficiency", "kinetic", "potential", "thermal", "energy", "energies",
    "conduction", "convection", "radiation", "evaporation", "boiling", "melting",
    "condensation", "solidification", "latent", "specific", "capacity", "temperature",
    "thermometer", "thermocouple", "expansion", "contraction", "brownian", "molecules",
    "particles", "vibrate", "vibrating", "vibration", "vibrations", "collide",
    "collisions", "collision", "randomly", "random", "continuous", "range", "values",
    "wave", "waves", "wavefront", "wavefronts", "wavelength", "wavelengths",
    "frequency", "frequencies", "amplitude", "period", "transverse", "longitudinal",
    "rarefaction", "rarefactions", "compressions", "diffraction", "refraction",
    "refractive", "reflection", "reflected", "incident", "incidence", "normal",
    "critical", "optical", "fibre", "fibres", "dispersion", "spectrum", "monochromatic",
    "electromagnetic", "microwaves", "microwave", "infrared", "ultraviolet", "gamma",
    "ultrasound", "echo", "echoes", "converging", "diverging", "principal", "focus",
    "focal", "magnified", "diminished", "inverted", "upright", "virtual", "real",
    "telecommunication", "telecommunications", "electricity", "electrostatic",
    "electrostatics", "charge", "charges", "charged", "coulomb", "coulombs",
    "current", "currents", "ampere", "amperes", "voltage", "voltages", "volt",
    "volts", "potential", "difference", "electromotive", "resistance", "resistor",
    "resistors", "filament", "diode", "thermistor", "relay", "solenoid", "circuit",
    "circuits", "series", "parallel", "components", "earthing", "breaker", "fuse",
    "magnet", "magnets", "magnetic", "magnetised", "magnetized", "magnetism",
    "electromagnet", "electromagnets", "electromagnetism", "commutator", "brushes",
    "armature", "induction", "induced", "induces", "inducing", "alternating", "direct",
    "generator", "generators", "transformer", "transformers", "primary", "secondary",
    "transmission", "national", "nuclear", "nucleus", "nuclei", "proton", "protons",
    "neutron", "neutrons", "electron", "electrons", "isotope", "isotopes", "nuclide",
    "alpha", "beta", "radioactive", "radioactivity", "ionising", "ionizing",
    "background", "detector", "geiger", "halflife", "decay", "decays", "fission",
    "fusion", "rutherford", "scattering", "subatomic", "thickness", "count", "counts",
    "astronomy", "universe", "galaxy", "galaxies", "redshift", "hubble", "protostar",
    "supernova", "supergiant", "nebula", "planetary", "elliptical", "geostationary",
    "satellite", "satellites", "orbital", "orbit", "orbits", "lightyear", "cosmic",
    "statement", "statements", "correct", "incorrect", "explained", "explanation",
    "increases", "increase", "increased", "increasing", "decreases", "decrease",
    "decreased", "decreasing", "remains", "constant", "unchanged", "between",
    "horizontal", "vertical", "direction", "directions", "experiences", "experience",
    "accelerated", "accelerates", "accelerating", "decelerates", "decelerating",
    "powered", "rocket", "engine", "finishing", "parachute", "switched", "readings",
    "reading", "materials", "material", "contacts", "reversing", "reverse", "reverses",
    "reaches", "reached", "reaching", "becomes", "become", "becoming", "changes",
    "changing", "changed", "towards", "away", "across", "through", "because", "causes",
    "causing", "caused", "smaller", "larger", "greater", "higher", "lower", "faster",
    "slower", "closer", "further", "stronger", "weaker", "heavier", "lighter", "always",
    "never", "sometimes", "however", "therefore", "although", "provided", "without",
    "another", "together", "against", "around", "inside", "outside", "surface",
    "surfaces", "liquid", "liquids", "solid", "solids", "gases", "object", "objects",
    "moving", "motion", "travel", "travels", "travelling", "distance", "distances",
    "seconds", "second", "minutes", "minute", "hours", "metres", "meters", "kilograms",
    "grams", "length", "height", "width", "depth", "radius", "diameter", "area",
    "angle", "angles", "region", "regions", "rocks", "buildings", "drag", "car",
    "racing", "opens", "open", "line", "axis", "axes", "graph", "graphs", "curve",
    "gradient", "slope", "straight", "uniform", "nonuniform", "instantaneous",
    "average", "total", "initial", "final", "maximum", "minimum", "zero", "positive",
    "negative", "equal", "opposite", "same", "different", "ratio", "product",
    "formula", "equation", "calculate", "calculated", "calculating", "determine",
    "explain", "describe", "state", "define", "compare", "contrast", "suggest",
    "sketch", "draw", "label", "plot", "measure", "measured", "measuring", "instrument",
    "ruler", "cylinder", "micrometer", "vernier", "caliper", "stopwatch", "balance",
    "spring", "pendulum", "oscillation", "oscillations", "amplitude", "trough", "crest",
    "propagation", "where", "overall", "combined", "absorber", "emitter", "reflector",
    "silver", "black", "dull", "shiny", "polished", "copper", "aluminium", "steel",
    "iron", "plastic", "rubber", "glass", "water", "mercury", "alcohol", "oxygen",
    "hydrogen", "helium", "uranium", "carbon", "plutonium", "lead", "aluminium",
    "paper", "foil", "sheet", "sheets", "cardboard", "concrete", "wood", "wooden",
    "wet", "road", "braking", "wall", "ball", "hard", "near", "registered",
    "counter", "rebounds", "upwards", "downwards", "impact", "varies", "vary",
    "curved", "shows", "showing", "its", "their", "them", "they", "this", "that",
    "with", "from", "into", "when", "where", "which", "while", "what", "how",
    "is", "are", "was", "were", "the", "and", "for", "of", "to", "in", "on", "at",
    "by", "as", "an", "or", "if", "so", "be", "has", "have", "had", "can", "will",
    "not", "no", "it", "do", "does", "did", "at", "up", "out", "off", "all", "one",
    "two", "three", "four", "five", "six", "ten", "time", "times", "line", "lines",
}

# Explicit typo / spacing fixes observed in PDF/OCR extracts
EXACT_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\boveral\b", "overall"),
    (r"\bdirection\s+f\s+the\b", "direction of the"),
    (r"\bwave\s+were\s+particles\b", "wave where particles"),
    (r"\bwhere\s+particles\s+vibrate\b", "where particles vibrate"),
    (r"\bElectromagnetics\b", "Electromagnetic"),
    (r"\bparticlesincreases\b", "particles increases"),
    (r"\bparticlesmove\b", "particles move"),
    (r"\bparticlesvibrate\b", "particles vibrate"),
    (r"\bparticlescollide\b", "particles collide"),
    (r"\bexperiencesaforce\b", "experiences a force"),
    (r"\bthemagneticfield\b", "the magnetic field"),
    (r"\bcontactsreversing\b", "contacts reversing"),
    (r"\bcurvedwavefronts\b", "curved wavefronts"),
    (r"\bfrequencyunchanged\b", "frequency unchanged"),
    (r"\bmomentumaftercollision\b", "momentum after collision"),
    (r"\bWhichstatementiscorrect\b", "Which statement is correct"),
    (r"\bsubatomicparticles\b", "subatomic particles"),
    (r"\brocksORbuildings\b", "rocks OR buildings"),
    (r"\bacceleratesORspeedincreases\b", "accelerates OR speed increases"),
    (r"\baccelerationORlessacceleration\b", "acceleration OR less acceleration"),
    (r"\belectromagneticradiation\b", "electromagnetic radiation"),
    (r"\benergyaselectromagneticradiation\b", "energy as electromagnetic radiation"),
    (r"\bregionofelectromagnetic\b", "region of electromagnetic"),
    (r"\bacontinuousrangeofvalues\b", "a continuous range of values"),
    (r"\btimetakenforhalfthenucleiofasampletodecay\b", "time taken for half the nuclei of a sample to decay"),
    (r"\bsocountratechangeswiththickness\b", "so count rate changes with thickness"),
    (r"\btwocorrectraysfrom\b", "two correct rays from"),
    (r"\bmagneticmaterials\b", "magnetic materials"),
    (r"\bcompressionbeing\b", "compression being"),
    (r"\bAdragcar\b", "A drag car"),
    (r"\bdragcar\b", "drag car"),
    (r"\bThisdragcar\b", "This drag car"),
    (r"\bThecardeceleratesuntil\b", "The car decelerates until"),
    (r"\bHowisthedeceleration\b", "How is the deceleration"),
    (r"\bthedifferencebetweentworeadingsonthehorizontal\b", "the difference between two readings on the horizontal"),
    (r"\bisthedifferencebetweentworeadingsonthevertical\b", "is the difference between two readings on the vertical"),
]


def build_dictionary(chunks: List[dict], mark_schemes: List[dict]) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for c in chunks:
        for w in re.findall(r"\b[a-z]{2,15}\b", (c.get("text") or "").lower()):
            counts[w] += 1
    for ms in mark_schemes:
        for q in ms.get("questions", []):
            for w in re.findall(r"\b[a-z]{2,15}\b", (q.get("summary_answer") or "").lower()):
                counts[w] += 1

    # Keep only words that appear cleanly >= 2 times, or are in ENGLISH_STOP_WORDS or PHYSICS_VOCAB
    vocab_freq: Dict[str, int] = {}
    for w, cnt in counts.items():
        if cnt >= 2 and len(w) <= 15:
            vocab_freq[w] = cnt
    for w in ENGLISH_STOP_WORDS:
        vocab_freq[w] = max(vocab_freq.get(w, 0), 50)
    for w in PHYSICS_VOCAB:
        vocab_freq[w] = max(vocab_freq.get(w, 0), 80)
    # Remove single-letter junk except a/i
    for bad in list(vocab_freq.keys()):
        if len(bad) == 1 and bad not in ("a", "i"):
            del vocab_freq[bad]
    vocab_freq["a"] = 500
    vocab_freq["i"] = 200

    total = sum(vocab_freq.values()) or 1
    return {w: math.log(cnt / total) for w, cnt in vocab_freq.items()}


def dp_segment(word: str, log_probs: Dict[str, float]) -> str:
    low = word.lower()
    if low in log_probs or len(low) <= 9:
        return word
    n = len(low)
    dp = [(-1e18, 0)] * (n + 1)
    dp[0] = (0.0, 0)
    for i in range(1, n + 1):
        for j in range(max(0, i - 18), i):
            sub = low[j:i]
            if sub in log_probs:
                # Penalize 2-char splits slightly unless really needed so real long words stay intact
                pen = -2.2 if len(sub) == 2 else (-3.5 if len(sub) == 1 else 0.0)
                score = dp[j][0] + log_probs[sub] + pen - 1.5
                if score > dp[i][0]:
                    dp[i] = (score, j)
    if dp[n][0] < -1e15:
        return word
    parts: List[str] = []
    idx = n
    while idx > 0:
        prev = dp[idx][1]
        parts.append(word[prev:idx])
        idx = prev
    parts.reverse()
    if len(parts) <= 1:
        return word
    # Avoid breaking into lots of tiny 1-2 char fragments
    tiny = sum(1 for p in parts if len(p) <= 2)
    # Allow tiny words if there are also multi-word meaningful parts (for long glued sentences)
    if tiny > len(parts) * 0.65 and len(word) < 22:
        return word
    return " ".join(parts)


def clean_text(text: str, log_probs: Dict[str, float]) -> str:
    if not text:
        return ""
    s = str(text)
    for pat, rep in EXACT_REPLACEMENTS:
        s = re.sub(pat, rep, s)
    # Split camelCase / lowercase-to-OR / lowercase-to-Capital boundaries (e.g. rocksORbuildings, TheCar)
    s = re.sub(r"([a-z])(OR)([a-zA-Z])", r"\1 \2 \3", s)
    s = re.sub(r"([a-z]{2,})([A-Z][a-z]{2,})", r"\1 \2", s)
    # Add space after comma/semicolon if missing before a letter (keep numbers like 1,000 intact)
    s = re.sub(r"([a-zA-Z])([,;:])([a-zA-Z])", r"\1\2 \3", s)

    # Segment long glued alpha tokens (> 11 chars) not in dictionary
    def _seg_token(m: re.Match) -> str:
        tok = m.group(0)
        if tok.lower() in log_probs or len(tok) < 11:
            return tok
        return dp_segment(tok, log_probs)

    s = re.sub(r"\b[A-Za-z]{11,}\b", _seg_token, s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def fix_unit_and_chapter(unit: str, chapter: str) -> Tuple[str, str]:
    ch_low = (chapter or "").lower()
    if "circuit components" in ch_low:
        return "Unit 3: Electricity", "U3 Ch4: Circuit Components"
    if "generator" in ch_low or "electromagnetic induction" in ch_low:
        return "Unit 4: Electromagnetism", re.sub(r"^U\d+\s+Ch\d+:?", "U4 Ch3:", chapter)
    if "transformers" in ch_low:
        return "Unit 4: Electromagnetism", chapter
    if "atomic structure" in ch_low:
        return "Unit 5: Nuclear Physics", "U5 Ch1: Atomic Structure"
    if "alpha, beta" in ch_low:
        return "Unit 5: Nuclear Physics", chapter
    if "half life" in ch_low:
        return "Unit 5: Nuclear Physics", "U5 Ch3: Half Life"
    if "water waves" in ch_low:
        return "Unit 6: Waves", "U6 Ch1: Water Waves"
    if "sound waves" in ch_low:
        return "Unit 6: Waves", "U6 Ch2: Sound Waves"
    if "light reflection" in ch_low:
        return "Unit 6: Waves", "U6 Ch3: Light Reflection"
    if "light refraction" in ch_low:
        return "Unit 6: Waves", "U6 Ch4: Light Refraction"
    if "light dispersion" in ch_low:
        return "Unit 6: Waves", "U6 Ch5: Light Dispersion"
    if "lenses" in ch_low:
        return "Unit 6: Waves", "U6 Ch6: Lenses"
    if "earth, moon" in ch_low or "solar system" in ch_low or "stars and galaxies" in ch_low or "universe" in ch_low:
        return "Unit 7: Astronomy", chapter
    return unit, chapter


def main() -> None:
    chunks = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
    mark_schemes = json.loads(MS_PATH.read_text(encoding="utf-8"))
    videos = json.loads(VIDEOS_PATH.read_text(encoding="utf-8"))

    log_probs = build_dictionary(chunks, mark_schemes)

    # 1. Clean chunks text + fix mislabeled units/topics
    fixed_chunk_units = 0
    for c in chunks:
        c["text"] = clean_text(c.get("text", ""), log_probs)
        u, t = fix_unit_and_chapter(c.get("unit", ""), c.get("topic", ""))
        if u != c.get("unit", ""):
            fixed_chunk_units += 1
        c["unit"] = u
        c["topic"] = t

    # 2. Clean mark schemes
    for ms in mark_schemes:
        for q in ms.get("questions", []):
            u, ch = fix_unit_and_chapter("", q.get("chapter", ""))
            if ch:
                q["chapter"] = ch
            q["summary_answer"] = clean_text(q.get("summary_answer", ""), log_probs)
            for pt in q.get("points", []):
                pt["answer"] = clean_text(pt.get("answer", ""), log_probs)
                if pt.get("guidance"):
                    pt["guidance"] = clean_text(pt["guidance"], log_probs)

    # 3. Fix mislabeled units/chapters & trailing dots in video_solutions.json
    fixed_vid_units = 0
    for v in videos:
        old_u = v.get("unit", "")
        new_u, new_ch = fix_unit_and_chapter(old_u, v.get("chapter", ""))
        if new_u != old_u:
            fixed_vid_units += 1
        v["unit"] = new_u
        v["chapter"] = new_ch
        lbl = str(v.get("label") or "")
        lbl = re.sub(r"\.{3,}\s*[A-D]?\s*$", "", lbl).strip()
        lbl = clean_text(lbl, log_probs)
        v["label"] = lbl

    NOTES_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    MS_PATH.write_text(json.dumps(mark_schemes, ensure_ascii=False, indent=2), encoding="utf-8")
    VIDEOS_PATH.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Cleaned {len(chunks)} chunks (fixed {fixed_chunk_units} units), {len(mark_schemes)} MS books, {len(videos)} videos (fixed {fixed_vid_units} units).")
    metrics = train_all_models()
    print("Retrained models & rebuilt static/browser_kb.json:", metrics)


if __name__ == "__main__":
    main()
