"""Strict Physics-Only Domain Guardrail for Haroony AI.

Combines:
1. Prompt-injection & jailbreak pattern shield (cannot be bypassed with "ignore previous instructions").
2. Explicit non-physics domain blocklist (pure biology, pure chemistry reactions, coding, history,
   politics, creative writing, sports, recipes, entertainment, finance) unless grounded in genuine physics.
3. Trained ML probability classifier (`models/physics_guardrail.joblib` — word + char TF-IDF LogisticRegression)
   trained directly on Harony Physics Clinic Notes, Cambridge 0625 Mark Schemes, Physics calculations,
   and diverse negative non-physics queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib

from backend.config import MODELS_DIR

GUARDRAIL_MODEL_PATH = MODELS_DIR / "physics_guardrail.joblib"

JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|rules|prompts)", re.I),
    re.compile(r"forget\s+(?:your|all)\s+(?:instructions|rules|constraints)", re.I),
    re.compile(r"\b(?:you\s+are\s+now\s+dan|developer\s+mode|jailbreak|system\s+override|act\s+as\s+a\s+general)\b", re.I),
    re.compile(r"pretend\s+(?:you\s+are\s+not|this\s+is\s+related\s+to\s+physics)", re.I),
]

GREETING_IDENTITY_RE = re.compile(
    r"^\s*(?:hi|hello|hey|salam|assalamu\s*alaikum|good\s+(?:morning|afternoon|evening)|"
    r"who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help(?:\s+me)?)\s*[?!.]*\s*$",
    re.I,
)

FOLLOWUP_PROBES_RE = re.compile(
    r"^\s*(?:why|how|explain\s+more|show\s+(?:the\s+)?(?:working|steps|formula|diagram|page)|"
    r"give\s+(?:me\s+)?(?:an?\s+)?example|what\s+is\s+the\s+unit|can\s+you\s+simplify|"
    r"test\s+me|quiz\s+me|give\s+me\s+a\s+practice\s+question|correct\s+(?:this|my\s+answer)|"
    r"mark\s+(?:this|my\s+answer|it)|grade\s+(?:this|it))\b",
    re.I,
)

# Strong Physics vocabulary & symbols (IGCSE 0625 + A-Level + Harony Notes)
PHYSICS_KEYWORDS = {
    "physics", "physical", "igcse", "0625", "cambridge", "edexcel", "mark scheme", "harony", "haroony",
    "scalar", "vector", "velocity", "speed", "acceleration", "deceleration", "displacement", "distance",
    "kinematics", "gradient", "speed-time", "distance-time", "velocity-time", "freefall", "terminal velocity",
    "force", "newton", "mass", "weight", "gravity", "gravitational", "density", "hooke", "spring",
    "extension", "limit of proportionality", "friction", "drag", "air resistance", "centripetal", "circular motion",
    "moment", "pivot", "fulcrum", "equilibrium", "clockwise", "anticlockwise", "centre of gravity", "center of gravity",
    "centre of mass", "momentum", "impulse", "collision", "conservation",
    "work", "energy", "kinetic", "potential", "joule", "watt", "power", "efficiency", "renewable", "geothermal",
    "hydroelectric", "tidal", "solar panel", "nuclear power", "san-key", "sankey",
    "pressure", "pascal", "barometer", "manometer", "hydraulic", "boyle",
    "thermal", "kelvin", "celsius", "thermometer", "thermocouple", "brownian", "evaporation", "boiling",
    "melting", "solidification", "condensation", "thermal expansion", "bimetallic", "conduction", "convection",
    "radiation", "infrared", "emitter", "absorber", "specific heat capacity", "latent heat", "internal energy",
    "wave", "transverse", "longitudinal", "wavelength", "frequency", "hertz", "amplitude", "period",
    "wavefront", "ripple tank", "reflection", "refraction", "diffraction", "echo", "ultrasound", "audible",
    "compression", "rarefaction", "refractive index", "snell", "critical angle", "total internal reflection",
    "optical fibre", "optical fiber", "converging lens", "diverging lens", "focal length", "principal focus",
    "real image", "virtual image", "dispersion", "prism", "monochromatic", "electromagnetic spectrum",
    "radio waves", "microwaves", "ultraviolet", "x-ray", "x-rays", "gamma",
    "electrostatics", "charge", "coulomb", "electric field", "conductor", "insulator", "current", "ampere",
    "ammeter", "voltmeter", "voltage", "potential difference", "electromotive force", "emf", "resistance",
    "resistor", "ohm", "thermistor", "ldr", "light-dependent resistor", "diode", "led", "relay", "fuse",
    "circuit", "series", "parallel", "potential divider", "kilowatt-hour", "kwh", "earth wire", "live wire", "neutral wire",
    "magnet", "magnetism", "magnetic field", "solenoid", "electromagnet", "fleming", "left-hand rule",
    "right-hand rule", "commutator", "split-ring", "slip ring", "dc motor", "d.c. motor", "ac generator",
    "a.c. generator", "electromagnetic induction", "lenz", "faraday", "transformer", "step-up", "step-down",
    "national grid",
    "atom", "nucleus", "proton", "neutron", "electron", "nucleon", "isotope", "alpha", "beta",
    "radioactive", "radioactivity", "ionising", "ionizing", "geiger", "background radiation", "half-life",
    "half life", "fission", "fusion", "rutherford", "gold foil",
    "astronomy", "solar system", "orbit", "orbital speed", "moon", "planet", "comet", "asteroid", "galaxy",
    "milky way", "protostar", "main sequence", "red giant", "red supergiant", "white dwarf", "supernova",
    "neutron star", "black hole", "light-year", "light year", "redshift", "red-shift", "cosmic microwave",
    "cmbr", "hubble", "big bang",
    "micrometer", "caliper", "vernier", "parallax", "meniscus", "pendulum", "oscillation", "significant figures",
    "pressure cooker", "swimming pool", "feel cold", "sweat", "windy", "winter coat", "double glazing",
    "sea breeze", "land breeze", "vacuum flask", "thermos", "ice float", "wooden spoon", "metal spoon",
    "thrown vertically", "vertically upwards", "highest point", "railway track", "bimetallic",
    "painted white", "dull black", "shiny surface", "matt black", "irregular stone", "irregular lamina",
    "displacement can", "eureka can", "measuring cylinder", "plumb line", "skydiver", "astronaut",
}

# Explicit off-topic triggers (blocked unless strong physics context outweighs them)
STRONG_OFF_TOPIC_PATTERNS = [
    (re.compile(r"\b(?:recipe|bake|ingredients|biryani|pasta|pizza|cake|chicken|chocolate|how\s+to\s+cook|cook\s+(?:pasta|rice|chicken|steak|dinner|lunch))\b", re.I), "cooking & food"),
    (re.compile(r"\b(?:poem|poetry|sonnet|haiku|love\s+letter|bedtime\s+story|shakespeare|hamlet|macbeth|novel)\b", re.I), "literature & creative writing"),
    (re.compile(r"\b(?:javascript|typescript|react|nextjs|css|html|sql|mongodb|discord\s+bot|scrape|leetcode|dockerfile|kubernetes)\b", re.I), "programming & software development"),
    (re.compile(r"\b(?:world\s+war|roman\s+empire|napoleon|cleopatra|ottawa|capital\s+of|president\s+of|prime\s+minister|election|fifa|world\s+cup|champions\s+league|messi|ronaldo|nba|netflix|tiktok|minecraft|fortnite|gta)\b", re.I), "general trivia, history, politics, sports & entertainment"),
    (re.compile(r"\b(?:mitosis|meiosis|photosynthesis|chloroplast|digestive\s+system|insulin|glucagon|nephron|alveoli|transpiration|dna\s+replication|mrna|ribosome|enzyme\s+denatur)\b", re.I), "Biology (non-physics)"),
    (re.compile(r"\b(?:alkane|alkene|polymerisation|titration|phenolphthalein|haber\s+process|contact\s+process|covalent\s+bond|esterification|carboxylic\s+acid|homologous\s+series|electrolysis\s+of\s+brine)\b", re.I), "Chemistry (non-physics)"),
    (re.compile(r"\b(?:bitcoin|ethereum|crypto|stock\s+market|forex|horoscope|zodiac|astrology)\b", re.I), "finance & astrology"),
]

PHYSICS_FORMULA_PATTERN = re.compile(
    r"(?:\b(?:m/s\^?2|m/s|km/h|kg/m\^?3|g/cm\^?3|n\s*m|n/m\^?2|kpa|mpa|kj|mj|kw|kwh|ma|k\s*ohm|hz|khz|mhz)\b|"
    r"\b(?:v\s*=\s*u\s*\+\s*a\s*t|f\s*=\s*m\s*a|w\s*=\s*m\s*g|p\s*=\s*f\s*/\s*a|v\s*=\s*i\s*r|v\s*=\s*f\s*λ|e\s*=\s*m\s*c)\b)",
    re.I,
)

REFUSAL_MESSAGE = (
    "I am **Haroony AI** (Harony Physics Clinic), and I am trained **strictly for Physics** "
    "(IGCSE / A-Level concepts, calculations, past-paper photos, and Mark Scheme grading).\n\n"
    "I cannot answer questions outside Physics. Try asking me a Physics question instead — for example:\n"
    "- *\"Explain how a D.C. motor's split-ring commutator works\"*\n"
    "- *\"A 820 kg aircraft travels at 60 m/s. Calculate its kinetic energy.\"*\n"
    "- Or upload a **photo of your Physics answer + Mark Scheme** in the **Correct My Answer** tab!"
)


@dataclass
class GuardrailDecision:
    allowed: bool
    category: str  # "physics" | "greeting" | "off_topic" | "jailbreak"
    confidence: float
    reason: str
    refusal_message: Optional[str] = None


class PhysicsGuardrail:
    def __init__(self) -> None:
        self.pipeline = None
        self.load()

    def load(self) -> bool:
        if GUARDRAIL_MODEL_PATH.exists():
            try:
                self.pipeline = joblib.load(GUARDRAIL_MODEL_PATH)
                return True
            except Exception:
                self.pipeline = None
        return False

    def count_physics_signals(self, text: str) -> int:
        low = text.lower()
        hits = sum(1 for kw in PHYSICS_KEYWORDS if kw in low)
        if PHYSICS_FORMULA_PATTERN.search(low):
            hits += 2
        return hits

    def check(self, text: str, has_attachment: bool = False, has_prior_physics_context: bool = False) -> GuardrailDecision:
        cleaned = (text or "").strip()

        # If student uploaded a photo/MS with empty or short prompt ("mark this", ""), allow so OCR runs first!
        if not cleaned and has_attachment:
            return GuardrailDecision(True, "physics", 0.99, "Uploaded Physics attachment for analysis/grading.")

        # 1. Jailbreak / prompt-injection shield
        for jb in JAILBREAK_PATTERNS:
            if jb.search(cleaned):
                return GuardrailDecision(
                    False,
                    "jailbreak",
                    0.99,
                    "Prompt-injection / off-topic override attempt blocked.",
                    REFUSAL_MESSAGE,
                )

        # 2. Pure greeting / "who are you?"
        if GREETING_IDENTITY_RE.match(cleaned) and not has_attachment:
            return GuardrailDecision(True, "greeting", 0.98, "Greeting / Haroony AI identity question.")

        phys_hits = self.count_physics_signals(cleaned)

        # 3. Check explicit off-topic patterns (even if student tries sneaking a physics word into a poem/coding request)
        for pat, domain_label in STRONG_OFF_TOPIC_PATTERNS:
            if pat.search(cleaned):
                # Only allow if it's clearly a real physics word problem (e.g., "a pizza van of mass 1200 kg accelerates at 2 m/s^2")
                has_real_physics_problem = phys_hits >= 3 or bool(PHYSICS_FORMULA_PATTERN.search(cleaned))
                is_creative_or_code = domain_label in ("literature & creative writing", "programming & software development")
                if is_creative_or_code or not has_real_physics_problem:
                    return GuardrailDecision(
                        False,
                        "off_topic",
                        0.97,
                        f"Outside Physics domain ({domain_label}).",
                        REFUSAL_MESSAGE,
                    )

        # 4. Short follow-up in an ongoing physics session or with an uploaded photo
        if (has_attachment or has_prior_physics_context) and FOLLOWUP_PROBES_RE.match(cleaned):
            return GuardrailDecision(True, "physics", 0.95, "Contextual Physics follow-up or grading instruction.")

        # 5. Strong physics keyword presence -> fast-path allow
        if phys_hits >= 1:
            return GuardrailDecision(True, "physics", min(0.99, 0.78 + 0.06 * phys_hits), "Matched Physics curriculum terminology.")

        # 6. Trained ML Classifier (`models/physics_guardrail.joblib`)
        if self.pipeline is not None and cleaned:
            try:
                proba = float(self.pipeline.predict_proba([cleaned])[0][1])
                if proba >= 0.56:
                    return GuardrailDecision(True, "physics", round(proba, 3), "Classified as Physics by Haroony Guardrail ML model.")
                return GuardrailDecision(
                    False,
                    "off_topic",
                    round(1.0 - proba, 3),
                    "Classified as outside Physics by Haroony Guardrail ML model.",
                    REFUSAL_MESSAGE,
                )
            except Exception:
                pass

        # Conservative fallback if model isn't loaded yet: refuse if zero physics signals
        if has_attachment:
            return GuardrailDecision(True, "physics", 0.75, "Attachment present; proceeding to OCR physics check.")

        return GuardrailDecision(
            False,
            "off_topic",
            0.85,
            "No Physics concepts detected.",
            REFUSAL_MESSAGE,
        )
