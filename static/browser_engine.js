/* Haroony AI — Smart In-Browser Physics Engine + Symbolic Solver + Neural RAG/Grader
 * Works both standalone on GitHub Pages (https://haroony-ai.github.io) and alongside
 * HaroonyNeuralAI (WebLLM Qwen2.5-Instruct / Transformers.js WASM + PDF.js + Tesseract OCR).
 */
(function (global) {
  "use strict";

  let kbPromise = null;
  let kbData = null;
  let qaPromise = null;
  let qaData = null;

  const REFUSAL_MSG =
    "I'm **Haroony AI**, and I specialize exclusively in **Physics** (IGCSE & A-Level Physics tutoring).\n\n" +
    "Ask me any question on *Mechanics, Thermal Physics, Waves, Light, Electricity, Electromagnetism, Nuclear Physics, or Space Physics*, or attach/paste (`Ctrl+V`) a screenshot of a Physics problem!";

  const JAILBREAK_RE =
    /\b(ignore\s+(?:all\s+|previous\s+|prior\s+)?instructions|disregard\s+(?:your\s+)?rules|pretend\s+you\s+are|act\s+as\s+a\s+general|bypass\s+guardrail|reveal\s+system\s+prompt)\b/i;

  const GREETING_RE =
    /^(?:hi+|hello+|hey+|yo|good\s+(?:morning|afternoon|evening)|salam|assalamu\s*alaikum|who\s+are\s+you|what\s+can\s+you\s+do|help|thanks|thank\s+you)\s*[!?.]*$/i;

  const HARD_NON_PHYSICS_RE =
    /\b(recipe|bake|cake|pasta|pizza|biryani|chicken|burger|how\s+to\s+cook|poem|poetry|haiku|sonnet|romeo|juliet|shakespeare|fifa|world\s+cup|football\s+match|basketball|oscar|netflix|movie\s+review|horoscope|zodiac|astrology|bitcoin|crypto|ethereum|stock\s+market|forex|makeup|skincare|mitosis|meiosis|photosynthesis|chloroplast|digestive\s+system|nephron|auxin|enzyme|dna\s+replication|titration|haber\s+process|alkane|alkene|polymerisation|covalent\s+bond|electrolysis\s+of\s+brine|scrape|beautifulsoup|flexbox|react\s+component|sql\s+query|discord\s+bot|roman\s+empire|french\s+revolution|world\s+war|capital\s+of)\b/i;

  const STRONG_PHYSICS_RE =
    /\b(physics|forces?|newtons?|velocit(?:y|ies)|accelerations?|decelerations?|speeds?|distance|displacement|scalar|vector|free\s*fall|terminal\s+velocity|thrown\s+vertically|vertically\s+upwards|highest\s+point|skydiver|astronaut|momentum|impulse|collisions?|kinetic|potential\s+energy|gravitational|gpe|work\s+done|power|watts?|joules?|efficiency|densit(?:y|ies)|mass|volume|weight|gravity|moments?|torque|pivot|equilibrium|centre\s+of\s+gravity|center\s+of\s+mass|irregular\s+(?:stone|lamina)|plumb\s+line|displacement\s+can|hooke|spring\s+constant|extension|limit\s+of\s+proportionality|pressures?|pressure\s+cooker|pascal|barometer|manometer|hydraulic|boyle|kelvin|absolute\s+zero|brownian|evaporation|evaporate|boiling|melting|thermal|conduction|convection|infrared|radiation|emitter|absorber|painted\s+white|dull\s+black|railway\s+tracks?|bimetallic|specific\s+heat|latent\s+heat|swimming\s+pool|windy|sweat|feel\s+cold|winter\s+coat|double\s+glazing|vacuum\s+flask|thermos|sea\s+breeze|land\s+breeze|waves?|transverse|longitudinal|wavelengths?|frequenc(?:y|ies)|amplitude|period|hertz|wavefronts?|diffraction|refraction|reflection|total\s+internal|critical\s+angle|snell|refractive\s+index|lens(?:es)?|focal\s+length|converging|diverging|principal\s+focus|dispersion|monochromatic|electromagnetic\s+spectrum|microwaves?|ultraviolet|x-rays|gamma|ultrasound|echo(?:es)?|sonar|pitch|loudness|charges?|coulombs?|electrostatic|electric\s+field|currents?|amperes?|amps|voltages?|volts?|emf|potential\s+difference|resistance|ohms?|resistors?|filament\s+lamp|diode|ldr|thermistor|series\s+circuit|parallel\s+circuit|circuits?|potential\s+divider|kilowatt|kwh|fuse|earth\s+wire|circuit\s+breaker|magnets?|magnetic|solenoid|electromagnets?|fleming|motor\s+effect|commutator|electromagnetic\s+induction|lenz|faraday|generators?|slip\s+ring|transformers?|step-up|step-down|national\s+grid|atoms?|nucle(?:us|i)|protons?|neutrons?|electrons?|isotopes?|nuclide|alpha|beta|radioactive|radioactivity|ionising|ionizing|background\s+radiation|geiger|half-life|decay|fission|fusion|rutherford|gold\s+foil|solar\s+system|orbits?|orbital\s+speed|planets?|comets?|satellites?|geostationary|protostar|main\s+sequence|red\s+giant|supergiant|white\s+dwarf|supernova|neutron\s+star|black\s+hole|galax(?:y|ies)|milky\s+way|light-year|redshift|big\s+bang|cosmic\s+microwave|hubble|0625|9702|igcse|mark\s+scheme|harony|haroony)\b/i;

  const FOLLOWUP_PHYSICS_RE =
    /\b(example|another|explain\s+more|simpler|why|how\s+come|what\s+if|formula|equation|unit|units|step|steps|rearrange|double|halve|increase|decrease|graph|diagram|quiz|question|solve|calculate|check)\b/i;

  const STOPWORDS = new Set([
    "in", "on", "of", "to", "is", "it", "at", "by", "as", "or", "an", "be", "do", "no", "so", "if", "up", "vs",
    "me", "my", "we", "he", "us", "am", "go",
    "the", "and", "for", "with", "from", "that", "this", "what", "how", "why",
    "explain", "define", "state", "describe", "calculate", "find", "determine",
    "into", "onto", "are", "was", "were", "has", "have", "had", "when", "where", "which",
    "your", "about", "does", "can", "you", "give", "tell", "please", "physics",
    "question", "questions", "answer", "marks", "mark", "compare", "difference", "between",
    "law", "laws", "effect", "effects", "rule", "rules", "principle", "theory",
    "work", "works", "first", "second", "third", "fluid", "fluids", "step", "steps",
  ]);

  // Built-in Physics Concept Synthesizer fallback knowledge (used ONLY if WebGPU/WASM CDN is unreachable)
  const CORE_CONCEPTS = [
    {
      match: /\b(ohm'?s?\s+law|v\s*=\s*i\s*r|voltage.*current.*resistance)\b/i,
      title: "Ohm's Law & Electrical Resistance",
      summary:
        "**Ohm's Law** states that the electric current (`I`) flowing through an ohmic conductor is **directly proportional** to the potential difference (`V`) across its ends, provided **temperature (and other physical conditions) remain constant**.",
      formulas: ["V = I × R", "R = V / I", "I = V / R"],
      points: [
        "**Units:** Potential difference `V` in volts (`V`), current `I` in amperes (`A`), resistance `R` in ohms (`Ω`).",
        "**I–V Graph of an Ohmic Resistor:** Straight line passing directly through the origin `(0, 0)` with constant gradient (`1/R`).",
        "**Filament Lamp Exception:** As current increases, the filament heats up; lattice ions vibrate more vigorously, colliding more frequently with electrons, so **resistance increases** and the `I–V` curve flattens.",
      ],
    },
    {
      match: /\b(newton'?s?\s+(?:first|second|third|1st|2nd|3rd|laws?)|f\s*=\s*m\s*a|resultant\s+force)\b/i,
      title: "Newton's Laws of Motion & Resultant Force",
      summary:
        "**Newton's Laws** link resultant force to the motion and momentum of an object:",
      formulas: ["F = m × a", "F = Δp / Δt = (mv − mu) / t"],
      points: [
        "**First Law (Inertia):** An object remains at rest or continues at **constant velocity** (constant speed in a straight line) unless acted upon by a non-zero **resultant force**.",
        "**Second Law:** Resultant force is directly proportional to acceleration (`F = ma`), acting in the same direction as the acceleration.",
        "**Third Law:** Whenever body A exerts a force on body B, body B exerts an **equal in magnitude, opposite in direction** force of the **same type** on body A.",
      ],
    },
    {
      match: /\b(transformer|step-up|step-down|primary\s+coil|secondary\s+coil)\b/i,
      title: "Transformers & Electromagnetic Induction",
      summary:
        "A **transformer** changes the size of an **alternating voltage (a.c.)** using two coils wound around a **soft-iron core**.",
      formulas: ["Vp / Vs = Np / Ns", "Vp × Ip = Vs × Is  (100% efficient)", "Ploss = I² × R"],
      points: [
        "**How it works:** Alternating current in the primary coil produces a **continuously changing magnetic field** in the soft-iron core, which cuts the secondary coil and **induces an alternating e.m.f.** across it.",
        "**Why d.c. fails:** Direct current creates a **stationary (constant) magnetic field**, so no field lines continuously cut the secondary coil (`ΔΦ/Δt = 0`).",
        "**High-Voltage Transmission:** Stepping up voltage reduces current (`I = P/V`), drastically cutting thermal power wasted in cables (`P = I²R`).",
      ],
    },
    {
      match: /\b(refraction|snell|critical\s+angle|total\s+internal\s+reflection|optical\s+fibre)\b/i,
      title: "Refraction & Total Internal Reflection",
      summary:
        "**Refraction** is the bending of a wave when it passes from one medium into another due to a **change in wave speed** (while **frequency stays constant**).",
      formulas: ["n = sin i / sin r", "n = c / v", "sin c = 1 / n"],
      points: [
        "**Fast → Slow (e.g., air to glass):** Speed decreases, wavelength decreases, and the ray bends **towards the normal** (`r < i`).",
        "**Slow → Fast (e.g., glass to air):** Speed increases, wavelength increases, and the ray bends **away from the normal** (`r > i`).",
        "**Total Internal Reflection (TIR):** Occurs when light travels from a **more optically dense to a less dense medium** AND the angle of incidence **exceeds the critical angle (`i > c`)**.",
      ],
    },
    {
      match: /\b(moment|equilibrium|pivot|anticlockwise|clockwise)\b/i,
      title: "Moments & Conditions for Equilibrium",
      summary:
        "The **moment of a force** measures its turning effect about a pivot.",
      formulas: ["Moment = Force × perpendicular distance from pivot  (M = F × d)"],
      points: [
        "**Principle of Moments:** For an object in rotational equilibrium, **total clockwise moment = total anticlockwise moment** about the same pivot.",
        "**Two Conditions for Complete Equilibrium:** (1) No resultant force in any direction (`ΣF = 0`), and (2) No resultant moment about any point (`ΣM = 0`).",
        "**SI Unit:** Newton-metre (`N m`). Note: Moment is a vector (direction: clockwise or anticlockwise), distinct from work done in joules.",
      ],
    },
    {
      match: /\b(convection|conduction|thermal\s+radiation|heat\s+transmission)\b/i,
      title: "Thermal Energy Transfer (Conduction, Convection & Radiation)",
      summary:
        "Thermal energy transfers by **conduction** (mainly in solids), **convection** (in fluids: liquids and gases), and **infrared radiation** (even through a vacuum).",
      formulas: ["Density = Mass / Volume  (ρ = m / V)"],
      points: [
        "**Convection (Fluids):** Heated fluid **expands**, its **density decreases**, and the warmer, less dense fluid **rises**; cooler, denser fluid **sinks** to take its place, forming a **convection current**.",
        "**Conduction (Metals vs Non-metals):** Lattice ions vibrate faster and pass energy to neighbours; in metals, **delocalised (free) electrons** move rapidly through the lattice and collide with ions, transferring energy much faster.",
        "**Infrared Radiation:** **Dull, black surfaces** are the best emitters and absorbers; **shiny, silver/white surfaces** are poor emitters and good reflectors.",
      ],
    },
    {
      match: /\b(transverse|longitudinal|diffraction|wave\s+properties|ripple\s+tank)\b/i,
      title: "General Wave Properties & Diffraction",
      summary:
        "Waves transfer **energy without transferring matter**. They are classified as **transverse** or **longitudinal** based on particle vibration direction.",
      formulas: ["v = f × λ", "f = 1 / T"],
      points: [
        "**Transverse Waves (e.g. light, water, EM waves):** Particles vibrate **perpendicular (at 90°)** to the direction of wave energy propagation (forming crests and troughs).",
        "**Longitudinal Waves (e.g. sound):** Particles vibrate **parallel** to the direction of energy propagation, forming alternating **compressions** (high pressure) and **rarefactions** (low pressure).",
        "**Diffraction:** Waves spread out when passing through a gap or around an edge. **Maximum diffraction** occurs when the **gap width ≈ wavelength (`λ`)**; wavelength, frequency, and speed stay **unchanged**.",
      ],
    },
    {
      match: /\b(sound\s+waves?|echo(?:es)?|ultrasound|pitch|loudness)\b/i,
      title: "Sound Waves, Echoes & Ultrasound",
      summary:
        "Sound is a **longitudinal mechanical wave** requiring a medium (solid, liquid, or gas) and cannot travel through a vacuum.",
      formulas: ["v = 2d / t  (for an echo)", "Human hearing range: 20 Hz to 20,000 Hz"],
      points: [
        "**Pitch & Loudness:** Higher **frequency** gives a higher **pitch**; larger **amplitude** gives a **louder** sound.",
        "**Speed of Sound:** Fastest in **solids**, slower in **liquids**, slowest in **gases** (≈ `330–340 m/s` in air).",
        "**Ultrasound (`f > 20 kHz`):** Used for medical fetal scanning, SONAR depth measurement (`d = v × t / 2`), and non-destructive crack detection in metals.",
      ],
    },
    {
      match: /\b(brownian\s+motion|kinetic\s+particle|boyle'?s?\s+law|gas\s+pressure)\b/i,
      title: "Kinetic Particle Theory, Brownian Motion & Gas Laws",
      summary:
        "Matter consists of continuously moving particles whose average kinetic energy depends on **absolute (Kelvin) temperature**.",
      formulas: ["T(K) = θ(°C) + 273", "p₁ V₁ = p₂ V₂  (constant T)"],
      points: [
        "**Brownian Motion:** Random, jerky zig-zag motion of visible smoke/pollen particles caused by **fast-moving, invisible air/water molecules colliding randomly** with them from all directions.",
        "**Gas Pressure:** Caused by gas molecules colliding with the container walls and exerting a **force per unit area** (rate of change of momentum).",
        "**Heating a Fixed Volume of Gas:** Molecules move **faster**, collide with the walls **more frequently** and **more violently**, so pressure increases.",
      ],
    },
    {
      match: /\b(half[\s-]*life|nuclear\s+fission|nuclear\s+fusion|alpha.*beta.*gamma|radioactiv)\b/i,
      title: "Radioactivity, Half-Life, Fission & Fusion",
      summary:
        "Radioactive decay is **random and spontaneous** — unaffected by temperature, pressure, or chemical bonding.",
      formulas: ["N = N₀ × (½)^(t / t_½)"],
      points: [
        "**Half-Life (`t_½`):** The **average time taken** for half the unstable nuclei in a radioactive sample to decay (or for the corrected count rate to fall by half).",
        "**Nuclear Fission:** A heavy unstable nucleus (e.g. `U-235`) absorbs a neutron and **splits** into two smaller nuclei, releasing energy and more neutrons (used in nuclear power stations).",
        "**Nuclear Fusion:** Two light nuclei (e.g. hydrogen isotopes) **join together** at extremely high temperature and pressure to form a heavier nucleus (helium), releasing huge energy (powers the Sun & stars).",
      ],
    },
    {
      match: /\b(redshift|red[\s-]*shift|big\s+bang|hubble|cosmic\s+microwave|cmbr)\b/i,
      title: "Redshift, Hubble's Law & The Big Bang",
      summary:
        "Astronomical observations show that distant galaxies are moving away from us, providing direct evidence that the Universe is **expanding**.",
      formulas: ["v = H₀ × d", "Age of Universe ≈ 1 / H₀  (H₀ ≈ 2.2 × 10⁻¹⁸ s⁻¹)"],
      points: [
        "**Redshift:** Light from distant galaxies shows an **increase in observed wavelength** (shifted towards the red end of the spectrum), proving galaxies are receding; **further galaxies move faster**.",
        "**Big Bang Theory:** The Universe began ~13.8 billion years ago from a **hot, dense point** and has been expanding and cooling ever since.",
        "**CMBR Evidence:** **Cosmic Microwave Background Radiation** fills all of space uniformly in the microwave region — the cooled leftover thermal radiation from the early hot Universe.",
      ],
    },
  ];

  async function ensureKB() {
    if (kbData) return kbData;
    if (!kbPromise) {
      kbPromise = fetch("static/browser_kb.json")
        .then((r) => {
          if (!r.ok) throw new Error("Failed to load static/browser_kb.json");
          return r.json();
        })
        .then((d) => {
          kbData = d;
          return kbData;
        });
    }
    return kbPromise;
  }

  async function ensureQA() {
    if (qaData) return qaData;
    if (!qaPromise) {
      qaPromise = fetch("static/browser_qa.json")
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => [])
        .then((arr) => {
          qaData = Array.isArray(arr) ? arr : [];
          return qaData;
        });
    }
    return qaPromise;
  }

  function searchQuestionBank(qaList, query, topK = 2, preferredUnit = null) {
    if (!Array.isArray(qaList) || !qaList.length) return [];
    const qToks = Array.from(new Set(tokenize(query).filter((w) => !STOPWORDS.has(w) && w.length >= 3)));
    if (!qToks.length) return [];
    const isPaperCode = /\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+/i.test(query);
    const scored = [];
    for (const item of qaList) {
      if (preferredUnit && !isPaperCode && item.u && item.u !== preferredUnit) continue;
      const hay = `${item.ch || ""} ${(item.pc || []).join(" ")} ${item.q || ""}`.toLowerCase();
      let hits = 0;
      for (const tok of qToks) {
        if (hay.includes(tok)) hits += tok.length >= 6 || /\d/.test(tok) ? 1.8 : 1.0;
      }
      const ratio = hits / Math.max(qToks.length, 2);
      if (hits >= 2.2 && ratio >= 0.42) {
        scored.push({ s: ratio, item });
      }
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, topK);
  }

  function tokenize(s) {
    return String(s || "")
      .toLowerCase()
      .match(/[a-z0-9ΔθλραβγΩ²³]{2,}/g) || [];
  }

  function fmtNum(x, sf = 3) {
    if (!Number.isFinite(x)) return String(x);
    if (x === 0) return "0";
    const abs = Math.abs(x);
    if (abs >= 1e5 || (abs > 0 && abs < 1e-3)) {
      const p = x.toExponential(sf - 1);
      const [coef, exp] = p.split("e");
      return `${Number(coef)} × 10^${ parseInt(exp, 10) }`;
    }
    return String(Number(x.toPrecision(sf)));
  }

  // ==================== GUARDRAIL ====================
  function checkGuardrail(text, model, hasPriorContext) {
    const t = String(text || "").trim();
    if (!t) return { allowed: false, category: "empty", refusal: REFUSAL_MSG };
    if (JAILBREAK_RE.test(t)) return { allowed: false, category: "jailbreak", refusal: REFUSAL_MSG };
    if (GREETING_RE.test(t)) return { allowed: true, category: "greeting", refusal: "" };
    if (HARD_NON_PHYSICS_RE.test(t) && !STRONG_PHYSICS_RE.test(t)) {
      return { allowed: false, category: "off_topic", refusal: REFUSAL_MSG };
    }
    if (STRONG_PHYSICS_RE.test(t)) return { allowed: true, category: "physics", refusal: "" };

    // Physics equation / unit pattern (e.g., "12 V and 4 ohms", "20 m/s for 5 s", "500 g block")
    if (/\b\d+(?:\.\d+)?\s*(?:m\/s2?|km\/h|kg|cm3|m3|pa|kpa|hz|khz|mhz|kv|mv|ma|kw|wh|kwh|ohm|Ω|°c|kelvin|joules?|watts?|newtons?|volts?|amps?)\b/i.test(t)) {
      return { allowed: true, category: "physics", refusal: "" };
    }

    if (hasPriorContext && (FOLLOWUP_PHYSICS_RE.test(t) || tokenize(t).length <= 8)) {
      return { allowed: true, category: "followup", refusal: "" };
    }

    const toks = tokenize(t);
    if (model && model.weights && toks.length > 0) {
      let score = Number(model.intercept || 0);
      const unique = Array.from(new Set(toks));
      const norm = Math.sqrt(unique.length || 1);
      for (const w of unique) {
        if (model.weights[w] !== undefined) {
          score += model.weights[w] / norm;
        }
      }
      const prob = 1 / (1 + Math.exp(-score));
      if (prob >= (hasPriorContext ? 0.32 : 0.50)) {
        return { allowed: true, category: "physics", refusal: "" };
      }
    }
    return { allowed: false, category: "off_topic", refusal: REFUSAL_MSG };
  }

  // ==================== 25+ EQUATION SYMBOLIC PHYSICS SOLVER ====================
  function parseQty(text) {
    const supMap = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+" };
    const q = String(text || "")
      .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]/g, (c) => supMap[c] || c)
      .replace(/\s*[×x*]\s*10\s*\^?\s*([+-]?\d+)/gi, "e$1")
      .replace(/−/g, "-");

    function allMatches(re, scale = 1) {
      const arr = [];
      let m;
      while ((m = re.exec(q)) !== null) {
        const val = parseFloat(m[1]);
        if (Number.isFinite(val)) arr.push(val * scale);
      }
      return arr;
    }

    const masses = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kg\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*g\b(?!\s*\/\s*cm)/gi, 1e-3),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*tonnes?\b/gi, 1000),
    ];
    const speeds = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m\s*\/\s*s\b(?!\s*(?:2|²|\^2))/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*km\s*\/\s*h\b/gi, 1 / 3.6),
    ];
    const accels = allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m\s*\/\s*s\s*(?:2|²|\^2)\b/gi, 1);
    const forces = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kn\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:n|newtons?)\b(?!\s*m\b|\s*\/\s*(?:m|cm|kg))/gi, 1),
    ];
    const times = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:s|sec|seconds?)\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:min|minutes?)\b/gi, 60),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:h|hr|hours?)\b(?!\s*\b)/gi, 3600),
    ];
    const dists = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*km\b(?!\s*\/)/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m\b(?!\s*(?:\/|s\b|2|3|²|³|\^|a\b|v\b|hz|w\b))/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*cm\b(?!\s*(?:2|3|²|³|\^))/gi, 0.01),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mm\b(?!\s*(?:2|3|²|³|\^))/gi, 0.001),
    ];
    const energies = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mj\b/gi, 1e6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kj\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:j|joules?)\b(?!\s*\/)/gi, 1),
    ];
    const powers = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mw\b/gi, 1e6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kw\b(?!\s*h)/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:w|watts?)\b/gi, 1),
    ];
    const volts = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kv\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mv\b/gi, 1e-3),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:v|volts?)\b/gi, 1),
    ];
    const currents = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*ma\b/gi, 1e-3),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:a|amps?|amperes?)\b/gi, 1),
    ];
    const resistances = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:kΩ|kohm|kohms|kiloohms?)\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:Ω|ohms?)\b/gi, 1),
    ];
    const charges = allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?:c|coulombs?)\b(?!\s*(?:m\b|°))/gi, 1);
    const freqs = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mhz\b/gi, 1e6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*khz\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*hz\b/gi, 1),
    ];
    const areas = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])m\s*(?:2|²|\^2)\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])cm\s*(?:2|²|\^2)\b/gi, 1e-4),
    ];
    const volumes = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])km\s*(?:3|³|\^3)\b/gi, 1e9),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])m\s*(?:3|³|\^3)\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])cm\s*(?:3|³|\^3)\b/gi, 1e-6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(?<![a-zA-Z/])mm\s*(?:3|³|\^3)\b/gi, 1e-9),
    ];
    const pressures = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mpa\b/gi, 1e6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kpa\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*pa\b/gi, 1),
    ];
    let temps = allMatches(/([+-]?\d+(?:\.\d+)?)\s*°\s*c\b/gi, 1);
    if (temps.length === 0 && /\b(?:temp|heat|heated|cool|cooled|thermal|celsius|specific\s+heat|water|boil|melt)\b/i.test(q)) {
      temps = allMatches(/([+-]?\d+(?:\.\d+)?)\s*(?:°\s*)?(?:c|celsius)\b/gi, 1);
    }
    const shc = allMatches(/([+-]?\d+(?:\.\d+)?)\s*j\s*\/\s*\(?\s*kg/gi, 1);
    const densities = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?)\s*kg\s*\/\s*m\s*(?:3|³|\^3)/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?)\s*g\s*\/\s*cm\s*(?:3|³|\^3)/gi, 1000),
    ];
    const springK = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?)\s*n\s*\/\s*m\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?)\s*n\s*\/\s*cm\b/gi, 100),
    ];
    const turns = allMatches(/([+-]?\d+)\s*turns?\b/gi, 1);
    const angles = allMatches(/([+-]?\d+(?:\.\d+)?)\s*(?:°|degrees?)\b(?!\s*c)/gi, 1);

    return {
      masses, speeds, accels, forces, times, dists, energies, powers,
      volts, currents, resistances, charges, freqs, areas, volumes,
      pressures, temps, shc, densities, springK, turns, angles,
    };
  }

  function makeStepCard(topic, formula, given, sub, val, unit, note = "") {
    return {
      topic,
      formula,
      value: val,
      unit,
      markdown:
        `### ⚡ Verified Physics Calculation — ${topic}\n` +
        `- **1. Formula:** \`${formula}\`\n` +
        `- **2. Given (SI Units):** ${given}\n` +
        `- **3. Substitution & Working:** \`${sub}\`\n` +
        `- **4. Final Answer:** **\`${fmtNum(val, 3)} ${unit}\`** *(2 s.f.: \`${fmtNum(val, 2)} ${unit}\`)*` +
        (note ? `\n- *Note:* ${note}` : ""),
    };
  }

  function solvePhysicsProblem(rawText) {
    const text = String(rawText || "");
    const low = text.toLowerCase();
    const u = parseQty(text);
    const g = 9.8;

    // 1. Kinetic Energy (Ek = 1/2 m v^2)
    if (/\bkinetic\s+(?:energy|store)\b/i.test(low)) {
      if (u.masses.length && u.speeds.length) {
        const m = u.masses[0], v = u.speeds[0];
        const ek = 0.5 * m * v * v;
        return makeStepCard("Kinetic Energy", "Ek = ½ m v²", `m = ${fmtNum(m)} kg, v = ${fmtNum(v)} m/s`, `Ek = 0.5 × ${fmtNum(m)} × (${fmtNum(v)})² = ${fmtNum(ek, 4)} J`, ek, "J");
      }
      if (u.energies.length && u.masses.length && /\b(speed|velocity)\b/i.test(low)) {
        const ek = u.energies[0], m = u.masses[0];
        const v = Math.sqrt((2 * ek) / m);
        return makeStepCard("Speed from Kinetic Energy", "v = √(2 Ek / m)", `Ek = ${fmtNum(ek)} J, m = ${fmtNum(m)} kg`, `v = √(2 × ${fmtNum(ek)} / ${fmtNum(m)})`, v, "m/s");
      }
    }

    // 2. Gravitational Potential Energy (ΔEp = m g Δh)
    if (/\b(gravitational\s+potential|g\.?p\.?e\.?|potential\s+energy)\b/i.test(low) && u.masses.length && u.dists.length) {
      const m = u.masses[0], h = u.dists[0];
      const ep = m * g * h;
      return makeStepCard("Gravitational Potential Energy", "ΔEp = m g Δh", `m = ${fmtNum(m)} kg, g = 9.8 N/kg, Δh = ${fmtNum(h)} m`, `ΔEp = ${fmtNum(m)} × 9.8 × ${fmtNum(h)}`, ep, "J");
    }

    // 3. Specific Heat Capacity (ΔE = m c ΔT)
    if ((/\bspecific\s+heat\b/i.test(low) || u.shc.length || (u.temps.length >= 2 && /\bheat/i.test(low))) && u.masses.length && u.temps.length >= 2) {
      const m = u.masses[0];
      const c = u.shc.length ? u.shc[0] : 4200;
      const dt = Math.abs(u.temps[1] - u.temps[0]);
      const E = m * c * dt;
      return makeStepCard("Thermal Energy (Specific Heat Capacity)", "ΔE = m c Δθ", `m = ${fmtNum(m)} kg, c = ${c} J/(kg °C), Δθ = ${fmtNum(dt)} °C`, `ΔE = ${fmtNum(m)} × ${c} × ${fmtNum(dt)}`, E, "J");
    }

    // 4. Ohm's Law & Circuits (V = I R, Series / Parallel Resistors, Electric Power P = V I)
    if (/\b(parallel)\b/i.test(low) && u.resistances.length >= 2 && /\bresist/i.test(low)) {
      const inv = u.resistances.reduce((acc, r) => acc + 1 / r, 0);
      const rt = 1 / inv;
      const desc = u.resistances.map((r) => `1/${fmtNum(r)}`).join(" + ");
      return makeStepCard("Combined Parallel Resistance", "1/R_total = 1/R₁ + 1/R₂ + …", `Resistors: ${u.resistances.map((r) => fmtNum(r) + " Ω").join(", ")}`, `1/R_total = ${desc} ⇒ R_total = ${fmtNum(rt, 4)} Ω`, rt, "Ω");
    }
    if (/\b(series)\b/i.test(low) && u.resistances.length >= 2 && /\b(total|combined|effective)\s+resistance\b/i.test(low)) {
      const rt = u.resistances.reduce((a, b) => a + b, 0);
      return makeStepCard("Combined Series Resistance", "R_total = R₁ + R₂ + …", `Resistors: ${u.resistances.map((r) => fmtNum(r) + " Ω").join(", ")}`, `R_total = ${u.resistances.map((r) => fmtNum(r)).join(" + ")}`, rt, "Ω");
    }
    if (/\btransformer\b/i.test(low) && u.turns.length >= 2 && u.volts.length >= 1) {
      const np = u.turns[0], ns = u.turns[1], vp = u.volts[0];
      const vs = vp * (ns / np);
      return makeStepCard("Ideal Transformer Voltage", "Vp / Vs = Np / Ns", `Np = ${np} turns, Ns = ${ns} turns, Vp = ${fmtNum(vp)} V`, `Vs = ${fmtNum(vp)} × (${ns} / ${np})`, vs, "V");
    }
    if (/\b(power)\b/i.test(low) && u.volts.length && u.currents.length) {
      const v = u.volts[0], i = u.currents[0];
      return makeStepCard("Electrical Power", "P = I × V", `V = ${fmtNum(v)} V, I = ${fmtNum(i)} A`, `P = ${fmtNum(i)} × ${fmtNum(v)}`, v * i, "W");
    }
    if (/\b(power)\b/i.test(low) && u.currents.length && u.resistances.length) {
      const i = u.currents[0], r = u.resistances[0];
      return makeStepCard("Electrical Power Dissipated", "P = I² × R", `I = ${fmtNum(i)} A, R = ${fmtNum(r)} Ω`, `P = (${fmtNum(i)})² × ${fmtNum(r)}`, i * i * r, "W");
    }
    if (u.volts.length && u.resistances.length && /\bcurrent\b/i.test(low)) {
      const v = u.volts[0], r = u.resistances[0];
      return makeStepCard("Ohm's Law (Current)", "I = V / R", `V = ${fmtNum(v)} V, R = ${fmtNum(r)} Ω`, `I = ${fmtNum(v)} / ${fmtNum(r)}`, v / r, "A");
    }
    if (u.currents.length && u.resistances.length && /\b(voltage|potential\s+difference|p\.?d\.?|emf)\b/i.test(low)) {
      const i = u.currents[0], r = u.resistances[0];
      return makeStepCard("Ohm's Law (Potential Difference)", "V = I × R", `I = ${fmtNum(i)} A, R = ${fmtNum(r)} Ω`, `V = ${fmtNum(i)} × ${fmtNum(r)}`, i * r, "V");
    }
    if (u.volts.length && u.currents.length && /\bresistance\b/i.test(low)) {
      const v = u.volts[0], i = u.currents[0];
      return makeStepCard("Electrical Resistance", "R = V / I", `V = ${fmtNum(v)} V, I = ${fmtNum(i)} A`, `R = ${fmtNum(v)} / ${fmtNum(i)}`, v / i, "Ω");
    }

    // 5. Charge & Current (Q = I t)
    if (/\bcharge\b/i.test(low) && u.currents.length && u.times.length) {
      const i = u.currents[0], t = u.times[0];
      return makeStepCard("Electric Charge", "Q = I × t", `I = ${fmtNum(i)} A, t = ${fmtNum(t)} s`, `Q = ${fmtNum(i)} × ${fmtNum(t)}`, i * t, "C");
    }
    if (/\bcurrent\b/i.test(low) && u.charges.length && u.times.length) {
      const q = u.charges[0], t = u.times[0];
      return makeStepCard("Electric Current", "I = Q / t", `Q = ${fmtNum(q)} C, t = ${fmtNum(t)} s`, `I = ${fmtNum(q)} / ${fmtNum(t)}`, q / t, "A");
    }

    // 6. Waves & Echoes (v = f λ, d = v t / 2)
    if (/\b(echo|sonar|ultrasound|pulse\s+reflect)\b/i.test(low) && u.speeds.length && u.times.length) {
      const v = u.speeds[0], t = u.times[0];
      const d = (v * t) / 2;
      return makeStepCard("Echo Reflection Distance", "d = (v × t) / 2", `v = ${fmtNum(v)} m/s, round-trip time t = ${fmtNum(t)} s`, `d = (${fmtNum(v)} × ${fmtNum(t)}) / 2`, d, "m", "Divided by 2 because the wave travels there and back.");
    }
    if (/\b(wave|frequency|wavelength)\b/i.test(low)) {
      if (u.freqs.length && u.dists.length && /\b(speed|velocity)\b/i.test(low)) {
        const f = u.freqs[0], lam = u.dists[0];
        return makeStepCard("Wave Speed", "v = f × λ", `f = ${fmtNum(f)} Hz, λ = ${fmtNum(lam)} m`, `v = ${fmtNum(f)} × ${fmtNum(lam)}`, f * lam, "m/s");
      }
      if (u.speeds.length && u.freqs.length && /\bwavelength\b/i.test(low)) {
        const v = u.speeds[0], f = u.freqs[0];
        return makeStepCard("Wavelength", "λ = v / f", `v = ${fmtNum(v)} m/s, f = ${fmtNum(f)} Hz`, `λ = ${fmtNum(v)} / ${fmtNum(f)}`, v / f, "m");
      }
      if (u.speeds.length && u.dists.length && /\bfrequency\b/i.test(low)) {
        const v = u.speeds[0], lam = u.dists[0];
        return makeStepCard("Wave Frequency", "f = v / λ", `v = ${fmtNum(v)} m/s, λ = ${fmtNum(lam)} m`, `f = ${fmtNum(v)} / ${fmtNum(lam)}`, v / lam, "Hz");
      }
    }

    // 7. Optics: Snell's Law & Critical Angle
    if (/\bcritical\s+angle\b/i.test(low)) {
      const nMatch = text.match(/refractive\s+index\s*(?:is|of|=)?\s*(\d+(?:\.\d+)?)/i);
      if (nMatch) {
        const n = parseFloat(nMatch[1]);
        if (n > 1) {
          const cDeg = (Math.asin(1 / n) * 180) / Math.PI;
          return makeStepCard("Critical Angle", "sin c = 1 / n", `n = ${n}`, `c = sin⁻¹(1 / ${n})`, cDeg, "°");
        }
      }
    }
    if (/\brefractive\s+index\b/i.test(low) && u.angles.length >= 2) {
      const iDeg = Math.max(u.angles[0], u.angles[1]);
      const rDeg = Math.min(u.angles[0], u.angles[1]);
      const n = Math.sin((iDeg * Math.PI) / 180) / Math.sin((rDeg * Math.PI) / 180);
      return makeStepCard("Snell's Law (Refractive Index)", "n = sin i / sin r", `i = ${iDeg}°, r = ${rDeg}°`, `n = sin(${iDeg}°) / sin(${rDeg}°)`, n, "");
    }

    // 8. Mechanics: Newton's 2nd Law, Acceleration, Weight, Momentum, Hooke's Law, Moments, Work, Power, Efficiency, Density, Pressure
    if (/\bmomentum\b/i.test(low) && u.masses.length && u.speeds.length) {
      const m = u.masses[0], v = u.speeds[0];
      return makeStepCard("Linear Momentum", "p = m × v", `m = ${fmtNum(m)} kg, v = ${fmtNum(v)} m/s`, `p = ${fmtNum(m)} × ${fmtNum(v)}`, m * v, "kg m/s");
    }
    if (/\bmoment\b/i.test(low) && !/\bmomentum\b/i.test(low) && u.forces.length && u.dists.length) {
      const f = u.forces[0], d = u.dists[0];
      return makeStepCard("Moment of a Force", "M = F × d", `F = ${fmtNum(f)} N, perpendicular distance d = ${fmtNum(d)} m`, `M = ${fmtNum(f)} × ${fmtNum(d)}`, f * d, "N m");
    }
    if (/\b(spring|extension|hooke)\b/i.test(low)) {
      if (u.forces.length && u.dists.length && /\bspring\s+constant\b/i.test(low)) {
        const f = u.forces[0], x = u.dists[0];
        return makeStepCard("Hooke's Law (Spring Constant)", "k = F / x", `F = ${fmtNum(f)} N, x = ${fmtNum(x)} m`, `k = ${fmtNum(f)} / ${fmtNum(x)}`, f / x, "N/m");
      }
      if (u.springK.length && u.dists.length) {
        const k = u.springK[0], x = u.dists[0];
        return makeStepCard("Hooke's Law (Force)", "F = k × x", `k = ${fmtNum(k)} N/m, x = ${fmtNum(x)} m`, `F = ${fmtNum(k)} × ${fmtNum(x)}`, k * x, "N");
      }
    }
    if (/\bdensity\b/i.test(low)) {
      const rawG = text.match(/(\d+(?:\.\d+)?)\s*g\b(?!\s*\/)/i);
      const rawCm3 = text.match(/(\d+(?:\.\d+)?)\s*(?<![a-zA-Z/])cm\s*(?:3|³|\^3)\b/i);
      if (rawG && rawCm3 && !/\bkg\/m/i.test(low) && /\b(calculate|find|what\s+is)\s+(?:the\s+)?density\b/i.test(low)) {
        const mg = parseFloat(rawG[1]), vc = parseFloat(rawCm3[1]);
        return makeStepCard("Density", "ρ = m / V", `m = ${mg} g, V = ${vc} cm³`, `ρ = ${mg} / ${vc}`, mg / vc, "g/cm³", `Equivalent SI density = ${fmtNum((mg / vc) * 1000)} kg/m³.`);
      }
      if (u.densities.length && u.volumes.length && /\bmass\b/i.test(low)) {
        const rho = u.densities[0], v = u.volumes[0];
        return makeStepCard("Mass from Density & Volume", "m = ρ × V", `ρ = ${fmtNum(rho)} kg/m³, V = ${fmtNum(v)} m³`, `m = ${fmtNum(rho)} × ${fmtNum(v)}`, rho * v, "kg", "Converted km³/cm³ to m³ (`1 km³ = 10⁹ m³`).");
      }
      if (u.densities.length && u.masses.length && /\bvolume\b/i.test(low)) {
        const rho = u.densities[0], m = u.masses[0];
        return makeStepCard("Volume from Mass & Density", "V = m / ρ", `m = ${fmtNum(m)} kg, ρ = ${fmtNum(rho)} kg/m³`, `V = ${fmtNum(m)} / ${fmtNum(rho)}`, m / rho, "m³");
      }
      if (u.masses.length && u.volumes.length) {
        const m = u.masses[0], v = u.volumes[0];
        return makeStepCard("Density", "ρ = m / V", `m = ${fmtNum(m)} kg, V = ${fmtNum(v)} m³`, `ρ = ${fmtNum(m)} / ${fmtNum(v)}`, m / v, "kg/m³");
      }
    }
    if (/\bboyle/i.test(low) || (u.pressures.length >= 1 && u.volumes.length >= 2 && /\bpressure\b/i.test(low))) {
      if (u.pressures.length >= 1 && u.volumes.length >= 2) {
        const p1 = u.pressures[0], v1 = u.volumes[0], v2 = u.volumes[1];
        const p2 = (p1 * v1) / v2;
        return makeStepCard("Boyle's Law (Isothermal Gas)", "p₁ V₁ = p₂ V₂", `p₁ = ${fmtNum(p1)} Pa, V₁ = ${fmtNum(v1)} m³, V₂ = ${fmtNum(v2)} m³`, `p₂ = (${fmtNum(p1)} × ${fmtNum(v1)}) / ${fmtNum(v2)}`, p2, "Pa");
      }
    }
    if (/\bpressure\b/i.test(low)) {
      if (u.densities.length && u.dists.length) {
        const rho = u.densities[0], h = u.dists[0];
        return makeStepCard("Hydrostatic Liquid Pressure", "Δp = ρ g Δh", `ρ = ${fmtNum(rho)} kg/m³, g = 9.8 N/kg, Δh = ${fmtNum(h)} m`, `Δp = ${fmtNum(rho)} × 9.8 × ${fmtNum(h)}`, rho * g * h, "Pa");
      }
      if (u.forces.length && u.areas.length) {
        const f = u.forces[0], a = u.areas[0];
        return makeStepCard("Solid Contact Pressure", "p = F / A", `F = ${fmtNum(f)} N, A = ${fmtNum(a)} m²`, `p = ${fmtNum(f)} / ${fmtNum(a)}`, f / a, "Pa");
      }
    }
    if (/\befficiency\b/i.test(low) && (u.energies.length >= 2 || u.powers.length >= 2)) {
      const vals = u.energies.length >= 2 ? u.energies : u.powers;
      const total = Math.max(vals[0], vals[1]);
      const useful = Math.min(vals[0], vals[1]);
      const eff = (useful / total) * 100;
      return makeStepCard("Percentage Efficiency", "Efficiency = (Useful Output / Total Input) × 100%", `Useful = ${fmtNum(useful)}, Total Input = ${fmtNum(total)}`, `(${fmtNum(useful)} / ${fmtNum(total)}) × 100%`, eff, "%");
    }
    if (/\bwork\s+done\b/i.test(low) && u.forces.length && u.dists.length) {
      const f = u.forces[0], d = u.dists[0];
      return makeStepCard("Mechanical Work Done", "W = F × d", `F = ${fmtNum(f)} N, d = ${fmtNum(d)} m`, `W = ${fmtNum(f)} × ${fmtNum(d)}`, f * d, "J");
    }
    if (/\bpower\b/i.test(low) && u.energies.length && u.times.length) {
      const e = u.energies[0], t = u.times[0];
      return makeStepCard("Power", "P = ΔE / t", `ΔE = ${fmtNum(e)} J, t = ${fmtNum(t)} s`, `P = ${fmtNum(e)} / ${fmtNum(t)}`, e / t, "W");
    }
    if (/\bweight\b/i.test(low) && u.masses.length && !u.forces.length) {
      const m = u.masses[0];
      return makeStepCard("Weight in Gravitational Field", "W = m × g", `m = ${fmtNum(m)} kg, g = 9.8 N/kg`, `W = ${fmtNum(m)} × 9.8`, m * g, "N");
    }
    if (/\b(acceleration|deceleration)\b/i.test(low)) {
      if (u.speeds.length >= 2 && u.times.length) {
        const u0 = u.speeds[0], v1 = u.speeds[1], t = u.times[0];
        const a = (v1 - u0) / t;
        return makeStepCard("Acceleration", "a = (v − u) / t", `u = ${fmtNum(u0)} m/s, v = ${fmtNum(v1)} m/s, t = ${fmtNum(t)} s`, `a = (${fmtNum(v1)} − ${fmtNum(u0)}) / ${fmtNum(t)}`, a, "m/s²");
      }
      if (/from\s+rest/i.test(low) && u.speeds.length === 1 && u.times.length) {
        const v1 = u.speeds[0], t = u.times[0];
        return makeStepCard("Acceleration from Rest", "a = (v − u) / t", `u = 0 m/s, v = ${fmtNum(v1)} m/s, t = ${fmtNum(t)} s`, `a = ${fmtNum(v1)} / ${fmtNum(t)}`, v1 / t, "m/s²");
      }
      if (u.forces.length && u.masses.length) {
        const fNet = u.forces.length >= 2 ? Math.abs(u.forces[0] - u.forces[1]) : u.forces[0];
        const m = u.masses[0];
        return makeStepCard("Newton's Second Law (Acceleration)", "a = F_resultant / m", `F = ${fmtNum(fNet)} N, m = ${fmtNum(m)} kg`, `a = ${fmtNum(fNet)} / ${fmtNum(m)}`, fNet / m, "m/s²");
      }
    }
    if (/\b(force)\b/i.test(low) && u.masses.length && u.accels.length) {
      const m = u.masses[0], a = u.accels[0];
      return makeStepCard("Newton's Second Law (Resultant Force)", "F = m × a", `m = ${fmtNum(m)} kg, a = ${fmtNum(a)} m/s²`, `F = ${fmtNum(m)} × ${fmtNum(a)}`, m * a, "N");
    }
    // Multi-segment Average Speed vs Average Velocity (e.g. "100 m north then 100 m south in 20 s")
    const dirMatches = [...text.matchAll(/([+-]?\d+(?:\.\d+)?)\s*(km|cm|mm|m)\s+(north|south|east|west|up|down|forward|forwards|back|backward|backwards|left|right)\b/gi)];
    if (dirMatches.length >= 2 && u.times.length && /\b(speed|velocity)\b/i.test(low)) {
      let dx = 0, dy = 0, totalD = 0;
      const segs = [];
      for (const dm of dirMatches) {
        const val = parseFloat(dm[1]);
        const ul = dm[2].toLowerCase();
        const dM = ul === "km" ? val * 1000 : ul === "cm" ? val / 100 : ul === "mm" ? val / 1000 : val;
        totalD += dM;
        const dl = dm[3].toLowerCase();
        if (dl === "north" || dl === "up" || dl === "forward" || dl === "forwards") dy += dM;
        else if (dl === "south" || dl === "down" || dl === "back" || dl === "backward" || dl === "backwards") dy -= dM;
        else if (dl === "east" || dl === "right") dx += dM;
        else if (dl === "west" || dl === "left") dx -= dM;
        segs.push(`${fmtNum(dM)} m ${dl}`);
      }
      const netDisp = Math.hypot(dx, dy);
      const tTot = u.times[u.times.length - 1];
      if (tTot > 0) {
        const avgSpd = totalD / tTot;
        const avgVel = netDisp / tTot;
        const askVelOnly = /\bvelocity\b/i.test(low) && !/\bspeed\b/i.test(low);
        return {
          topic: "Average Speed vs Average Velocity",
          formula: "Average Speed = Total Distance / Time | Average Velocity = Net Displacement / Time",
          value: askVelOnly ? avgVel : avgSpd,
          unit: "m/s",
          markdown:
            `### ⚡ Verified Physics Calculation — Average Speed & Average Velocity\n` +
            `- **1. Given:** ${segs.join(", ")}, total time = \`${fmtNum(tTot)} s\`\n` +
            `- **2. Average Speed *(scalar — total path length)*:** \`Total distance = ${fmtNum(totalD)} m\` → \`${fmtNum(totalD)} / ${fmtNum(tTot)}\` = **\`${fmtNum(avgSpd, 3)} m/s\`**\n` +
            `- **3. Average Velocity *(vector — net displacement)*:** \`Net displacement = ${fmtNum(netDisp)} m\` → \`${fmtNum(netDisp)} / ${fmtNum(tTot)}\` = **\`${fmtNum(avgVel, 3)} m/s\`**\n` +
            `- *Note:* Speed adds all distances regardless of direction; opposite directions cancel in displacement.`,
        };
      }
    }
    // Orbital Speed: v = 2πr / T (must precede single-segment v = d / t)
    if (/\b(orbital\s+speed|orbit)\b/i.test(low) && u.dists.length && u.times.length) {
      const r = u.dists[0], t = u.times[0];
      if (t > 0) {
        const vOrb = (2 * Math.PI * r) / t;
        return makeStepCard("Orbital Speed (Unit 7: Space Physics)", "v = 2π r / T", `r = ${fmtNum(r)} m, T = ${fmtNum(t)} s`, `v = (2 × π × ${fmtNum(r)}) / ${fmtNum(t)}`, vOrb, "m/s");
      }
    }
    if (/\b(average\s+speed|speed)\b/i.test(low) && u.dists.length === 1 && u.times.length && !u.freqs.length && !u.speeds.length && !/\b(velocity|orbit)\b/i.test(low)) {
      const d = u.dists[0], t = u.times[0];
      return makeStepCard("Average Speed", "v = d / t", `d = ${fmtNum(d)} m, t = ${fmtNum(t)} s`, `v = ${fmtNum(d)} / ${fmtNum(t)}`, d / t, "m/s");
    }
    if (/\bdistance\b/i.test(low) && u.speeds.length && u.times.length) {
      const v = u.speeds[0], t = u.times[0];
      return makeStepCard("Distance Travelled", "d = v × t", `v = ${fmtNum(v)} m/s, t = ${fmtNum(t)} s`, `d = ${fmtNum(v)} × ${fmtNum(t)}`, v * t, "m");
    }

    // 9. Nuclear Half-Life
    if (/\bhalf[\s-]*life\b/i.test(low)) {
      const hlMatch = text.match(/half[\s-]*life\s+(?:of\s+|is\s+)?(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?|years?)/i);
      const afterMatch = text.match(/after\s+(\d+(?:\.\d+)?)\s*(seconds?|minutes?|hours?|days?|years?)/i);
      const initMatch = text.match(/(\d+(?:\.\d+)?)\s*(bq|counts?(?:\/s|\/min)?|g|mg|kg|nuclei|atoms)/i);
      if (hlMatch && afterMatch && initMatch && hlMatch[2].toLowerCase().slice(0, 3) === afterMatch[2].toLowerCase().slice(0, 3)) {
        const tHalf = parseFloat(hlMatch[1]);
        const tTotal = parseFloat(afterMatch[1]);
        const n0 = parseFloat(initMatch[1]);
        const unit = initMatch[2];
        const nHalves = tTotal / tHalf;
        const rem = n0 * Math.pow(0.5, nHalves);
        return makeStepCard("Radioactive Half-Life Decay", "N = N₀ × (½)^(t / t_½)", `N₀ = ${n0} ${unit}, t_½ = ${tHalf} ${hlMatch[2]}, t = ${tTotal} ${afterMatch[2]} (${fmtNum(nHalves)} half-lives)`, `N = ${n0} × (0.5)^${fmtNum(nHalves)}`, rem, unit);
      }
    }

    return null;
  }

  // ==================== RELEVANCE-FILTERED HARONY 2026/2027 SEARCH ====================
  const CHUNK_SYNONYMS = {
    ohm: ["resistance", "voltage", "current", "circuits", "electric"],
    ohms: ["resistance", "voltage", "current", "circuits", "electric"],
    newton: ["forces", "inertia", "acceleration", "kinematics", "mechanics"],
    convection: ["heat", "transmission", "thermal", "density"],
    conduction: ["heat", "transmission", "thermal", "electrons"],
    brownian: ["kinetic", "particle", "theory", "molecules", "thermal"],
    diffraction: ["waves", "water", "wavefronts", "gap"],
    redshift: ["universe", "hubble", "big", "bang", "astronomy"],
    bang: ["universe", "redshift", "astronomy"],
    transformer: ["transformers", "electromagnetism", "primary", "secondary"],
  };

  function searchChunksScored(kb, query, topK) {
    const baseToks = tokenize(query).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
    if (!baseToks.length) return [];
    const expanded = [];
    for (const t of baseToks) {
      if (!expanded.includes(t)) expanded.push(t);
      for (const syn of CHUNK_SYNONYMS[t] || []) {
        if (!expanded.includes(syn)) expanded.push(syn);
      }
    }
    const uniqueQ = expanded;
    const isPaperCode = /\b20\d{2}\b|\bp[24]\b|\bq\d+/i.test(query);

    const scored = [];
    for (const ch of kb.chunks) {
      const topicWords = new Set(tokenize((ch.unit || "") + " " + (ch.topic || "")));
      const bodyWords = new Set(tokenize(ch.text || ""));
      let s = 0;
      let matchedBase = 0;
      for (const tok of uniqueQ) {
        const isBase = baseToks.includes(tok);
        let hit = false;
        if (topicWords.has(tok)) {
          s += isBase ? 4.2 : 2.0;
          hit = true;
        }
        if (bodyWords.has(tok)) {
          s += isBase ? 1.8 : 0.8;
          hit = true;
        }
        if (hit && isBase) matchedBase++;
      }
      const coverage = matchedBase / baseToks.length;
      if (!isPaperCode && ch.source === "Harony Notes 2026 Final" && s > 0) s += 1.2;
      if (s >= 2.8 && (coverage >= 0.4 || matchedBase >= 2)) {
        scored.push({ s, coverage, ch });
      }
    }
    scored.sort((a, b) => b.s - a.s);
    if (scored.length && scored[0].s >= 4.0) {
      const primaryUnit = scored[0].ch.unit;
      const locked = scored.filter((x) => !primaryUnit || x.ch.unit === primaryUnit);
      return locked.slice(0, topK);
    }
    return scored.slice(0, topK);
  }

  function searchMsTips(kb, query, topK = 2) {
    const qToks = Array.from(new Set(tokenize(query).filter((w) => !STOPWORDS.has(w) && w.length >= 4)));
    if (!qToks.length || !kb.mark_schemes) return [];
    const scored = [];
    for (const ms of kb.mark_schemes) {
      for (const q of ms.questions || []) {
        const hay = ((q.title || "") + " " + (q.summary_answer || "")).toLowerCase();
        let hits = 0;
        for (const t of qToks) {
          if (hay.includes(t)) hits++;
        }
        if (hits >= 2 || (qToks.length === 1 && hits === 1 && (q.title || "").toLowerCase().includes(qToks[0]))) {
          scored.push({ hits, msTitle: ms.title, q });
        }
      }
    }
    scored.sort((a, b) => b.hits - a.hits);
    return scored.slice(0, topK);
  }

  const VIDEO_IGNORE = new Set([
    "youtube", "video", "videos", "link", "links", "watch", "solve", "solving",
    "solution", "solutions", "give", "send", "show", "find", "please", "any",
    "topic", "topics", "about", "with", "from", "can", "you", "get", "me", "want", "need",
  ]);

  const VIDEO_SYNONYMS = {
    convection: ["heat", "transmission"],
    conduction: ["heat", "transmission"],
    radiation: ["heat", "transmission"],
    transverse: ["waves", "water"],
    longitudinal: ["waves", "water", "sound"],
    transformer: ["transformers"],
    "step-up": ["transformers"],
    "step-down": ["transformers"],
    ohm: ["electric", "circuits", "resistance"],
    ohms: ["electric", "circuits", "resistance"],
    newton: ["forces", "kinematics"],
    inertia: ["forces"],
    redshift: ["universe", "redshift", "hubble"],
    bang: ["universe"],
    optical: ["refraction", "total", "internal"],
    fibres: ["refraction", "total", "internal"],
    fibre: ["refraction", "total", "internal"],
    tir: ["total", "internal", "reflection"],
    echo: ["sound"],
    echoes: ["sound"],
    ultrasound: ["sound"],
    half: ["half", "life"],
    halflife: ["half", "life"],
    fission: ["alpha", "beta", "gamma", "nuclear", "fission"],
    fusion: ["nuclear", "fusion", "stars"],
  };

  function searchVideos(kb, query, topK, preferredUnit = null, preferredTopic = null) {
    const rawToks = tokenize(query).filter((w) => !STOPWORDS.has(w) && !VIDEO_IGNORE.has(w) && w.length >= 2);
    const qToks = [];
    for (const raw of rawToks) {
      const t = raw === "may" ? "jun" : raw;
      if (!qToks.includes(t)) qToks.push(t);
      for (const syn of VIDEO_SYNONYMS[t] || []) {
        if (!qToks.includes(syn)) qToks.push(syn);
      }
    }
    if (!qToks.length && preferredTopic) {
      for (const t of tokenize(preferredTopic)) {
        if (!STOPWORDS.has(t) && t.length >= 3 && !qToks.includes(t)) qToks.push(t);
      }
    }
    if (!qToks.length || !kb.videos) return [];

    const isPaperCode = /\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+/i.test(query);
    const scored = [];
    for (const v of kb.videos) {
      const vUnit = v.unit || "";
      if (preferredUnit && !isPaperCode && vUnit && vUnit !== preferredUnit) continue;

      const labelWords = new Set(tokenize(v.label || ""));
      const nkWords = new Set(tokenize(v.norm_key || ""));
      const chapWords = new Set(tokenize(v.chapter || ""));
      const unitWords = new Set(tokenize(vUnit));

      let hits = 0;
      let matchedDistinct = 0;
      for (const t of qToks) {
        if (labelWords.has(t) || nkWords.has(t)) {
          hits += 3.0;
          matchedDistinct++;
        } else if (chapWords.has(t)) {
          hits += 2.2;
          matchedDistinct++;
        } else if (unitWords.has(t) && t.length >= 4) {
          hits += 1.0;
        }
      }
      if (hits >= 2.2 && matchedDistinct >= 1) {
        scored.push({ hits, v });
      }
    }
    scored.sort((a, b) => b.hits - a.hits);
    const out = [];
    const seen = new Set();
    for (const item of scored) {
      if (item.v.video_url && !seen.has(item.v.video_url)) {
        seen.add(item.v.video_url);
        out.push(item.v);
        if (out.length >= topK) break;
      }
    }
    return out;
  }

  function cleanNoteBullets(rawText, topic) {
    const lines = String(rawText || "")
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 4 && !/^\d{1,3}$/.test(l));
    const merged = [];
    for (let ln of lines) {
      ln = ln.replace(/^[•o\-*]\s*/, "").trim();
      if (!ln) continue;
      if (merged.length < 2 && ln.length < 36 && topic && ln.toLowerCase() === topic.toLowerCase()) continue;
      // Skip slide outline header items like "1- Explain evaporation in terms of kinetic theory."
      if (/^\d+\s*[-.)]\s*(?:explain|compare|how|examples|define|state|what|list)\b/i.test(ln)) continue;
      if (
        merged.length &&
        !/[.:?!)]$/.test(merged[merged.length - 1]) &&
        (/^[a-z]/.test(ln) || /\b(of|to|in|the|a|an|and|or|is|are|with|by|from|that|as)$/i.test(merged[merged.length - 1]))
      ) {
        merged[merged.length - 1] += " " + ln;
      } else {
        merged.push(ln);
      }
    }
    return merged;
  }

  function formatNoteRefs(scoredChunks) {
    return (scoredChunks || [])
      .map((x) => x.ch)
      .filter((c) => c && (c.image_url || c.text))
      .map((c) => ({
        id: c.id,
        page: c.page,
        unit: c.unit,
        topic: c.topic,
        source: c.source,
        image_url: c.image_url,
        snippet: (c.text || "").slice(0, 280),
        revision_points: cleanNoteBullets(c.text, c.topic).slice(0, 4),
      }));
  }

  function matchConceptAtlas(kb, query) {
    const atlas = kb && Array.isArray(kb.concept_atlas) ? kb.concept_atlas : [];
    if (!atlas.length || !query) return null;
    const low = String(query).toLowerCase();
    const qWords = tokenize(low);
    const stem = (w) => (w.length >= 5 && w.endsWith("s") ? w.slice(0, -1) : w);
    const qStems = new Set(qWords.map(stem));
    let best = null;
    let bestScore = 0;
    for (const entry of atlas) {
      let sc = 0;
      for (const kw of entry.keywords || []) {
        if (low.includes(kw)) {
          const wc = kw.split(/\s+/).length;
          sc += wc >= 2 ? 3.8 * wc : 2.6;
        }
      }
      for (const tw of tokenize(entry.topic || "")) {
        if (tw.length >= 4 && (qStems.has(tw) || qStems.has(stem(tw)))) sc += 1.6;
      }
      if (sc > bestScore) {
        bestScore = sc;
        best = entry;
      }
    }
    return bestScore >= 3.0 ? best : null;
  }

  // ==================== NEURAL TUTOR PROMPT BUILDER ====================
  function buildTutorMessages(userQuestion, history, solved, scoredChunks, msTips, ocrText, conceptEntry) {
    const ctxLines = [];
    if (solved) {
      ctxLines.push(
        `VERIFIED EXACT CALCULATION (use these exact steps & numbers — do NOT recalculate differently):\n` +
          `Formula: ${solved.formula}\nResult: ${fmtNum(solved.value, 3)} ${solved.unit} (2 s.f.: ${fmtNum(solved.value, 2)} ${solved.unit})\n${solved.markdown}`
      );
    }
    if (conceptEntry) {
      ctxLines.push(
        `[Harony Concept Guide — ${conceptEntry.topic} (${conceptEntry.unit})]: ${conceptEntry.summary} ` +
          (conceptEntry.points || []).join(" ") +
          (conceptEntry.formulas && conceptEntry.formulas.length ? ` Formulas: ${conceptEntry.formulas.join(", ")}` : "")
      );
    }
    for (const item of scoredChunks.slice(0, 2)) {
      const ch = item.ch;
      const cleanTxt = cleanNoteBullets(ch.text, ch.topic).slice(0, 6).join("; ");
      ctxLines.push(`[Harony Notes — ${ch.unit} / ${ch.topic} (p.${ch.page || "?"})]: ${cleanTxt}`);
    }
    for (const mt of msTips.slice(0, 2)) {
      ctxLines.push(`[Cambridge Mark Scheme Phrase (${mt.msTitle} Q${mt.q.question_id})]: ${mt.q.summary_answer}`);
    }

    const sysPrompt =
      "You are Haroony AI, Mr. Haroony's expert, friendly IGCSE & A-Level Physics tutor from Harony Physics Clinic.\n" +
      "STRICT RULES:\n" +
      "1. Answer ONLY Physics questions. Never answer non-Physics topics.\n" +
      "2. REASON FROM FIRST PRINCIPLES FOR ANY QUESTION: Answer the student's EXACT question directly, accurately, and naturally in your own words — even if it is a novel scenario not word-for-word in the reference context. NEVER parrot unrelated slide fragments or force an irrelevant snippet.\n" +
      "3. For calculations: state the formula, convert given values to SI units, show clear substitution, and give the final answer with proper SI units (and 2–3 significant figures). If a VERIFIED EXACT CALCULATION is supplied in context, use its exact numbers.\n" +
      "4. Highlight key Cambridge IGCSE 0625 examiner phrases in **bold** so the student scores full marks.\n" +
      "5. Keep formatting crisp with short sections or bullet points.";

    const msgs = [{ role: "system", content: sysPrompt }];
    if (Array.isArray(history)) {
      for (const h of history.slice(-4)) {
        if (h && (h.role === "user" || h.role === "assistant") && h.content) {
          msgs.push({ role: h.role, content: String(h.content).slice(0, 700) });
        }
      }
    }

    let userContent = userQuestion;
    if (ocrText) {
      userContent = `Student note: ${userQuestion || "Please solve/explain this Physics image/document."}\n\nExtracted text from student's uploaded Physics image/PDF:\n${ocrText.slice(0, 1400)}`;
    }
    if (ctxLines.length) {
      userContent += `\n\n---\nReference Physics Context (use only if directly relevant to the student's question):\n${ctxLines.join("\n")}`;
    }
    msgs.push({ role: "user", content: userContent });
    return msgs;
  }

  function tryReasonPhysicsVariation(query) {
    const low = String(query || "").toLowerCase();

    // Everyday & Core Causal Physics ("Why does...", "Explain why...")
    if (low.includes("pressure cooker")) {
      return (
        "### ⚡ Why a Pressure Cooker Cooks Food Faster *(Unit 2: Thermal Physics)*\n\n" +
        "- A pressure cooker seals in steam so the **gas pressure above the water rises** above atmospheric pressure.\n" +
        "- Higher pressure **raises the boiling point of water** above `100 °C` (typically `~120 °C`).\n" +
        "- Water and steam reach a **higher temperature**, transferring thermal energy into food faster so it cooks in less time."
      );
    }
    if ((/\b(swimming|pool|sweat|wet|out\s+of\s+water|step\s+out)\b/.test(low) && /\b(cold|cool|wind|windy|breeze)\b/.test(low)) || (low.includes("wind") && low.includes("evaporat"))) {
      return (
        "### ⚡ Evaporation Cooling & Effect of Wind *(Unit 2: Thermal Physics)*\n\n" +
        "- Water on your skin **evaporates**: the **fastest, highest-kinetic-energy molecules** overcome intermolecular forces and escape from the liquid surface.\n" +
        "- This lowers the **average kinetic energy** of the remaining water molecules (absorbing latent heat from your skin), causing **cooling**.\n" +
        "- **Wind** sweeps away humid water vapour above the skin, **increasing the rate of evaporation** and cooling you faster."
      );
    }
    if ((low.includes("ice") && low.includes("float")) || (low.includes("ice") && low.includes("dense"))) {
      return (
        "### ⚡ Why Ice Floats on Water *(Unit 1 & 2)*\n\n" +
        "- When water freezes, its molecules form an open lattice taking up **more volume** for the same mass.\n" +
        "- Ice therefore has a **lower density** (`≈ 0.92 g/cm³`) than liquid water (`1.0 g/cm³`), so it floats."
      );
    }
    if ((low.includes("spoon") || low.includes("metal")) && (low.includes("wood") || low.includes("plastic")) && /\b(cold|colder|warm|hot|touch|feel)\b/.test(low)) {
      return (
        "### ⚡ Why Metal Feels Colder Than Wood at Room Temperature *(Unit 2: Conduction)*\n\n" +
        "- Both are at the **same room temperature**, but metal has **delocalised (free) electrons** making it a **good thermal conductor**.\n" +
        "- Metal conducts thermal energy **away from your warm hand much faster** than wood (an insulator)."
      );
    }
    if (/\b(coat|sweater|blanket|jacket|double\s+glazing|fur|feathers)\b/.test(low) && /\b(warm|heat|insulat|cold)\b/.test(low)) {
      return (
        "### ⚡ How Coats & Double Glazing Reduce Heat Loss *(Unit 2: Thermal Physics)*\n\n" +
        "- Wool, feathers, and double glazing **trap small pockets of air**.\n" +
        "- **Air is a poor thermal conductor** (reduces conduction) and trapped pockets prevent air circulating (**reduces convection**)."
      );
    }
    if (low.includes("bulb") && low.includes("parallel") && /\b(bright|brightness|same|add)\b/.test(low)) {
      return (
        "### ⚡ Adding a Bulb in Parallel *(Unit 3: Electric Circuits)*\n\n" +
        "- In a **parallel circuit**, every branch sits directly across the **full supply potential difference (voltage)**.\n" +
        "- Adding another bulb in parallel leaves the p.d. across existing bulbs **unchanged**, so they keep the **same brightness** (while total battery current increases)."
      );
    }
    if (low.includes("total internal reflection") || (/\btir\b/.test(low) && low.includes("condition"))) {
      return (
        "### ⚡ Total Internal Reflection (TIR) *(Unit 6: Light & Optics)*\n\n" +
        "- **Definition:** Complete reflection of light back inside a denser medium at the boundary with a less dense medium.\n" +
        "- **Two Conditions:** (1) Light travels from an **optically denser to a less dense medium** (e.g. glass → air), AND (2) **angle of incidence > critical angle (`i > c`)**, where `sin c = 1 / n`.\n" +
        "- **Uses:** Optical fibres (telecommunications & medical endoscopes) and reflecting prisms."
      );
    }
    if (/\b(thrown\s+vertically|vertically\s+upwards|highest\s+point)\b/.test(low)) {
      return (
        "### ⚡ Ball Thrown Vertically Upwards *(Unit 1: Kinematics & Gravity)*\n\n" +
        "- While rising, **weight acts downwards**, so the ball **decelerates at `9.8 m/s²`** as kinetic energy converts into gravitational potential energy.\n" +
        "- **At the highest point:** velocity = **`0 m/s`**, kinetic energy = **`0 J`**, GPE is maximum, and **acceleration is STILL `9.8 m/s²` downwards** (not zero!)."
      );
    }
    if (low.includes("railway") && (low.includes("gap") || low.includes("track") || low.includes("expand"))) {
      return (
        "### ⚡ Gaps in Railway Tracks *(Unit 2: Thermal Expansion)*\n\n" +
        "- On hot days, metal rails heat up and **expand (increase in length)** as particles vibrate with larger amplitude.\n" +
        "- Expansion gaps allow the rails to expand safely without **buckling or bending**."
      );
    }
    if ((low.includes("white") && /\b(houses?|buildings?|hot\s+countr|paint)/i.test(low)) || (low.includes("dull black") && low.includes("shiny"))) {
      return (
        "### ⚡ Why Houses in Hot Countries Are Painted White *(Unit 2: Infrared Radiation)*\n\n" +
        "- **White / shiny surfaces** are **poor absorbers** and **good reflectors** of **infrared radiation** from the Sun, keeping buildings cooler.\n" +
        "- *(Dull black surfaces are the best absorbers and best emitters of IR radiation.)*"
      );
    }
    if (low.includes("steel") && low.includes("ship") && low.includes("float")) {
      return (
        "### ⚡ Why a Steel Ship Floats on Water *(Unit 1: Density)*\n\n" +
        "- Solid steel is denser than water (`~7800 kg/m³`), but a ship is **hollow with large air spaces**.\n" +
        "- Its **overall average density (`total mass ÷ total volume`) is less than water (`1000 kg/m³`)**, so it floats."
      );
    }
    if (low.includes("irregular") && /\b(stone|rock|object)\b/.test(low) && low.includes("density")) {
      return (
        "### ⚡ Density of an Irregular Stone *(Unit 1: Practical)*\n\n" +
        "1. Measure **mass `m`** on a balance.\n" +
        "2. Read initial water volume `V₁` in a measuring cylinder (bottom of meniscus at eye level), submerge the stone gently, and read `V₂` (`V = V₂ − V₁`) *(or use a displacement can)*.\n" +
        "3. Calculate **`ρ = m / V`**."
      );
    }
    if (low.includes("irregular") && low.includes("lamina")) {
      return (
        "### ⚡ Centre of Gravity of an Irregular Lamina *(Unit 1: Practical)*\n\n" +
        "1. Suspend the lamina freely from a pin near the edge and hang a **plumb line** from the same pin.\n" +
        "2. Mark the vertical line, repeat from another hole, and find where the **lines intersect** → **centre of gravity**."
      );
    }
    if (low.includes("vacuum flask") || low.includes("thermos")) {
      return (
        "### ⚡ How a Vacuum Flask Reduces Heat Loss *(Unit 2: Thermal Physics)*\n\n" +
        "- **Vacuum gap:** Stops **conduction and convection** (no particles).\n" +
        "- **Silvered walls:** Reflect infrared radiation (**reduces radiation**).\n" +
        "- **Insulating stopper:** Reduces **conduction**, **convection**, and **evaporation**."
      );
    }
    if (low.includes("sea breeze") || low.includes("land breeze")) {
      return (
        "### ⚡ Sea Breeze & Land Breeze *(Unit 2: Convection)*\n\n" +
        "- **Day (Sea Breeze):** Land heats up faster than water (lower SHC); warm air above land expands, becomes **less dense and rises**, and cooler air from the sea flows in.\n" +
        "- **Night (Land Breeze):** Land cools faster; warmer air over the sea rises and cooler air from land flows out."
      );
    }

    if (/\b(?:difference|compare|distinguish|contrast|vs\.?|versus)\b/.test(low)) {
      if (low.includes("evaporat") && low.includes("boil")) {
        return (
          "### ⚡ Evaporation vs Boiling *(Unit 3 — Thermal Physics)*\n\n" +
          "- **Temperature:** Evaporation occurs at **any temperature** below the boiling point; boiling occurs only at a **fixed boiling point**.\n" +
          "- **Location:** Evaporation happens **only at the surface**; boiling happens **throughout the entire liquid** (bubbles form inside).\n" +
          "- **Cooling Effect:** Faster, more energetic molecules escape from the surface during evaporation, lowering the average kinetic energy of remaining molecules."
        );
      }
      if (low.includes("mass") && low.includes("weight")) {
        return (
          "### ⚡ Mass vs Weight *(Unit 1 — Mechanics)*\n\n" +
          "- **Mass (`m`):** Measure of quantity of matter (inertia); **scalar**, measured in **kg**; **constant everywhere**.\n" +
          "- **Weight (`W = m × g`):** **Gravitational force** on a mass; **vector**, measured in **N**; changes with gravitational field strength `g`."
        );
      }
      if (low.includes("scalar") && low.includes("vector")) {
        return (
          "### ⚡ Scalar vs Vector Quantities *(Unit 1 — Mechanics)*\n\n" +
          "- **Scalar:** Magnitude only (e.g. distance, speed, mass, time, energy).\n" +
          "- **Vector:** Both magnitude **and direction** (e.g. displacement, velocity, acceleration, force, weight, momentum)."
        );
      }
      if (low.includes("speed") && low.includes("velocity")) {
        return (
          "### ⚡ Speed vs Velocity & Distance vs Displacement *(Unit 1 — Mechanics)*\n\n" +
          "- **Distance / Speed** are **scalars** (magnitude only). **Displacement / Velocity** are **vectors** (magnitude **and direction**).\n" +
          "- An object moving in a **circle at constant speed** has **changing velocity** because its **direction** is continuously changing (so it is accelerating towards the centre)."
        );
      }
      if (low.includes("series") && low.includes("parallel")) {
        return (
          "### ⚡ Series vs Parallel Circuits *(Unit 5 — Electricity)*\n\n" +
          "- **Series:** **Same current** everywhere (`I₁ = I₂`); total p.d. **shared** (`V = V₁ + V₂`); `R = R₁ + R₂`.\n" +
          "- **Parallel:** **Same p.d.** across every branch (`V₁ = V₂`); current **splits** (`I = I₁ + I₂`); `1/R = 1/R₁ + 1/R₂`."
        );
      }
      if ((low.includes("a.c") || low.includes("alternating")) && (low.includes("d.c") || low.includes("direct"))) {
        return (
          "### ⚡ d.c. vs a.c. *(Unit 5/6)*\n\n" +
          "- **d.c.:** Flows in **one direction only** (e.g. battery).\n" +
          "- **a.c.:** Continuously **reverses direction** periodically (needed for **transformers**)."
        );
      }
      if (low.includes("fission") && low.includes("fusion")) {
        return (
          "### ⚡ Nuclear Fission vs Fusion *(Unit 5 — Nuclear Physics)*\n\n" +
          "- **Fission:** A **heavy unstable nucleus** (e.g. U-235) **splits** into smaller nuclei + neutrons + energy.\n" +
          "- **Fusion:** Two **light nuclei** (hydrogen isotopes) **join** at extreme temperature & pressure to form helium + energy."
        );
      }
      if (low.includes("conduction") && (low.includes("convection") || low.includes("radiation"))) {
        return (
          "### ⚡ Conduction, Convection & Infrared Radiation *(Unit 2 — Thermal Physics)*\n\n" +
          "- **Conduction (solids):** Atoms vibrate and pass kinetic energy; metals conduct fastest via **delocalised (free) electrons**.\n" +
          "- **Convection (liquids & gases only):** Heated fluid expands, becomes **less dense and rises**, while cooler denser fluid sinks (**convection current**).\n" +
          "- **Infrared Radiation (no medium needed):** EM wave transfer — works through a **vacuum**; dull black surfaces are best emitters/absorbers."
        );
      }
      if (low.includes("transverse") && low.includes("longitudinal")) {
        return (
          "### ⚡ Transverse vs Longitudinal Waves *(Unit 6 — Waves)*\n\n" +
          "- **Transverse:** Vibrations are **perpendicular (90°)** to wave travel (light, water, EM waves).\n" +
          "- **Longitudinal:** Vibrations are **parallel** to wave travel via **compressions & rarefactions** (sound waves)."
        );
      }
      if (low.includes("soft iron") && low.includes("steel")) {
        return (
          "### ⚡ Soft Iron vs Steel *(Unit 4 — Electromagnetism)*\n\n" +
          "- **Soft Iron:** Magnetises and demagnetises **easily** (temporary magnet → electromagnet & transformer cores).\n" +
          "- **Steel:** Retains magnetism (permanent magnet → bar magnets & compass needles)."
        );
      }
      if (low.includes("real") && low.includes("virtual") && (low.includes("image") || low.includes("lens") || low.includes("mirror"))) {
        return (
          "### ⚡ Real Image vs Virtual Image *(Unit 6 — Light & Lenses)*\n\n" +
          "- **Real Image:** Formed where light rays actually intersect; **can be projected onto a screen** (inverted).\n" +
          "- **Virtual Image:** Light rays only appear to come from behind the mirror/lens; **cannot be formed on a screen** (upright)."
        );
      }
    }

    if (/\b(what\s+happens|how\s+does|effect\s+on|change\s+in|doubles?|triples?|halves?|halved|doubled|tripled|increased?|decreased?)\b/.test(low)) {
      let k = null, kTxt = "";
      if (/\b(doubles?|twice|2\s*times|x\s*2)\b/.test(low)) { k = 2; kTxt = "doubled (×2)"; }
      else if (/\b(triples?|three\s+times|3\s*times|x\s*3)\b/.test(low)) { k = 3; kTxt = "tripled (×3)"; }
      else if (/\b(halved?|halves|half)\b/.test(low)) { k = 0.5; kTxt = "halved (×½)"; }
      if (k !== null) {
        if (low.includes("kinetic energy") && (low.includes("speed") || low.includes("velocity"))) {
          return `### ⚡ Effect on Kinetic Energy (\`KE = ½ m v²\`)\n\n- Kinetic energy is proportional to **speed²** (\`KE ∝ v²\`).\n- If speed is **${kTxt}**, kinetic energy changes by **\`${k * k}×\`**.`;
        }
        if (low.includes("resistance") && (low.includes("length") || low.includes("longer") || low.includes("shorter"))) {
          return `### ⚡ Effect on Resistance (\`R = ρ L / A\`)\n\n- Resistance is **directly proportional to length** (\`R ∝ L\`).\n- If length is **${kTxt}**, resistance changes by **\`${k}×\`**.`;
        }
        if (low.includes("resistance") && (low.includes("diameter") || low.includes("radius"))) {
          return `### ⚡ Effect on Resistance (\`R = ρ L / A\`)\n\n- Resistance is **inversely proportional to diameter²** (\`R ∝ 1 / d²\`).\n- If diameter is **${kTxt}**, area changes by \`${k * k}×\`, so resistance becomes **\`${1 / (k * k)}×\`**.`;
        }
        if (low.includes("pressure") && low.includes("volume")) {
          return `### ⚡ Boyle's Law (\`p₁V₁ = p₂V₂\`)\n\n- At constant temperature, pressure is **inversely proportional to volume** (\`p ∝ 1 / V\`).\n- If volume is **${kTxt}**, pressure becomes **\`${1 / k}×\`**.`;
        }
        if (low.includes("pressure") && low.includes("area")) {
          return `### ⚡ Solid Pressure (\`p = F / A\`)\n\n- For a constant force, pressure is **inversely proportional to area** (\`p ∝ 1 / A\`).\n- If area is **${kTxt}**, pressure becomes **\`${1 / k}×\`**.`;
        }
        if (low.includes("wavelength") && low.includes("frequency")) {
          return `### ⚡ Wave Equation (\`v = f × λ\`)\n\n- At constant wave speed, frequency and wavelength are **inversely proportional**.\n- If frequency is **${kTxt}**, wavelength becomes **\`${1 / k}×\`**.`;
        }
        if (low.includes("current") && low.includes("resistance")) {
          return `### ⚡ Ohm's Law (\`I = V / R\`)\n\n- At constant voltage (p.d.), current is **inversely proportional to resistance** (\`I ∝ 1 / R\`).\n- If resistance is **${kTxt}**, current becomes **\`${1 / k}×\`**.`;
        }
      }
    }
    return null;
  }

  function buildSynthesizedFallbackReply(query, solved, scoredChunks, msTips, ocrText, conceptEntry, qbHits = []) {
    const sections = [];
    if (ocrText) {
      const prev = ocrText.slice(0, 320) + (ocrText.length > 320 ? "…" : "");
      sections.push(`**📷 Read from your uploaded photo/PDF:**\n> ${prev.replace(/\n/g, "\n> ")}`);
    }
    const reasoned = solved ? null : tryReasonPhysicsVariation(query);
    if (solved) {
      sections.push(solved.markdown);
    } else if (reasoned) {
      sections.push(reasoned);
    }
    if (conceptEntry && !solved && !reasoned) {
      const fams = (conceptEntry.formulas || []).map((f) => `- \`${f}\``).join("\n");
      const pts = (conceptEntry.points || []).map((p) => `- ${p}`).join("\n");
      let blk = `### ⚡ ${conceptEntry.topic} *(${conceptEntry.unit})*\n\n${conceptEntry.summary}\n\n**Key Exam Points:**\n${pts}`;
      if (fams) blk += `\n\n**Key Equations:**\n${fams}`;
      if (conceptEntry.examiner_trap) blk += `\n\n> **Examiner Trap:** ${conceptEntry.examiner_trap}`;
      sections.push(blk);
    } else {
      // Check built-in core concept fallback
      let matchedCore = null;
      for (const c of CORE_CONCEPTS) {
        if (c.match.test(query)) {
          matchedCore = c;
          break;
        }
      }
      if (matchedCore && !solved && !reasoned) {
        const fams = matchedCore.formulas.map((f) => `- \`${f}\``).join("\n");
        const pts = matchedCore.points.map((p) => `- ${p}`).join("\n");
        sections.push(`### 📘 ${matchedCore.title}\n\n${matchedCore.summary}\n\n**Key Equations:**\n${fams}\n\n**Key Exam Points:**\n${pts}`);
      } else if (scoredChunks.length && !(solved || reasoned)) {
        const qWords = new Set(tokenize(query).filter((w) => w.length >= 3 && !STOPWORDS.has(w)));
        let bestTop = scoredChunks[0].ch;
        let bestBullets = cleanNoteBullets(bestTop.text, bestTop.topic);
        let bestMaxHit = -1;
        for (const item of scoredChunks) {
          const bList = cleanNoteBullets(item.ch.text, item.ch.topic);
          const maxH = bList.reduce((mx, b) => {
            const bl = b.toLowerCase();
            let c = 0;
            for (const w of qWords) if (bl.includes(w)) c++;
            return Math.max(mx, c);
          }, 0);
          if (maxH > bestMaxHit) {
            bestMaxHit = maxH;
            bestTop = item.ch;
            bestBullets = bList;
          }
        }
        if (bestMaxHit >= 1) {
          for (const item of scoredChunks) {
            if (item.ch !== bestTop && item.ch.unit === bestTop.unit) {
              for (const b of cleanNoteBullets(item.ch.text, item.ch.topic)) {
                if (!bestBullets.includes(b)) bestBullets.push(b);
              }
            }
          }
          const ranked = bestBullets
            .map((b, idx) => {
              const bl = b.toLowerCase();
              let h = 0;
              for (const w of qWords) if (bl.includes(w)) h++;
              return { h, idx, b };
            })
            .sort((a, b) => b.h - a.h || a.idx - b.idx)
            .slice(0, 6)
            .map((x) => `- ${x.b}`)
            .join("\n");
          const cite = bestTop.page ? `${bestTop.source} p.${bestTop.page}` : bestTop.source;
          sections.push(`### 📘 ${bestTop.topic} *(${bestTop.unit} — ${cite})*\n\n${ranked}`);
        }
      }
    }
    const isPaperCodeQ = /\b20\d{2}\b|\bp[24]\b|\bv[123]\b|\bq\d+/i.test(query);
    if (qbHits && qbHits.length && !solved && !reasoned && (ocrText || isPaperCodeQ || qbHits[0].s >= 0.85)) {
      const qb0 = qbHits[0].item;
      const msRows = (qb0.ms || []).filter((m) => m.a).map((m) => `- **${m.id}** *(${m.mk} marks)*: ${m.a}`);
      if (msRows.length) {
        sections.push(
          `### 📝 Matched Harony 2027 Past-Paper & Official Mark Scheme *(${qb0.bk} — ${qb0.ch})*\n${msRows.join("\n")}`
        );
      }
    }
    if (msTips.length && !solved && !reasoned && isPaperCodeQ) {
      const tip = msTips[0];
      const pts = (tip.q.points || [])
        .slice(0, 3)
        .map((p) => `- **[${p.code}]** ${p.answer}`)
        .join("\n");
      if (pts) {
        sections.push(`### 🎯 Examiner Marking Keywords *(${tip.msTitle})*\n${pts}`);
      }
    }
    if (!sections.length) {
      sections.push(
        "### Physics Tutor Guidance\n" +
          "Please share the exact Physics topic, values with units, or upload a photo of the problem so I can walk through the formula and step-by-step working with you!"
      );
    }
    return sections.join("\n\n---\n\n");
  }

  // ==================== MARK SCHEME & AI EXAMINER GRADING ====================
  function extractNumbers(s) {
    const out = [];
    const str = String(s || "").replace(/×/g, "x").replace(/−/g, "-");
    const sciRe = /([+-]?\d+(?:\.\d+)?)\s*[x*]\s*10\s*\^?\s*([+-]?\d+)/gi;
    let m;
    while ((m = sciRe.exec(str)) !== null) {
      out.push(parseFloat(m[1]) * Math.pow(10, parseInt(m[2], 10)));
    }
    const numRe = /(?<![a-zA-Z0-9_])([+-]?\d+(?:\.\d+)?)(?![a-zA-Z0-9_])/g;
    while ((m = numRe.exec(str)) !== null) {
      out.push(parseFloat(m[1]));
    }
    return out;
  }

  const SYNONYM_GROUPS = [
    ["decrease", "decreases", "drop", "drops", "fall", "falls", "less", "lower", "reduces", "reduced", "smaller"],
    ["increase", "increases", "rise", "rises", "more", "greater", "higher", "larger", "bigger"],
    ["constant", "uniform", "steady", "unchanged", "same"],
    ["collide", "collides", "collision", "collisions", "hit", "hits", "impact", "strike"],
    ["average", "overall", "mean", "total", "combined"],
    ["perpendicular", "normal", "90", "right angle"],
    ["air", "hollow", "empty space"],
  ];

  function expandWithSynonyms(wordSet) {
    const expanded = new Set(wordSet);
    for (const grp of SYNONYM_GROUPS) {
      if (grp.some((w) => expanded.has(w))) {
        for (const w of grp) expanded.add(w);
      }
    }
    return expanded;
  }

  // Detect explicit Physics contradictions (e.g., saying "greater" when MS requires "less")
  function hasOppositeContradiction(expectedLow, stuLow) {
    const pairs = [
      [/\bless\s+than\b|\bsmaller\s+than\b|\blower\s+than\b/, /\b(greater|more|higher|heavier|larger)\s+than\b/],
      [/\bgreater\s+than\b|\bmore\s+than\b|\bhigher\s+than\b/, /\b(less|smaller|lower|lighter)\s+than\b/],
      [/\bincreases?\b/, /\bdecreases?\b/],
      [/\bdecreases?\b/, /\bincreases?\b/],
      [/\btowards\s+the\s+normal\b/, /\baway\s+from\s+the\s+normal\b/],
      [/\baway\s+from\s+the\s+normal\b/, /\btowards\s+the\s+normal\b/],
    ];
    for (const [expRe, oppRe] of pairs) {
      if (expRe.test(expectedLow) && oppRe.test(stuLow) && !expRe.test(stuLow)) {
        return true;
      }
    }
    return false;
  }

  function extractStudentAnswerOnly(rawText) {
    const str = String(rawText || "");
    const m = str.match(/(?:^|\n|\.\s+)(?:my\s+answer|student\s+answer|answer|solution|working)\s*[:\-]\s*([\s\S]+)$/i);
    if (m && m[1].trim().length >= 2) {
      return m[1].trim();
    }
    return str;
  }

  function gradeAgainstQuestion(q, rawStudentText) {
    const studentText = extractStudentAnswerOnly(rawStudentText);
    const stuLow = studentText.toLowerCase();
    const stuToks = expandWithSynonyms(new Set(tokenize(studentText)));
    const stuNums = extractNumbers(studentText);
    let awarded = 0;
    const rows = [];
    const pts = q.points || [];

    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      const mtype = pt.mark_type || "B";
      const mval = Number(pt.marks || 1);
      const ans = String(pt.answer || "");
      const ansClean = ans.replace(/\([^)]*\)/g, " ");
      const ansWords = tokenize(ansClean).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
      const ansNums = extractNumbers(ansClean);

      let hit = false;
      const contradicted = hasOppositeContradiction(ansClean.toLowerCase(), stuLow);

      if (!contradicted && mtype === "A" && ansNums.length > 0) {
        const target = ansNums[0];
        hit = stuNums.some((sn) => {
          if (target === 0) return Math.abs(sn) < 1e-6;
          return Math.abs(sn - target) / Math.abs(target) <= 0.045;
        });
        if (hit) {
          awarded += mval;
          rows.push({
            code: pt.code,
            status: "awarded",
            marks_awarded: mval,
            max_marks: mval,
            expected: ans,
            comment: `✓ Numerical answer matches \`${ans}\` — Full ${mval}/${mval} marks awarded.`,
          });
          while (i + 1 < pts.length && pts[i + 1].mark_type === "C") i++;
          continue;
        }
      } else if (!contradicted && /option\s+([abcd])\b/i.test(ans)) {
        const opt = ans.match(/option\s+([abcd])\b/i)[1].toLowerCase();
        hit = new RegExp(`\\b${opt}\\b`, "i").test(studentText);
      } else if (!contradicted && ansWords.length > 0) {
        // Split OR / slash alternatives so matching ANY valid Cambridge alternative awards the mark!
        const alts = ansClean.split(/\bOR\b|\.\s*\/|;/i).map((s) => s.trim()).filter(Boolean);
        for (const alt of alts) {
          const altLow = alt.toLowerCase();
          // If alternative explicitly requires negation ("not solid"), require negation or hollow/air in student answer
          if (/\bnot\b/.test(altLow) && !/\b(not|no|never|hollow|air|empty)\b/.test(stuLow)) {
            continue;
          }
          const wList = tokenize(alt).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
          if (!wList.length) continue;
          const matched = wList.filter((w) => stuToks.has(w) || stuLow.includes(w)).length;
          if (matched / wList.length >= (wList.length <= 2 ? 0.5 : 0.6)) {
            hit = true;
            break;
          }
        }
      }

      if (hit) {
        awarded += mval;
        rows.push({
          code: pt.code,
          status: "awarded",
          marks_awarded: mval,
          max_marks: mval,
          expected: ans,
          comment: `✓ Marking point (\`${pt.code}\`) satisfied: \`${ans}\``,
        });
      } else {
        rows.push({
          code: pt.code,
          status: contradicted ? "contradiction" : "missed",
          marks_awarded: 0,
          max_marks: mval,
          expected: ans,
          comment: contradicted
            ? `✗ Contradiction on (\`${pt.code}\`): expected \`${ans}\`.`
            : `✗ Missing point (\`${pt.code}\`): \`${ans}\``,
        });
      }
    }

    const maxM = Number(q.max_marks || Math.max(1, pts.length));
    awarded = Math.min(awarded, maxM);
    return {
      question_id: q.question_id,
      title: q.title || q.question_id,
      video_url: q.video_url || null,
      awarded_marks: awarded,
      max_marks: maxM,
      points: rows,
      model_answer: pts.map((p) => `[${p.code}] ${p.answer}`).join(" ; "),
    };
  }

  function gradeLetter(pct) {
    if (pct >= 85) return "A*";
    if (pct >= 73) return "A";
    if (pct >= 62) return "B";
    if (pct >= 50) return "C";
    if (pct >= 40) return "D";
    if (pct >= 30) return "E";
    return "U (Needs Revision)";
  }

  function splitQuestionAndAnswer(rawText) {
    const str = String(rawText || "").trim();
    const m = str.match(/(?:^|\n|\.\s+)(?:my\s+answer|student(?:'s)?\s+answer|answer|solution|working)\s*[:\-]\s*([\s\S]+)$/i);
    if (m && m[1].trim().length >= 1) {
      const qPart = str.slice(0, m.index).trim();
      const aPart = m[1].trim();
      return { questionPart: qPart || str, answerPart: aPart };
    }
    return { questionPart: str, answerPart: str };
  }

  function buildDeterministicFeedback(awarded, maxM, pts, noteRefs) {
    const missed = (pts || []).filter((p) => p.status !== "awarded");
    const topNote = noteRefs && noteRefs[0];
    const revStr = topNote ? ` Revise **${topNote.topic}** (Harony Notes p.${topNote.page || "?"}).` : "";
    if (awarded >= maxM && maxM > 0) {
      return `Excellent work! You earned full marks (${awarded}/${maxM}) by hitting all required Cambridge marking points.`;
    }
    const hint = missed[0] ? (missed[0].expected || missed[0].comment || "").replace(/^[✓✗💡]\s*/, "") : "";
    if (awarded === 0) {
      return `0/${maxM} marks awarded. Make sure your answer directly covers the required physics points${hint ? ` (e.g., ${hint})` : ""}.${revStr}`;
    }
    return `Good start (${awarded}/${maxM} marks)! To gain full marks, also include${hint ? `: ${hint}` : " the missing points above"}.${revStr}`;
  }

  function findAutoMatchedQuestion(questionPart, kb) {
    const low = questionPart.toLowerCase();
    const codeMatch = low.match(/\b(20\d{2})\s+(mar(?:ch)?|may|jun(?:e)?|nov(?:ember)?)\s+p([24])\s+v(\d)\s+q?0?(\d+)/i);
    if (codeMatch && kb.mark_schemes) {
      const yr = codeMatch[1];
      const mon = codeMatch[2].startsWith("ma") && codeMatch[2] !== "march" ? "jun" : codeMatch[2].slice(0, 3);
      const pp = "p" + codeMatch[3];
      const vr = "v" + codeMatch[4];
      const qn = codeMatch[5];
      for (const m of kb.mark_schemes) {
        const hit = (m.questions || []).find((q) => {
          const idLow = (q.question_id || "").toLowerCase();
          return (
            idLow.includes(yr) &&
            idLow.includes(mon) &&
            idLow.includes(pp) &&
            idLow.includes(vr) &&
            (idLow.includes("q0" + qn) || idLow.includes("q" + qn))
          );
        });
        if (hit) return { msTitle: m.title, q: hit };
      }
    }

    const qToks = Array.from(new Set(tokenize(questionPart).filter((w) => !STOPWORDS.has(w) && w.length >= 4)));
    if (qToks.length < 2 || !kb.mark_schemes) return null;
    let best = null;
    let bestHits = 0;
    for (const ms of kb.mark_schemes) {
      if ((ms.id || "").includes("p2")) continue;
      for (const q of ms.questions || []) {
        const hay = `${q.title || ""} ${q.chapter || ""} ${q.summary_answer || ""}`.toLowerCase();
        const hits = qToks.reduce((acc, t) => acc + (hay.includes(t) ? 1 : 0), 0);
        const ratio = hits / qToks.length;
        if (hits >= 3 && ratio >= 0.45 && hits > bestHits) {
          bestHits = hits;
          best = { msTitle: ms.title, q };
        }
      }
    }
    return best;
  }

  function runAutoGradeDeterministic(studentText, kb) {
    const { questionPart, answerPart } = splitQuestionAndAnswer(studentText);
    const solved = solvePhysicsProblem(questionPart) || solvePhysicsProblem(studentText);
    const matchedMs = findAutoMatchedQuestion(questionPart, kb);
    const notesSeed = [
      questionPart,
      solved ? solved.topic : "",
      matchedMs && matchedMs.q ? `${matchedMs.q.title || ""} ${matchedMs.q.chapter || ""}` : "",
    ]
      .filter(Boolean)
      .join(" ");
    const scoredChunks = searchChunksScored(
      { chunks: (kb.chunks || []).filter((c) => c.source === "Harony Notes 2026 Final") },
      notesSeed,
      2
    );
    const noteRefs = formatNoteRefs(scoredChunks.length ? scoredChunks : searchChunksScored(kb, notesSeed, 2));
    const relatedVids = searchVideos(kb, notesSeed, 3);

    // 1. Calculation Question (100% deterministic 3 marks)
    if (solved) {
      const eqParts = answerPart.split("=");
      const finalChunk = eqParts.length >= 2 ? eqParts[eqParts.length - 1] : answerPart;
      const stuNums = extractNumbers(finalChunk);
      const hasAns = stuNums.some((n) => Math.abs(n - solved.value) <= 0.04 * Math.max(1, Math.abs(solved.value)));
      const hasFormulaOrSub = eqParts.length >= 2 || solved.formula.split("=")[0].trim().toLowerCase() === answerPart.trim().toLowerCase();
      const maxM = 3;
      const pts = [];
      let awd = 0;
      if (hasAns) {
        awd = 3;
        pts.push({
          code: "A3",
          status: "awarded",
          expected: `${fmtNum(solved.value, 3)} ${solved.unit}`,
          comment: `✓ Correct calculation! Matches \`${fmtNum(solved.value, 3)} ${solved.unit}\` using \`${solved.formula}\`.`,
        });
      } else {
        if (hasFormulaOrSub) {
          awd = 1;
          pts.push({
            code: "C1",
            status: "awarded",
            expected: solved.formula,
            comment: `✓ Formula/working structure shown (\`${solved.formula}\`).`,
          });
        } else {
          pts.push({
            code: "C1",
            status: "missed",
            expected: solved.formula,
            comment: `✗ Write the formula \`${solved.formula}\` first.`,
          });
        }
        pts.push({
          code: "A2",
          status: "missed",
          expected: `${fmtNum(solved.value, 3)} ${solved.unit}`,
          comment: `✗ Expected final value: \`${fmtNum(solved.value, 3)} ${solved.unit}\`.`,
        });
      }
      const pct = Math.round((1000 * awd) / maxM) / 10;
      const vidUrl = (matchedMs && matchedMs.q && matchedMs.q.video_url) || (relatedVids[0] && relatedVids[0].video_url) || null;
      return {
        ok: true,
        mode: "symbolic_check",
        mark_scheme_title: `Haroony AI Physics Examiner (${solved.topic})`,
        awarded_marks: awd,
        max_marks: maxM,
        percentage: pct,
        grade: gradeLetter(pct),
        examiner_feedback: buildDeterministicFeedback(awd, maxM, pts, noteRefs),
        questions: [
          {
            question_id: "Calculation",
            title: solved.topic,
            video_url: vidUrl,
            awarded_marks: awd,
            max_marks: maxM,
            points: pts,
            model_answer: `${solved.formula} → ${fmtNum(solved.value, 3)} ${solved.unit}`,
          },
        ],
        related_videos: relatedVids,
        extracted_student_text: studentText,
        notes_references: noteRefs,
      };
    }

    // 2. Official Harony Mark Scheme Question Match (100% deterministic!)
    if (matchedMs) {
      const gq = gradeAgainstQuestion(matchedMs.q, answerPart);
      if (!gq.video_url && relatedVids[0]) gq.video_url = relatedVids[0].video_url;
      const pct = Math.round((1000 * gq.awarded_marks) / Math.max(1, gq.max_marks)) / 10;
      return {
        ok: true,
        mode: "auto_matched_ms",
        mark_scheme_title: `${matchedMs.msTitle} (Auto-Matched)`,
        awarded_marks: gq.awarded_marks,
        max_marks: gq.max_marks,
        percentage: pct,
        grade: gradeLetter(pct),
        examiner_feedback: buildDeterministicFeedback(gq.awarded_marks, gq.max_marks, gq.points, noteRefs),
        questions: [gq],
        related_videos: relatedVids,
        extracted_student_text: studentText,
        notes_references: noteRefs,
      };
    }

    // 3. Clean Harony Notes Concept Rubric ONLY when strong topic match (never fake a score on ambiguous fragments!)
    const meaningfulWords = tokenize(studentText).filter(
      (w) => w.length >= 3 && !STOPWORDS.has(w) && !["question", "answer", "my", "is", "the"].includes(w)
    );
    if (scoredChunks.length > 0 && scoredChunks[0].s >= 4.0 && meaningfulWords.length >= 4) {
      const top = scoredChunks[0].ch;
      const bullets = cleanNoteBullets(top.text, top.topic).slice(0, 3);
      const stuToks = expandWithSynonyms(new Set(tokenize(answerPart)));
      const ansLow = answerPart.toLowerCase();
      const pts = [];
      let awd = 0;
      for (const b of bullets) {
        const bWords = tokenize(b).filter((w) => !STOPWORDS.has(w) && w.length >= 4);
        if (!bWords.length) continue;
        const contradicted = hasOppositeContradiction(b.toLowerCase(), ansLow);
        const hitCount = bWords.filter((w) => stuToks.has(w)).length;
        const ok = !contradicted && hitCount / bWords.length >= 0.35;
        if (ok) awd++;
        pts.push({
          code: "B1",
          status: ok ? "awarded" : "missed",
          expected: b,
          comment: ok
            ? `✓ Covered key concept: ${b}`
            : contradicted
            ? `✗ Contradiction on concept: ${b}`
            : `✗ Missing/incomplete concept: ${b}`,
        });
      }
      const maxM = Math.max(1, pts.length);
      const pct = Math.round((1000 * awd) / maxM) / 10;
      const vidUrl = (relatedVids[0] && relatedVids[0].video_url) || null;
      return {
        ok: true,
        mode: "concept_check",
        mark_scheme_title: `Harony Notes Concept Check — ${top.topic}`,
        awarded_marks: awd,
        max_marks: maxM,
        percentage: pct,
        grade: gradeLetter(pct),
        examiner_feedback: buildDeterministicFeedback(awd, maxM, pts, noteRefs),
        questions: [
          {
            question_id: "Concept",
            title: top.topic,
            video_url: vidUrl,
            awarded_marks: awd,
            max_marks: maxM,
            points: pts,
            model_answer: bullets.join(" ; "),
          },
        ],
        related_videos: relatedVids,
        extracted_student_text: studentText,
        notes_references: noteRefs,
      };
    }

    return {
      ok: false,
      extracted_student_text: studentText,
      error:
        "I read your submission, but without the **question text** or a selected **Mark Scheme**, I won't guess a score. " +
        "Please include the question above your answer, or pick the exact book & question in **Harony MS** / **Upload MS** / **Paste MS**.",
    };
  }

  // ==================== BROWSER CONTINUOUS ONLINE ML LEARNER ====================
  const ONLINE_ML_KEY = "harony_online_ml_v1";
  let memoryCache = null;

  function loadOnlineMemory() {
    if (memoryCache) return memoryCache;
    try {
      if (typeof localStorage !== "undefined") {
        const raw = localStorage.getItem(ONLINE_ML_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") {
            memoryCache = parsed;
            return memoryCache;
          }
        }
      }
    } catch (_) {}
    memoryCache = { unitWeights: {}, interactions: [] };
    return memoryCache;
  }

  function saveOnlineMemory() {
    try {
      if (typeof localStorage !== "undefined" && memoryCache) {
        localStorage.setItem(ONLINE_ML_KEY, JSON.stringify(memoryCache));
      }
    } catch (_) {}
  }

  function predictOnlineUnit(query) {
    const mem = loadOnlineMemory();
    const toks = tokenize(query).filter((w) => w.length >= 3 && !STOPWORDS.has(w));
    if (!toks.length) return null;
    const scores = {};
    for (const t of toks) {
      const uw = mem.unitWeights[t];
      if (!uw) continue;
      for (const [u, w] of Object.entries(uw)) {
        scores[u] = (scores[u] || 0) + w;
      }
    }
    let bestU = null;
    let bestS = 0;
    for (const [u, s] of Object.entries(scores)) {
      if (s > bestS) {
        bestS = s;
        bestU = u;
      }
    }
    return bestS >= 2.0 ? bestU : null;
  }

  function recordOnlineInteraction(query, unit, topic, confidenceScore = 4.0) {
    const toks = Array.from(new Set(tokenize(query).filter((w) => w.length >= 3 && !STOPWORDS.has(w))));
    const id = "br-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 6);
    if (!toks.length) return id;
    const mem = loadOnlineMemory();
    if (unit && confidenceScore >= 3.8) {
      for (const t of toks) {
        if (!mem.unitWeights[t]) mem.unitWeights[t] = {};
        mem.unitWeights[t][unit] = (mem.unitWeights[t][unit] || 0) + 1;
      }
    }
    mem.interactions.push({ id, query: String(query).slice(0, 240), unit: unit || "", topic: topic || "", score: 0 });
    if (mem.interactions.length > 250) mem.interactions = mem.interactions.slice(-250);
    saveOnlineMemory();
    return id;
  }

  function applyOnlineFeedback(interactionId, helpful, queryFallback, unitFallback) {
    const mem = loadOnlineMemory();
    const entry = mem.interactions.find((x) => x.id === interactionId);
    const q = (entry ? entry.query : queryFallback) || "";
    const u = (entry ? entry.unit : unitFallback) || "";
    if (entry) entry.score += helpful ? 2 : -2;
    if (helpful && q && u) {
      for (const t of Array.from(new Set(tokenize(q).filter((w) => w.length >= 3 && !STOPWORDS.has(w))))) {
        if (!mem.unitWeights[t]) mem.unitWeights[t] = {};
        mem.unitWeights[t][u] = (mem.unitWeights[t][u] || 0) + 2;
      }
    }
    saveOnlineMemory();
    return { online_ml_active: true, total_learned_questions: mem.interactions.length };
  }

  // ==================== PUBLIC ENGINE API ====================
  global.HaroonyBrowserEngine = {
    solvePhysicsProblem,

    async getMarkSchemeQuestions(msId) {
      const kb = await ensureKB();
      const ms = (kb.mark_schemes || []).find((m) => m.id === msId);
      if (!ms) return [];
      return (ms.questions || []).map((q) => ({
        question_id: q.question_id,
        title: q.title || q.question_id,
        chapter: q.chapter || "",
        max_marks: q.max_marks || 1,
        video_url: q.video_url || null,
      }));
    },

    async getStatus() {
      const kb = await ensureKB();
      return {
        ok: true,
        mode: "github_pages_browser",
        name: "Haroony AI",
        clinic: "Harony Physics Clinic",
        free_unlimited: true,
        physics_only_guardrail: true,
        notes_chunks_count: kb.chunks.length,
        video_solutions_count: (kb.videos || []).length,
        mark_schemes: (kb.mark_schemes || []).map((m) => ({
          id: m.id,
          title: m.title,
          paper_code: m.paper_code,
          session: m.session,
          total_marks: m.total_marks,
          question_count: m.question_count,
        })),
        curriculum: kb.curriculum || [],
        metrics: kb.metrics || {},
      };
    },

    async chat(message, history, fileObj, callbacks = {}) {
      const { onProgress = null, onToken = null } = callbacks;
      const [kb, qaList] = await Promise.all([ensureKB(), ensureQA()]);

      let userMsg = String(message || "").trim();
      let ocrText = "";

      if (fileObj && global.HaroonyNeuralAI) {
        ocrText = await global.HaroonyNeuralAI.extractDocumentText(fileObj, onProgress);
      }

      const combined = `${userMsg}\n${ocrText}`.trim();
      if (!combined) {
        return {
          ok: true,
          refused: false,
          category: "empty",
          reply: fileObj
            ? "I couldn't read clear text from that file. Please upload a clearer photo/PDF or type the Physics question alongside it!"
            : "Please type a Physics question or attach a photo/PDF of a problem.",
          notes_cards: [],
          video_links: [],
          engine: "Haroony AI",
        };
      }

      const hasPrior = Array.isArray(history) && history.length > 0;
      const g = checkGuardrail(combined, kb.guardrail_model, hasPrior);
      if (!g.allowed) {
        return {
          ok: true,
          refused: true,
          category: g.category,
          reply: g.refusal,
          notes_cards: [],
          video_links: [],
          engine: "Haroony Physics Guardrail",
        };
      }

      if (g.category === "greeting" && !ocrText) {
        return {
          ok: true,
          refused: false,
          category: "greeting",
          reply:
            "Hello! Welcome to **Haroony AI**.\n\n" +
            "Ask me any **IGCSE or A-Level Physics** question, request **YouTube video solutions** for any topic, or attach / paste (`Ctrl+V`) a photo or PDF of a Physics problem!",
          notes_cards: [],
          video_links: [],
          engine: "Haroony AI",
        };
      }

      // Only inherit previous turn for genuine anaphoric follow-ups (never trap new topics!)
      const PURE_FOLLOWUP_RE =
        /^\s*(?:why\??|how\??|explain\s+more|tell\s+me\s+more|simpler|simplify|give\s+(?:me\s+)?(?:an?\s+)?example|another\s+example|show\s+(?:the\s+)?(?:formula|equation|working|steps?)|what(?:'s|\s+is)\s+(?:the|its)\s+unit\??|(?:youtube\s+)?videos?\s+for\s+(?:this|that|it)\??)\s*[?!.]*\s*$/i;
      let searchQuery = combined;
      if (hasPrior && PURE_FOLLOWUP_RE.test(combined)) {
        const prevUsers = history.filter((h) => h && h.role === "user" && h.content);
        if (prevUsers.length) {
          searchQuery = prevUsers[prevUsers.length - 1].content + " " + combined;
        }
      }

      // Direct YouTube video solution requests for any topic or paper code!
      if (/\b(?:youtube|youtu\.be|videos?|video\s+solution|watch)\b/i.test(combined)) {
        const noteHits = searchChunksScored(kb, searchQuery, 2);
        const prefUnit = noteHits.length && noteHits[0].s >= 4.0 ? noteHits[0].ch.unit : null;
        const prefTopic = noteHits.length ? noteHits[0].ch.topic : null;
        const vidHits = searchVideos(kb, searchQuery, 8, prefUnit, prefTopic);
        const strongCards = noteHits
          .filter((x) => x.s >= 4.2)
          .slice(0, 2)
          .map((x) => x.ch);
        let replyMd;
        if (vidHits.length) {
          const lines = vidHits.map((v, idx) => {
            const loc = `${v.book || "Harony 2027"} • ${v.chapter || v.unit || ""}${v.page ? ` (p.${v.page})` : ""}`;
            return `${idx + 1}. **${v.label || "Video Solution"}** *(${loc})* — ${v.video_url}`;
          });
          replyMd =
            "### 🎥 Harony Physics YouTube Video Solutions\n\n" +
            "Here are the matching Mr. Haroony video walkthroughs from **Harony ClassWork 2027**, **P2 Classified 2027**, and **P4 Classified 2027**:\n\n" +
            lines.join("\n");
        } else {
          replyMd =
            "### 🎥 YouTube Video Solutions\n\n" +
            "Tell me the specific IGCSE Physics topic (e.g. *density, moments, thermal, waves, diffraction, electricity, transformers, half-life, astronomy*) " +
            "or paper code (e.g. *2023 March P4 V2 Q02*) and I will send you Mr. Haroony's exact YouTube walkthrough links!";
        }
        return {
          ok: true,
          refused: false,
          category: "physics",
          reply: replyMd,
          notes_cards: strongCards,
          video_links: vidHits,
          ocr_text: ocrText || null,
          engine: "Haroony Video Library (2,793 Solutions)",
        };
      }

      if (/\b(?:quiz\s+me|test\s+me|practice\s+question|give\s+me\s+a\s+question)\b/i.test(combined)) {
        const qbPicks = searchQuestionBank(qaList, searchQuery, 5).filter((x) => x.item.ms && x.item.ms.length && (x.item.q || "").length >= 60);
        if (qbPicks.length) {
          const picked = qbPicks[Math.floor(Math.random() * qbPicks.length)].item;
          const qClean = String(picked.q || "").replace(/https?:\/\/\S+/g, "").trim().slice(0, 460);
          const pc = (picked.pc && picked.pc[0]) || picked.ch || "Cambridge 0625";
          const qVids = (picked.yt || []).map((u) => ({
            book: picked.bk || "Harony 2027",
            unit: picked.u || "",
            chapter: picked.ch || "",
            label: pc,
            video_url: u,
          }));
          return {
            ok: true,
            refused: false,
            category: "physics",
            reply:
              `### 📝 IGCSE Physics Past-Paper Practice — ${picked.ch || picked.u}\n` +
              `**Source:** \`${picked.bk}\` *(${pc})*\n\n> ${qClean}\n\n` +
              `*Reply here with your working and answer and I will walk you through the official Cambridge 0625 mark scheme!*`,
            notes_cards: [],
            video_links: qVids,
            ocr_text: null,
            engine: "Haroony Physics AI Examiner (Full 2027 Bank)",
          };
        }
      }

      const solved = solvePhysicsProblem(combined);
      const scoredChunks = searchChunksScored(kb, searchQuery, 3);
      const primaryUnit = scoredChunks.length && scoredChunks[0].s >= 4.0 ? scoredChunks[0].ch.unit : null;
      const primaryTopic = scoredChunks.length ? scoredChunks[0].ch.topic : null;
      const conceptEntry = matchConceptAtlas(kb, searchQuery);
      const qbHits = searchQuestionBank(qaList, searchQuery, 2, primaryUnit);
      const msTips = searchMsTips(kb, searchQuery, 2);
      const vids = searchVideos(kb, searchQuery, 3, primaryUnit, primaryTopic);
      for (const qbWrap of qbHits) {
        for (const ytUrl of qbWrap.item.yt || []) {
          if (vids.length < 3 && !vids.some((v) => v.video_url === ytUrl)) {
            vids.push({
              book: qbWrap.item.bk || "Harony 2027",
              unit: qbWrap.item.u || "",
              chapter: qbWrap.item.ch || "",
              label: (qbWrap.item.pc && qbWrap.item.pc[0]) || qbWrap.item.ch || "Past Paper Walkthrough",
              video_url: ytUrl,
            });
          }
        }
      }

      // Only attach visual reference cards if strongly relevant to the question!
      const strongCards = scoredChunks
        .filter((x) => x.s >= 5.0)
        .slice(0, 2)
        .map((x) => x.ch);

      let finalReply = null;
      if (global.HaroonyNeuralAI && global.HaroonyNeuralAI.isReady && global.HaroonyNeuralAI.isReady()) {
        try {
          const messages = buildTutorMessages(userMsg, history, solved, scoredChunks, msTips, ocrText, conceptEntry);
          finalReply = await global.HaroonyNeuralAI.generate(messages, {
            maxTokens: 480,
            temperature: 0.2,
            onProgress,
            onToken,
          });
        } catch (err) {
          console.warn("Neural chat fallback:", err);
        }
      }

      if (!finalReply) {
        finalReply = buildSynthesizedFallbackReply(searchQuery, solved, scoredChunks, msTips, ocrText, conceptEntry, qbHits);
      } else if (ocrText && !finalReply.includes("Read from")) {
        const prev = ocrText.slice(0, 240) + (ocrText.length > 240 ? "…" : "");
        finalReply = `**📷 Read from your uploaded file:**\n> ${prev.replace(/\n/g, "\n> ")}\n\n${finalReply}`;
      }

      const resolvedUnit = primaryUnit || (conceptEntry && conceptEntry.unit) || predictOnlineUnit(searchQuery) || "";
      const resolvedTopic = primaryTopic || (conceptEntry && conceptEntry.topic) || "Physics";
      const interactionId = recordOnlineInteraction(combined, resolvedUnit, resolvedTopic, scoredChunks.length ? scoredChunks[0].s : 4.0);

      return {
        ok: true,
        refused: false,
        category: "physics",
        reply: finalReply,
        notes_cards: strongCards,
        video_links: vids,
        ocr_text: ocrText || null,
        interaction_id: interactionId,
        unit: resolvedUnit,
        topic: resolvedTopic,
        engine: "Haroony AI",
      };
    },

    recordFeedback(interactionId, helpful, queryText, unitName) {
      return applyOnlineFeedback(interactionId, helpful, queryText, unitName);
    },

    async getMarkSchemeById(msId) {
      const kb = await ensureKB();
      return (kb.mark_schemes || []).find((m) => m.id === msId) || null;
    },

    async grade({ studentText, studentFile, studentFiles, msId, msText, msFile, questionId, onProgress }, progressCb) {
      const kb = await ensureKB();
      const prog = onProgress || progressCb;

      let text = String(studentText || "").trim();
      const filesToRead = Array.isArray(studentFiles) && studentFiles.length
        ? studentFiles
        : studentFile
        ? [studentFile]
        : [];
      for (const sf of filesToRead) {
        if (sf && global.HaroonyNeuralAI) {
          const stuOcr = await global.HaroonyNeuralAI.extractDocumentText(sf, prog);
          if (stuOcr) text = `${text}\n${stuOcr}`.trim();
        }
      }
      if (!text) {
        return {
          ok: false,
          error: "Please write your Physics answer or upload a clear photo/PDF of your work in Step 1.",
        };
      }

      let customMsContent = String(msText || "").trim();
      if (msFile && global.HaroonyNeuralAI) {
        const msOcr = await global.HaroonyNeuralAI.extractDocumentText(msFile, prog);
        if (msOcr) customMsContent = `${customMsContent}\n${msOcr}`.trim();
      }

      const g = checkGuardrail(text, kb.guardrail_model, Boolean(msId || customMsContent));
      if (!g.allowed && (g.category === "jailbreak" || (!msId && !customMsContent))) {
        return { ok: false, refused: true, error: g.refusal };
      }

      let msObj = null;
      if (customMsContent) {
        const lines = customMsContent
          .split(/\n|;/)
          .map((l) => l.trim())
          .filter((l) => l.length >= 2);
        const pts = lines.map((ln) => {
          const m = ln.match(/\b([BMCA])([1-5])\b/);
          const code = m ? m[1] + m[2] : "B1";
          const mval = m ? parseInt(m[2], 10) : 1;
          return {
            mark_type: m ? m[1] : "B",
            marks: mval,
            code,
            answer: ln.replace(/\[?\b[BMCA][1-5]\b\]?/g, "").trim() || ln,
          };
        });
        const maxM = pts.reduce((a, p) => a + p.marks, 0) || 1;
        msObj = {
          title: msFile ? `Uploaded Mark Scheme (${msFile.name})` : "Custom Mark Scheme",
          questions: [{ question_id: questionId || "Custom", title: "Mark Scheme Evaluation", max_marks: maxM, points: pts }],
        };
      } else if (msId) {
        msObj = (kb.mark_schemes || []).find((m) => m.id === msId) || null;
      } else {
        // Check if student text explicitly names a Harony paper code like "2023 March P4 V2 Q02"
        const codeMatch = text.match(/\b(20\d{2})\s+(Mar(?:ch)?|May|Jun(?:e)?|Nov(?:ember)?)\s+P([24])\s+V(\d)\s+Q?0?(\d+)/i);
        if (codeMatch && kb.mark_schemes) {
          const mon = codeMatch[2].toLowerCase().startsWith("ma") && codeMatch[2].toLowerCase() !== "march" ? "jun" : codeMatch[2].toLowerCase().slice(0, 3);
          for (const m of kb.mark_schemes) {
            const hit = (m.questions || []).find((q) => {
              const idLow = (q.question_id || "").toLowerCase();
              return (
                idLow.includes(codeMatch[1]) &&
                idLow.includes(mon) &&
                idLow.includes("p" + codeMatch[3]) &&
                idLow.includes("v" + codeMatch[4]) &&
                (idLow.includes("q0" + codeMatch[5]) || idLow.includes("q" + codeMatch[5]))
              );
            });
            if (hit) {
              msObj = { title: m.title, questions: [hit] };
              break;
            }
          }
        }
      }

      if (msObj && msObj.questions && msObj.questions.length) {
        let qs = msObj.questions;
        if (questionId && questionId.trim() && qs.length > 1) {
          const qlow = questionId.trim().toLowerCase().replace(/^q/, "");
          const exact = qs.filter(
            (q) =>
              q.question_id.toLowerCase() === qlow ||
              q.question_id.toLowerCase() === "q" + qlow ||
              q.question_id.toLowerCase().endsWith("-q0" + qlow) ||
              q.question_id.toLowerCase().endsWith("-q" + qlow)
          );
          if (exact.length) {
            qs = exact.slice(0, 4);
          } else {
            const qtoks = qlow.match(/[a-z0-9]+/g) || [];
            const filtered = qs.filter((q) => {
              const hay = ((q.title || "") + " " + q.question_id).toLowerCase();
              return hay.includes(qlow) || (qtoks.length >= 2 && qtoks.every((t) => hay.includes(t)));
            });
            qs = filtered.length ? filtered.slice(0, 4) : qs.slice(0, 2);
          }
        } else if (qs.length > 1) {
          const { questionPart } = splitQuestionAndAnswer(text);
          const sToks = tokenize(questionPart).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
          const ranked = qs
            .map((q) => {
              const hay = ((q.title || "") + " " + (q.summary_answer || "")).toLowerCase();
              const sc = sToks.reduce((acc, t) => acc + (hay.includes(t) ? 1 : 0), 0);
              return { sc, q };
            })
            .sort((a, b) => b.sc - a.sc);
          qs = ranked.slice(0, 1).map((x) => x.q);
        }

        const { answerPart } = splitQuestionAndAnswer(text);
        const graded = qs.map((q) => gradeAgainstQuestion(q, answerPart));
        const awarded = graded.reduce((a, x) => a + x.awarded_marks, 0);
        const maxM = Math.max(1, graded.reduce((a, x) => a + x.max_marks, 0));
        const pct = Math.round((1000 * awarded) / maxM) / 10;

        const seedQuery = graded.map((g) => g.model_answer).join(" ") + " " + text;
        const lectureChunks = searchChunksScored(
          { chunks: (kb.chunks || []).filter((c) => c.source === "Harony Notes 2026 Final") },
          seedQuery,
          2
        );
        const noteRefs = formatNoteRefs(lectureChunks.length ? lectureChunks : searchChunksScored(kb, seedQuery, 2));
        const relatedVids = searchVideos(kb, seedQuery, 3);
        for (const gq of graded) {
          if (!gq.video_url && relatedVids[0]) gq.video_url = relatedVids[0].video_url;
        }
        const allPts = graded.flatMap((g) => g.points || []);

        return {
          ok: true,
          mode: "mark_scheme",
          mark_scheme_title: msObj.title,
          awarded_marks: awarded,
          max_marks: maxM,
          percentage: pct,
          grade: gradeLetter(pct),
          examiner_feedback: buildDeterministicFeedback(awarded, maxM, allPts, noteRefs),
          questions: graded,
          related_videos: relatedVids,
          extracted_student_text: text,
          notes_references: noteRefs,
        };
      }

      // Mode 1: Fast, Deterministic, Real Physics Examiner Auto-Check!
      return runAutoGradeDeterministic(text, kb);
    },

    async searchNotes(q, unit) {
      const kb = await ensureKB();
      let pool = kb.chunks.filter((c) => c.source === "Harony Notes 2026 Final");
      if (unit) pool = pool.filter((c) => (c.unit || "").toLowerCase() === unit.toLowerCase());
      if (q && q.trim()) {
        const subKb = { chunks: pool };
        return { ok: true, results: searchChunksScored(subKb, q, 16).map((x) => x.ch) };
      }
      return { ok: true, results: pool.slice(3, 27) };
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
