/* Haroony AI — Smart In-Browser Physics Engine + Symbolic Solver + Neural RAG/Grader
 * Works both standalone on GitHub Pages (https://haroony-ai.github.io) and alongside
 * HaroonyNeuralAI (WebLLM Qwen2.5-Instruct / Transformers.js WASM + PDF.js + Tesseract OCR).
 */
(function (global) {
  "use strict";

  let kbPromise = null;
  let kbData = null;

  const REFUSAL_MSG =
    "I'm **Haroony AI**, and I specialize exclusively in **Physics** (IGCSE & A-Level Physics tutoring and exam answer correction).\n\n" +
    "Ask me any question on *Mechanics, Thermal Physics, Waves, Light, Electricity, Electromagnetism, Nuclear Physics, or Space Physics*, or open **Check Answer** to grade your Physics work.";

  const JAILBREAK_RE =
    /\b(ignore\s+(?:all\s+|previous\s+|prior\s+)?instructions|disregard\s+(?:your\s+)?rules|pretend\s+you\s+are|act\s+as\s+a\s+general|bypass\s+guardrail|reveal\s+system\s+prompt)\b/i;

  const GREETING_RE =
    /^(?:hi+|hello+|hey+|yo|good\s+(?:morning|afternoon|evening)|salam|assalamu\s*alaikum|who\s+are\s+you|what\s+can\s+you\s+do|help|thanks|thank\s+you)\s*[!?.]*$/i;

  const HARD_NON_PHYSICS_RE =
    /\b(recipe|bake|cake|pasta|pizza|biryani|chicken|burger|poem|poetry|haiku|sonnet|romeo|juliet|shakespeare|fifa|world\s+cup|football\s+match|basketball|oscar|netflix|movie\s+review|horoscope|zodiac|astrology|bitcoin|crypto|ethereum|stock\s+market|forex|makeup|skincare|mitosis|meiosis|photosynthesis|chloroplast|digestive\s+system|nephron|auxin|enzyme|dna\s+replication|titration|haber\s+process|alkane|alkene|polymerisation|covalent\s+bond|electrolysis\s+of\s+brine|scrape|beautifulsoup|flexbox|react\s+component|sql\s+query|discord\s+bot|roman\s+empire|french\s+revolution|world\s+war|capital\s+of)\b/i;

  const STRONG_PHYSICS_RE =
    /\b(physics|force|newton|velocity|acceleration|deceleration|speed|distance|displacement|scalar|vector|free\s*fall|terminal\s+velocity|momentum|impulse|collision|kinetic|potential\s+energy|gravitational|gpe|work\s+done|power|watt|joule|efficiency|density|mass|volume|weight|gravity|moment|torque|pivot|equilibrium|centre\s+of\s+gravity|center\s+of\s+mass|hooke|spring\s+constant|extension|limit\s+of\s+proportionality|pressure|pascal|barometer|manometer|hydraulic|boyle|kelvin|absolute\s+zero|brownian|evaporation|boiling|melting|thermal|conduction|convection|infrared|radiation|emitter|absorber|specific\s+heat|latent\s+heat|wave|transverse|longitudinal|wavelength|frequency|amplitude|period|hertz|wavefront|diffraction|refraction|reflection|total\s+internal|critical\s+angle|snell|refractive\s+index|lens|focal\s+length|converging|diverging|principal\s+focus|dispersion|monochromatic|electromagnetic\s+spectrum|microwaves|ultraviolet|x-rays|gamma|ultrasound|echo|sonar|pitch|loudness|charge|coulomb|electrostatic|electric\s+field|current|ampere|amps|voltage|volt|emf|potential\s+difference|resistance|ohm|resistor|filament\s+lamp|diode|ldr|thermistor|series\s+circuit|parallel\s+circuit|potential\s+divider|kilowatt|kwh|fuse|earth\s+wire|circuit\s+breaker|magnet|magnetic|solenoid|electromagnet|fleming|motor\s+effect|commutator|electromagnetic\s+induction|lenz|faraday|generator|slip\s+ring|transformer|step-up|step-down|national\s+grid|atom|nucleus|proton|neutron|electron|isotope|nuclide|alpha|beta|radioactive|ionising|ionizing|background\s+radiation|geiger|half-life|decay|fission|fusion|rutherford|gold\s+foil|solar\s+system|orbit|orbital\s+speed|planet|comet|satellite|geostationary|protostar|main\s+sequence|red\s+giant|supergiant|white\s+dwarf|supernova|neutron\s+star|black\s+hole|galaxy|milky\s+way|light-year|redshift|big\s+bang|cosmic\s+microwave|hubble|0625|9702|igcse|mark\s+scheme|harony|haroony)\b/i;

  const FOLLOWUP_PHYSICS_RE =
    /\b(example|another|explain\s+more|simpler|why|how\s+come|what\s+if|formula|equation|unit|units|step|steps|rearrange|double|halve|increase|decrease|graph|diagram|quiz|question|solve|calculate|check)\b/i;

  const STOPWORDS = new Set([
    "the", "and", "for", "with", "from", "that", "this", "what", "how", "why",
    "explain", "define", "state", "describe", "calculate", "find", "determine",
    "into", "are", "was", "were", "has", "have", "when", "where", "which",
    "your", "about", "does", "can", "you", "give", "tell", "please", "physics",
    "question", "answer", "marks", "mark"
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
    const q = String(text || "")
      .replace(/×\s*10\s*\^?\s*([+-]?\d+)/gi, "e$1")
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
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m\s*(?:2|²|\^2)\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*cm\s*(?:2|²|\^2)\b/gi, 1e-4),
    ];
    const volumes = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*m\s*(?:3|³|\^3)\b/gi, 1),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*cm\s*(?:3|³|\^3)\b/gi, 1e-6),
    ];
    const pressures = [
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*mpa\b/gi, 1e6),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*kpa\b/gi, 1000),
      ...allMatches(/([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*pa\b/gi, 1),
    ];
    const temps = allMatches(/([+-]?\d+(?:\.\d+)?)\s*°\s*c\b/gi, 1);
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
      const rawG = text.match(/(\d+(?:\.\d+)?)\s*g\b/i);
      const rawCm3 = text.match(/(\d+(?:\.\d+)?)\s*cm\s*(?:3|³|\^3)\b/i);
      if (rawG && rawCm3 && !/\bkg\/m/i.test(low)) {
        const mg = parseFloat(rawG[1]), vc = parseFloat(rawCm3[1]);
        return makeStepCard("Density", "ρ = m / V", `m = ${mg} g, V = ${vc} cm³`, `ρ = ${mg} / ${vc}`, mg / vc, "g/cm³", `Equivalent SI density = ${fmtNum((mg / vc) * 1000)} kg/m³.`);
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
    if (/\b(average\s+speed|speed)\b/i.test(low) && u.dists.length && u.times.length && !u.freqs.length) {
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
  function searchChunksScored(kb, query, topK) {
    const qToks = tokenize(query).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
    if (!qToks.length) return [];
    const uniqueQ = Array.from(new Set(qToks));
    const isPaperCode = /\b20\d{2}\b|\bp[24]\b|\bq\d+/i.test(query);

    const scored = [];
    for (const ch of kb.chunks) {
      const topicLow = ((ch.unit || "") + " " + (ch.topic || "")).toLowerCase();
      const bodyLow = (ch.text || "").toLowerCase();
      let s = 0;
      let matchedTerms = 0;
      for (const tok of uniqueQ) {
        let hit = false;
        if (topicLow.includes(tok)) {
          s += 3.6;
          hit = true;
        }
        if (bodyLow.includes(tok)) {
          s += 1.5;
          hit = true;
        }
        if (hit) matchedTerms++;
      }
      const coverage = matchedTerms / uniqueQ.length;
      if (!isPaperCode && ch.source === "Harony Notes 2026 Final" && s > 0) s += 1.0;
      // Require meaningful semantic match so unrelated slides are NEVER returned!
      if (s >= 2.8 && (coverage >= 0.4 || matchedTerms >= 2)) {
        scored.push({ s, coverage, ch });
      }
    }
    scored.sort((a, b) => b.s - a.s);
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

  function searchVideos(kb, query, topK) {
    const qToks = Array.from(new Set(tokenize(query).filter((w) => !STOPWORDS.has(w) && w.length >= 3)));
    if (!qToks.length || !kb.videos) return [];
    const scored = [];
    for (const v of kb.videos) {
      const labelLow = (v.label || "").toLowerCase();
      const hay = `${labelLow} ${v.chapter || ""} ${v.unit || ""}`.toLowerCase();
      let hits = 0;
      for (const t of qToks) {
        if (labelLow.includes(t)) hits += 2.5;
        else if (hay.includes(t)) hits += 1.0;
      }
      if (hits >= 3.0 || (qToks.length === 1 && labelLow.includes(qToks[0]))) {
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

  // ==================== NEURAL TUTOR PROMPT BUILDER ====================
  function buildTutorMessages(userQuestion, history, solved, scoredChunks, msTips, ocrText) {
    const ctxLines = [];
    if (solved) {
      ctxLines.push(
        `VERIFIED EXACT CALCULATION (use these exact steps & numbers — do NOT recalculate differently):\n` +
          `Formula: ${solved.formula}\nResult: ${fmtNum(solved.value, 3)} ${solved.unit} (2 s.f.: ${fmtNum(solved.value, 2)} ${solved.unit})\n${solved.markdown}`
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
      "2. Answer the student's EXACT question directly, clearly, and naturally in your own words. NEVER dump raw slide fragments or answer an unrelated question.\n" +
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

  function buildSynthesizedFallbackReply(query, solved, scoredChunks, msTips, ocrText) {
    const sections = [];
    if (ocrText) {
      const prev = ocrText.slice(0, 320) + (ocrText.length > 320 ? "…" : "");
      sections.push(`**📷 Read from your uploaded photo/PDF:**\n> ${prev.replace(/\n/g, "\n> ")}`);
    }
    if (solved) {
      sections.push(solved.markdown);
    }
    // Check built-in crisp concept explanations
    let matchedCore = null;
    for (const c of CORE_CONCEPTS) {
      if (c.match.test(query)) {
        matchedCore = c;
        break;
      }
    }
    if (matchedCore && !solved) {
      const fams = matchedCore.formulas.map((f) => `- \`${f}\``).join("\n");
      const pts = matchedCore.points.map((p) => `- ${p}`).join("\n");
      sections.push(`### 📘 ${matchedCore.title}\n\n${matchedCore.summary}\n\n**Key Equations:**\n${fams}\n\n**Key Exam Points:**\n${pts}`);
    } else if (scoredChunks.length && (!solved || scoredChunks[0].s >= 5.5)) {
      const top = scoredChunks[0].ch;
      const bullets = cleanNoteBullets(top.text, top.topic)
        .slice(0, solved ? 4 : 7)
        .map((b) => `- ${b}`)
        .join("\n");
      const cite = top.page ? `${top.source} p.${top.page}` : top.source;
      sections.push(`### 📘 ${top.topic} *(${top.unit} — ${cite})*\n\n${bullets}`);
    }
    if (msTips.length && !solved) {
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

  function extractJsonBlock(raw) {
    if (!raw) return null;
    const str = String(raw).trim();
    const firstBrace = str.indexOf("{");
    const lastBrace = str.lastIndexOf("}");
    if (firstBrace === -1 || lastBrace <= firstBrace) return null;
    try {
      return JSON.parse(str.slice(firstBrace, lastBrace + 1));
    } catch (_) {
      return null;
    }
  }

  async function runNeuralAutoGrade(studentText, kb, onProgress) {
    const solved = solvePhysicsProblem(studentText);
    const scoredChunks = searchChunksScored(kb, studentText, 2);
    const msTips = searchMsTips(kb, studentText, 2);

    const refLines = [];
    if (solved) {
      refLines.push(`Verified Calculation: Formula ${solved.formula}, Exact Answer = ${fmtNum(solved.value, 3)} ${solved.unit}`);
    }
    for (const sc of scoredChunks) {
      refLines.push(`[Harony Notes - ${sc.ch.topic}]: ${cleanNoteBullets(sc.ch.text, sc.ch.topic).slice(0, 5).join("; ")}`);
    }
    for (const mt of msTips) {
      refLines.push(`[Official Harony MS (${mt.msTitle} Q${mt.q.question_id}, ${mt.q.max_marks} marks)]: ${mt.q.summary_answer}`);
    }

    // Try Neural AI Examiner JSON evaluation first!
    if (global.HaroonyNeuralAI) {
      try {
        const sys =
          "You are a strict, fair Cambridge IGCSE / A-Level Physics Examiner from Harony Physics Clinic.\n" +
          "Evaluate the student's Physics answer accurately. NEVER award marks for wrong Physics, opposite relationships, wrong units, or missing key explanations.\n" +
          "Respond with ONLY valid JSON in this exact format:\n" +
          "{\n" +
          '  "topic": "Short Physics Topic Name",\n' +
          '  "max_marks": 3,\n' +
          '  "awarded_marks": 2,\n' +
          '  "feedback": "1-3 clear sentences directly explaining what the student got right and any mistake or missing Physics concept.",\n' +
          '  "points": [\n' +
          '    {"code": "B1", "status": "awarded", "comment": "✓ Correctly stated ..."},\n' +
          '    {"code": "B1", "status": "missed", "comment": "✗ Did not mention ..."}\n' +
          "  ],\n" +
          '  "model_answer": "Complete full-mark Cambridge Physics answer."\n' +
          "}";

        const usr =
          `Student Submission:\n"""\n${studentText}\n"""\n\n` +
          (refLines.length ? `Verified Physics Ground Truth:\n${refLines.join("\n")}\n\n` : "") +
          "Grade this submission strictly using Cambridge Physics marking standards and return ONLY the JSON object.";

        const rawJson = await global.HaroonyNeuralAI.generate(
          [
            { role: "system", content: sys },
            { role: "user", content: usr },
          ],
          { maxTokens: 460, temperature: 0.1, onProgress }
        );
        const parsed = extractJsonBlock(rawJson);
        if (parsed && typeof parsed.max_marks === "number" && typeof parsed.awarded_marks === "number") {
          const maxM = Math.max(1, Math.min(10, Math.round(parsed.max_marks)));
          const awdM = Math.max(0, Math.min(maxM, Math.round(parsed.awarded_marks)));
          const pct = Math.round((1000 * awdM) / maxM) / 10;
          const pts = Array.isArray(parsed.points) && parsed.points.length
            ? parsed.points.map((p) => ({
                code: p.code || "B1",
                status: p.status === "awarded" ? "awarded" : "missed",
                comment: p.comment || "",
              }))
            : [
                {
                  code: "B1",
                  status: awdM > 0 ? "awarded" : "missed",
                  comment: parsed.feedback || "Evaluated by Haroony AI Examiner.",
                },
              ];
          return {
            ok: true,
            mode: "ai_examiner",
            mark_scheme_title: "Haroony AI Physics Examiner",
            awarded_marks: awdM,
            max_marks: maxM,
            percentage: pct,
            grade: gradeLetter(pct),
            examiner_feedback: parsed.feedback || "",
            questions: [
              {
                question_id: "AI Check",
                title: parsed.topic || (scoredChunks[0] ? scoredChunks[0].ch.topic : "Physics Evaluation"),
                awarded_marks: awdM,
                max_marks: maxM,
                points: pts,
                model_answer: parsed.model_answer || (solved ? `${solved.formula} => ${fmtNum(solved.value, 3)} ${solved.unit}` : ""),
              },
            ],
            extracted_student_text: studentText,
            notes_references: scoredChunks.map((x) => x.ch).filter((c) => c.image_url),
          };
        }
      } catch (err) {
        console.warn("Neural auto-grade fallback:", err);
      }
    }

    // Deterministic Smart Fallback if Neural JSON unavailable:
    // 1. If it's a calculation question, check student's answer numbers against exact symbolic solver!
    if (solved) {
      const ansOnly = extractStudentAnswerOnly(studentText);
      const eqParts = ansOnly.split("=");
      const finalChunk = eqParts.length >= 2 ? eqParts[eqParts.length - 1] : ansOnly;
      const stuNums = extractNumbers(finalChunk);
      const hasAns = stuNums.some((n) => Math.abs(n - solved.value) <= 0.04 * Math.max(1, Math.abs(solved.value)));
      const hasFormulaOrSub = eqParts.length >= 2 && (hasAns || /[\/×*+\-]/.test(ansOnly));
      const awd = hasAns ? 2 : hasFormulaOrSub && !stuNums.length ? 1 : 0;
      const pct = (awd / 2) * 100;
      return {
        ok: true,
        mode: "symbolic_check",
        mark_scheme_title: `Physics Calculation Check (${solved.topic})`,
        awarded_marks: awd,
        max_marks: 2,
        percentage: pct,
        grade: gradeLetter(pct),
        examiner_feedback: hasAns
          ? `Great work! Your final answer matches the exact value (${fmtNum(solved.value, 3)} ${solved.unit}) using ${solved.formula}.`
          : `Check your formula and substitution: using ${solved.formula}, the correct result is ${fmtNum(solved.value, 3)} ${solved.unit}.`,
        questions: [
          {
            question_id: "Calc",
            title: solved.topic,
            awarded_marks: awd,
            max_marks: 2,
            points: [
              {
                code: "C1",
                status: hasAns ? "awarded" : "missed",
                comment: `Formula required: \`${solved.formula}\``,
              },
              {
                code: "A1",
                status: hasAns ? "awarded" : "missed",
                comment: hasAns
                  ? `✓ Correct numerical answer matching \`${fmtNum(solved.value, 3)} ${solved.unit}\`.`
                  : `✗ Expected final value \`${fmtNum(solved.value, 3)} ${solved.unit}\`.`,
              },
            ],
            model_answer: `${solved.formula} → ${fmtNum(solved.value, 3)} ${solved.unit}`,
          },
        ],
        extracted_student_text: studentText,
        notes_references: scoredChunks.map((x) => x.ch).filter((c) => c.image_url),
      };
    }

    // 2. If a closely matching MS question exists in Harony 2026/2027, grade against its real rubric!
    if (msTips.length > 0 && msTips[0].hits >= 3) {
      const best = msTips[0];
      const gq = gradeAgainstQuestion(best.q, studentText);
      const pct = Math.round((1000 * gq.awarded_marks) / Math.max(1, gq.max_marks)) / 10;
      return {
        ok: true,
        mode: "auto_matched_ms",
        mark_scheme_title: `${best.msTitle} (Auto-Matched)`,
        awarded_marks: gq.awarded_marks,
        max_marks: gq.max_marks,
        percentage: pct,
        grade: gradeLetter(pct),
        questions: [gq],
        extracted_student_text: studentText,
        notes_references: scoredChunks.map((x) => x.ch).filter((c) => c.image_url),
      };
    }

    // 3. Evaluate key concepts from Harony Notes (checking real keyword overlap + contradictions, NEVER blind 100%!)
    if (scoredChunks.length > 0) {
      const top = scoredChunks[0].ch;
      const bullets = cleanNoteBullets(top.text, top.topic).slice(0, 3);
      const stuToks = expandWithSynonyms(new Set(tokenize(studentText)));
      const pts = [];
      let awd = 0;
      for (const b of bullets) {
        const bWords = tokenize(b).filter((w) => !STOPWORDS.has(w) && w.length >= 4);
        if (!bWords.length) continue;
        const contradicted = hasOppositeContradiction(b.toLowerCase(), studentText.toLowerCase());
        const hitCount = bWords.filter((w) => stuToks.has(w)).length;
        const ok = !contradicted && hitCount / bWords.length >= 0.35;
        if (ok) awd++;
        pts.push({
          code: "B1",
          status: ok ? "awarded" : "missed",
          comment: ok ? `✓ Covered key concept: ${b}` : `✗ Missing/incomplete concept: ${b}`,
        });
      }
      const maxM = Math.max(1, pts.length);
      const pct = Math.round((1000 * awd) / maxM) / 10;
      return {
        ok: true,
        mode: "concept_check",
        mark_scheme_title: `Harony Notes Concept Check — ${top.topic}`,
        awarded_marks: awd,
        max_marks: maxM,
        percentage: pct,
        grade: gradeLetter(pct),
        questions: [
          {
            question_id: "Concept",
            title: top.topic,
            awarded_marks: awd,
            max_marks: maxM,
            points: pts,
            model_answer: bullets.join(" ; "),
          },
        ],
        extracted_student_text: studentText,
        notes_references: scoredChunks.map((x) => x.ch).filter((c) => c.image_url),
      };
    }

    return {
      ok: false,
      error: "Please include the Physics question or topic along with your answer (or select a Mark Scheme in Step 2) so I can grade it accurately.",
    };
  }

  // ==================== PUBLIC ENGINE API ====================
  global.HaroonyBrowserEngine = {
    solvePhysicsProblem,

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
      const kb = await ensureKB();

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
            "Ask me any **IGCSE or A-Level Physics** question, upload a photo or PDF of a Physics problem, or open **Check Answer** above to mark your work!",
          notes_cards: [],
          video_links: [],
          engine: "Haroony AI",
        };
      }

      // Expand short follow-up queries using last user turn for retrieval
      let searchQuery = combined;
      if (tokenize(combined).length <= 6 && hasPrior) {
        const prevUsers = history.filter((h) => h && h.role === "user" && h.content);
        if (prevUsers.length) {
          searchQuery = prevUsers[prevUsers.length - 1].content + " " + combined;
        }
      }

      const solved = solvePhysicsProblem(combined);
      const scoredChunks = searchChunksScored(kb, searchQuery, 3);
      const msTips = searchMsTips(kb, searchQuery, 2);
      const vids = searchVideos(kb, searchQuery, 2);

      // Only attach visual reference cards if strongly relevant to the question!
      const strongCards = scoredChunks
        .filter((x) => x.s >= 5.0)
        .slice(0, 2)
        .map((x) => x.ch);

      let finalReply = null;
      if (global.HaroonyNeuralAI) {
        try {
          const messages = buildTutorMessages(userMsg, history, solved, scoredChunks, msTips, ocrText);
          finalReply = await global.HaroonyNeuralAI.generate(messages, {
            maxTokens: 560,
            temperature: 0.2,
            onProgress,
            onToken,
          });
        } catch (err) {
          console.warn("Neural chat fallback:", err);
        }
      }

      if (!finalReply) {
        finalReply = buildSynthesizedFallbackReply(searchQuery, solved, scoredChunks, msTips, ocrText);
      } else if (ocrText && !finalReply.includes("Read from")) {
        const prev = ocrText.slice(0, 240) + (ocrText.length > 240 ? "…" : "");
        finalReply = `**📷 Read from your uploaded file:**\n> ${prev.replace(/\n/g, "\n> ")}\n\n${finalReply}`;
      }

      return {
        ok: true,
        refused: false,
        category: "physics",
        reply: finalReply,
        notes_cards: strongCards,
        video_links: vids,
        ocr_text: ocrText || null,
        engine: "Haroony AI",
      };
    },

    async grade({ studentText, studentFile, msId, msText, msFile, questionId, onProgress }) {
      const kb = await ensureKB();

      let text = String(studentText || "").trim();
      if (studentFile && global.HaroonyNeuralAI) {
        const stuOcr = await global.HaroonyNeuralAI.extractDocumentText(studentFile, onProgress);
        if (stuOcr) text = `${text}\n${stuOcr}`.trim();
      }
      if (!text) {
        return {
          ok: false,
          error: "Please write your Physics answer or upload a clear photo/PDF of your work in Step 1.",
        };
      }

      let customMsContent = String(msText || "").trim();
      if (msFile && global.HaroonyNeuralAI) {
        const msOcr = await global.HaroonyNeuralAI.extractDocumentText(msFile, onProgress);
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
        const codeMatch = text.match(/\b(20\d{2})\s+(March|June|Nov(?:ember)?)\s+P([24])\s+V(\d)\s+Q?0?(\d+)/i);
        if (codeMatch && kb.mark_schemes) {
          for (const m of kb.mark_schemes) {
            const hit = (m.questions || []).find((q) => {
              const idLow = q.question_id.toLowerCase();
              return (
                idLow.includes(codeMatch[1]) &&
                idLow.includes(codeMatch[2].toLowerCase().slice(0, 3)) &&
                idLow.includes("q0" + codeMatch[5]) || idLow.includes("q" + codeMatch[5])
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
          const sToks = tokenize(text).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
          const ranked = qs
            .map((q) => {
              const hay = ((q.title || "") + " " + (q.summary_answer || "")).toLowerCase();
              const sc = sToks.reduce((acc, t) => acc + (hay.includes(t) ? 1 : 0), 0);
              return { sc, q };
            })
            .sort((a, b) => b.sc - a.sc);
          qs = ranked.slice(0, 1).map((x) => x.q);
        }

        const graded = qs.map((q) => gradeAgainstQuestion(q, text));
        const awarded = graded.reduce((a, x) => a + x.awarded_marks, 0);
        const maxM = Math.max(1, graded.reduce((a, x) => a + x.max_marks, 0));
        const pct = Math.round((1000 * awarded) / maxM) / 10;

        // Generate personalized AI Examiner coaching comment when Neural AI is ready!
        let examinerFeedback = "";
        if (global.HaroonyNeuralAI) {
          try {
            const rubSummary = graded
              .map((g) => `Q ${g.title} (${g.awarded_marks}/${g.max_marks}): Expected: ${g.model_answer}`)
              .join("\n");
            examinerFeedback =
              (await global.HaroonyNeuralAI.generate(
                [
                  {
                    role: "system",
                    content:
                      "You are Mr. Haroony's IGCSE Physics Examiner. Write a concise, encouraging 2-sentence personalized coaching summary explaining why the student earned their score and how to capture any missed mark-scheme keywords.",
                  },
                  {
                    role: "user",
                    content: `Student Answer:\n${text.slice(0, 800)}\n\nMark Scheme Breakdown:\n${rubSummary}`,
                  },
                ],
                { maxTokens: 160, temperature: 0.2, onProgress }
              )) || "";
          } catch (_) {}
        }

        const refs = searchChunksScored(kb, graded.map((g) => g.model_answer).join(" ") + " " + text, 2)
          .map((x) => x.ch)
          .filter((c) => c.image_url);

        return {
          ok: true,
          mode: "mark_scheme",
          mark_scheme_title: msObj.title,
          awarded_marks: awarded,
          max_marks: maxM,
          percentage: pct,
          grade: gradeLetter(pct),
          examiner_feedback: examinerFeedback,
          questions: graded,
          extracted_student_text: text,
          notes_references: refs,
        };
      }

      // Mode 1: Smart AI Examiner Auto-Check (No Mark Scheme Needed!)
      return await runNeuralAutoGrade(text, kb, onProgress);
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
})(window);
