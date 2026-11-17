"""Instant First-Principles Physics Reasoner for Proportionality, Comparisons & Definitions.

Supplements the SymPy calculation solver and Harony 2026/2027 RAG so Haroony AI answers
'what happens if...', 'compare...', and 'difference between...' questions accurately and instantly
even without an external cloud API key.
"""
from __future__ import annotations

import re
from typing import Optional


def _factor_from_text(text: str) -> Optional[tuple[float, str]]:
    low = text.lower()
    if re.search(r"\b(doubled?|twice|2\s*times|x\s*2)\b", low):
        return 2.0, "doubled (×2)"
    if re.search(r"\b(tripled?|three\s+times|3\s*times|x\s*3)\b", low):
        return 3.0, "tripled (×3)"
    if re.search(r"\b(quadrupled?|four\s+times|4\s*times)\b", low):
        return 4.0, "quadrupled (×4)"
    if re.search(r"\b(halved?|half)\b", low):
        return 0.5, "halved (×½)"
    if re.search(r"\b(quartered?|one\s+quarter)\b", low):
        return 0.25, "quartered (×¼)"
    if re.search(r"\b(increased?|greater|larger|higher|more)\b", low):
        return 1.5, "increased"
    if re.search(r"\b(decreased?|reduced?|smaller|lower|less)\b", low):
        return 0.7, "decreased"
    return None


def try_reason_physics_variation(query: str) -> Optional[str]:
    low = (query or "").lower()

    # 0. Everyday & Core Causal Physics ("Why does...", "Explain why...")
    if "pressure cooker" in low:
        return (
            "### ⚡ Why a Pressure Cooker Cooks Food Faster *(Unit 2: Thermal Physics)*\n\n"
            "- A pressure cooker has a **sealed lid** that traps steam above the water, **increasing the gas pressure** above normal atmospheric pressure.\n"
            "- Higher pressure **raises the boiling point of water** above `100 °C` (typically `~120 °C`).\n"
            "- Because the water and steam reach a **higher temperature**, thermal energy transfers into the food more rapidly, cooking it faster."
        )
    if (any(w in low for w in ("swimming", "pool", "sweat", "wet", "out of water", "step out")) and any(w in low for w in ("cold", "cool", "wind", "windy", "breeze"))) or (
        "wind" in low and "evaporat" in low
    ):
        return (
            "### ⚡ Evaporation Cooling & Effect of Wind *(Unit 2: Thermal Physics)*\n\n"
            "- Water on your skin **evaporates**: the **fastest, highest-kinetic-energy molecules** overcome intermolecular forces and escape from the liquid surface.\n"
            "- This lowers the **average kinetic energy** of the remaining water molecules (and absorbs latent heat from your skin), causing **cooling**.\n"
            "- **Wind** continuously blows away humid air (water vapour) from above the skin, **increasing the rate of evaporation** and making you feel colder faster."
        )
    if ("ice" in low and "float" in low) or ("ice" in low and "dense" in low):
        return (
            "### ⚡ Why Ice Floats on Liquid Water *(Unit 1 & 2)*\n\n"
            "- When water freezes, its molecules form an open crystal lattice that takes up **more volume** for the same mass.\n"
            "- Therefore ice has a **lower density** (`≈ 0.92 g/cm³`) than liquid water (`1.0 g/cm³`), so it floats."
        )
    if ("spoon" in low or "metal" in low) and ("wood" in low or "plastic" in low) and any(w in low for w in ("cold", "colder", "warm", "hot", "touch", "feel")):
        return (
            "### ⚡ Why Metal Feels Colder Than Wood at Room Temperature *(Unit 2: Conduction)*\n\n"
            "- Both are at the **exact same room temperature**.\n"
            "- Metal has **delocalised (free) electrons**, making it a **good thermal conductor** that conducts heat **away from your warm hand much faster** than wood (a thermal insulator)."
        )
    if any(w in low for w in ("coat", "sweater", "blanket", "jacket", "double glazing", "fur", "feathers")) and any(w in low for w in ("warm", "heat", "insulat", "cold")):
        return (
            "### ⚡ How Coats, Fur & Double Glazing Reduce Heat Loss *(Unit 2: Thermal Physics)*\n\n"
            "- Wool, feathers, and double-glazed windows **trap small pockets of air**.\n"
            "- **Air is a poor thermal conductor** (reduces conduction), and trapping it in small pockets stops it circulating (**reduces convection**)."
        )
    if "sea breeze" in low or "land breeze" in low:
        return (
            "### ⚡ Sea Breeze & Land Breeze *(Unit 2: Convection)*\n\n"
            "- **Day (Sea Breeze):** Land has a lower specific heat capacity than water, so land heats up faster. Air above the warm land expands, becomes **less dense and rises**, while cooler denser air from over the sea flows in to replace it.\n"
            "- **Night (Land Breeze):** Land cools faster than the sea; warmer air over the sea rises and cooler air from the land flows out towards the sea."
        )
    if "bulb" in low and "parallel" in low and any(w in low for w in ("bright", "brightness", "same", "add")):
        return (
            "### ⚡ Adding a Bulb in Parallel *(Unit 3: Electric Circuits)*\n\n"
            "- In a **parallel circuit**, every branch is connected directly across the **full supply potential difference (voltage)**.\n"
            "- Adding another bulb in parallel does **not** change the p.d. across the existing bulbs, so each bulb keeps the **same current and same brightness** (while total current from the battery increases)."
        )
    if "total internal reflection" in low or ("tir" in low.split() and "condition" in low):
        return (
            "### ⚡ Total Internal Reflection (TIR) *(Unit 6: Light & Optics)*\n\n"
            "- **Definition:** Complete reflection of light back inside a denser medium at the boundary with a less dense medium (zero refraction out).\n"
            "- **Two Required Conditions:**\n"
            "  1. Light must travel from an **optically denser medium to a less dense medium** (e.g. glass/water → air).\n"
            "  2. The **angle of incidence must be greater than the critical angle** (`i > c`, where `sin c = 1 / n`).\n"
            "- **Applications:** Optical fibres (high-speed internet & medical endoscopes) and reflecting prisms in periscopes/binoculars."
        )

    # 1. Comparisons / Differences
    if re.search(r"\b(?:difference|compare|distinguish|contrast|vs\.?|versus)\b", low):
        if "evaporat" in low and "boil" in low:
            return (
                "### ⚡ Evaporation vs Boiling *(Unit 3 — Thermal Physics)*\n\n"
                "- **Temperature:** Evaporation occurs at **any temperature** below the boiling point; boiling occurs only at a **fixed boiling point**.\n"
                "- **Location:** Evaporation happens **only at the surface** of the liquid; boiling happens **throughout the entire liquid** (forming bubbles inside).\n"
                "- **Rate & Cooling:** Faster, more energetic molecules escape from the surface during evaporation, lowering the average kinetic energy of remaining molecules (**cooling effect**)."
            )
        if "mass" in low and "weight" in low:
            return (
                "### ⚡ Mass vs Weight *(Unit 1 — Mechanics)*\n\n"
                "- **Mass (`m`):** Measure of the quantity of matter in an object (and its resistance to change in motion); **scalar**, measured in **kg** with a balance; **constant everywhere**.\n"
                "- **Weight (`W = m × g`):** **Gravitational force** acting on a mass; **vector** acting downwards, measured in **N** with a newtonmeter; **changes** if gravitational field strength `g` changes."
            )
        if "scalar" in low and "vector" in low:
            return (
                "### ⚡ Scalar vs Vector Quantities *(Unit 1 — Mechanics)*\n\n"
                "- **Scalar:** Has **magnitude (size) only** — e.g. distance, speed, mass, time, energy, temperature.\n"
                "- **Vector:** Has **both magnitude and direction** — e.g. displacement, velocity, acceleration, force, weight, momentum."
            )
        if ("speed" in low and "velocity" in low) or ("distance" in low and "displacement" in low):
            return (
                "### ⚡ Speed vs Velocity & Distance vs Displacement *(Unit 1 — Kinematics)*\n\n"
                "- **Distance / Speed** are **scalars** (magnitude only). **Displacement / Velocity** are **vectors** (magnitude **and direction**).\n"
                "- An object moving in a **circle at constant speed** has **changing velocity** because its **direction** is continuously changing (so it is accelerating towards the centre)."
            )
        if "series" in low and "parallel" in low:
            return (
                "### ⚡ Series vs Parallel Circuits *(Unit 5 — Electricity)*\n\n"
                "- **Series:** **Same current** everywhere (`I₁ = I₂`); total p.d. is **shared** (`V = V₁ + V₂`); combined resistance increases (`R = R₁ + R₂`).\n"
                "- **Parallel:** **Same p.d.** across every branch (`V₁ = V₂`); total current **splits** at junctions (`I = I₁ + I₂`); combined resistance is **less than the smallest individual resistor** (`1/R = 1/R₁ + 1/R₂`)."
            )
        if ("a.c" in low or "ac " in low or "alternating" in low) and ("d.c" in low or "dc " in low or "direct" in low):
            return (
                "### ⚡ Direct Current (d.c.) vs Alternating Current (a.c.) *(Unit 5/6)*\n\n"
                "- **d.c.:** Flows in **one direction only** (e.g. from a battery or cell).\n"
                "- **a.c.:** Continuously **reverses direction** periodically (e.g. UK mains at **50 Hz**, producing a changing magnetic field needed for **transformers**)."
            )
        if ("fission" in low and "fusion" in low):
            return (
                "### ⚡ Nuclear Fission vs Nuclear Fusion *(Unit 7 — Nuclear Physics)*\n\n"
                "- **Nuclear Fission:** A **heavy unstable nucleus** (e.g. Uranium-235) **splits** into two smaller daughter nuclei + neutrons + energy.\n"
                "- **Nuclear Fusion:** Two **light nuclei** (e.g. hydrogen isotopes) **join together** at extremely high temperature & pressure to form a heavier nucleus (helium) + energy (powers the Sun/stars)."
            )
        if ("alpha" in low or "beta" in low or "gamma" in low) and sum(k in low for k in ("alpha", "beta", "gamma")) >= 2:
            return (
                "### ⚡ Alpha (α), Beta (β) and Gamma (γ) Radiation *(Unit 5 — Nuclear Physics)*\n\n"
                "- **Alpha (α, ⁴₂He):** High mass, `+2` charge → **strongest ionisation**, **least penetrating** (stopped by paper / few cm of air).\n"
                "- **Beta (β⁻, fast electron):** Small mass, `-1` charge → **moderate ionisation**, stopped by **few mm of aluminium**.\n"
                "- **Gamma (γ, EM wave):** No mass, no charge → **weakest ionisation**, **most penetrating** (reduced by thick lead/concrete)."
            )
        if "conduction" in low and ("convection" in low or "radiation" in low):
            return (
                "### ⚡ Conduction, Convection & Infrared Radiation *(Unit 2 — Thermal Physics)*\n\n"
                "- **Conduction (solids):** Particles vibrate and pass energy to neighbours; metals conduct fastest via **delocalised (free) electrons**.\n"
                "- **Convection (liquids & gases only):** Heated fluid expands, becomes **less dense and rises**, while cooler denser fluid sinks (**convection current**).\n"
                "- **Infrared Radiation (no medium needed):** Electromagnetic wave transfer — works through a **vacuum**; dull black surfaces are best emitters/absorbers."
            )
        if "transverse" in low and "longitudinal" in low:
            return (
                "### ⚡ Transverse vs Longitudinal Waves *(Unit 6 — Waves)*\n\n"
                "- **Transverse:** Vibrations are **perpendicular (at 90°)** to wave direction (peaks & troughs) — e.g. light, water waves, all electromagnetic waves.\n"
                "- **Longitudinal:** Vibrations are **parallel** to wave direction (**compressions and rarefactions**) — e.g. sound waves, ultrasound."
            )
        if "soft iron" in low and "steel" in low:
            return (
                "### ⚡ Soft Iron vs Steel *(Unit 4 — Electromagnetism)*\n\n"
                "- **Soft Iron:** Magnetises and demagnetises **easily** (temporary magnet) → used for **electromagnet cores & transformer cores**.\n"
                "- **Steel:** Hard to magnetise and **retains magnetism** (permanent magnet) → used for bar magnets & compass needles."
            )
        if ("real" in low and "virtual" in low) and ("image" in low or "lens" in low or "mirror" in low):
            return (
                "### ⚡ Real Image vs Virtual Image *(Unit 6 — Light & Lenses)*\n\n"
                "- **Real Image:** Formed where real light rays **actually converge/intersect**; **can be projected onto a screen** (inverted).\n"
                "- **Virtual Image:** Light rays only **appear to diverge** from behind the mirror/lens; **cannot be formed on a screen** (upright)."
            )

    # 2. Proportionality / "What happens if ..."
    if re.search(r"\b(what\s+happens|how\s+does|effect\s+on|change\s+in|doubles?|triples?|halves?|halved|doubled|tripled|increased?|decreased?)\b", low):
        fac = _factor_from_text(low)
        if fac:
            k, k_txt = fac
            if "kinetic energy" in low and any(w in low for w in ("speed", "velocity")):
                out_k = k * k
                return (
                    f"### ⚡ Effect on Kinetic Energy (`KE = ½ m v²`)\n\n"
                    f"- Kinetic energy is **directly proportional to the square of speed** (`KE ∝ v²`).\n"
                    f"- If speed is **{k_txt}**, kinetic energy changes by a factor of `({k:g})² = {out_k:g}×` (assuming constant mass)."
                )
            if "resistance" in low and any(w in low for w in ("length", "longer", "shorter")):
                return (
                    f"### ⚡ Effect on Resistance (`R = ρ L / A`)\n\n"
                    f"- Resistance of a uniform wire is **directly proportional to its length** (`R ∝ L`).\n"
                    f"- If length is **{k_txt}**, the resistance changes by **{k:g}×** (assuming constant cross-sectional area and temperature)."
                )
            if "resistance" in low and any(w in low for w in ("diameter", "radius")):
                out_k = 1.0 / (k * k)
                return (
                    f"### ⚡ Effect on Resistance (`R = ρ L / A`)\n\n"
                    f"- Cross-sectional area `A = π r² = π (d/2)²`, so `A ∝ d²`, and resistance is **inversely proportional to area** (`R ∝ 1 / d²`).\n"
                    f"- If diameter/radius is **{k_txt}**, area changes by `{k*k:g}×`, so resistance becomes **`{out_k:g}×`** its original value."
                )
            if "resistance" in low and "area" in low:
                out_k = 1.0 / k
                return (
                    f"### ⚡ Effect on Resistance (`R = ρ L / A`)\n\n"
                    f"- Resistance is **inversely proportional to cross-sectional area** (`R ∝ 1 / A`).\n"
                    f"- If area is **{k_txt}**, resistance changes by **`{out_k:g}×`**."
                )
            if "pressure" in low and "volume" in low and any(w in low for w in ("gas", "boyle", "temperature")):
                out_k = 1.0 / k
                return (
                    f"### ⚡ Boyle's Law Effect on Gas Pressure (`p₁V₁ = p₂V₂`)\n\n"
                    f"- At **constant temperature**, pressure of a fixed mass of gas is **inversely proportional to volume** (`p ∝ 1 / V`).\n"
                    f"- If volume is **{k_txt}**, particles collide with container walls more/less frequently, so pressure becomes **`{out_k:g}×`**."
                )
            if "pressure" in low and "area" in low:
                out_k = 1.0 / k
                return (
                    f"### ⚡ Effect on Solid Pressure (`p = F / A`)\n\n"
                    f"- For a constant force, pressure is **inversely proportional to contact area** (`p ∝ 1 / A`).\n"
                    f"- If area is **{k_txt}**, pressure becomes **`{out_k:g}×`**."
                )
            if "wavelength" in low and "frequency" in low:
                out_k = 1.0 / k
                return (
                    f"### ⚡ Wave Equation Relationship (`v = f × λ`)\n\n"
                    f"- At **constant wave speed** (same medium), frequency and wavelength are **inversely proportional** (`λ = v / f`).\n"
                    f"- If frequency is **{k_txt}**, wavelength becomes **`{out_k:g}×`**."
                )
            if "transformer" in low and "turns" in low:
                return (
                    f"### ⚡ Transformer Turns Ratio (`V_p / V_s = N_p / N_s`)\n\n"
                    f"- Secondary voltage is **directly proportional to the number of secondary turns** (`V_s ∝ N_s`).\n"
                    f"- If secondary turns are **{k_txt}**, output voltage `V_s` changes by **`{k:g}×`**."
                )

    return None
