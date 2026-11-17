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
