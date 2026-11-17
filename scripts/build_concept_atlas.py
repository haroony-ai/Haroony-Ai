"""Build a comprehensive Cambridge IGCSE (0625) & A-Level Physics Concept Atlas
grounded in Harony Notes 2026 Final (243 pages), ClassWork 2027, P2 CLS 2027, and P4 CLS 2027.
Saved to data/harony_concept_atlas.json and bundled into static/browser_kb.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import DATA_DIR

ATLAS = [
    # ==================== UNIT 1: GENERAL PHYSICS ====================
    {
        "id": "u1_measurements_scalars_vectors",
        "unit": "Unit 1: General Physics",
        "topic": "Measurements, Scalars & Vectors",
        "keywords": ["scalar", "vector", "parallax", "meniscus", "micrometer", "vernier", "pendulum", "oscillation", "resultant", "measuring cylinder"],
        "summary": "A **scalar** has magnitude only (e.g. mass, speed, distance, energy, time, temperature). A **vector** has both magnitude and direction (e.g. force, velocity, acceleration, weight, displacement, momentum).",
        "formulas": ["Vector resultant (perpendicular): R = √(F₁² + F₂²)", "tan θ = F_y / F_x", "Period T = total time / N oscillations"],
        "points": [
            "**Measuring Cylinder & Parallax Error:** Read the **bottom of the meniscus** with your eye level **perpendicular** to the scale.",
            "**Simple Pendulum:** Time **20 oscillations** from the centre/fiducial mark and divide by 20 to reduce human reaction-time percentage error.",
            "**Micrometer Screw Gauge:** Precision `0.01 mm` (main scale + thimble reading minus zero error).",
        ],
        "examiner_trap": "Remember: **speed** and **distance** are scalars, whereas **velocity**, **displacement**, and **momentum** are vectors!",
        "pages": [4, 5, 6, 7],
    },
    {
        "id": "u1_kinematics_motion_graphs",
        "unit": "Unit 1: General Physics",
        "topic": "Kinematics & Motion Graphs",
        "keywords": ["distance-time", "speed-time", "velocity-time", "motion graph", "acceleration", "deceleration", "gradient", "slope", "area under", "terminal velocity", "free fall", "freefall", "uniform acceleration"],
        "summary": "Motion graphs describe speed, acceleration, and distance travelled using **gradient (slope)** and **area under the graph**.",
        "formulas": ["v = d / t (constant speed)", "a = (v − u) / t", "d = ((u + v) / 2) × t (constant acceleration)", "Distance = Area under speed–time graph"],
        "points": [
            "**Distance–Time Graph:** Gradient = **speed**. Horizontal line = at rest (`speed = 0`); straight line sloping up = constant speed; curve getting steeper = acceleration; curve getting flatter = deceleration.",
            "**Speed–Time Graph:** Gradient = **acceleration**; **Area under graph** = **distance travelled**. Horizontal line = constant speed (`a = 0`); straight upward line = uniform acceleration; downward line = uniform deceleration.",
            "**Terminal Velocity (Falling Object):** Initially `weight > air resistance`, so object accelerates (`a = g = 9.8 m/s²` at `t = 0`). As speed rises, air resistance increases, resultant force and acceleration decrease until **air resistance = weight** (`resultant force = 0`), reaching constant **terminal velocity**.",
        ],
        "examiner_trap": "Never say 'the object stops' for a horizontal line on a **speed–time** graph unless speed = 0; a non-zero horizontal line means **constant speed**!",
        "pages": [14, 15, 16, 17, 18],
    },
    {
        "id": "u1_mass_weight_density",
        "unit": "Unit 1: General Physics",
        "topic": "Mass, Weight & Density",
        "keywords": ["mass", "weight", "density", "float", "sink", "ship", "displacement can", "gravitational field strength", "kilogram", "newton"],
        "summary": "**Mass** (`kg`) is the amount of matter (resists change in motion / inertia) and stays constant everywhere. **Weight** (`N`) is the gravitational force on a mass (`W = m × g`). **Density** (`ρ = m / V`) determines whether an object floats or sinks.",
        "formulas": ["W = m × g  (g ≈ 9.8 N/kg on Earth)", "ρ = m / V"],
        "points": [
            "**Why a steel ship floats:** Although steel is denser than water, the ship is hollow and contains **air spaces**, so the **overall average density** of the ship is **less than the density of water**.",
            "**Irregular Solid Density:** Measure mass on a digital balance; find volume using a measuring cylinder or **displacement (Eureka) can**; calculate `ρ = m / V`.",
            "**Unit Conversion:** `1 g/cm³ = 1000 kg/m³`.",
        ],
        "examiner_trap": "Do not confuse mass (`kg`) with weight (`N`). On the Moon, **mass stays the same**, while **weight decreases** because `g` is smaller.",
        "pages": [20, 21, 22, 23],
    },
    {
        "id": "u1_forces_newtons_laws_hooke",
        "unit": "Unit 1: General Physics",
        "topic": "Forces, Newton's Laws & Hooke's Law",
        "keywords": ["newton", "inertia", "resultant force", "hooke", "spring constant", "limit of proportionality", "extension", "circular motion", "centripetal", "friction"],
        "summary": "A **resultant force** changes an object's velocity (`F = m × a`), direction (circular motion), or shape (Hooke's law).",
        "formulas": ["F = m × a", "F = k × x  (Hooke's Law)", "k = F / x"],
        "points": [
            "**Newton's 1st Law:** An object remains at rest or moves at **constant velocity** unless acted upon by a non-zero **resultant force**.",
            "**Newton's 2nd Law:** `F = m × a` — resultant force equals mass × acceleration (in the same direction).",
            "**Newton's 3rd Law:** Body A exerts a force on B; B exerts an **equal and opposite** force of the **same type** on A.",
            "**Hooke's Law:** Extension (`x`) is directly proportional to load (`F`) up to the **limit of proportionality** (straight line through the origin on a force–extension graph).",
            "**Circular Motion:** Even at constant speed, **velocity changes** because **direction changes continuously**, requiring a **resultant centripetal force towards the centre**.",
        ],
        "examiner_trap": "On an extension–load graph, **limit of proportionality** is where the straight line starts to curve.",
        "pages": [24, 25, 26, 27, 28, 29, 30],
    },
    {
        "id": "u1_moments_equilibrium_cog",
        "unit": "Unit 1: General Physics",
        "topic": "Moments, Equilibrium & Centre of Gravity",
        "keywords": ["moment", "pivot", "fulcrum", "clockwise", "anticlockwise", "equilibrium", "centre of gravity", "center of gravity", "topple", "stability", "plumb line"],
        "summary": "The **moment of a force** is the turning effect about a pivot: `Moment = Force × perpendicular distance from pivot`.",
        "formulas": ["M = F × d  (unit: N m)", "Equilibrium: Σ Clockwise moments = Σ Anticlockwise moments"],
        "points": [
            "**Two Conditions for Equilibrium:** (1) **No resultant force** (`ΣF = 0`), and (2) **No resultant moment** (`ΣM = 0`) about any pivot.",
            "**Stability:** An object is more stable when it has a **lower centre of gravity** and a **wider base**. It topples when the vertical line through its centre of gravity falls outside its base.",
            "**Irregular Lamina Centre of Gravity:** Hang freely from a pin and use a **plumb line** to mark vertical lines from two suspension holes; their intersection is the centre of gravity.",
        ],
        "examiner_trap": "Always state **perpendicular distance** from the pivot — just writing `distance` often loses the definition mark!",
        "pages": [33, 34, 35, 36],
    },
    {
        "id": "u1_momentum_impulse",
        "unit": "Unit 1: General Physics",
        "topic": "Momentum & Impulse",
        "keywords": ["momentum", "impulse", "collision", "explosion", "airbag", "crumple zone", "seatbelt", "conservation of momentum"],
        "summary": "**Momentum** (`p = m × v`) is conserved in closed systems where no external resultant force acts.",
        "formulas": ["p = m × v  (unit: kg m/s or N s)", "Impulse = F × Δt = Δp = mv − mu", "Total p_before = Total p_after"],
        "points": [
            "**Safety Features (Airbags / Crumple Zones):** Increase the **collision time (`Δt`)** for the same change in momentum (`Δp`), so the **rate of change of momentum (`F = Δp/Δt`) decreases**, reducing the force on the passenger.",
            "**Direction Matters:** Momentum is a **vector** — assign `+` to one direction and `−` to the opposite direction.",
        ],
        "examiner_trap": "Never say airbags 'reduce momentum' — the change in momentum is the same; airbags **increase the time taken** so the **force decreases**.",
        "pages": [38, 39, 40],
    },
    {
        "id": "u1_work_energy_power_efficiency",
        "unit": "Unit 1: General Physics",
        "topic": "Work, Energy, Power & Efficiency",
        "keywords": ["work done", "kinetic energy", "gravitational potential", "gpe", "power", "efficiency", "conservation of energy", "renewable", "non-renewable", "sankey"],
        "summary": "Energy cannot be created or destroyed, only transferred between stores (**kinetic, gravitational potential, elastic, thermal, chemical, nuclear**).",
        "formulas": ["W = F × d = ΔE  (J)", "E_k = ½ m v²", "ΔE_p = m g Δh", "P = E / t = W / t  (W)", "Efficiency = (useful output / total input) × 100%"],
        "points": [
            "**Doubling Speed:** Because `E_k = ½ m v²`, doubling speed increases kinetic energy by **4×** (`2²`).",
            "**Power:** Rate of energy transfer / work done per unit time (`1 W = 1 J/s`).",
            "**Wasted Energy:** Usually dissipated to the surroundings as **thermal (internal) energy** via friction or air resistance.",
        ],
        "examiner_trap": "Remember distance `d` in `W = F × d` must be **in the direction of the force**.",
        "pages": [42, 43, 44, 45, 46, 47],
    },
    {
        "id": "u1_pressure_liquids_barometer",
        "unit": "Unit 1: General Physics",
        "topic": "Pressure in Solids & Liquids",
        "keywords": ["pressure", "pascal", "hydraulic", "barometer", "manometer", "depth", "atmosphere", "atmospheric pressure"],
        "summary": "**Pressure** is force per unit perpendicular area (`p = F / A`). In a liquid, pressure increases with depth and density (`Δp = ρ g Δh`).",
        "formulas": ["p = F / A  (Pa = N/m²)", "Δp = ρ × g × Δh"],
        "points": [
            "**Liquid Pressure:** Acts equally in all directions at the same depth and is independent of container shape.",
            "**Hydraulic System:** Liquid is incompressible, so pressure is transmitted equally (`F₁/A₁ = F₂/A₂`); a larger output piston area multiplies force.",
            "**Mercury Barometer:** Measures atmospheric pressure (`p = ρ g h`, `≈ 760 mmHg ≈ 1.0 × 10⁵ Pa`).",
        ],
        "examiner_trap": "In `p = F/A`, convert `cm²` to `m²` by dividing by `10,000` (`1 m² = 10⁴ cm²`) to get Pascals (`Pa`).",
        "pages": [51, 52, 53, 54],
    },
    # ==================== UNIT 2: THERMAL PHYSICS ====================
    {
        "id": "u2_kinetic_particle_brownian_boyle",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Kinetic Particle Theory, Brownian Motion & Gas Laws",
        "keywords": ["brownian", "smoke cell", "kinetic particle", "gas pressure", "boyle", "kelvin", "absolute zero", "evaporation", "boiling", "molecules"],
        "summary": "All matter is made of moving particles. Average kinetic energy of molecules is directly proportional to **Kelvin temperature**.",
        "formulas": ["T (K) = θ (°C) + 273", "p₁ V₁ = p₂ V₂  (at constant temperature)", "p / T(K) = constant  (at constant volume)"],
        "points": [
            "**Brownian Motion:** Visible smoke/pollen specks jiggle randomly because **invisible, fast-moving air/water molecules collide randomly** with them from all sides.",
            "**Gas Pressure Explanation:** Gas molecules collide with container walls, changing momentum and exerting a **force over the wall area**. Heating increases molecular speed → **more frequent and harder collisions** → higher pressure.",
            "**Evaporation vs Boiling:** Evaporation happens at **any temperature below boiling point** from the **surface only** as the **most energetic molecules escape**, lowering the average KE of remaining molecules (**cooling effect**).",
        ],
        "examiner_trap": "In **Boyle's Law (`p₁V₁ = p₂V₂`)**, temperature must stay **constant**; for Charles's/Pressure Law, temperature MUST be converted to **Kelvin (`+273`)**!",
        "pages": [58, 59, 60, 61, 62, 63, 64, 65],
    },
    {
        "id": "u2_shc_latent_heat_expansion",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Thermal Expansion, Specific Heat Capacity & Latent Heat",
        "keywords": ["specific heat capacity", "latent heat", "thermal expansion", "bimetallic", "melting", "boiling", "heating curve", "cooling curve"],
        "summary": "**Specific Heat Capacity (`c`)** is the energy needed to raise the temperature of `1 kg` of a substance by `1 °C` (`1 K`). During a change of state, temperature stays constant because energy breaks intermolecular bonds (**latent heat**).",
        "formulas": ["E = m × c × Δθ", "c = E / (m × Δθ)", "E = P × t = V × I × t"],
        "points": [
            "**Why Temperature Stays Constant During Melting/Boiling:** Supplied thermal energy increases the **internal potential energy** to weaken/break **intermolecular forces/bonds**, rather than increasing the average kinetic energy of molecules.",
            "**Thermal Expansion:** Gases expand most, then liquids, then solids (least). Used in **bimetallic strips** (thermostats) and requires gaps in railway tracks/bridges.",
        ],
        "examiner_trap": "Always convert mass from **grams to kilograms (`÷ 1000`)** when `c` is given in `J/(kg °C)`!",
        "pages": [67, 68, 69, 70, 71],
    },
    {
        "id": "u2_conduction_convection_radiation",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Conduction, Convection & Infrared Radiation",
        "keywords": ["conduction", "convection", "radiation", "infrared", "emitter", "absorber", "reflector", "vacuum flask", "free electrons", "heat lost", "heat gained"],
        "summary": "Thermal energy transfers by **conduction** (solids), **convection** (fluids: liquids & gases), and **infrared radiation** (electromagnetic waves — travels through a **vacuum**).",
        "formulas": ["Rate of heat supplied vs rate of heat lost determines whether temperature rises, falls, or stays constant"],
        "points": [
            "**Conduction:** Atoms vibrate faster and collide with neighbours. **Metals** conduct far better because they contain **delocalised (free) electrons** that move rapidly through the lattice and transfer kinetic energy.",
            "**Convection (Fluids Only):** Heated fluid **expands** → its **density decreases** → warmer, less dense fluid **rises** while cooler, denser fluid **sinks**, creating a **convection current**.",
            "**Infrared Radiation:** **Dull, black surfaces** are the **best emitters and absorbers**; **shiny, silver/white surfaces** are **poor emitters/absorbers and good reflectors**.",
            "**Heat Lost vs Heat Gained (Notes p.80):** If heat supplied `<` heat lost → temperature **decreases**; equal → **constant**; greater → **increases**.",
        ],
        "examiner_trap": "Convection **cannot** happen in solids (particles fixed) or in a vacuum (no particles); **radiation** is the only method that works in a vacuum.",
        "pages": [73, 74, 75, 76, 77, 78, 79, 80],
    },
    # ==================== UNIT 3: ELECTRICITY ====================
    {
        "id": "u3_charge_current_voltage_ohm",
        "unit": "Unit 3: Electricity",
        "topic": "Charge, Current, Voltage & Ohm's Law",
        "keywords": ["charge", "coulomb", "current", "voltage", "potential difference", "emf", "ohm", "resistance", "filament lamp", "diode", "electrostatic", "electric field"],
        "summary": "**Current (`I`)** is the rate of flow of charge (`I = Q/t`). **Potential difference (`V`)** is work done per unit charge (`V = E/Q`). **Ohm's Law** states `I ∝ V` at **constant temperature**.",
        "formulas": ["Q = I × t", "V = E / Q", "V = I × R", "R ∝ L / A  (wire length & cross-sectional area)"],
        "points": [
            "**Metals vs Conventional Current:** Electrons flow from **negative to positive**; conventional current is drawn from **positive to negative**.",
            "**Ohmic Resistor `I–V` Graph:** Straight line through the origin `(0,0)` → constant resistance.",
            "**Filament Lamp `I–V` Graph:** Curves/flattens as `V` increases because temperature rises → metal ions vibrate more → more frequent electron collisions → **resistance increases**.",
            "**Semiconductor Diode:** Conducts current in **one direction only** (very low resistance in forward bias after threshold, huge resistance in reverse bias).",
            "**Wire Dimensions:** Doubling length `L` → **doubles** resistance; doubling diameter `d` → area increases `4×` → resistance becomes **`¼`**.",
        ],
        "examiner_trap": "Always connect an **ammeter in series** and a **voltmeter in parallel** across the component.",
        "pages": [82, 83, 84, 85, 86, 87, 88, 89, 90],
    },
    {
        "id": "u3_circuits_ldr_thermistor_safety",
        "unit": "Unit 3: Electricity",
        "topic": "Series & Parallel Circuits, LDR, Thermistor & Electrical Safety",
        "keywords": ["series", "parallel", "potential divider", "ldr", "thermistor", "fuse", "earth wire", "circuit breaker", "power", "kilowatt-hour", "kwh"],
        "summary": "In **series**, current is the same everywhere and voltages add (`V = V₁ + V₂`). In **parallel**, voltage across each branch is equal and currents add (`I = I₁ + I₂`).",
        "formulas": ["Series: R_total = R₁ + R₂", "Parallel: 1/R_total = 1/R₁ + 1/R₂  (or R₁R₂/(R₁+R₂))", "P = I × V = I² R = V² / R", "E = P × t = V × I × t", "Cost = kW × hours × price"],
        "points": [
            "**LDR (Light-Dependent Resistor):** As **light intensity increases**, **resistance decreases**.",
            "**NTC Thermistor:** As **temperature increases**, **resistance decreases**.",
            "**Potential Divider:** `V₁ / V₂ = R₁ / R₂` — a component with larger resistance takes a larger share of the supply voltage.",
            "**Fuse & Earth Wire:** Fuse is placed on the **live wire**; if a fault causes excessive current, the fuse wire melts and **breaks the circuit**. The **earth wire** connects metal casing to ground so fault current flows safely to earth instead of through a user.",
        ],
        "examiner_trap": "Remember: **double insulation** (plastic casing) means no earth wire is needed because the outer case cannot become live.",
        "pages": [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    },
    # ==================== UNIT 4: ELECTROMAGNETISM ====================
    {
        "id": "u4_magnetism_motor_fleming",
        "unit": "Unit 4: Electromagnetism",
        "topic": "Magnetic Fields, Electromagnets & D.C. Motor",
        "keywords": ["magnet", "magnetic field", "solenoid", "electromagnet", "fleming", "left-hand rule", "motor", "commutator", "split-ring", "force on a wire"],
        "summary": "A current-carrying conductor in a magnetic field experiences a **force** perpendicular to both current and field (**Fleming's Left-Hand Rule**).",
        "formulas": ["Fleming's Left-Hand Rule: ThuMb = Motion/Force, First finger = Field (N→S), seCond finger = Current (+→−)"],
        "points": [
            "**Magnetic Field Lines:** Point from **North to South** outside a magnet; closer lines mean a **stronger field**.",
            "**Soft Iron vs Steel:** **Soft iron** magnetises and demagnetises easily (used in **electromagnet cores & transformers**); **steel** retains magnetism (used for **permanent magnets**).",
            "**D.C. Motor & Split-Ring Commutator:** The split-ring commutator **reverses the current direction in the coil every half-turn** (when vertical), so the turning moment/force stays in the **same rotational direction** continuously.",
            "**Increase Motor Speed/Force:** Increase current, use stronger magnets, add more turns on the coil, or wind on a soft-iron core.",
        ],
        "examiner_trap": "Use **Fleming's Left-Hand Rule** for **Motors** (Force from current + field), and **Right-Hand Rule** for **Generators** (Induced current from motion + field)!",
        "pages": [132, 133, 134, 135, 136, 137, 138, 139, 140],
    },
    {
        "id": "u4_induction_generator_transformer",
        "unit": "Unit 4: Electromagnetism",
        "topic": "Electromagnetic Induction, A.C. Generator & Transformers",
        "keywords": ["electromagnetic induction", "lenz", "faraday", "generator", "slip ring", "1st half cycle", "2nd half cycle", "transformer", "step-up", "step-down", "national grid"],
        "summary": "An e.m.f. is induced whenever a conductor **cuts magnetic field lines** (or magnetic flux through a coil changes). **Lenz's Law** states the induced current opposes the change producing it.",
        "formulas": ["Vp / Vs = Np / Ns", "Vp × Ip = Vs × Is  (ideal 100% efficient transformer)", "Power loss in cables: P_loss = I² × R"],
        "points": [
            "**A.C. Generator (1st & 2nd Half Cycles — Notes p.148–150):** Uses **slip rings** & brushes. When the coil is **vertical**, sides move parallel to field lines → **zero cutting → zero current** (`0°, 180°, 360°`). When **horizontal**, sides cut field lines fastest → **maximum current** (`90°` positive, `270°` negative in 2nd half cycle).",
            "**Transformers:** Need **a.c.** so the magnetic field in the **soft-iron core** continuously changes and cuts the secondary coil. **d.c. fails** because its magnetic field is constant (`ΔΦ = 0`).",
            "**National Grid:** Step-up transformers raise voltage to **reduce current** (`I = P/V`), minimising thermal power loss `I²R` in transmission cables.",
        ],
        "examiner_trap": "A transformer changes **voltage and current**, NOT power or frequency (`Pin = Pout` for ideal transformer).",
        "pages": [144, 145, 146, 147, 148, 149, 150, 151, 152, 153],
    },
    # ==================== UNIT 5: NUCLEAR PHYSICS ====================
    {
        "id": "u5_atom_radioactivity_halflife",
        "unit": "Unit 5: Nuclear Physics",
        "topic": "Nuclear Atom, Alpha/Beta/Gamma, Half-Life, Fission & Fusion",
        "keywords": ["alpha", "beta", "gamma", "half-life", "half life", "radioactive", "background radiation", "geiger", "rutherford", "gold foil", "fission", "fusion", "isotope", "nuclide"],
        "summary": "Radioactive decay is **random and spontaneous**. Unstable nuclei emit **alpha (`⁴₂He`)**, **beta (`⁰₋₁e`)**, or **gamma (`γ`)** radiation to become more stable.",
        "formulas": ["N = N₀ × (½)^(t / t_½)", "Alpha: A − 4, Z − 2", "Beta⁻: A unchanged, Z + 1", "Corrected count rate = Measured − Background"],
        "points": [
            "**Rutherford Gold Foil Experiment:** Most α-particles passed straight through (atom is **mostly empty space**); a few deflected/bounced back (**tiny, dense, positively charged nucleus**).",
            "**Penetration & Ionisation:** Alpha is **most ionising** (stopped by paper/skin); Beta is moderate (stopped by `~3 mm` aluminium); Gamma is **most penetrating** (reduced by thick lead/concrete).",
            "**Half-Life (`t_½`):** Average time for half the unstable nuclei to decay (subtract **background radiation** first!).",
            "**Fission vs Fusion:** **Fission** splits a heavy nucleus (e.g. `U-235`) using a neutron; **Fusion** joins two light nuclei (e.g. hydrogen → helium) at extreme temperature & pressure (powers stars).",
        ],
        "examiner_trap": "Always subtract **background count rate** BEFORE halving to find half-life, and add it back if asked for a detector reading!",
        "pages": [155, 156, 157, 158, 159, 160, 161, 162, 163, 164],
    },
    # ==================== UNIT 6: WAVES & LIGHT ====================
    {
        "id": "u6_waves_diffraction_em_sound",
        "unit": "Unit 6: Waves",
        "topic": "Wave Properties, Diffraction, Sound & EM Spectrum",
        "keywords": ["wave", "transverse", "longitudinal", "wavelength", "frequency", "diffraction", "ripple tank", "sound", "echo", "ultrasound", "electromagnetic spectrum", "compression", "rarefaction"],
        "summary": "Waves transfer **energy without transferring matter**. Speed `v = f × λ`.",
        "formulas": ["v = f × λ", "f = 1 / T", "Echo speed: v = 2d / t", "Audible range: 20 Hz – 20,000 Hz"],
        "points": [
            "**Transverse vs Longitudinal:** Transverse vibrates **perpendicular** to energy transfer (light, water, all EM waves); Longitudinal vibrates **parallel** via compressions & rarefactions (sound).",
            "**Diffraction:** Waves spread out through a gap/around an obstacle. **Greatest diffraction** when **gap width ≈ wavelength (`λ`)**. Wavelength, frequency, and speed stay **unchanged**.",
            "**Refraction in Ripple Tank (Deep → Shallow):** Speed **decreases**, wavelength **decreases**, **frequency stays constant**, wavefronts bend **towards normal**.",
            "**EM Spectrum Order (decreasing λ, increasing f):** Radio → Microwaves → Infrared → Visible Light → Ultraviolet → X-rays → Gamma rays (all travel at `3.0 × 10⁸ m/s` in vacuum).",
        ],
        "examiner_trap": "When any wave enters a different medium/depth, **frequency NEVER changes** (only speed and wavelength change)!",
        "pages": [105, 106, 107, 108, 109, 110, 111, 112, 113, 128, 129],
    },
    {
        "id": "u6_light_refraction_tir_lenses",
        "unit": "Unit 6: Waves",
        "topic": "Reflection, Refraction, Critical Angle, TIR & Lenses",
        "keywords": ["reflection", "refraction", "snell", "refractive index", "critical angle", "total internal reflection", "tir", "optical fibre", "converging lens", "focal length", "virtual image", "real image", "dispersion", "prism"],
        "summary": "Light obeys reflection (`i = r`), refraction (`n = sin i / sin r`), and **Total Internal Reflection** (`i > c` from dense to less dense medium).",
        "formulas": ["n = sin i / sin r = c_air / v_medium", "sin c = 1 / n"],
        "points": [
            "**Conditions for Total Internal Reflection (TIR):** (1) Light travels from **optically denser to less dense medium** (e.g. glass → air), AND (2) angle of incidence **greater than critical angle (`i > c`)**.",
            "**Converging Lens (`f`):** Object beyond `2F` → real, inverted, diminished (camera); between `F` and `2F` → real, inverted, magnified (projector); **closer than `F` (`u < f`)** → **virtual, upright, magnified** (magnifying glass).",
            "**Dispersion (Glass Prism):** White light splits into spectrum because **violet** refracts most (slowest in glass, shortest λ) and **red** refracts least.",
        ],
        "examiner_trap": "In `n = sin i / sin r`, angle `i` is always in **air/vacuum** and `r` in the **medium** (both measured to the **normal**, never the surface!).",
        "pages": [114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],
    },
    # ==================== UNIT 7: ASTRONOMY / SPACE PHYSICS ====================
    {
        "id": "u7_space_orbits_stars_bigbang",
        "unit": "Unit 7: Astronomy",
        "topic": "Solar System, Orbital Speed, Stellar Life Cycle, Redshift & Big Bang",
        "keywords": ["orbit", "orbital speed", "planet", "comet", "star", "protostar", "main sequence", "red giant", "supernova", "white dwarf", "neutron star", "black hole", "redshift", "big bang", "hubble", "cmbr", "light-year"],
        "summary": "Gravity provides the centripetal force keeping planets, moons, and comets in orbit. Redshift and CMBR provide evidence for the **expanding Universe** and **Big Bang**.",
        "formulas": ["v = 2πr / T", "v = H₀ × d", "Age of Universe ≈ 1 / H₀  (H₀ ≈ 2.2 × 10⁻¹⁸ s⁻¹)", "1 light-year ≈ 9.5 × 10¹⁵ m"],
        "points": [
            "**Comet Orbits:** Highly elliptical; moves **fastest when closest to the Sun** (where gravitational pull is strongest and GPE transfers to KE).",
            "**Life Cycle of Stars:** Nebula → Protostar → Main Sequence (stable: inward gravity = outward radiation pressure from H→He fusion). **Sun-like star:** Red Giant → Planetary Nebula + **White Dwarf**. **Massive star:** Red Supergiant → **Supernova** → **Neutron Star** or **Black Hole**.",
            "**Redshift & Big Bang:** Light from distant galaxies has **increased wavelength** (shifted towards red), showing galaxies are moving away — **further galaxies recede faster** (`v = H₀ d`).",
            "**CMBR:** Cosmic Microwave Background Radiation is uniform microwave radiation filling all of space — leftover thermal radiation from the hot early Universe stretched to microwaves by expansion.",
        ],
        "examiner_trap": "Always say redshift is an **increase in observed wavelength** (not 'the galaxy turns red').",
        "pages": [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243],
    },
]


def main() -> None:
    out_path = DATA_DIR / "harony_concept_atlas.json"
    out_path.write_text(json.dumps(ATLAS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(ATLAS)} structured Harony Physics concepts across all 7 Units to {out_path}")


if __name__ == "__main__":
    main()
