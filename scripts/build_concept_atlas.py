"""Build a comprehensive 30-module Cambridge IGCSE (0625) & A-Level Physics Concept & Reasoning Atlas
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
        "keywords": ["scalar", "vector", "parallax", "meniscus", "micrometer", "vernier", "pendulum", "oscillation", "measuring cylinder"],
        "summary": "A **scalar** has magnitude only (mass, speed, distance, energy, time, temperature). A **vector** has both magnitude and direction (force, velocity, acceleration, weight, displacement, momentum).",
        "formulas": ["Resultant (perpendicular vectors): R = √(F₁² + F₂²)", "Period T = total time / N oscillations"],
        "points": [
            "**Measuring Cylinder & Parallax Error:** Read the **bottom of the meniscus** with your eye **perpendicular** to the scale.",
            "**Simple Pendulum:** Time **20 oscillations** from the centre fiducial mark and divide by 20 to reduce human reaction-time error.",
            "**Micrometer Screw Gauge:** Precision `0.01 mm` (main scale + thimble reading minus any zero error).",
        ],
        "examiner_trap": "Speed and distance are scalars; velocity, displacement, force, weight, and momentum are vectors.",
        "pages": [4, 5, 6, 7],
    },
    {
        "id": "u1_kinematics_motion_graphs",
        "unit": "Unit 1: General Physics",
        "topic": "Kinematics & Motion Graphs",
        "keywords": ["distance-time", "speed-time", "velocity-time", "motion graph", "acceleration", "deceleration", "gradient", "slope", "area under", "uniform acceleration"],
        "summary": "Motion graphs describe speed, acceleration, and distance travelled using **gradient (slope)** and **area under the graph**.",
        "formulas": ["v = d / t (constant speed)", "a = (v − u) / t", "d = ((u + v) / 2) × t", "Distance = Area under speed–time graph"],
        "points": [
            "**Distance–Time Graph:** Gradient = **speed**. Horizontal line = at rest (`speed = 0`); straight line sloping up = constant speed; curve getting steeper = acceleration; curve getting flatter = deceleration.",
            "**Speed–Time Graph:** Gradient = **acceleration**; **Area under graph** = **distance travelled**. Horizontal line = constant speed (`a = 0`); straight upward line = uniform acceleration; downward line = uniform deceleration.",
        ],
        "examiner_trap": "A horizontal line on a **speed–time** graph means **constant speed** (not at rest, unless `speed = 0`).",
        "pages": [14, 15, 16, 17, 18],
    },
    {
        "id": "u1_freefall_terminal_velocity_projectile",
        "unit": "Unit 1: General Physics",
        "topic": "Free Fall, Vertical Throw & Terminal Velocity",
        "keywords": ["terminal velocity", "free fall", "freefall", "thrown upward", "thrown up", "highest point", "parachute", "air resistance", "weightless", "astronaut"],
        "summary": "Gravity accelerates all objects near Earth downwards at `g ≈ 9.8 m/s²` regardless of mass (when air resistance is negligible).",
        "formulas": ["W = m × g", "Resultant force F_net = W − F_air", "v² = u² + 2 a s"],
        "points": [
            "**Ball Thrown Vertically Upwards (at the highest point):** Instantaneous **velocity is `0 m/s`**, but **acceleration is STILL `9.8 m/s²` downwards** because Earth's gravitational force (weight) still acts downwards.",
            "**Terminal Velocity:** Initially `weight > air resistance` so object accelerates downwards. As speed rises, air resistance increases, reducing resultant force and acceleration until **air resistance = weight** (`resultant force = 0`, constant terminal velocity).",
            "**Opening a Parachute:** Surface area increases sharply → air resistance suddenly becomes **greater than weight** → net upward force → object decelerates until a new, lower terminal velocity is reached.",
            "**Why Astronauts Feel Weightless in Orbit:** Gravity STILL acts on them (providing centripetal force), but both the astronaut and spacecraft are in **continuous free fall** together around Earth, so there is zero normal contact force.",
        ],
        "examiner_trap": "Never write 'gravity becomes zero' at the top of a throw or in Earth orbit — gravity still acts!",
        "pages": [16, 17, 25],
    },
    {
        "id": "u1_mass_weight_density",
        "unit": "Unit 1: General Physics",
        "topic": "Mass, Weight & Density",
        "keywords": ["mass", "weight", "density", "float", "sink", "ship", "displacement can", "gravitational field strength"],
        "summary": "**Mass** (`kg`) is the amount of matter (stays constant everywhere). **Weight** (`N`) is the gravitational force on a mass (`W = m × g`). **Density** (`ρ = m / V`) determines whether an object floats or sinks.",
        "formulas": ["W = m × g  (g ≈ 9.8 N/kg on Earth)", "ρ = m / V", "1 g/cm³ = 1000 kg/m³"],
        "points": [
            "**Why a steel ship floats:** The ship is hollow and contains **air spaces**, so its **overall average density** is **less than the density of water**.",
            "**Irregular Solid Density:** Measure mass on a digital balance; measure volume by water displacement in a measuring cylinder or **displacement (Eureka) can**; compute `ρ = m / V`.",
            "**Earth vs Moon:** **Mass stays constant**, whereas **weight decreases** on the Moon because `g` is smaller (`≈ 1.6 N/kg`).",
        ],
        "examiner_trap": "Mass is measured in `kg` on a balance; weight is a force measured in `N` with a newton-meter.",
        "pages": [20, 21, 22, 23],
    },
    {
        "id": "u1_forces_newtons_laws_hooke",
        "unit": "Unit 1: General Physics",
        "topic": "Forces, Newton's Laws, Hooke's Law & Circular Motion",
        "keywords": ["newton", "inertia", "resultant force", "hooke", "spring constant", "limit of proportionality", "extension", "circular motion", "centripetal", "friction"],
        "summary": "A **resultant force** changes an object's speed, direction (circular motion), or shape (Hooke's law).",
        "formulas": ["F = m × a", "F = k × x  (Hooke's Law)", "k = F / x"],
        "points": [
            "**Newton's 1st Law (Inertia):** An object stays at rest or continues at **constant velocity** unless acted upon by a non-zero **resultant force**.",
            "**Newton's 2nd Law:** `F = m × a` — resultant force equals mass × acceleration (in the same direction).",
            "**Newton's 3rd Law:** Body A exerts a force on B; B exerts an **equal in magnitude, opposite in direction** force of the **same type** on A.",
            "**Hooke's Law:** Extension (`x`) is directly proportional to load (`F`) up to the **limit of proportionality**.",
            "**Circular Motion:** Even at constant speed, **velocity changes** because **direction changes continuously**, requiring a **resultant centripetal force towards the centre**.",
        ],
        "examiner_trap": "On an extension–load graph, the **limit of proportionality** is the point where the straight line starts to curve.",
        "pages": [24, 25, 26, 27, 28, 29, 30],
    },
    {
        "id": "u1_moments_equilibrium_cog",
        "unit": "Unit 1: General Physics",
        "topic": "Moments, Equilibrium & Centre of Gravity",
        "keywords": ["moment", "pivot", "fulcrum", "clockwise", "anticlockwise", "equilibrium", "centre of gravity", "center of gravity", "topple", "stability", "plumb line"],
        "summary": "The **moment of a force** measures its turning effect about a pivot: `Moment = Force × perpendicular distance from pivot`.",
        "formulas": ["M = F × d  (unit: N m)", "Equilibrium: Σ Clockwise moments = Σ Anticlockwise moments"],
        "points": [
            "**Two Conditions for Equilibrium:** (1) **No resultant force** (`ΣF = 0`), and (2) **No resultant moment** (`ΣM = 0`) about any pivot.",
            "**Stability:** Lower **centre of gravity** and wider **base** give greater stability. An object topples when the vertical line through its centre of gravity falls outside its base.",
            "**Irregular Lamina Centre of Gravity:** Hang freely from a pin and draw vertical lines using a **plumb line** from two different holes; their intersection is the centre of gravity.",
        ],
        "examiner_trap": "Always state **perpendicular distance** from the pivot — omitting 'perpendicular' loses the definition mark.",
        "pages": [33, 34, 35, 36],
    },
    {
        "id": "u1_momentum_impulse",
        "unit": "Unit 1: General Physics",
        "topic": "Momentum & Impulse",
        "keywords": ["momentum", "impulse", "collision", "explosion", "airbag", "crumple zone", "seatbelt", "conservation of momentum"],
        "summary": "**Momentum** (`p = m × v`) is conserved in closed systems where no external resultant force acts.",
        "formulas": ["p = m × v  (kg m/s or N s)", "Impulse = F × Δt = Δp = mv − mu", "Total p_before = Total p_after"],
        "points": [
            "**Airbags & Crumple Zones:** Increase the **collision time (`Δt`)** for the same change in momentum (`Δp`), so the **rate of change of momentum (`F = Δp/Δt`) decreases**, reducing the force on the passenger.",
            "**Vector Sign:** Assign `+` to one direction and `−` to the opposite direction during collisions or explosions.",
        ],
        "examiner_trap": "Never say airbags 'reduce momentum' — the momentum change is unchanged; airbags **increase impact time** so **force decreases**.",
        "pages": [38, 39, 40],
    },
    {
        "id": "u1_work_energy_power_efficiency",
        "unit": "Unit 1: General Physics",
        "topic": "Work, Energy, Power & Efficiency",
        "keywords": ["work done", "kinetic energy", "gravitational potential", "gpe", "power", "efficiency", "conservation of energy", "renewable", "non-renewable", "sankey"],
        "summary": "Energy is conserved and transferred between stores (**kinetic, gravitational potential, elastic, thermal, chemical, nuclear**).",
        "formulas": ["W = F × d = ΔE  (J)", "E_k = ½ m v²", "ΔE_p = m g Δh", "P = E / t = W / t  (W)", "Efficiency = (useful output / total input) × 100%"],
        "points": [
            "**Speed vs Kinetic Energy:** Because `E_k = ½ m v²`, doubling speed increases kinetic energy by **4×** (`2²`).",
            "**Power:** Work done / energy transferred per second (`1 W = 1 J/s`).",
            "**Wasted Energy:** Usually transferred to the surroundings as **thermal (internal) energy** due to friction or air resistance.",
        ],
        "examiner_trap": "Distance `d` in `W = F × d` must be moved **in the direction of the force**.",
        "pages": [42, 43, 44, 45, 46, 47],
    },
    {
        "id": "u1_pressure_liquids_barometer",
        "unit": "Unit 1: General Physics",
        "topic": "Pressure in Solids, Liquids & Hydraulics",
        "keywords": ["pressure", "pascal", "hydraulic", "barometer", "manometer", "depth", "atmosphere", "atmospheric pressure", "sharp knife", "snowshoes"],
        "summary": "**Pressure** is force per unit perpendicular area (`p = F / A`). Liquid pressure increases with depth and density (`Δp = ρ g Δh`).",
        "formulas": ["p = F / A  (Pa = N/m²)", "Δp = ρ × g × Δh", "Hydraulic: F₁ / A₁ = F₂ / A₂"],
        "points": [
            "**Small Area vs Large Area:** A sharp knife concentrates force onto a tiny area (`high pressure`); snowshoes spread weight over a large area (`low pressure`).",
            "**Liquid Pressure:** Acts equally in all directions at the same depth and does not depend on container shape.",
            "**Hydraulic Jack:** Liquids are incompressible, so pressure is transmitted equally throughout the fluid; a larger output piston area produces a larger output force.",
        ],
        "examiner_trap": "Convert `cm²` to `m²` by dividing by `10,000` (`1 m² = 10⁴ cm²`) before computing Pascals (`Pa`).",
        "pages": [51, 52, 53, 54],
    },
    # ==================== UNIT 2: THERMAL PHYSICS ====================
    {
        "id": "u2_kinetic_particle_brownian_boyle",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Kinetic Particle Theory, Brownian Motion & Gas Pressure",
        "keywords": ["brownian", "smoke cell", "kinetic particle", "gas pressure", "boyle", "kelvin", "absolute zero", "molecules"],
        "summary": "Particles in matter are in continuous random motion; average kinetic energy of molecules increases with **Kelvin temperature**.",
        "formulas": ["T (K) = θ (°C) + 273", "p₁ V₁ = p₂ V₂  (constant T)"],
        "points": [
            "**Brownian Motion:** Visible smoke/pollen particles move randomly in zig-zag paths because **invisible, fast-moving air/water molecules collide randomly** with them from all directions.",
            "**Gas Pressure:** Gas molecules collide with container walls, changing momentum and exerting a **force per unit area**. Heating a gas makes molecules move **faster** → **more frequent and harder collisions** with the walls → pressure increases.",
            "**Compressing Gas at Constant Temperature (`p₁V₁ = p₂V₂`):** Smaller volume means molecules travel shorter distances between collisions → **more frequent collisions** with walls → higher pressure.",
        ],
        "examiner_trap": "Always mention BOTH **more frequent** collisions AND **greater force / harder** collisions when temperature rises!",
        "pages": [58, 59, 60, 61, 62, 63],
    },
    {
        "id": "u2_evaporation_vs_boiling",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Evaporation, Cooling Effect & Boiling",
        "keywords": ["evaporation", "evaporate", "sweat", "cooling effect", "boiling", "surface area", "draught", "humidity"],
        "summary": "**Evaporation** happens at any temperature below boiling point when the **most energetic molecules escape from the liquid surface**.",
        "formulas": ["Faster evaporation: higher temperature, larger surface area, air draught/wind, lower humidity"],
        "points": [
            "**Why Evaporation Causes Cooling:** Only the **fastest (highest kinetic energy) molecules** overcome intermolecular forces and escape from the surface, so the **average kinetic energy** of the remaining molecules decreases → **temperature drops**.",
            "**Evaporation vs Boiling:** Evaporation happens at **any temperature** and **at the surface only** (no bubbles). Boiling happens at a **fixed boiling point** throughout the **entire liquid** (bubbles form).",
        ],
        "examiner_trap": "State that **average** kinetic energy of the remaining molecules decreases (not that all molecules stop moving).",
        "pages": [64, 65],
    },
    {
        "id": "u2_shc_latent_heat_expansion",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Thermal Expansion, Specific Heat Capacity & Change of State",
        "keywords": ["specific heat capacity", "latent heat", "thermal expansion", "bimetallic", "melting", "freezing", "heating curve", "cooling curve", "temperature stay constant", "temperature stays constant"],
        "summary": "**Specific Heat Capacity (`c`)** is the thermal energy needed to raise the temperature of `1 kg` of a substance by `1 °C`. During melting or boiling, temperature stays constant because energy breaks intermolecular bonds.",
        "formulas": ["E = m × c × Δθ", "c = E / (m × Δθ)", "E = P × t = V × I × t"],
        "points": [
            "**Why Temperature Stays Constant During Melting/Boiling:** Supplied thermal energy increases **internal potential energy** to overcome/break **intermolecular forces/bonds**, instead of increasing the average kinetic energy of molecules.",
            "**Thermal Expansion:** Gases expand most, then liquids, then solids (least). Applied in **bimetallic strips** (thermostats, fire alarms) and requires expansion gaps in railway lines and bridges.",
        ],
        "examiner_trap": "Convert mass from **grams to kilograms (`÷ 1000`)** when `c` is in `J/(kg °C)`.",
        "pages": [67, 68, 69, 70, 71],
    },
    {
        "id": "u2_conduction_convection_radiation",
        "unit": "Unit 2: Thermal Physics",
        "topic": "Conduction, Convection & Infrared Radiation",
        "keywords": ["conduction", "convection", "radiation", "infrared", "emitter", "absorber", "reflector", "vacuum flask", "free electrons", "heat lost", "heat gained", "metal spoon", "wooden spoon", "sea breeze"],
        "summary": "Thermal energy transfers by **conduction** (mainly solids), **convection** (fluids: liquids & gases), and **infrared radiation** (EM waves — works in a vacuum).",
        "formulas": ["Convection requires fluid density change; Radiation requires no medium"],
        "points": [
            "**Conduction (Metals vs Non-Metals):** Atoms vibrate faster and pass kinetic energy to neighbours. Metals conduct far faster because they have **delocalised (free) electrons** that move rapidly through the lattice and collide with ions.",
            "**Why a Metal Spoon Feels Colder Than Wood at Room Temperature:** Both are at the same temperature, but metal is a **good thermal conductor**, so it conducts heat **away from your hand much faster** than wood (an insulator).",
            "**Convection (Fluids Only):** Heated fluid **expands** → its **density decreases** → warmer, less dense fluid **rises** while cooler, denser fluid **sinks**, creating a **convection current**.",
            "**Infrared Radiation:** **Dull, black surfaces** are the **best emitters and absorbers**; **shiny, silver/white surfaces** are **poor emitters/absorbers and good reflectors**.",
            "**Heat Lost vs Heat Gained (Notes p.80):** Heat supplied `<` heat lost → temperature falls; equal → constant; greater → rises.",
        ],
        "examiner_trap": "Convection **never** occurs in solids (particles fixed) or in a vacuum (no particles); **radiation** is the only transfer method through a vacuum.",
        "pages": [72, 73, 74, 75, 76, 77, 78, 79, 80],
    },
    # ==================== UNIT 3: ELECTRICITY ====================
    {
        "id": "u3_charge_current_voltage_ohm",
        "unit": "Unit 3: Electricity",
        "topic": "Charge, Current, Voltage, Resistance & I–V Graphs",
        "keywords": ["charge", "coulomb", "current", "voltage", "potential difference", "emf", "ohm", "resistance", "filament lamp", "diode", "electrostatic", "electric field", "wire length", "diameter"],
        "summary": "**Current (`I = Q/t`)** is rate of flow of charge. **Potential difference (`V = E/Q`)** is work done per unit charge. **Ohm's Law** states `I ∝ V` at constant temperature.",
        "formulas": ["Q = I × t", "V = E / Q", "V = I × R", "R ∝ L / A"],
        "points": [
            "**Ohmic Resistor `I–V` Graph:** Straight line through origin `(0, 0)` → constant resistance.",
            "**Filament Lamp `I–V` Graph:** As current increases, the filament heats up → metal lattice ions vibrate more vigorously → **more frequent collisions with free electrons** → **resistance increases** and the graph flattens.",
            "**Semiconductor Diode:** Conducts in **one direction only** (forward bias); very high resistance in reverse bias.",
            "**Wire Length & Diameter:** Doubling length `L` doubles resistance; doubling diameter `d` increases area `4×` (`A = πr²`), reducing resistance to **`¼`**.",
        ],
        "examiner_trap": "Connect an **ammeter in series** (very low resistance) and a **voltmeter in parallel** (very high resistance).",
        "pages": [82, 83, 84, 85, 86, 87, 88, 89, 90],
    },
    {
        "id": "u3_circuits_ldr_thermistor_safety",
        "unit": "Unit 3: Electricity",
        "topic": "Series & Parallel Circuits, Potential Divider, LDR, Thermistor & Safety",
        "keywords": ["series", "parallel", "brightness", "bulb", "potential divider", "ldr", "thermistor", "fuse", "earth wire", "circuit breaker", "kilowatt-hour", "kwh", "double insulation"],
        "summary": "In **series**, current is identical everywhere and voltages add (`V = V₁ + V₂`). In **parallel**, voltage across each branch equals the supply voltage and currents add (`I = I₁ + I₂`).",
        "formulas": ["Series: R_T = R₁ + R₂", "Parallel: 1/R_T = 1/R₁ + 1/R₂", "P = I V = I² R = V² / R", "E = P t = V I t"],
        "points": [
            "**Adding a Bulb in Parallel:** Each parallel branch still receives the **full supply voltage**, so existing bulbs keep the **same brightness** (and if one breaks, the others stay on), while total current from the battery increases.",
            "**LDR & NTC Thermistor:** Increasing **light intensity** decreases an **LDR's** resistance; increasing **temperature** decreases a **thermistor's** resistance.",
            "**Fuse & Earth Wire:** The **fuse** is on the **live wire** and melts to break the circuit if current exceeds its rating. The **earth wire** connects metal cases to ground so fault current flows safely to earth and blows the fuse.",
        ],
        "examiner_trap": "A plastic-cased appliance has **double insulation** and needs no earth wire.",
        "pages": [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    },
    # ==================== UNIT 4: ELECTROMAGNETISM ====================
    {
        "id": "u4_magnetism_motor_fleming",
        "unit": "Unit 4: Electromagnetism",
        "topic": "Magnetic Fields, Electromagnets & D.C. Motor",
        "keywords": ["magnet", "magnetic field", "solenoid", "electromagnet", "fleming", "left-hand rule", "motor", "commutator", "split-ring", "soft iron", "steel"],
        "summary": "A current-carrying wire in a magnetic field experiences a **force** perpendicular to both current and field (**Fleming's Left-Hand Rule**).",
        "formulas": ["Fleming's Left-Hand Rule: ThuMb = Motion/Force, First finger = Field (N→S), seCond finger = Current (+→−)"],
        "points": [
            "**Soft Iron vs Steel:** **Soft iron** magnetises and demagnetises easily (electromagnets & transformer cores); **steel** retains magnetism (permanent magnets).",
            "**D.C. Motor & Split-Ring Commutator:** The split-ring commutator **reverses the current direction in the coil every half-turn** (when vertical) so the turning force stays in the **same rotational direction** continuously.",
            "**Increase Motor Turning Effect:** Increase current, use stronger magnets, add more coil turns, or insert a soft-iron core.",
        ],
        "examiner_trap": "Use **Left-Hand Rule** for **Motors** (force from current), and **Right-Hand Rule** for **Generators** (induced current from motion).",
        "pages": [132, 133, 134, 135, 136, 137, 138, 139, 140],
    },
    {
        "id": "u4_induction_generator_transformer",
        "unit": "Unit 4: Electromagnetism",
        "topic": "Electromagnetic Induction, A.C. Generator & Transformers",
        "keywords": ["electromagnetic induction", "lenz", "faraday", "generator", "slip ring", "1st half cycle", "2nd half cycle", "transformer", "step-up", "step-down", "national grid", "why dc"],
        "summary": "An e.m.f. is induced whenever a conductor **cuts magnetic field lines** (or magnetic flux changes). **Lenz's Law** states the induced current opposes the change causing it.",
        "formulas": ["Vp / Vs = Np / Ns", "Vp × Ip = Vs × Is  (ideal transformer)", "Power loss in cables: P_loss = I² × R"],
        "points": [
            "**A.C. Generator (1st & 2nd Half Cycles — Notes p.148–150):** Uses **slip rings**. Vertical coil → sides move parallel to field → **zero cutting, zero current**. Horizontal coil → cuts field lines at max rate → **maximum current** (reverses direction every half-turn).",
            "**Why Transformers Need a.c. (and Fail With d.c.):** Alternating current creates a **continuously changing magnetic field** in the **soft-iron core** that cuts the secondary coil and induces e.m.f. Direct current creates a **constant magnetic field** (`no cutting` → zero induced voltage).",
            "**High-Voltage Power Transmission:** Step-up transformers raise voltage to **reduce current** (`I = P/V`), drastically cutting thermal power loss **`I²R`** in transmission cables.",
        ],
        "examiner_trap": "Step-up transformers reduce **current** (not resistance) in cables, reducing `I²R` heating loss.",
        "pages": [144, 145, 146, 147, 148, 149, 150, 151, 152, 153],
    },
    # ==================== UNIT 5: NUCLEAR PHYSICS ====================
    {
        "id": "u5_atom_radioactivity_halflife",
        "unit": "Unit 5: Nuclear Physics",
        "topic": "Nuclear Atom, Alpha/Beta/Gamma, Half-Life, Fission & Fusion",
        "keywords": ["alpha", "beta", "gamma", "half-life", "half life", "radioactive", "background radiation", "geiger", "rutherford", "gold foil", "fission", "fusion", "isotope", "nuclide"],
        "summary": "Radioactive decay is **random and spontaneous**. Unstable nuclei emit **alpha (`⁴₂He`)**, **beta (`⁰₋₁e`)**, or **gamma (`γ`)** radiation.",
        "formulas": ["N = N₀ × (½)^(t / t_½)", "Alpha: A − 4, Z − 2", "Beta⁻: A unchanged, Z + 1", "Corrected count = Measured − Background"],
        "points": [
            "**Rutherford Gold Foil Experiment:** Most α-particles passed straight through (**atom is mostly empty space**); a few deflected/bounced back (**tiny, dense, positively charged nucleus**).",
            "**Penetration & Ionisation:** α is **most ionising** (stopped by paper/skin); β is moderate (stopped by `~3 mm` aluminium); γ is **most penetrating** (reduced by thick lead/concrete).",
            "**Half-Life (`t_½`):** Average time for half the unstable nuclei to decay (always subtract **background radiation** first!).",
            "**Fission vs Fusion:** **Fission** splits a heavy nucleus (`U-235`) using a neutron; **Fusion** joins two light nuclei (hydrogen → helium) at extreme temperature & pressure (powers the Sun & stars).",
        ],
        "examiner_trap": "Subtract **background count rate** BEFORE calculating half-life, and add it back if asked for a detector reading.",
        "pages": [155, 156, 157, 158, 159, 160, 161, 162, 163, 164],
    },
    # ==================== UNIT 6: WAVES & LIGHT ====================
    {
        "id": "u6_waves_diffraction_em_sound",
        "unit": "Unit 6: Waves",
        "topic": "Wave Properties, Diffraction, Sound & EM Spectrum",
        "keywords": ["wave", "transverse", "longitudinal", "wavelength", "frequency", "diffraction", "ripple tank", "sound", "echo", "ultrasound", "electromagnetic spectrum", "compression", "rarefaction", "pitch", "loudness"],
        "summary": "Waves transfer **energy without transferring matter**. Wave speed `v = f × λ`.",
        "formulas": ["v = f × λ", "f = 1 / T", "Echo speed: v = 2d / t", "Human hearing: 20 Hz – 20,000 Hz"],
        "points": [
            "**Transverse vs Longitudinal:** Transverse vibrates **perpendicular** to wave direction (light, water, all EM waves); Longitudinal vibrates **parallel** via compressions & rarefactions (sound).",
            "**Diffraction:** Waves spread out through a gap/around an edge. **Greatest diffraction** when **gap width ≈ wavelength (`λ`)**. Wavelength, frequency, and speed stay **unchanged**.",
            "**Deep → Shallow Water (Ripple Tank):** Speed **decreases**, wavelength **decreases**, **frequency stays constant**, wavefronts bend **towards normal**.",
            "**Sound Pitch & Loudness:** Higher **frequency** → higher **pitch**; larger **amplitude** → **louder** sound. Sound travels fastest in **solids**, slowest in **gases**, and cannot travel in a **vacuum**.",
        ],
        "examiner_trap": "During refraction or change of water depth, **frequency NEVER changes** (only speed and wavelength change).",
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
            "**Total Internal Reflection (TIR):** Occurs ONLY when (1) light travels from **optically denser to less dense medium** (e.g. glass → air), AND (2) **angle of incidence > critical angle (`i > c`)**.",
            "**Optical Fibres:** Light travels inside a high-index glass core reflecting repeatedly by **TIR** — used in high-speed internet and medical **endoscopes**.",
            "**Converging Lens (`f`):** `u > 2f` → real, inverted, diminished; `f < u < 2f` → real, inverted, magnified; **`u < f`** → **virtual, upright, magnified** (magnifying glass).",
            "**Dispersion:** White light splits in a prism because **violet** travels slowest in glass and refracts **most**, while **red** refracts **least**.",
        ],
        "examiner_trap": "Measure angles `i` and `r` from the **normal** (perpendicular to the surface), never from the glass surface.",
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
            "**Comet Orbits:** Elliptical orbit; comet moves **fastest when closest to the Sun** (GPE transfers to KE as gravitational attraction is strongest).",
            "**Life Cycle of Stars:** Nebula → Protostar → Main Sequence (stable: inward gravity = outward radiation pressure). **Sun-like star:** Red Giant → Planetary Nebula + **White Dwarf**. **Massive star:** Red Supergiant → **Supernova** → **Neutron Star** or **Black Hole**.",
            "**Redshift & Big Bang:** Light from distant galaxies shows an **increase in observed wavelength**, proving galaxies are receding — **further galaxies move faster** (`v = H₀ d`).",
            "**CMBR:** Cosmic Microwave Background Radiation fills all of space uniformly — cooled leftover thermal radiation from the hot early Universe.",
        ],
        "examiner_trap": "Say redshift is an **increase in observed wavelength** (not 'the galaxy turns red').",
        "pages": [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243],
    },
]


def main() -> None:
    out_path = DATA_DIR / "harony_concept_atlas.json"
    out_path.write_text(json.dumps(ATLAS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(ATLAS)} structured Harony Physics concepts across all 7 Units to {out_path}")


if __name__ == "__main__":
    main()
