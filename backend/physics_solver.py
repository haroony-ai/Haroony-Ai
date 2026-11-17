"""Symbolic & Numerical IGCSE / A-Level Physics Problem Solver (SymPy-powered).

Detects quantitative physics word problems, extracts given quantities with SI unit conversions,
selects the matching IGCSE 0625 equation, solves symbolically/numerically, and formats
a complete Cambridge examiner-style solution:
  1. Given Quantities & SI Conversions
  2. Formula [C1 / M1]
  3. Substitution & Working [C1]
  4. Final Answer (exact + rounded to 2–3 significant figures with SI unit) [A1]
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SolvedPhysicsProblem:
    topic: str
    formula_display: str
    formula_words: str
    given_lines: List[str]
    working_lines: List[str]
    exact_value: float
    rounded_2sf: str
    rounded_3sf: str
    unit: str
    exam_tip: str

    def to_markdown(self) -> str:
        given_md = "\n".join(f"- {g}" for g in self.given_lines)
        work_md = "\n".join(f"  {w}" for w in self.working_lines)
        ans_line = f"**{self.rounded_2sf} {self.unit}**"
        if self.rounded_3sf != self.rounded_2sf:
            ans_line += f" *(3 s.f.: `{self.rounded_3sf} {self.unit}`, exact: `{format_clean_float(self.exact_value)} {self.unit}`)*"
        return (
            f"### Step-by-Step IGCSE Physics Calculation ({self.topic})\n\n"
            f"**1. Given Quantities & SI Unit Check:**\n{given_md}\n\n"
            f"**2. Formula *(earns C1 / M1 mark)*:**\n"
            f"- `{self.formula_display}` — *{self.formula_words}*\n\n"
            f"**3. Substitution & Working *(earns C1 mark)*:**\n"
            f"{work_md}\n\n"
            f"**4. Final Answer *(earns A1 mark — include unit & 2–3 s.f.)*:**\n"
            f"- {ans_line}\n\n"
            f"> **Examiner Tip:** {self.exam_tip}"
        )


def format_sig_figs(val: float, sig: int = 2) -> str:
    if val == 0 or math.isnan(val) or math.isinf(val):
        return "0"
    abs_v = abs(val)
    if abs_v >= 1e5 or (0 < abs_v < 1e-2):
        # Scientific / standard form e.g. 1.5 × 10⁶
        exp = int(math.floor(math.log10(abs_v)))
        mant = val / (10 ** exp)
        mant_r = round(mant, sig - 1)
        if abs(mant_r) >= 10.0:
            mant_r /= 10.0
            exp += 1
        fmt = f"{mant_r:.{max(0, sig - 1)}f}"
        return f"{fmt} × 10^{exp}"
    digits = sig - int(math.floor(math.log10(abs_v))) - 1
    rounded = round(val, digits)
    if digits <= 0:
        return str(int(round(rounded)))
    return f"{rounded:.{digits}f}"


def format_clean_float(val: float) -> str:
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e12:
        return str(int(round(val)))
    return f"{val:.6g}"


NUM_PATTERN = r"([+-]?\d+(?:\.\d+)?(?:\s*[×x*]\s*10\s*\^?\s*[+-]?\d+|e[+-]?\d+)?)"


def parse_num(s: str) -> float:
    s = s.strip().replace(" ", "")
    s = re.sub(r"[×x*]10\^?([+-]?\d+)", r"e\1", s, flags=re.I)
    return float(s)


def find_quantity(text: str, unit_regex: str) -> List[Tuple[float, str, str]]:
    """Return list of (raw_value, matched_unit, full_snippet) for numbers followed by unit_regex."""
    pat = re.compile(NUM_PATTERN + r"\s*(" + unit_regex + r")(?![a-zA-Z/])", re.I)
    out = []
    for m in pat.finditer(text):
        try:
            val = parse_num(m.group(1))
            out.append((val, m.group(2), m.group(0)))
        except Exception:
            continue
    return out


def extract_si_quantities(text: str) -> Dict[str, List[Tuple[float, str]]]:
    """Extract physical quantities converted to SI base units alongside readable conversion notes."""
    q: Dict[str, List[Tuple[float, str]]] = {
        "mass": [],
        "speed": [],
        "accel": [],
        "time": [],
        "length": [],
        "area": [],
        "volume": [],
        "force": [],
        "energy": [],
        "power": [],
        "pressure": [],
        "freq": [],
        "current": [],
        "voltage": [],
        "resistance": [],
        "charge": [],
        "temp_c": [],
        "shc": [],
        "angle_deg": [],
        "density": [],
    }

    # Normalize unicode superscripts & power-of-10 notation for clean matching
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    norm = (
        text.replace("×", "x")
        .replace("−", "-")
        .replace("Ω", "ohm")
        .replace("Ω", "ohm")
    )
    norm = re.sub(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "10^" + m.group(1).translate(sup_map), norm)
    norm = re.sub(r"([mkc]?m|s)([²³⁻⁰¹⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "^" + m.group(2).translate(sup_map), norm)
    norm = norm.translate(sup_map)

    # Specific heat capacity (J/(kg °C) or J/g°C)
    for val, u, raw in find_quantity(norm, r"J\s*/\s*\(?\s*(?:kg|g)\s*[°o]?\s*[CK]\s*\)?"):
        if "/g" in u.lower().replace(" ", "") and "kg" not in u.lower():
            si = val * 1000.0
            q["shc"].append((si, f"c = {raw} = {format_clean_float(si)} J/(kg °C)"))
        else:
            q["shc"].append((val, f"c = {format_clean_float(val)} J/(kg °C)"))

    # Acceleration (m/s^2)
    for val, u, raw in find_quantity(norm, r"m\s*/\s*s\^?2|m\s*s\s*-\s*2"):
        q["accel"].append((val, f"a = {format_clean_float(val)} m/s²"))

    # Speed / Velocity (m/s, km/h, cm/s)
    for val, u, raw in find_quantity(norm, r"m\s*/\s*s|km\s*/\s*h(?:r)?|cm\s*/\s*s"):
        ul = u.lower().replace(" ", "")
        if ul.startswith("km"):
            si = val / 3.6
            q["speed"].append((si, f"v = {raw} = {format_clean_float(si)} m/s (÷ 3.6)"))
        elif ul.startswith("cm"):
            si = val / 100.0
            q["speed"].append((si, f"v = {raw} = {format_clean_float(si)} m/s"))
        else:
            q["speed"].append((val, f"v = {format_clean_float(val)} m/s"))

    # Mass (kg, g, mg, tonnes)
    for val, u, raw in find_quantity(norm, r"kg|kilograms?|tonnes?|mg|g|grams?"):
        ul = u.lower()
        if ul.startswith("kg") or ul.startswith("kilogram"):
            q["mass"].append((val, f"m = {format_clean_float(val)} kg"))
        elif ul.startswith("tonne"):
            si = val * 1000.0
            q["mass"].append((si, f"m = {raw} = {format_clean_float(si)} kg"))
        elif ul == "mg":
            si = val * 1e-6
            q["mass"].append((si, f"m = {raw} = {format_clean_float(si)} kg"))
        else:
            si = val / 1000.0
            q["mass"].append((si, f"m = {raw} = {format_clean_float(si)} kg (÷ 1000)"))

    # Area (m^2, cm^2, mm^2)
    for val, u, raw in find_quantity(norm, r"m\^2|cm\^2|mm\^2"):
        ul = u.lower()
        if ul.startswith("cm"):
            si = val * 1e-4
            q["area"].append((si, f"A = {raw} = {format_clean_float(si)} m² (× 10⁻⁴)"))
        elif ul.startswith("mm"):
            si = val * 1e-6
            q["area"].append((si, f"A = {raw} = {format_clean_float(si)} m² (× 10⁻⁶)"))
        else:
            q["area"].append((val, f"A = {format_clean_float(val)} m²"))

    # Density (kg/m^3, g/cm^3)
    for val, u, raw in find_quantity(norm, r"kg\s*/\s*m\^?3|kg\s*m\s*\^?-3|g\s*/\s*cm\^?3|g\s*cm\s*\^?-3"):
        ul = u.lower().replace(" ", "")
        if ul.startswith("g"):
            si = val * 1000.0
            q["density"].append((si, f"ρ = {raw} = {format_clean_float(si)} kg/m³ (× 1000)"))
        else:
            q["density"].append((val, f"ρ = {format_clean_float(val)} kg/m³"))

    # Volume (km^3, m^3, cm^3, mm^3, litres)
    for val, u, raw in find_quantity(norm, r"km\^?3|m\^?3|cm\^?3|mm\^?3|litres?|L"):
        ul = u.lower()
        if ul.startswith("km"):
            si = val * 1e9
            q["volume"].append((si, f"V = {raw} = {format_clean_float(si)} m³ (× 10⁹)"))
        elif ul.startswith("cm"):
            si = val * 1e-6
            q["volume"].append((si, f"V = {raw} = {format_clean_float(si)} m³ (× 10⁻⁶)"))
        elif ul.startswith("mm"):
            si = val * 1e-9
            q["volume"].append((si, f"V = {raw} = {format_clean_float(si)} m³ (× 10⁻⁹)"))
        elif ul.startswith("l"):
            si = val * 1e-3
            q["volume"].append((si, f"V = {raw} = {format_clean_float(si)} m³ (× 10⁻³)"))
        else:
            q["volume"].append((val, f"V = {format_clean_float(val)} m³"))

    # Length / Distance / Wavelength / Height (km, m, cm, mm, nm) — avoid matching m/s, m^2, m^3, min, ms, mA
    len_pat = re.compile(NUM_PATTERN + r"\s*(km|cm|mm|nm|metres?|meters?|m)(?![a-zA-Z/^0-9])", re.I)
    for m in len_pat.finditer(norm):
        val = parse_num(m.group(1))
        u = m.group(2).lower()
        raw = m.group(0)
        if u == "km":
            si = val * 1000.0
            q["length"].append((si, f"d = {raw} = {format_clean_float(si)} m (× 1000)"))
        elif u == "cm":
            si = val / 100.0
            q["length"].append((si, f"d = {raw} = {format_clean_float(si)} m (÷ 100)"))
        elif u == "mm":
            si = val / 1000.0
            q["length"].append((si, f"d = {raw} = {format_clean_float(si)} m (÷ 1000)"))
        elif u == "nm":
            si = val * 1e-9
            q["length"].append((si, f"λ = {raw} = {format_clean_float(si)} m"))
        else:
            q["length"].append((val, f"d = {format_clean_float(val)} m"))

    # Time (ms, s, min, hours, years)
    time_pat = re.compile(NUM_PATTERN + r"\s*(ms|milliseconds?|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h)(?![a-zA-Z/])", re.I)
    for m in time_pat.finditer(norm):
        # Skip if preceded by '/' like m/s
        prefix = norm[max(0, m.start() - 3): m.start()]
        if "/" in prefix:
            continue
        val = parse_num(m.group(1))
        u = m.group(2).lower()
        raw = m.group(0)
        if u.startswith("ms") or u.startswith("millisec"):
            si = val / 1000.0
            q["time"].append((si, f"t = {raw} = {format_clean_float(si)} s"))
        elif u.startswith("min"):
            si = val * 60.0
            q["time"].append((si, f"t = {raw} = {format_clean_float(si)} s (× 60)"))
        elif u.startswith("h"):
            si = val * 3600.0
            q["time"].append((si, f"t = {raw} = {format_clean_float(si)} s (× 3600)"))
        else:
            q["time"].append((val, f"t = {format_clean_float(val)} s"))

    # Force / Weight (MN, kN, N)
    for val, u, raw in find_quantity(norm, r"MN|kN|newtons?|N"):
        if u == "kN":
            si = val * 1000.0
            q["force"].append((si, f"F = {raw} = {format_clean_float(si)} N"))
        elif u == "MN":
            si = val * 1e6
            q["force"].append((si, f"F = {raw} = {format_clean_float(si)} N"))
        else:
            q["force"].append((val, f"F = {format_clean_float(val)} N"))

    # Energy / Work (MJ, kJ, J)
    en_pat = re.compile(NUM_PATTERN + r"\s*(MJ|kJ|joules?|J)(?![a-zA-Z/])", re.I)
    for m in en_pat.finditer(norm):
        val = parse_num(m.group(1))
        u = m.group(2)
        raw = m.group(0)
        if u.upper() == "KJ":
            si = val * 1000.0
            q["energy"].append((si, f"E = {raw} = {format_clean_float(si)} J (× 1000)"))
        elif u.upper() == "MJ":
            si = val * 1e6
            q["energy"].append((si, f"E = {raw} = {format_clean_float(si)} J (× 10⁶)"))
        else:
            q["energy"].append((val, f"E = {format_clean_float(val)} J"))

    # Power (MW, kW, W)
    for val, u, raw in find_quantity(norm, r"MW|kW|watts?|W"):
        if u.upper() == "KW":
            si = val * 1000.0
            q["power"].append((si, f"P = {raw} = {format_clean_float(si)} W (× 1000)"))
        elif u.upper() == "MW":
            si = val * 1e6
            q["power"].append((si, f"P = {raw} = {format_clean_float(si)} W (× 10⁶)"))
        else:
            q["power"].append((val, f"P = {format_clean_float(val)} W"))

    # Pressure (MPa, kPa, Pa)
    for val, u, raw in find_quantity(norm, r"MPa|kPa|pascals?|Pa"):
        if u.lower() == "kpa":
            si = val * 1000.0
            q["pressure"].append((si, f"p = {raw} = {format_clean_float(si)} Pa (× 1000)"))
        elif u.lower() == "mpa":
            si = val * 1e6
            q["pressure"].append((si, f"p = {raw} = {format_clean_float(si)} Pa (× 10⁶)"))
        else:
            q["pressure"].append((val, f"p = {format_clean_float(val)} Pa"))

    # Frequency (MHz, kHz, Hz)
    for val, u, raw in find_quantity(norm, r"MHz|kHz|hertz|Hz"):
        if u.lower() == "khz":
            si = val * 1000.0
            q["freq"].append((si, f"f = {raw} = {format_clean_float(si)} Hz"))
        elif u.lower() == "mhz":
            si = val * 1e6
            q["freq"].append((si, f"f = {raw} = {format_clean_float(si)} Hz"))
        else:
            q["freq"].append((val, f"f = {format_clean_float(val)} Hz"))

    # Current (mA, A)
    for val, u, raw in find_quantity(norm, r"mA|amperes?|amps?|A"):
        if u == "mA":
            si = val / 1000.0
            q["current"].append((si, f"I = {raw} = {format_clean_float(si)} A (÷ 1000)"))
        else:
            q["current"].append((val, f"I = {format_clean_float(val)} A"))

    # Voltage (kV, mV, V)
    for val, u, raw in find_quantity(norm, r"kV|mV|volts?|V"):
        if u.lower() == "kv":
            si = val * 1000.0
            q["voltage"].append((si, f"V = {raw} = {format_clean_float(si)} V"))
        elif u.lower() == "mv":
            si = val / 1000.0
            q["voltage"].append((si, f"V = {raw} = {format_clean_float(si)} V"))
        else:
            q["voltage"].append((val, f"V = {format_clean_float(val)} V"))

    # Resistance (kohm, ohm)
    for val, u, raw in find_quantity(norm, r"k\s*ohms?|ohms?"):
        if u.lower().startswith("k"):
            si = val * 1000.0
            q["resistance"].append((si, f"R = {raw} = {format_clean_float(si)} Ω (× 1000)"))
        else:
            q["resistance"].append((val, f"R = {format_clean_float(val)} Ω"))

    # Charge (mC, μC, C)
    for val, u, raw in find_quantity(norm, r"mC|coulombs?|C"):
        if u == "mC":
            si = val / 1000.0
            q["charge"].append((si, f"Q = {raw} = {format_clean_float(si)} C"))
        else:
            q["charge"].append((val, f"Q = {format_clean_float(val)} C"))

    # Temperature (°C)
    for val, u, raw in find_quantity(norm, r"°\s*C|degrees?\s*C(?:elsius)?|deg\s*C"):
        q["temp_c"].append((val, f"θ = {format_clean_float(val)} °C"))

    # Angles (degrees)
    for val, u, raw in find_quantity(norm, r"°|degrees?"):
        if "C" not in raw:
            q["angle_deg"].append((val, f"angle = {format_clean_float(val)}°"))

    return q


def try_solve_physics_problem(question_text: str) -> Optional[SolvedPhysicsProblem]:
    """Attempt to deterministically solve a numerical IGCSE Physics calculation problem."""
    low = question_text.lower()
    is_calc_request = any(
        kw in low
        for kw in (
            "calculate", "find", "determine", "what is the", "work out", "how much",
            "how far", "how long", "how fast", "compute", "solve",
        )
    ) or "?" in question_text

    if not is_calc_request:
        return None

    q = extract_si_quantities(question_text)

    def make_res(topic: str, f_disp: str, f_words: str, given: List[str], work: List[str], val: float, unit: str, tip: str) -> SolvedPhysicsProblem:
        return SolvedPhysicsProblem(
            topic=topic,
            formula_display=f_disp,
            formula_words=f_words,
            given_lines=given,
            working_lines=work,
            exact_value=val,
            rounded_2sf=format_sig_figs(val, 2),
            rounded_3sf=format_sig_figs(val, 3),
            unit=unit,
            exam_tip=tip,
        )

    # 1. Kinetic Energy: Ek = 1/2 m v^2
    if ("kinetic" in low or "e_k" in low or "ek " in low) and q["mass"] and q["speed"]:
        m, m_str = q["mass"][0]
        v, v_str = q["speed"][0]
        ek = 0.5 * m * (v ** 2)
        return make_res(
            "Kinetic Energy (Unit 1: Work, Energy & Power)",
            "E_k = ½ m v²",
            "kinetic energy = ½ × mass × speed²",
            [m_str, v_str],
            [f"`E_k = 0.5 × {format_clean_float(m)} × ({format_clean_float(v)})²`", f"`E_k = {format_clean_float(ek)} J`"],
            ek,
            "J",
            "Remember to square ONLY the speed `v`, not `½ m v`! Always write `E_k = ½ m v²` first for the C1 mark.",
        )

    # 2. Gravitational Potential Energy / Height: Ep = m g h
    if any(k in low for k in ("potential energy", "gravitational store", "e_p", "ep ", "height")) and q["mass"]:
        g_val = 10.0 if re.search(r"g\s*=\s*10\b", low) else 9.8
        m, m_str = q["mass"][0]
        if ("height" in low or "how high" in low) and q["energy"]:
            ep, e_str = q["energy"][0]
            h = ep / (m * g_val)
            return make_res(
                "Gravitational Potential Energy — Finding Height (Unit 1)",
                "Δh = ΔE_p / (m g)",
                "change in height = change in gravitational potential energy ÷ (mass × g)",
                [e_str, m_str, f"g = {g_val} N/kg (Cambridge 0625 standard)"],
                [f"`Δh = {format_clean_float(ep)} / ({format_clean_float(m)} × {g_val})`", f"`Δh = {format_clean_float(h)} m`"],
                h,
                "m",
                "Convert kJ to J (× 1000) before dividing! Cambridge 0625 uses g = 9.8 N/kg unless stated otherwise.",
            )
        if q["length"]:
            h, h_str = q["length"][0]
            ep = m * g_val * h
            return make_res(
                "Gravitational Potential Energy (Unit 1)",
                "ΔE_p = m g Δh",
                "gravitational potential energy = mass × gravitational field strength × height",
                [m_str, h_str, f"g = {g_val} N/kg"],
                [f"`ΔE_p = {format_clean_float(m)} × {g_val} × {format_clean_float(h)}`", f"`ΔE_p = {format_clean_float(ep)} J`"],
                ep,
                "J",
                "Cambridge 0625 uses `g = 9.8 N/kg` (or `9.8 m/s²`). Always show the formula `ΔE_p = mgΔh` first.",
            )

    # 3. Specific Heat Capacity: ΔE = m c Δθ
    if ("specific heat" in low or "thermal energy" in low or "heat energy" in low or ("heats" in low and q["temp_c"])) and q["mass"]:
        m, m_str = q["mass"][0]
        temps = [t[0] for t in q["temp_c"]]
        d_theta = abs(temps[1] - temps[0]) if len(temps) >= 2 else (temps[0] if len(temps) == 1 else None)
        c_val = q["shc"][0][0] if q["shc"] else (4200.0 if "water" in low else None)
        if d_theta is not None and c_val is not None and ("energy" in low or "how much heat" in low):
            de = m * c_val * d_theta
            given = [m_str, f"c = {format_clean_float(c_val)} J/(kg °C)", f"Δθ = {format_clean_float(d_theta)} °C"]
            return make_res(
                "Specific Heat Capacity (Unit 2: Thermal Physics)",
                "ΔE = m c Δθ",
                "thermal energy = mass × specific heat capacity × temperature change",
                given,
                [f"`ΔE = {format_clean_float(m)} × {format_clean_float(c_val)} × {format_clean_float(d_theta)}`", f"`ΔE = {format_clean_float(de)} J`"],
                de,
                "J",
                "Make sure mass `m` matches the unit in `c` (`kg` with `4200 J/(kg °C)` OR `g` with `4.2 J/(g °C)` — never mix `0.2 kg` with `4.2`!).",
            )
        if d_theta is not None and q["energy"] and "specific heat capacity" in low:
            de, e_str = q["energy"][0]
            c_calc = de / (m * d_theta)
            return make_res(
                "Finding Specific Heat Capacity (Unit 2: Thermal Physics)",
                "c = ΔE / (m Δθ)",
                "specific heat capacity = thermal energy ÷ (mass × temperature change)",
                [e_str, m_str, f"Δθ = {format_clean_float(d_theta)} °C"],
                [f"`c = {format_clean_float(de)} / ({format_clean_float(m)} × {format_clean_float(d_theta)})`", f"`c = {format_clean_float(c_calc)} J/(kg °C)`"],
                c_calc,
                "J/(kg °C)",
                "Include the full unit `J/(kg °C)` or `J/(kg K)` for the final mark.",
            )

    # 4. Acceleration: a = (v - u) / t
    if ("acceleration" in low or "deceleration" in low) and q["time"] and q["speed"]:
        t, t_str = q["time"][-1]
        if len(q["speed"]) >= 2:
            u, u_str = q["speed"][0]
            v, v_str = q["speed"][1]
        else:
            u, u_str = 0.0, "u = 0 m/s (from rest)"
            v, v_str = q["speed"][0]
        if t > 0:
            a = (v - u) / t
            val = abs(a) if "deceleration" in low else a
            return make_res(
                "Acceleration (Unit 1: Kinematics)",
                "a = (v − u) / t",
                "acceleration = change in velocity ÷ time taken",
                [u_str, v_str, t_str],
                [f"`a = ({format_clean_float(v)} − {format_clean_float(u)}) / {format_clean_float(t)}`", f"`a = {format_clean_float(val)} m/s²`"],
                val,
                "m/s²",
                "Note: Deceleration is negative acceleration (so if a question asks for 'deceleration', write the positive magnitude in m/s²).",
            )

    # 4b. Multi-segment Average Speed vs Average Velocity (e.g. "100 m north then 100 m south in 20 s")
    dir_segs = re.findall(
        NUM_PATTERN + r"\s*(km|cm|mm|m)\s+(north|south|east|west|up|down|forward|forwards|back|backward|backwards|left|right)\b",
        question_text,
        re.I,
    )
    if len(dir_segs) >= 2 and q["time"] and ("speed" in low or "velocity" in low):
        dx, dy, total_d = 0.0, 0.0, 0.0
        seg_notes = []
        for num_s, unit_s, dir_s in dir_segs:
            val = parse_num(num_s)
            ul = unit_s.lower()
            d_m = val * 1000.0 if ul == "km" else val / 100.0 if ul == "cm" else val / 1000.0 if ul == "mm" else val
            total_d += d_m
            dl = dir_s.lower()
            if dl in ("north", "up", "forward", "forwards"):
                dy += d_m
            elif dl in ("south", "down", "back", "backward", "backwards"):
                dy -= d_m
            elif dl in ("east", "right"):
                dx += d_m
            elif dl in ("west", "left"):
                dx -= d_m
            seg_notes.append(f"{format_clean_float(d_m)} m {dl}")
        net_disp = math.hypot(dx, dy)
        t_tot, t_str = q["time"][-1]
        if t_tot > 0:
            avg_spd = total_d / t_tot
            avg_vel = net_disp / t_tot
            ask_vel_only = "velocity" in low and "speed" not in low
            primary_val = avg_vel if ask_vel_only else avg_spd
            return make_res(
                "Average Speed vs Average Velocity (Unit 1: Kinematics)",
                "Average Speed = Total Distance / Total Time  |  Average Velocity = Net Displacement / Total Time",
                "speed uses scalar total distance; velocity uses vector net displacement",
                [*seg_notes, t_str],
                [
                    f"`Total distance = {format_clean_float(total_d)} m` → **`Average speed = {format_clean_float(total_d)} / {format_clean_float(t_tot)} = {format_clean_float(avg_spd)} m/s`**",
                    f"`Net displacement = {format_clean_float(net_disp)} m` → **`Average velocity = {format_clean_float(net_disp)} / {format_clean_float(t_tot)} = {format_clean_float(avg_vel)} m/s`**",
                ],
                primary_val,
                "m/s",
                "**Distance / speed** are scalars (add all path lengths); **displacement / velocity** are vectors (opposite directions cancel!).",
            )

    # 4c. Single-segment Average Speed: v = d / t
    if "speed" in low and len(q["length"]) == 1 and q["time"] and not q["speed"] and "velocity" not in low:
        d, d_str = q["length"][0]
        t, t_str = q["time"][0]
        if t > 0:
            v = d / t
            return make_res(
                "Average Speed (Unit 1: Kinematics)",
                "v = d / t",
                "average speed = distance travelled ÷ time taken",
                [d_str, t_str],
                [f"`v = {format_clean_float(d)} / {format_clean_float(t)}`", f"`v = {format_clean_float(v)} m/s`"],
                v,
                "m/s",
                "Speed is a scalar (`distance / time`); convert `km` to `m` and `minutes` to `s` for `m/s`.",
            )

    # 5. Newton's Second Law: F = m a (or a = F / m)
    if ("resultant force" in low or "force" in low or "acceleration" in low) and q["mass"] and (q["accel"] or q["force"]):
        m, m_str = q["mass"][0]
        if q["accel"] and "force" in low:
            a, a_str = q["accel"][0]
            f = m * a
            return make_res(
                "Newton's Second Law (Unit 1: Forces)",
                "F = m a",
                "resultant force = mass × acceleration",
                [m_str, a_str],
                [f"`F = {format_clean_float(m)} × {format_clean_float(a)}`", f"`F = {format_clean_float(f)} N`"],
                f,
                "N",
                "`F` in `F = ma` is always the **resultant (net) force** in Newtons, and mass `m` must be in kilograms.",
            )
        if q["force"] and "acceleration" in low:
            f, f_str = q["force"][0]
            a = f / m
            return make_res(
                "Newton's Second Law — Finding Acceleration (Unit 1: Forces)",
                "a = F / m",
                "acceleration = resultant force ÷ mass",
                [f_str, m_str],
                [f"`a = {format_clean_float(f)} / {format_clean_float(m)}`", f"`a = {format_clean_float(a)} m/s²`"],
                a,
                "m/s²",
                "Always convert mass from grams to kilograms before dividing force by mass.",
            )

    # 6. Weight: W = m g
    if "weight" in low and q["mass"] and not q["length"]:
        g_val = 10.0 if re.search(r"g\s*=\s*10\b", low) else 9.8
        m, m_str = q["mass"][0]
        w = m * g_val
        return make_res(
            "Weight & Mass (Unit 1: Forces)",
            "W = m g",
            "weight = mass × gravitational field strength",
            [m_str, f"g = {g_val} N/kg"],
            [f"`W = {format_clean_float(m)} × {g_val}`", f"`W = {format_clean_float(w)} N`"],
            w,
            "N",
            "Weight is a **force** measured in **Newtons (N)**, never in kg!",
        )

    # 7. Density: ρ = m / V, m = ρ × V, V = m / ρ
    if "density" in low and q["mass"] and q["volume"]:
        m, m_str = q["mass"][0]
        vol, v_str = q["volume"][0]
        rho = m / vol
        return make_res(
            "Density (Unit 1: Measurements & Density)",
            "ρ = m / V",
            "density = mass ÷ volume",
            [m_str, v_str],
            [f"`ρ = {format_clean_float(m)} / {format_clean_float(vol)}`", f"`ρ = {format_clean_float(rho)} kg/m³`"],
            rho,
            "kg/m³",
            "Note: `1 g/cm³ = 1000 kg/m³`. Always check whether the question asks for `kg/m³` or `g/cm³`.",
        )
    if "density" in low and q["density"] and q["volume"] and "mass" in low:
        rho, r_str = q["density"][0]
        vol, v_str = q["volume"][0]
        m_val = rho * vol
        return make_res(
            "Mass from Density & Volume (Unit 1: Density)",
            "m = ρ × V",
            "mass = density × volume",
            [r_str, v_str],
            [f"`m = {format_clean_float(rho)} × {format_clean_float(vol)}`", f"`m = {format_clean_float(m_val)} kg`"],
            m_val,
            "kg",
            "Convert volume to `m³` first (`1 km³ = 10⁹ m³`, `1 cm³ = 10⁻⁶ m³`) when density is in `kg/m³`.",
        )
    if "density" in low and q["density"] and q["mass"] and "volume" in low:
        rho, r_str = q["density"][0]
        m_val, m_str = q["mass"][0]
        vol = m_val / rho
        return make_res(
            "Volume from Mass & Density (Unit 1: Density)",
            "V = m / ρ",
            "volume = mass ÷ density",
            [m_str, r_str],
            [f"`V = {format_clean_float(m_val)} / {format_clean_float(rho)}`", f"`V = {format_clean_float(vol)} m³`"],
            vol,
            "m³",
            "Ensure mass and density use matching units (`kg` with `kg/m³`, or `g` with `g/cm³`).",
        )

    # 8. Moment of a Force: M = F × d
    if "moment" in low and q["force"] and q["length"] and len(q["force"]) == 1:
        f, f_str = q["force"][0]
        d, d_str = q["length"][0]
        mom = f * d
        return make_res(
            "Moment of a Force (Unit 1: Moments)",
            "M = F × d",
            "moment = force × perpendicular distance from pivot",
            [f_str, d_str],
            [f"`M = {format_clean_float(f)} × {format_clean_float(d)}`", f"`M = {format_clean_float(mom)} N m`"],
            mom,
            "N m",
            "Distance `d` must be the **perpendicular** distance from the line of action of the force to the pivot!",
        )

    # 9. Momentum: p = m v
    if "momentum" in low and q["mass"] and q["speed"] and len(q["mass"]) == 1 and len(q["speed"]) == 1:
        m, m_str = q["mass"][0]
        v, v_str = q["speed"][0]
        mom = m * v
        return make_res(
            "Momentum (Unit 1: Momentum)",
            "p = m v",
            "momentum = mass × velocity",
            [m_str, v_str],
            [f"`p = {format_clean_float(m)} × {format_clean_float(v)}`", f"`p = {format_clean_float(mom)} kg m/s`"],
            mom,
            "kg m/s",
            "Momentum is a **vector** quantity — its unit is `kg m/s` (or `N s`, which is equivalent!).",
        )

    # 10. Solid Pressure: p = F / A (or A = F / p)
    if ("pressure" in low or "area" in low) and (q["force"] and (q["area"] or q["pressure"])):
        f, f_str = q["force"][0]
        if q["area"] and "pressure" in low:
            a, a_str = q["area"][0]
            pr = f / a
            return make_res(
                "Pressure (Unit 1: Pressure)",
                "p = F / A",
                "pressure = force ÷ contact area",
                [f_str, a_str],
                [f"`p = {format_clean_float(f)} / {format_clean_float(a)}`", f"`p = {format_clean_float(pr)} Pa`"],
                pr,
                "Pa",
                "`1 Pa = 1 N/m²`. If area is given in `cm²`, multiply by `10⁻⁴` to convert to `m²`.",
            )
        if q["pressure"] and "area" in low:
            pr, p_str = q["pressure"][0]
            a = f / pr
            return make_res(
                "Pressure — Finding Contact Area (Unit 1: Pressure)",
                "A = F / p",
                "area = force ÷ pressure",
                [f_str, p_str],
                [f"`A = {format_clean_float(f)} / {format_clean_float(pr)}`", f"`A = {format_clean_float(a)} m²`"],
                a,
                "m²",
                "Ensure pressure in kPa is converted to Pa (`× 1000`) so area comes out in `m²`.",
            )

    # 11. Boyle's Law: p1 V1 = p2 V2
    if len(q["pressure"]) == 2 and len(q["volume"]) == 1:
        p1, p1_s = q["pressure"][0]
        p2, p2_s = q["pressure"][1]
        v1, v1_s = q["volume"][0]
        v2 = (p1 * v1) / p2
        return make_res(
            "Boyle's Law at Constant Temperature (Unit 2: Kinetic Theory)",
            "p₁ V₁ = p₂ V₂",
            "initial pressure × initial volume = final pressure × final volume",
            [p1_s, v1_s, p2_s],
            [f"`V₂ = (p₁ × V₁) / p₂ = ({format_clean_float(p1)} × {format_clean_float(v1)}) / {format_clean_float(p2)}`", f"`V₂ = {format_clean_float(v2)} m³`"],
            v2,
            "m³",
            "Boyle's law applies to a fixed mass of gas at **constant temperature**.",
        )

    # 12. Mechanical / Electrical Power from Energy & Time: P = E / t
    if "power" in low and q["energy"] and q["time"]:
        e, e_str = q["energy"][0]
        t, t_str = q["time"][0]
        pw = e / t
        return make_res(
            "Power (Rate of Energy Transfer)",
            "P = ΔE / t",
            "power = energy transferred ÷ time taken",
            [e_str, t_str],
            [f"`P = {format_clean_float(e)} / {format_clean_float(t)}`", f"`P = {format_clean_float(pw)} W`"],
            pw,
            "W",
            "`1 Watt = 1 Joule per second (1 J/s)`. Convert minutes to seconds (`× 60`) before dividing!",
        )

    # 13. Work Done: W = F d
    if "work done" in low and q["force"] and q["length"]:
        f, f_str = q["force"][0]
        d, d_str = q["length"][0]
        w = f * d
        return make_res(
            "Mechanical Work Done (Unit 1: Work, Energy & Power)",
            "W = F d",
            "work done = force × distance moved in the direction of the force",
            [f_str, d_str],
            [f"`W = {format_clean_float(f)} × {format_clean_float(d)}`", f"`W = {format_clean_float(w)} J`"],
            w,
            "J",
            "Distance must be measured **in the direction of the applied force**.",
        )

    # 14. Wave Equation: v = f λ
    if any(k in low for k in ("wave", "wavelength", "frequency", "sound", "ripple")) and (
        (q["freq"] and q["length"]) or (q["speed"] and q["length"] and "frequency" in low) or (q["speed"] and q["freq"] and "wavelength" in low)
    ):
        if q["freq"] and q["length"] and ("speed" in low or "velocity" in low):
            f, f_str = q["freq"][0]
            lam, l_str = q["length"][0]
            v = f * lam
            return make_res(
                "Wave Speed Equation (Unit 6: Waves)",
                "v = f λ",
                "wave speed = frequency × wavelength",
                [f_str, l_str.replace("d =", "λ =")],
                [f"`v = {format_clean_float(f)} × {format_clean_float(lam)}`", f"`v = {format_clean_float(v)} m/s`"],
                v,
                "m/s",
                "Convert wavelength from `cm` or `mm` into metres (`m`) so wave speed is in `m/s`.",
            )
        if q["speed"] and q["length"] and "frequency" in low:
            v, v_str = q["speed"][0]
            lam, l_str = q["length"][0]
            f = v / lam
            return make_res(
                "Wave Equation — Finding Frequency (Unit 6: Waves)",
                "f = v / λ",
                "frequency = wave speed ÷ wavelength",
                [v_str, l_str.replace("d =", "λ =")],
                [f"`f = {format_clean_float(v)} / {format_clean_float(lam)}`", f"`f = {format_clean_float(f)} Hz`"],
                f,
                "Hz",
                "Frequency is measured in **Hertz (Hz)**, which equals waves per second (`s⁻¹`).",
            )
        if q["speed"] and q["freq"] and "wavelength" in low:
            v, v_str = q["speed"][0]
            f, f_str = q["freq"][0]
            lam = v / f
            return make_res(
                "Wave Equation — Finding Wavelength (Unit 6: Waves)",
                "λ = v / f",
                "wavelength = wave speed ÷ frequency",
                [v_str, f_str],
                [f"`λ = {format_clean_float(v)} / {format_clean_float(f)}`", f"`λ = {format_clean_float(lam)} m`"],
                lam,
                "m",
                "Check kHz (`× 10³ Hz`) or MHz (`× 10⁶ Hz`) conversions carefully.",
            )

    # 15. Echo Sounding / Ultrasound Distance: d = v t / 2
    if "echo" in low and q["speed"] and q["time"]:
        v, v_str = q["speed"][0]
        t, t_str = q["time"][0]
        d = (v * t) / 2.0
        return make_res(
            "Echo / Ultrasound Distance (Unit 6: Sound Waves)",
            "d = (v × t) / 2",
            "distance to reflector = (speed of sound × echo time) ÷ 2",
            [v_str, t_str],
            [f"`d = ({format_clean_float(v)} × {format_clean_float(t)}) / 2`", f"`d = {format_clean_float(d)} m`"],
            d,
            "m",
            "Never forget to **divide by 2** for an echo because the wave travels there AND back!",
        )

    # 16. Snell's Law Refractive Index: n = sin i / sin r
    if "refractive index" in low and len(q["angle_deg"]) >= 2:
        i_deg, i_str = q["angle_deg"][0]
        r_deg, r_str = q["angle_deg"][1]
        if 0 < r_deg < 90:
            n = math.sin(math.radians(i_deg)) / math.sin(math.radians(r_deg))
            return make_res(
                "Snell's Law — Refractive Index (Unit 6: Light Refraction)",
                "n = sin i / sin r",
                "refractive index = sin(angle of incidence) ÷ sin(angle of refraction)",
                [f"i = {format_clean_float(i_deg)}°", f"r = {format_clean_float(r_deg)}°"],
                [f"`n = sin({format_clean_float(i_deg)}°) / sin({format_clean_float(r_deg)}°)`", f"`n = {format_clean_float(n)}`"],
                n,
                "(no unit)",
                "Refractive index `n` is a ratio and has **no unit**! Always measure angles `i` and `r` from the **normal**, never from the surface.",
            )

    # 17. Electric Current & Charge: I = Q / t or Q = I t
    if ("current" in low or "charge" in low) and ((q["charge"] and q["time"]) or (q["current"] and q["time"])):
        if q["charge"] and q["time"] and "current" in low:
            ch, c_str = q["charge"][0]
            t, t_str = q["time"][0]
            curr = ch / t
            return make_res(
                "Electric Current & Charge (Unit 3: Electricity)",
                "I = Q / t",
                "current = charge ÷ time",
                [c_str, t_str],
                [f"`I = {format_clean_float(ch)} / {format_clean_float(t)}`", f"`I = {format_clean_float(curr)} A`"],
                curr,
                "A",
                "Remember to convert time into **seconds** (`1 min = 60 s`) so current is in Amperes (`A`).",
            )
        if q["current"] and q["time"] and "charge" in low:
            curr, i_str = q["current"][0]
            t, t_str = q["time"][0]
            ch = curr * t
            return make_res(
                "Electric Charge (Unit 3: Electricity)",
                "Q = I t",
                "charge = current × time",
                [i_str, t_str],
                [f"`Q = {format_clean_float(curr)} × {format_clean_float(t)}`", f"`Q = {format_clean_float(ch)} C`"],
                ch,
                "C",
                "Charge is measured in **Coulombs (C)**.",
            )

    # 18. Potential Divider / Series & Parallel Resistors / Ohm's Law
    if len(q["resistance"]) == 2 and ("parallel" in low or "series" in low) and "total resistance" in low or "combined resistance" in low:
        if len(q["resistance"]) >= 2:
            r1, r1_s = q["resistance"][0]
            r2, r2_s = q["resistance"][1]
            if "parallel" in low:
                rt = (r1 * r2) / (r1 + r2)
                return make_res(
                    "Resistors in Parallel (Unit 3: Electric Circuits)",
                    "1/R_T = 1/R₁ + 1/R₂  ⇒  R_T = (R₁ × R₂) / (R₁ + R₂)",
                    "combined parallel resistance = product ÷ sum",
                    [r1_s, r2_s],
                    [f"`R_T = ({format_clean_float(r1)} × {format_clean_float(r2)}) / ({format_clean_float(r1)} + {format_clean_float(r2)})`", f"`R_T = {format_clean_float(rt)} Ω`"],
                    rt,
                    "Ω",
                    "Sanity check: combined resistance in parallel is **always smaller** than the smallest individual resistor!",
                )
            if "series" in low:
                rt = r1 + r2
                return make_res(
                    "Resistors in Series (Unit 3: Electric Circuits)",
                    "R_T = R₁ + R₂",
                    "combined series resistance = sum of resistances",
                    [r1_s, r2_s],
                    [f"`R_T = {format_clean_float(r1)} + {format_clean_float(r2)}`", f"`R_T = {format_clean_float(rt)} Ω`"],
                    rt,
                    "Ω",
                    "In a series circuit, current is the same everywhere and resistances add directly.",
                )

    if any(k in low for k in ("ohm", "resistance", "voltage", "potential difference", "current")) and (
        bool(q["voltage"]) + bool(q["current"]) + bool(q["resistance"]) == 2
    ):
        if q["current"] and q["resistance"] and ("voltage" in low or "potential difference" in low or "p.d." in low):
            curr, i_str = q["current"][0]
            res, r_str = q["resistance"][0]
            v = curr * res
            return make_res(
                "Ohm's Law — Finding Voltage (Unit 3: Electricity)",
                "V = I R",
                "potential difference = current × resistance",
                [i_str, r_str],
                [f"`V = {format_clean_float(curr)} × {format_clean_float(res)}`", f"`V = {format_clean_float(v)} V`"],
                v,
                "V",
                "Convert `mA` to `A` (`÷ 1000`) and `kΩ` to `Ω` (`× 1000`) before multiplying.",
            )
        if q["voltage"] and q["resistance"] and "current" in low and len(q["resistance"]) == 1:
            v, v_str = q["voltage"][0]
            res, r_str = q["resistance"][0]
            curr = v / res
            return make_res(
                "Ohm's Law — Finding Current (Unit 3: Electricity)",
                "I = V / R",
                "current = potential difference ÷ resistance",
                [v_str, r_str],
                [f"`I = {format_clean_float(v)} / {format_clean_float(res)}`", f"`I = {format_clean_float(curr)} A`"],
                curr,
                "A",
                "Always state `I = V / R` (or `V = I R`) to secure the C1 formula mark.",
            )
        if q["voltage"] and q["current"] and "resistance" in low:
            v, v_str = q["voltage"][0]
            curr, i_str = q["current"][0]
            res = v / curr
            return make_res(
                "Ohm's Law — Finding Resistance (Unit 3: Electricity)",
                "R = V / I",
                "resistance = potential difference ÷ current",
                [v_str, i_str],
                [f"`R = {format_clean_float(v)} / {format_clean_float(curr)}`", f"`R = {format_clean_float(res)} Ω`"],
                res,
                "Ω",
                "Resistance is measured in **Ohms (Ω)**.",
            )

    # 19. Transformer Equation: Vp / Vs = Np / Ns
    if "transformer" in low and q["voltage"]:
        turns = [int(parse_num(m.group(1))) for m in re.finditer(r"(\d+)\s*turns", low)]
        if len(turns) == 2 and len(q["voltage"]) == 1:
            vp, vp_str = q["voltage"][0]
            np_t, ns_t = turns[0], turns[1]
            vs = vp * (ns_t / np_t)
            return make_res(
                "Ideal Transformer Equation (Unit 4: Electromagnetism)",
                "V_p / V_s = N_p / N_s",
                "primary voltage ÷ secondary voltage = primary turns ÷ secondary turns",
                [vp_str.replace("V =", "V_p ="), f"N_p = {np_t} turns", f"N_s = {ns_t} turns"],
                [f"`V_s = V_p × (N_s / N_p) = {format_clean_float(vp)} × ({ns_t} / {np_t})`", f"`V_s = {format_clean_float(vs)} V`"],
                vs,
                "V",
                "Step-down transformer: `N_s < N_p` so `V_s < V_p`. Step-up transformer: `N_s > N_p` so `V_s > V_p`.",
            )

    # 20. Distance from Constant Speed & Time: d = v t (when acceleration not asked)
    if ("distance" in low or "how far" in low) and q["speed"] and q["time"] and len(q["speed"]) == 1:
        v, v_str = q["speed"][0]
        t, t_str = q["time"][0]
        d = v * t
        return make_res(
            "Speed, Distance & Time (Unit 1: Kinematics)",
            "d = v × t",
            "distance travelled = average speed × time",
            [v_str, t_str],
            [f"`d = {format_clean_float(v)} × {format_clean_float(t)}`", f"`d = {format_clean_float(d)} m`"],
            d,
            "m",
            "Ensure speed (`m/s`) and time (`s`) use matching time units before multiplying!",
        )

    # 21. Momentum: p = m × v
    if "momentum" in low and q["mass"] and q["speed"] and "impulse" not in low:
        m, m_str = q["mass"][0]
        v, v_str = q["speed"][0]
        p = m * v
        return make_res(
            "Linear Momentum (Unit 1: Momentum & Impulse)",
            "p = m × v",
            "momentum = mass × velocity",
            [m_str, v_str],
            [f"`p = {format_clean_float(m)} × {format_clean_float(v)}`", f"`p = {format_clean_float(p)} kg m/s`"],
            p,
            "kg m/s",
            "Momentum is a **vector** quantity (unit: `kg m/s` or `N s`).",
        )

    # 22. Echo / Ultrasound Depth: d = (v × t) / 2
    if any(w in low for w in ("echo", "ultrasound", "sonar", "depth", "cliff", "seabed")) and q["speed"] and q["time"]:
        v, v_str = q["speed"][0]
        t, t_str = q["time"][0]
        d = (v * t) / 2.0
        return make_res(
            "Echo / Ultrasound Distance (Unit 6: Sound Waves)",
            "d = (v × t) / 2",
            "distance = (wave speed × echo time) ÷ 2",
            [v_str, t_str],
            [f"`d = ({format_clean_float(v)} × {format_clean_float(t)}) / 2`", f"`d = {format_clean_float(d)} m`"],
            d,
            "m",
            "Divide by **2** because the sound travels **there AND back**!",
        )

    # 23. Critical Angle from Refractive Index: sin c = 1 / n
    if "critical angle" in low and not q["mass"] and not q["speed"]:
        nm = re.search(r"refractive\s+index[^0-9]*([0-9]+(?:\.[0-9]+)?)", low)
        if nm:
            n_val = float(nm.group(1))
            if n_val > 1.0:
                c_deg = math.degrees(math.asin(1.0 / n_val))
                return make_res(
                    "Critical Angle & Total Internal Reflection (Unit 6: Light)",
                    "sin c = 1 / n",
                    "sine of critical angle = 1 ÷ refractive index",
                    [f"n = {format_clean_float(n_val)}"],
                    [f"`sin c = 1 / {format_clean_float(n_val)} = {format_clean_float(1.0 / n_val)}`", f"`c = arcsin({format_clean_float(1.0 / n_val)}) = {format_clean_float(c_deg)}°`"],
                    c_deg,
                    "°",
                    "Total internal reflection occurs when light travels from **denser to less dense** medium with angle of incidence **`i > c`**.",
                )

    # 24. Orbital Speed: v = 2 π r / T
    if ("orbital speed" in low or "orbit" in low) and q["length"] and q["time"]:
        r, r_str = q["length"][0]
        t_val, t_str = q["time"][0]
        if t_val > 0:
            v = (2.0 * math.pi * r) / t_val
            return make_res(
                "Orbital Speed (Unit 7: Space Physics)",
                "v = 2 π r / T",
                "orbital speed = circumference of orbit ÷ orbital period",
                [r_str.replace("d =", "r ="), t_str.replace("t =", "T =")],
                [f"`v = (2 × π × {format_clean_float(r)}) / {format_clean_float(t_val)}`", f"`v = {format_clean_float(v)} m/s`"],
                v,
                "m/s",
                "Convert orbital radius to **metres** and orbital period to **seconds** (e.g. `1 day = 86,400 s`).",
            )

    return None
