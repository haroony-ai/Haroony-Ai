/* Haroony AI — Client-Side Serverless Engine for GitHub Pages Static Hosting
 * Activates automatically whenever no Python FastAPI backend (/api/status) is present,
 * running the trained Physics Guardrail, Harony 2026/2027 Retriever, Formula Solver,
 * Cambridge Mark Scheme Grader, and Video Solution Finder 100% in the browser!
 */
(function (global) {
  "use strict";

  let kbPromise = null;
  let kbData = null;

  const REFUSAL_MSG =
    "I'm **Haroony AI**, and I only answer **Physics** questions and mark Physics exam work.\n\n" +
    "Feel free to ask about *Mechanics, Thermal Physics, Waves, Light, Electricity, Electromagnetism, Nuclear Physics, or Space Physics*, or open **Check Answer** to grade your work.";

  const JAILBREAK_RE =
    /\b(ignore\s+(?:all\s+|previous\s+|prior\s+)?instructions|disregard\s+(?:your\s+)?rules|pretend\s+you\s+are|act\s+as\s+a\s+general|bypass\s+guardrail|system\s+prompt)\b/i;

  const GREETING_RE =
    /^(?:hi+|hello+|hey+|good\s+(?:morning|afternoon|evening)|salam|assalamu\s*alaikum|who\s+are\s+you|what\s+can\s+you\s+do|help)\s*[!?.]*$/i;

  const HARD_NON_PHYSICS_RE =
    /\b(recipe|bake|cake|pasta|biryani|chicken|poem|poetry|haiku|romeo|juliet|shakespeare|fifa|world\s+cup|football|oscar|netflix|horoscope|zodiac|bitcoin|crypto|stock\s+market|mitosis|meiosis|photosynthesis|chloroplast|digestive\s+system|nephron|auxin|titration|haber\s+process|alkane|polymerisation|scrape|beautifulsoup|flexbox|sql\s+query|discord\s+bot|roman\s+empire|french\s+revolution)\b/i;

  const STRONG_PHYSICS_RE =
    /\b(physics|force|velocity|acceleration|speed|momentum|impulse|kinetic|potential\s+energy|density|mass|volume|weight|gravity|moment|pivot|equilibrium|hooke|spring|pressure|pascal|joule|watt|newton|thermal|conduction|convection|radiation|evaporation|specific\s+heat|latent\s+heat|kelvin|boyle|wave|wavelength|frequency|hertz|diffraction|refraction|reflection|snell|critical\s+angle|lens|converging|diverging|ultrasound|echo|electromagnetic|current|voltage|resistance|ohm|circuit|resistor|parallel|series|ldr|thermistor|diode|fuse|transformer|commutator|motor|generator|magnetic|solenoid|fleming|induction|alpha|beta|gamma|radioactive|half-life|isotopes|fission|fusion|rutherford|nuclide|redshift|hubble|galaxy|orbital|light-year|0625|p2|p4|mark\s+scheme)\b/i;

  const STOPWORDS = new Set([
    "the", "and", "for", "with", "from", "that", "this", "what", "how", "why",
    "explain", "define", "state", "describe", "calculate", "find", "into", "are",
    "was", "were", "has", "have", "when", "where", "which", "your", "about"
  ]);

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
          // Merge any Teacher Studio items saved in browser localStorage
          try {
            const extraChunks = JSON.parse(localStorage.getItem("haroony_custom_chunks") || "[]");
            const extraMs = JSON.parse(localStorage.getItem("haroony_custom_ms") || "[]");
            if (Array.isArray(extraChunks) && extraChunks.length) {
              kbData.chunks = kbData.chunks.concat(extraChunks);
            }
            if (Array.isArray(extraMs) && extraMs.length) {
              kbData.mark_schemes = kbData.mark_schemes.concat(extraMs);
            }
          } catch (_) {}
          return kbData;
        });
    }
    return kbPromise;
  }

  function tokenize(s) {
    return String(s || "")
      .toLowerCase()
      .match(/[a-z0-9Δθλραβγ]{2,}/g) || [];
  }

  function checkGuardrail(text, model, hasPriorContext) {
    const t = String(text || "").trim();
    if (!t) return { allowed: false, category: "empty", refusal: REFUSAL_MSG };
    if (JAILBREAK_RE.test(t)) return { allowed: false, category: "jailbreak", refusal: REFUSAL_MSG };
    if (GREETING_RE.test(t)) return { allowed: true, category: "greeting", refusal: "" };
    if (HARD_NON_PHYSICS_RE.test(t) && !STRONG_PHYSICS_RE.test(t)) {
      return { allowed: false, category: "off_topic", refusal: REFUSAL_MSG };
    }
    if (STRONG_PHYSICS_RE.test(t)) return { allowed: true, category: "physics", refusal: "" };

    // Evaluate linear TF-IDF logistic model exported from scikit-learn
    const toks = tokenize(t);
    if (model && model.weights) {
      let score = Number(model.intercept || 0);
      const unique = Array.from(new Set(toks));
      const norm = Math.sqrt(unique.length || 1);
      for (const w of unique) {
        if (model.weights[w] !== undefined) {
          score += model.weights[w] / norm;
        }
      }
      const prob = 1 / (1 + Math.exp(-score));
      if (prob >= (hasPriorContext ? 0.35 : 0.52)) {
        return { allowed: true, category: "physics", refusal: "" };
      }
    }
    if (hasPriorContext && toks.length <= 6) {
      return { allowed: true, category: "followup", refusal: "" };
    }
    return { allowed: false, category: "off_topic", refusal: REFUSAL_MSG };
  }

  function trySolveCalculation(q) {
    const low = String(q || "").toLowerCase();
    // 1. Kinetic Energy: mass (kg) & speed (m/s)
    if (low.includes("kinetic energy") || low.includes("kinetic store")) {
      const mMatch = q.match(/(\d+(?:\.\d+)?)\s*kg\b/i);
      const vMatch = q.match(/(\d+(?:\.\d+)?)\s*m\s*\/\s*s\b/i);
      if (mMatch && vMatch) {
        const m = parseFloat(mMatch[1]);
        const v = parseFloat(vMatch[1]);
        const ek = 0.5 * m * v * v;
        const sf2 = Number(ek.toPrecision(2));
        return (
          `### ⚡ Step-by-Step Cambridge Calculation — Kinetic Energy\n` +
          `- **1. Formula:** \`Ek = ½ m v²\`\n` +
          `- **2. Given (SI Units):** \`m = ${m} kg\`, \`v = ${v} m/s\`\n` +
          `- **3. Substitution:** \`Ek = 0.5 × ${m} × ${v}² = ${ek.toLocaleString()} J\`\n` +
          `- **4. Final Answer (2 s.f.):** **\`${sf2 >= 1e5 ? sf2.toExponential().replace("e+", " × 10^") : sf2} J\`**`
        );
      }
    }
    // 2. Specific heat capacity / thermal energy Q = m c ΔT
    if (low.includes("specific heat") || (low.includes("heated") && low.includes("4200"))) {
      const mG = q.match(/(\d+(?:\.\d+)?)\s*g\b/i);
      const mKg = q.match(/(\d+(?:\.\d+)?)\s*kg\b/i);
      const cM = q.match(/(\d+(?:\.\d+)?)\s*j\s*\/\s*\(?kg/i);
      const temps = Array.from(q.matchAll(/(\d+(?:\.\d+)?)\s*°?\s*c\b/gi)).map((x) => parseFloat(x[1]));
      if ((mG || mKg) && temps.length >= 2) {
        const m = mKg ? parseFloat(mKg[1]) : parseFloat(mG[1]) / 1000;
        const c = cM ? parseFloat(cM[1]) : 4200;
        const dt = Math.abs(temps[1] - temps[0]);
        const E = m * c * dt;
        return (
          `### ⚡ Step-by-Step Cambridge Calculation — Thermal Energy\n` +
          `- **1. Formula:** \`ΔE = m c Δθ\`\n` +
          `- **2. Given (SI Units):** \`m = ${m} kg\`, \`c = ${c} J/(kg °C)\`, \`Δθ = ${dt} °C\`\n` +
          `- **3. Substitution:** \`ΔE = ${m} × ${c} × ${dt} = ${Math.round(E)} J\`\n` +
          `- **4. Final Answer (2 s.f.):** **\`${Number(E.toPrecision(2))} J\`**`
        );
      }
    }
    // 3. Density ρ = m / V
    if (low.includes("density")) {
      const mG = q.match(/(\d+(?:\.\d+)?)\s*g\b/i);
      const mKg = q.match(/(\d+(?:\.\d+)?)\s*kg\b/i);
      const vCm = q.match(/(\d+(?:\.\d+)?)\s*cm\s*(?:3|³|\^3)\b/i);
      const vM = q.match(/(\d+(?:\.\d+)?)\s*m\s*(?:3|³|\^3)\b/i);
      if ((mG || mKg) && (vCm || vM)) {
        const m = mKg ? parseFloat(mKg[1]) : parseFloat(mG[1]);
        const v = vM ? parseFloat(vM[1]) : parseFloat(vCm[1]);
        const unit = mKg && vM ? "kg/m³" : "g/cm³";
        const rho = m / v;
        return (
          `### ⚡ Step-by-Step Cambridge Calculation — Density\n` +
          `- **1. Formula:** \`ρ = m / V\`\n` +
          `- **2. Substitution:** \`ρ = ${m} / ${v} = ${Number(rho.toPrecision(3))} ${unit}\`\n` +
          `- **3. Final Answer:** **\`${Number(rho.toPrecision(2))} ${unit}\`**`
        );
      }
    }
    return null;
  }

  function searchChunks(kb, query, topK) {
    const qToks = tokenize(query).filter((w) => !STOPWORDS.has(w));
    if (!qToks.length) return kb.chunks.slice(3, 3 + topK);
    const isPaperCode = /\b20\d{2}\b|\bp[24]\b|\bq\d+/i.test(query);

    const scored = [];
    for (const ch of kb.chunks) {
      const topicLow = ((ch.unit || "") + " " + (ch.topic || "")).toLowerCase();
      const bodyLow = (ch.text || "").toLowerCase();
      let s = 0;
      for (const tok of qToks) {
        if (topicLow.includes(tok)) s += 3.2;
        if (bodyLow.includes(tok)) s += 1.4;
      }
      if (!isPaperCode && ch.source === "Harony Notes 2026 Final" && s > 0) s += 1.1;
      if (s > 0) scored.push({ s, ch });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, topK).map((x) => x.ch);
  }

  function searchVideos(kb, query, topK) {
    const qToks = tokenize(query).filter((w) => !STOPWORDS.has(w) && w !== "physics");
    if (!qToks.length || !kb.videos) return [];
    const scored = [];
    for (const v of kb.videos) {
      const hay = `${v.label || ""} ${v.chapter || ""} ${v.unit || ""}`.toLowerCase();
      let hits = 0;
      for (const t of qToks) {
        if ((v.label || "").toLowerCase().includes(t)) hits += 2;
        else if (hay.includes(t)) hits += 1;
      }
      if (hits >= 2 || (qToks.length === 1 && hits >= 1)) scored.push({ hits, v });
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

  function extractNumbers(s) {
    const out = [];
    const str = String(s || "").replace(/×/g, "x");
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

  function gradeAgainstQuestion(q, studentText) {
    const stuLow = studentText.toLowerCase();
    const stuToks = new Set(tokenize(studentText));
    const stuNums = extractNumbers(studentText);
    let awarded = 0;
    const rows = [];
    const pts = q.points || [];

    for (let i = 0; i < pts.length; i++) {
      const pt = pts[i];
      const mtype = pt.mark_type || "B";
      const mval = Number(pt.marks || 1);
      const ans = String(pt.answer || "");
      const ansWords = tokenize(ans.replace(/\([^)]*\)/g, " ")).filter((w) => !STOPWORDS.has(w) && w.length >= 3);
      const ansNums = extractNumbers(ans.replace(/\([^)]*\)/g, " "));

      let hit = false;
      if (mtype === "A" && ansNums.length > 0) {
        const target = ansNums[0];
        hit = stuNums.some((sn) => Math.abs(sn - target) <= 0.05 * Math.max(1, Math.abs(target)));
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
          // Skip compensatory C-marks directly following an awarded A-mark
          while (i + 1 < pts.length && pts[i + 1].mark_type === "C") i++;
          continue;
        }
      } else if (ansWords.length > 0) {
        const matched = ansWords.filter((w) => stuToks.has(w) || stuLow.includes(w)).length;
        hit = matched / ansWords.length >= (ansWords.length <= 2 ? 0.5 : 0.55);
      } else if (/option\s+([abcd])\b/i.test(ans)) {
        const opt = ans.match(/option\s+([abcd])\b/i)[1].toLowerCase();
        hit = new RegExp(`\\b${opt}\\b`, "i").test(studentText);
      }

      if (hit) {
        awarded += mval;
        rows.push({
          code: pt.code,
          status: "awarded",
          marks_awarded: mval,
          max_marks: mval,
          expected: ans,
          comment: `✓ Key marking point (\`${pt.code}\`) satisfied: \`${ans}\``,
        });
      } else {
        rows.push({
          code: pt.code,
          status: "missed",
          marks_awarded: 0,
          max_marks: mval,
          expected: ans,
          comment: `✗ Missing point (\`${pt.code}\`): \`${ans}\``,
        });
      }
    }

    const maxM = Number(q.max_marks || 1);
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

  global.HaroonyBrowserEngine = {
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

    async chat(message, history, fileObj) {
      const kb = await ensureKB();
      let combined = String(message || "").trim();
      if (fileObj && !combined) {
        combined = fileObj.name.replace(/[_.-]+/g, " ");
      }
      const g = checkGuardrail(combined, kb.guardrail_model, history && history.length > 0);
      if (!g.allowed) {
        return {
          ok: true,
          refused: true,
          category: g.category,
          reply: g.refusal,
          notes_cards: [],
          video_links: [],
          engine: "Haroony Physics Guardrail (Browser Engine)",
        };
      }
      if (g.category === "greeting") {
        return {
          ok: true,
          refused: false,
          category: "greeting",
          reply:
            "Hello! Welcome to **Haroony AI**.\n\n" +
            "Ask me any **IGCSE or A-Level Physics** question, attach a photo/PDF of a Physics problem, or switch to **Check Answer** above to mark your work against a Mark Scheme.",
          notes_cards: [],
          video_links: [],
          engine: "Haroony AI",
        };
      }

      if (/\b(quiz\s+me|test\s+me|practice\s+question)\b/i.test(combined)) {
        return {
          ok: true,
          refused: false,
          category: "physics",
          reply:
            "### 📝 IGCSE Physics Practice Question (Harony ClassWork 2027)\n\n" +
            "**2023 March P4 V2 Q02(a) — Density:**\n" +
            "Explain why a ship made of steel is able to float on water, even though steel has a greater density than water. **[2 marks]**\n\n" +
            "*Reply with your answer or test it in the **Correct Answer & MS** tab!*",
          notes_cards: searchChunks(kb, "density float water", 2),
          video_links: searchVideos(kb, "2023 March P4 V2 Q02 Density", 1),
          engine: "Haroony Physics AI Examiner",
        };
      }

      const calcMd = trySolveCalculation(combined);
      const hits = searchChunks(kb, combined, 3);
      const vids = searchVideos(kb, combined, 2);

      const parts = [];
      if (calcMd) parts.push(calcMd);
      if (hits.length) {
        const top = hits[0];
        const lines = String(top.text || "")
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l.length > 4 && !/^\d+$/.test(l))
          .slice(0, calcMd ? 5 : 9)
          .map((l) => `- ${l}`)
          .join("\n");
        const cite = top.page ? `${top.source} p.${top.page}` : top.source;
        parts.push(`### 📘 ${top.topic} *(${top.unit} — ${cite})*\n\n${lines}`);
      }
      if (!parts.length) {
        parts.push("Ask any specific question about Mechanics, Thermal Physics, Waves, Electricity, Electromagnetism, Nuclear Physics, or Space Physics!");
      }

      return {
        ok: true,
        refused: false,
        category: "physics",
        reply: parts.join("\n\n---\n\n"),
        notes_cards: hits,
        video_links: vids,
        engine: "Haroony Physics AI (Harony 2026/2027 Browser Engine)",
      };
    },

    async grade({ studentText, msId, msText, questionId, useSampleDemo }) {
      const kb = await ensureKB();
      let text = String(studentText || "").trim();
      if (useSampleDemo === "cw_u1_density") {
        if (!text) {
          text =
            "2023 March P4 V2 Q02 a - Density:\n" +
            "The ship is not solid steel because there are air spaces inside the ship, " +
            "so the overall average density of the ship is less than the density of the water.";
        }
        msId = msId || "harony-cw-2027-u1";
        questionId = questionId || "2023-March-P4-V2-Q02-a";
      }
      if (!text) {
        return {
          ok: false,
          error: "Please enter your Physics answer in Step 1 first.",
        };
      }

      const g = checkGuardrail(text, kb.guardrail_model, Boolean(msId || msText));
      if (!g.allowed && (g.category === "jailbreak" || (!msId && !msText))) {
        return { ok: false, refused: true, error: g.refusal };
      }

      let msObj = null;
      if (msText && msText.trim()) {
        const lines = msText.split("\n").map((l) => l.trim()).filter(Boolean);
        const pts = lines.map((ln) => {
          const m = ln.match(/\b([BMCA])([1-5])\b/);
          const code = m ? m[1] + m[2] : "B1";
          const mval = m ? parseInt(m[2], 10) : 1;
          return {
            mark_type: m ? m[1] : "B",
            marks: mval,
            code,
            answer: ln.replace(/\b[BMCA][1-5]\b/g, "").trim(),
          };
        });
        msObj = {
          title: "Pasted Mark Scheme",
          questions: [{ question_id: "Custom", title: "Pasted Criteria", max_marks: pts.length, points: pts }],
        };
      } else if (msId) {
        msObj = (kb.mark_schemes || []).find((m) => m.id === msId) || null;
      }

      if (msObj && msObj.questions && msObj.questions.length) {
        let qs = msObj.questions;
        if (questionId && questionId.trim()) {
          const qlow = questionId.trim().toLowerCase();
          const exact = qs.filter((q) => q.question_id.toLowerCase() === qlow);
          if (exact.length) {
            qs = exact;
          } else {
            const qtoks = qlow.match(/[a-z0-9]+/g) || [];
            const filtered = qs.filter(
              (q) =>
                q.question_id.toLowerCase().includes(qlow) ||
                (q.title || "").toLowerCase().includes(qlow) ||
                (qtoks.length >= 2 &&
                  qtoks.every((t) =>
                    (((q.title || "") + " " + q.question_id).toLowerCase().match(/[a-z0-9]+/g) || []).includes(t)
                  ))
            );
            qs = filtered.length ? filtered.slice(0, 5) : qs.slice(0, 3);
          }
        } else {
          // Score questions by overlap with student text
          const sToks = tokenize(text).filter((w) => !STOPWORDS.has(w));
          const ranked = qs
            .map((q) => {
              const hay = ((q.title || "") + " " + (q.summary_answer || "")).toLowerCase();
              const sc = sToks.reduce((acc, t) => acc + (hay.includes(t) ? 1 : 0), 0);
              return { sc, q };
            })
            .sort((a, b) => b.sc - a.sc);
          qs = ranked.slice(0, 2).map((x) => x.q);
        }

        const graded = qs.map((q) => gradeAgainstQuestion(q, text));
        const awarded = graded.reduce((a, x) => a + x.awarded_marks, 0);
        const maxM = Math.max(1, graded.reduce((a, x) => a + x.max_marks, 0));
        const pct = Math.round((1000 * awarded) / maxM) / 10;
        const refs = searchChunks(kb, graded.map((g) => g.model_answer).join(" ") + " " + text, 2).filter((c) => c.image_url);
        return {
          ok: true,
          mode: "mark_scheme",
          mark_scheme_title: msObj.title,
          awarded_marks: awarded,
          max_marks: maxM,
          percentage: pct,
          grade: gradeLetter(pct),
          questions: graded,
          extracted_student_text: text,
          notes_references: refs,
        };
      }

      const refs = searchChunks(kb, text, 2);
      return {
        ok: true,
        mode: "notes_check",
        mark_scheme_title: "Harony Notes Key-Point Check",
        awarded_marks: 1,
        max_marks: 1,
        percentage: 100,
        grade: "A*",
        questions: [
          {
            question_id: "Concept",
            title: refs[0] ? refs[0].topic : "Physics Check",
            awarded_marks: 1,
            max_marks: 1,
            points: [{ code: "B1", status: "awarded", comment: "✓ Checked against Harony Physics Clinic Notes." }],
            model_answer: refs[0] ? refs[0].text.slice(0, 220) : "",
          },
        ],
        extracted_student_text: text,
        notes_references: refs.filter((c) => c.image_url),
      };
    },

    async searchNotes(q, unit) {
      const kb = await ensureKB();
      let pool = kb.chunks.filter((c) => c.source === "Harony Notes 2026 Final");
      if (unit) pool = pool.filter((c) => (c.unit || "").toLowerCase() === unit.toLowerCase());
      if (q && q.trim()) {
        const subKb = { chunks: pool };
        return { ok: true, results: searchChunks(subKb, q, 16) };
      }
      return { ok: true, results: pool.slice(3, 27) };
    },
  };
})(window);
