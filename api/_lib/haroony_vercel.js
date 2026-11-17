"use strict";

const fs = require("fs");
const path = require("path");
const Busboy = require("busboy");

const ROOT = path.resolve(__dirname, "..", "..");
let _kbCache = null;
let _qaCache = null;
let _engineLoaded = false;

const HAROONY_SYSTEM_PROMPT =
  "You are Haroony AI, the official Cambridge IGCSE (0625) & A-Level (9702) Physics AI Assistant " +
  "for Mr. Haroony's Harony Physics Clinic. Never mention third-party model names or API providers.\n\n" +
  "STRICT RULES:\n" +
  "1. PHYSICS ONLY: Answer ONLY Physics questions. Politely refuse non-physics requests.\n" +
  "2. REASON FROM FIRST PRINCIPLES FOR ANY QUESTION: Answer the student's EXACT question directly and accurately, " +
  "even if it is a novel everyday scenario or 'why/what happens if' question not word-for-word in the context. " +
  "NEVER force an unrelated snippet into your reply, and NEVER repeat an earlier turn's answer when switching topics.\n" +
  "3. ALIGN WITH HARONY NOTES & CAMBRIDGE 0625 MARK SCHEMES: Use exact definitions, formulas, and bold key marking phrases.\n" +
  "4. 4-STEP CALCULATION FORMAT: Formula -> Given & SI Conversions -> Substitution -> Final Answer with Units (2-3 s.f.).\n" +
  "5. IMAGE ACCURACY: When an image is attached, read every label, graph axis, circuit symbol, and number carefully.\n";

function loadKB() {
  if (!_kbCache) {
    _kbCache = JSON.parse(fs.readFileSync(path.join(ROOT, "static", "browser_kb.json"), "utf8"));
  }
  return _kbCache;
}

function loadQA() {
  if (!_qaCache) {
    _qaCache = JSON.parse(fs.readFileSync(path.join(ROOT, "static", "browser_qa.json"), "utf8"));
  }
  return _qaCache;
}

const NATIVE_FETCH = globalThis.fetch.bind(globalThis);

function getEngine() {
  if (!globalThis.localStorage) {
    globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  }
  if (!globalThis.__haroonyFetchWrapped) {
    globalThis.__haroonyFetchWrapped = true;
    globalThis.fetch = async (url, opts) => {
      const s = String(url || "");
      if (s.includes("browser_qa.json")) {
        return { ok: true, json: async () => loadQA() };
      }
      if (s.includes("browser_kb.json")) {
        return { ok: true, json: async () => loadKB() };
      }
      return NATIVE_FETCH(url, opts);
    };
  }
  if (!_engineLoaded) {
    require(path.join(ROOT, "static", "browser_engine.js"));
    _engineLoaded = true;
  }
  return globalThis.HaroonyBrowserEngine;
}

function getKeyPool(...names) {
  const keys = [];
  const seen = new Set();
  for (const n of names) {
    const raw = (process.env[n] || "").trim();
    if (!raw) continue;
    for (const tok of raw.split(/[,;\s\n]+/)) {
      const k = tok.trim();
      if (k && !seen.has(k)) {
        seen.add(k);
        keys.push(k);
      }
    }
  }
  return keys;
}

function sanitizeProviderMentions(text) {
  if (!text) return text;
  return String(text).replace(
    /\b(?:Gemini|OpenAI|ChatGPT|GPT-4o?|GPT-3\.5|Claude|Anthropic|Google\s+AI)\b/gi,
    "Haroony AI"
  );
}

function scoreCandidate(ans, ctx) {
  if (!ans) return -1;
  let score = Math.min(ans.length, 1400) / 400;
  if (ans.includes("**")) score += 1.0;
  if (ans.includes("`")) score += 0.5;
  const m = String(ctx || "").match(/Final Answer\*?\*?:?\s*`?([0-9.+\-eE×^ ]+)/);
  if (m) {
    const nm = m[1].match(/[-+]?\d+(?:\.\d+)?/);
    if (nm && ans.includes(nm[0])) score += 3.0;
  }
  return score;
}

async function callGemini(messages, imageBuffer, imageMime, maxTokens = 1100) {
  const keys = getKeyPool("GEMINI_API_KEYS", "GEMINI_API_KEY");
  if (!keys.length) return null;
  const primary = (process.env.GEMINI_MODEL || "gemini-2.5-flash").trim() || "gemini-2.5-flash";
  const models = [...new Set([primary, "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"])];

  const systemParts = [];
  const contents = [];
  for (const m of messages) {
    if (m.role === "system") {
      systemParts.push({ text: m.content });
    } else {
      contents.push({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      });
    }
  }
  if (imageBuffer && contents.length) {
    contents[contents.length - 1].parts.push({
      inline_data: {
        mime_type: imageMime || "image/jpeg",
        data: imageBuffer.toString("base64"),
      },
    });
  }

  const payload = {
    contents,
    generationConfig: { temperature: 0.2, maxOutputTokens: maxTokens },
  };
  if (systemParts.length) payload.systemInstruction = { parts: systemParts };

  for (const apiKey of keys) {
    for (const model of models) {
      const modelPath = model.startsWith("models/") || model.startsWith("tunedModels/") ? model : `models/${model}`;
      const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:generateContent?key=${encodeURIComponent(apiKey)}`;
      try {
        const resp = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) continue;
        const data = await resp.json();
        const parts = (((data.candidates || [])[0] || {}).content || {}).parts || [];
        const txt = parts.map((p) => p.text || "").join("\n").trim();
        if (txt) return txt;
      } catch (_) {}
    }
  }
  return null;
}

async function callOpenAI(messages, imageBuffer, imageMime, maxTokens = 1100) {
  const keys = getKeyPool("OPENAI_API_KEYS", "OPENAI_API_KEY");
  if (!keys.length) return null;
  const primary = (process.env.OPENAI_MODEL || "gpt-4o-mini").trim() || "gpt-4o-mini";
  const models = [...new Set([primary, "gpt-4o-mini", "gpt-4o"])];

  const oaiMsgs = messages.map((m, idx) => {
    if (idx === messages.length - 1 && imageBuffer && m.role === "user") {
      return {
        role: "user",
        content: [
          { type: "text", text: m.content },
          { type: "image_url", image_url: { url: `data:${imageMime || "image/jpeg"};base64,${imageBuffer.toString("base64")}` } },
        ],
      };
    }
    return { role: m.role, content: m.content };
  });

  for (const apiKey of keys) {
    for (const model of models) {
      try {
        const resp = await fetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model,
            messages: oaiMsgs,
            temperature: 0.2,
            max_tokens: maxTokens,
          }),
        });
        if (!resp.ok) continue;
        const data = await resp.json();
        const txt = (((data.choices || [])[0] || {}).message || {}).content;
        if (txt && txt.trim()) return txt.trim();
      } catch (_) {}
    }
  }
  return null;
}

async function runDualLLM(userPrompt, contextBlock, history, imageBuffer, imageMime) {
  const messages = [{ role: "system", content: HAROONY_SYSTEM_PROMPT }];
  if (Array.isArray(history)) {
    for (const h of history.slice(-6)) {
      if ((h.role === "user" || h.role === "assistant") && h.content) {
        messages.push({ role: h.role, content: String(h.content).slice(0, 1400) });
      }
    }
  }
  messages.push({
    role: "user",
    content:
      `HARONY PHYSICS REFERENCE CONTEXT:\n${contextBlock || "Use Cambridge 0625 Physics principles."}\n\n` +
      `STUDENT QUESTION:\n${userPrompt || "Solve and explain the Physics problem shown in the attached image."}\n\n` +
      `Respond clearly in Markdown grounded in Physics & the Harony context above. ` +
      `If the context does not cover an everyday Physics question, answer cleanly from first principles.`,
  });

  const hasGem = getKeyPool("GEMINI_API_KEYS", "GEMINI_API_KEY").length > 0;
  const hasOai = getKeyPool("OPENAI_API_KEYS", "OPENAI_API_KEY").length > 0;
  const mode = (process.env.HAROONY_LLM_MODE || "dual").trim().toLowerCase();

  if (hasGem && hasOai && mode === "dual") {
    const [gRes, oRes] = await Promise.all([
      callGemini(messages, imageBuffer, imageMime),
      callOpenAI(messages, imageBuffer, imageMime),
    ]);
    if (gRes && oRes) {
      const best = scoreCandidate(gRes, contextBlock) >= scoreCandidate(oRes, contextBlock) ? gRes : oRes;
      return sanitizeProviderMentions(best);
    }
    if (gRes) return sanitizeProviderMentions(gRes);
    if (oRes) return sanitizeProviderMentions(oRes);
    return null;
  }
  if (mode === "openai_first") {
    const r1 = await callOpenAI(messages, imageBuffer, imageMime);
    if (r1) return sanitizeProviderMentions(r1);
    const r2 = await callGemini(messages, imageBuffer, imageMime);
    return r2 ? sanitizeProviderMentions(r2) : null;
  }
  const r1 = await callGemini(messages, imageBuffer, imageMime);
  if (r1) return sanitizeProviderMentions(r1);
  const r2 = await callOpenAI(messages, imageBuffer, imageMime);
  return r2 ? sanitizeProviderMentions(r2) : null;
}

async function runVisionOCR(imageBuffer, imageMime) {
  if (!imageBuffer) return null;
  const msgs = [{ role: "user", content: "Transcribe ALL text, numbers, units, and diagram/graph labels from this Physics problem image cleanly." }];
  return (await callGemini(msgs, imageBuffer, imageMime, 700)) || (await callOpenAI(msgs, imageBuffer, imageMime, 700));
}

function parseRequestBody(req) {
  return new Promise((resolve, reject) => {
    const ct = String(req.headers["content-type"] || "");
    if (ct.includes("multipart/form-data")) {
      const fields = {};
      let file = null;
      const bb = Busboy({ headers: req.headers, limits: { fileSize: 8 * 1024 * 1024 } });
      bb.on("field", (name, val) => {
        fields[name] = val;
      });
      bb.on("file", (name, stream, info) => {
        const chunks = [];
        stream.on("data", (d) => chunks.push(d));
        stream.on("end", () => {
          file = {
            fieldname: name,
            filename: info.filename || "upload.jpg",
            mimeType: info.mimeType || "image/jpeg",
            buffer: Buffer.concat(chunks),
          };
        });
      });
      bb.on("finish", () => resolve({ fields, file }));
      bb.on("error", reject);
      req.pipe(bb);
      return;
    }

    const raw = [];
    req.on("data", (c) => raw.push(Buffer.isBuffer(c) ? c : Buffer.from(String(c))));
    req.on("end", () => {
      const bodyStr = Buffer.concat(raw).toString("utf8");
      if (!bodyStr) return resolve({ fields: {}, file: null });
      if (ct.includes("application/json")) {
        try {
          return resolve({ fields: JSON.parse(bodyStr), file: null });
        } catch (_) {
          return resolve({ fields: {}, file: null });
        }
      }
      const params = new URLSearchParams(bodyStr);
      const fields = {};
      for (const [k, v] of params.entries()) fields[k] = v;
      resolve({ fields, file: null });
    });
    req.on("error", reject);
  });
}

function hasActiveLLM() {
  return (
    getKeyPool("GEMINI_API_KEYS", "GEMINI_API_KEY").length > 0 ||
    getKeyPool("OPENAI_API_KEYS", "OPENAI_API_KEY").length > 0
  );
}

module.exports = {
  ROOT,
  loadKB,
  loadQA,
  getEngine,
  getKeyPool,
  runDualLLM,
  runVisionOCR,
  parseRequestBody,
  hasActiveLLM,
};
