"use strict";

const fs = require("fs");
const path = require("path");
const { ROOT, getKeyPool, parseRequestBody } = require("./_lib/haroony_vercel");

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method === "OPTIONS") return res.status(204).end();

  const expectedPin = String(process.env.HAROONY_TEACHER_PIN || "0625").trim();
  const url = new URL(req.url, "http://localhost");
  const { fields } = req.method === "POST" ? await parseRequestBody(req) : { fields: {} };
  const pin = String(fields.pin || url.searchParams.get("pin") || req.headers["x-admin-pin"] || "").trim();

  if (!pin || pin !== expectedPin) {
    return res.status(401).json({ ok: false, detail: "Invalid Teacher/Admin PIN." });
  }

  const gemKey = getKeyPool("GEMINI_API_KEYS", "GEMINI_API_KEY")[0] || "";
  const oaiKey = getKeyPool("OPENAI_API_KEYS", "OPENAI_API_KEY")[0] || "";

  // GET: check status of existing fine-tuning jobs
  if (req.method === "GET") {
    const out = { ok: true, gemini: null, openai: null };
    const gemOp = url.searchParams.get("gem_model") || "";
    const oaiJob = url.searchParams.get("oai_job") || "";
    if (gemOp && gemKey) {
      const p = gemOp.startsWith("tunedModels/") ? gemOp : `tunedModels/${gemOp}`;
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/${p}?key=${encodeURIComponent(gemKey)}`);
      out.gemini = await r.json();
    }
    if (oaiJob && oaiKey) {
      const r = await fetch(`https://api.openai.com/v1/fine_tuning/jobs/${encodeURIComponent(oaiJob)}`, {
        headers: { Authorization: `Bearer ${oaiKey}` },
      });
      out.openai = await r.json();
    }
    return res.status(200).json(out);
  }

  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, detail: "Use POST to launch training or GET to check status." });
  }

  const results = { ok: true, gemini: { launched: false }, openai: { launched: false } };

  // 1. Launch Gemini Fine-Tuning on Harony materials
  if (gemKey) {
    try {
      const dsPath = path.join(ROOT, "data", "training", "gemini_sft_dataset.jsonl");
      const lines = fs.readFileSync(dsPath, "utf8").split("\n").filter(Boolean).slice(0, 1200);
      const examples = [];
      for (const ln of lines) {
        const obj = JSON.parse(ln);
        if (obj.contents && obj.contents.length === 2) {
          examples.push({
            text_input: String(obj.contents[0].parts[0].text || "").slice(0, 1800),
            output: String(obj.contents[1].parts[0].text || "").slice(0, 1800),
          });
        }
      }
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/tunedModels?key=${encodeURIComponent(gemKey)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: "Haroony-Physics-AI-2027",
          base_model: "models/gemini-1.5-flash-001-tuning",
          tuning_task: {
            hyperparameters: { batch_size: 4, learning_rate: 0.001, epoch_count: 3 },
            training_data: { examples: { examples } },
          },
        }),
      });
      const body = await r.json();
      results.gemini = { launched: r.ok, status: r.status, response: body };
    } catch (e) {
      results.gemini = { launched: false, error: String(e.message || e) };
    }
  } else {
    results.gemini = { launched: false, reason: "Missing GEMINI_API_KEYS env var" };
  }

  // 2. Launch OpenAI Fine-Tuning on Harony materials
  if (oaiKey) {
    try {
      const dsPath = path.join(ROOT, "data", "training", "openai_sft_dataset.jsonl");
      const buf = fs.readFileSync(dsPath);
      const fd = new FormData();
      fd.append("purpose", "fine-tune");
      fd.append("file", new Blob([buf], { type: "application/jsonl" }), "openai_sft_dataset.jsonl");

      const upResp = await fetch("https://api.openai.com/v1/files", {
        method: "POST",
        headers: { Authorization: `Bearer ${oaiKey}` },
        body: fd,
      });
      const upData = await upResp.json();
      if (!upResp.ok || !upData.id) {
        results.openai = { launched: false, stage: "upload", status: upResp.status, response: upData };
      } else {
        const jobResp = await fetch("https://api.openai.com/v1/fine_tuning/jobs", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${oaiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            training_file: upData.id,
            model: "gpt-4o-mini-2024-07-18",
            suffix: "haroony-physics-2027",
            hyperparameters: { n_epochs: 3 },
          }),
        });
        const jobData = await jobResp.json();
        results.openai = { launched: jobResp.ok, status: jobResp.status, file_id: upData.id, job: jobData };
      }
    } catch (e) {
      results.openai = { launched: false, error: String(e.message || e) };
    }
  } else {
    results.openai = { launched: false, reason: "Missing OPENAI_API_KEYS env var" };
  }

  return res.status(200).json(results);
};
