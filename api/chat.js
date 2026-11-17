"use strict";

const {
  getEngine,
  parseRequestBody,
  runDualLLM,
  runVisionOCR,
  hasActiveLLM,
} = require("./_lib/haroony_vercel");

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, detail: "Method not allowed" });
  }

  try {
    const { fields, file } = await parseRequestBody(req);
    const message = String(fields.message || "").trim();
    let history = [];
    try {
      history = fields.history ? JSON.parse(fields.history) : [];
    } catch (_) {
      history = [];
    }

    const hasImage = Boolean(file && file.buffer && file.buffer.length > 0 && String(file.mimeType || "").startsWith("image/"));
    let ocrText = "";
    if (hasImage && hasActiveLLM()) {
      ocrText = (await runVisionOCR(file.buffer, file.mimeType)) || "";
    }

    const combinedQuery = ocrText ? `${message}\n${ocrText}`.trim() : message;
    const engine = getEngine();
    const localRes = await engine.chat(combinedQuery, history, null, {});

    if (localRes.refused) {
      return res.status(200).json({ ...localRes, ocr_text: ocrText || null });
    }
    if (localRes.category === "greeting" && !hasImage) {
      return res.status(200).json(localRes);
    }

    if (hasActiveLLM()) {
      const llmReply = await runDualLLM(
        combinedQuery,
        localRes.rag_context || localRes.reply || "",
        history,
        hasImage ? file.buffer : null,
        hasImage ? file.mimeType : "image/jpeg"
      );
      if (llmReply) {
        return res.status(200).json({
          ...localRes,
          reply: llmReply,
          ocr_text: ocrText || null,
          neural_generated: true,
          engine: "Haroony AI (Dual AI + Harony 2026/2027 RAG)",
        });
      }
    }

    return res.status(200).json({ ...localRes, ocr_text: ocrText || null });
  } catch (err) {
    return res.status(500).json({ ok: false, detail: "Could not process request right now." });
  }
};
