"use strict";

const { loadKB, hasActiveLLM } = require("./_lib/haroony_vercel");

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Cache-Control", "no-store");
  const kb = loadKB();
  res.status(200).json({
    ok: true,
    mode: "server",
    name: "Haroony AI",
    clinic: "Harony Physics Clinic",
    free_unlimited: true,
    physics_only_guardrail: true,
    notes_chunks_count: (kb.chunks || []).length,
    video_solutions_count: (kb.videos || []).length,
    mark_schemes: (kb.mark_schemes || []).map((m) => ({
      id: m.id,
      title: m.title,
      paper_code: m.paper_code,
      session: m.session,
      total_marks: m.total_marks,
      question_count: m.question_count,
    })),
    server_llm_active: hasActiveLLM(),
  });
};
