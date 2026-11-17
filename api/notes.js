"use strict";

const { getEngine } = require("./_lib/haroony_vercel");

module.exports = async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  const url = new URL(req.url, "http://localhost");
  const q = url.searchParams.get("q") || "";
  const unit = url.searchParams.get("unit") || "";
  const engine = getEngine();
  const data = await engine.searchNotes(q, unit);
  res.status(200).json(data);
};
