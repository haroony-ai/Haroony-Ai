import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

env.allowLocalModels = false;

let generatorPromise = null;

async function getGenerator(onProgress) {
  if (!generatorPromise) {
    generatorPromise = pipeline(
      "text-generation",
      "onnx-community/Qwen2.5-0.5B-Instruct",
      {
        dtype: "q4",
        progress_callback: (info) => {
          if (onProgress && info && typeof info.progress === "number") {
            onProgress(Math.round(info.progress));
          }
        },
      }
    );
  }
  return generatorPromise;
}

self.onmessage = async (event) => {
  const { id, type, messages, maxTokens, temperature } = event.data || {};
  if (type === "warmup") {
    try {
      await getGenerator((pct) => {
        self.postMessage({ id, type: "progress", text: `Preparing Physics AI (${pct}%)…` });
      });
      self.postMessage({ id, type: "ready" });
    } catch (err) {
      self.postMessage({ id, type: "error", error: String(err && err.message ? err.message : err) });
    }
    return;
  }

  if (type === "generate") {
    try {
      const gen = await getGenerator((pct) => {
        self.postMessage({ id, type: "progress", text: `Preparing Physics AI (${pct}%)…` });
      });
      const out = await gen(messages, {
        max_new_tokens: maxTokens || 420,
        temperature: temperature ?? 0.25,
        do_sample: (temperature ?? 0.25) > 0.05,
      });
      let reply = "";
      if (Array.isArray(out) && out[0] && out[0].generated_text) {
        const gt = out[0].generated_text;
        if (Array.isArray(gt)) {
          const last = gt[gt.length - 1];
          reply = last && last.content ? String(last.content) : "";
        } else {
          reply = String(gt);
        }
      }
      self.postMessage({ id, type: "done", text: reply.trim() });
    } catch (err) {
      self.postMessage({ id, type: "error", error: String(err && err.message ? err.message : err) });
    }
  }
};
