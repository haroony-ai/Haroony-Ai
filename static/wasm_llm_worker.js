import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

env.allowLocalModels = false;
if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
  env.backends.onnx.wasm.proxy = false;
}

let generatorPromise = null;
let activeModelId = "onnx-community/Qwen2.5-0.5B-Instruct";

const WASM_MODELS = [
  "onnx-community/Qwen2.5-0.5B-Instruct",
  "HuggingFaceTB/SmolLM2-360M-Instruct",
];

async function getGenerator(requestedModel, onProgress) {
  const target = requestedModel || activeModelId;
  if (!generatorPromise || target !== activeModelId) {
    activeModelId = target;
    generatorPromise = (async () => {
      const tryList = [target, ...WASM_MODELS.filter((m) => m !== target)];
      let lastErr = null;
      for (const modelId of tryList) {
        try {
          activeModelId = modelId;
          return await pipeline("text-generation", modelId, {
            dtype: "q4",
            device: "wasm",
            progress_callback: (info) => {
              if (onProgress && info && typeof info.progress === "number") {
                onProgress(Math.round(info.progress));
              }
            },
          });
        } catch (err) {
          lastErr = err;
        }
      }
      throw lastErr || new Error("No open-source WASM model could be loaded.");
    })().catch((err) => {
      generatorPromise = null;
      throw err;
    });
  }
  return generatorPromise;
}

self.onmessage = async (event) => {
  const { id, type, messages, maxTokens, temperature, modelId } = event.data || {};
  if (type === "warmup") {
    try {
      await getGenerator(modelId, (pct) => {
        self.postMessage({ id, type: "progress", text: `Loading open-source AI (${pct}%)…`, progress: pct });
      });
      self.postMessage({ id, type: "ready", modelId: activeModelId });
    } catch (err) {
      self.postMessage({ id, type: "error", error: String(err && err.message ? err.message : err) });
    }
    return;
  }

  if (type === "generate") {
    try {
      const gen = await getGenerator(modelId, (pct) => {
        self.postMessage({ id, type: "progress", text: `Loading open-source AI (${pct}%)…`, progress: pct });
      });
      const out = await gen(messages, {
        max_new_tokens: maxTokens || 520,
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
