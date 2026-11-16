/* Haroony AI — Real In-Browser Neural LLM + Vision OCR/PDF Client
 * 1. WebGPU Neural Engine: @mlc-ai/web-llm (Qwen2.5-1.5B-Instruct / Qwen2.5-0.5B-Instruct)
 * 2. WASM CPU Neural Fallback: @huggingface/transformers (Qwen2.5-0.5B-Instruct q4)
 * 3. Browser Document & Vision Reader: PDF.js + Tesseract.js v5
 * Runs 100% free with zero API keys, zero logins, and persistent browser caching.
 */
(function (global) {
  "use strict";

  let engineState = "idle"; // idle | loading | ready | failed
  let backendType = null;   // "webllm" | "wasm"
  let mlcEngine = null;
  let wasmWorker = null;
  let initPromise = null;
  let activeProgressListeners = new Set();
  let reqCounter = 0;
  let pendingWasmCalls = new Map();

  function notifyProgress(text) {
    for (const fn of activeProgressListeners) {
      try {
        fn(text);
      } catch (_) {}
    }
  }

  function isMobileDevice() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  }

  async function tryInitWebLLM() {
    if (!navigator.gpu) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");

    const maxBuf = (adapter.limits && adapter.limits.maxStorageBufferBindingSize) || 0;
    const modelId =
      isMobileDevice() || (maxBuf > 0 && maxBuf < 268435456)
        ? "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
        : "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";

    const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.88");
    const worker = new Worker("static/llm_worker.js", { type: "module" });

    mlcEngine = await webllm.CreateWebWorkerMLCEngine(worker, modelId, {
      initProgressCallback: (report) => {
        if (!report) return;
        if (typeof report.progress === "number" && report.progress > 0 && report.progress < 1) {
          const pct = Math.round(report.progress * 100);
          notifyProgress(`Loading Physics AI brain (${pct}%)…`);
        } else if (report.text && report.text.toLowerCase().includes("cache")) {
          notifyProgress("Loading cached Physics AI…");
        }
      },
    });
    backendType = "webllm";
    engineState = "ready";
    return true;
  }

  async function tryInitWasmFallback() {
    return new Promise((resolve, reject) => {
      try {
        wasmWorker = new Worker("static/wasm_llm_worker.js", { type: "module" });
      } catch (err) {
        reject(err);
        return;
      }
      const warmupId = "warmup-0";
      wasmWorker.onmessage = (ev) => {
        const d = ev.data || {};
        if (d.type === "progress" && d.text) {
          notifyProgress(d.text);
          return;
        }
        if (d.id === warmupId) {
          if (d.type === "ready") {
            backendType = "wasm";
            engineState = "ready";
            resolve(true);
          } else if (d.type === "error") {
            reject(new Error(d.error || "WASM init failed"));
          }
          return;
        }
        const pending = pendingWasmCalls.get(d.id);
        if (pending) {
          pendingWasmCalls.delete(d.id);
          if (d.type === "done") pending.resolve(d.text || "");
          else pending.reject(new Error(d.error || "WASM generation error"));
        }
      };
      wasmWorker.onerror = (err) => {
        reject(err);
      };
      wasmWorker.postMessage({ id: warmupId, type: "warmup" });
    });
  }

  function ensureEngine(onProgress) {
    if (onProgress) activeProgressListeners.add(onProgress);
    if (engineState === "ready") {
      if (onProgress) activeProgressListeners.delete(onProgress);
      return Promise.resolve(true);
    }
    if (!initPromise) {
      engineState = "loading";
      initPromise = tryInitWebLLM()
        .catch(() => tryInitWasmFallback())
        .catch((err) => {
          console.warn("Neural engine unavailable, using symbolic+RAG synthesizer:", err);
          engineState = "failed";
          return false;
        });
    }
    return initPromise.finally(() => {
      if (onProgress) activeProgressListeners.delete(onProgress);
    });
  }

  // ==================== DOCUMENT & PHOTO OCR IN BROWSER ====================
  let scriptCache = {};
  function loadScriptOnce(src) {
    if (!scriptCache[src]) {
      scriptCache[src] = new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error("Failed to load " + src));
        document.head.appendChild(s);
      });
    }
    return scriptCache[src];
  }

  async function ocrImageSource(imgOrCanvas, onProgress) {
    await loadScriptOnce("https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js");
    if (!global.Tesseract) throw new Error("Tesseract unavailable");
    const res = await global.Tesseract.recognize(imgOrCanvas, "eng", {
      logger: (m) => {
        if (onProgress && m && m.status === "recognizing text" && typeof m.progress === "number") {
          onProgress(`Reading image text (${Math.round(m.progress * 100)}%)…`);
        }
      },
    });
    return (res && res.data && res.data.text ? res.data.text : "").trim();
  }

  async function extractPdfTextInBrowser(file, onProgress) {
    if (onProgress) onProgress("Reading PDF document…");
    await loadScriptOnce("https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js");
    const pdfjsLib = global.pdfjsLib;
    if (!pdfjsLib) throw new Error("PDF.js unavailable");
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";

    const buf = await file.arrayBuffer();
    const doc = await pdfjsLib.getDocument({ data: buf }).promise;
    const maxPages = Math.min(doc.numPages, 6);
    const pagesOut = [];

    for (let p = 1; p <= maxPages; p++) {
      const page = await doc.getPage(p);
      const tc = await page.getTextContent();
      const pageTxt = (tc.items || [])
        .map((it) => it.str || "")
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();

      if (pageTxt.length >= 25) {
        pagesOut.push(pageTxt);
      } else if (p <= 3) {
        // Scanned image PDF page -> render onto canvas & OCR!
        const viewport = page.getViewport({ scale: 1.7 });
        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        await page.render({ canvasContext: ctx, viewport }).promise;
        const ocrTxt = await ocrImageSource(canvas, onProgress);
        if (ocrTxt) pagesOut.push(ocrTxt);
      }
    }
    return pagesOut.join("\n\n").trim();
  }

  async function extractDocumentText(file, onProgress) {
    if (!file) return "";
    const name = (file.name || "").toLowerCase();
    const type = (file.type || "").toLowerCase();
    try {
      if (type === "application/pdf" || name.endsWith(".pdf")) {
        return await extractPdfTextInBrowser(file, onProgress);
      }
      if (type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif)$/.test(name)) {
        if (onProgress) onProgress("Reading handwriting & text from photo…");
        return await ocrImageSource(file, onProgress);
      }
      return (await file.text()).trim();
    } catch (err) {
      console.warn("Document extraction error:", err);
      return "";
    }
  }

  // ==================== GENERATE COMPLETION ====================
  async function generate(messages, opts = {}) {
    const { maxTokens = 520, temperature = 0.2, onProgress = null, onToken = null } = opts;
    const ready = await ensureEngine(onProgress);
    if (!ready) return null;

    if (backendType === "webllm" && mlcEngine) {
      if (onProgress) onProgress("Writing clear step-by-step explanation…");
      const chunks = await mlcEngine.chat.completions.create({
        messages,
        max_tokens: maxTokens,
        temperature,
        top_p: 0.9,
        stream: true,
      });
      let fullText = "";
      for await (const chunk of chunks) {
        const delta =
          chunk &&
          chunk.choices &&
          chunk.choices[0] &&
          chunk.choices[0].delta &&
          chunk.choices[0].delta.content;
        if (delta) {
          fullText += delta;
          if (onToken) onToken(fullText, delta);
        }
      }
      return fullText.trim();
    }

    if (backendType === "wasm" && wasmWorker) {
      if (onProgress) onProgress("Generating Physics explanation…");
      const id = "gen-" + ++reqCounter;
      return new Promise((resolve, reject) => {
        pendingWasmCalls.set(id, { resolve, reject });
        wasmWorker.postMessage({ id, type: "generate", messages, maxTokens, temperature });
      });
    }

    return null;
  }

  // Warm up quietly shortly after page load so AI is ready fast
  if (typeof window !== "undefined") {
    window.addEventListener("load", () => {
      setTimeout(() => {
        ensureEngine(null);
      }, 450);
    });
  }

  global.HaroonyNeuralAI = {
    ensureEngine,
    generate,
    extractDocumentText,
    getState: () => ({ state: engineState, backend: backendType }),
  };
})(window);
