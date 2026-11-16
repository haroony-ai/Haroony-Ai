/* Haroony AI — Real Canvas-Enhanced OCR/PDF + Optional Cached WebGPU / Custom LLM Client
 * 1. Instant-first: NEVER freezes or blocks waiting on uncached multi-hundred-MB downloads.
 * 2. Real Canvas-Enhanced OCR (grayscale + contrast boost + upscale) for handwritten photos & PDFs via Tesseract.js + PDF.js.
 * 3. Optional Real Neural LLM:
 *    - Browser WebGPU (@mlc-ai/web-llm Qwen2.5-0.5B-Instruct-q4f16_1-MLC)
 *    - Or user-configured local/cloud OpenAI-compatible endpoint (Ollama, Groq, OpenAI, etc. via localStorage)
 */
(function (global) {
  "use strict";

  const CUSTOM_CFG_KEY = "haroony_custom_llm_cfg";
  let engineState = "idle"; // idle | loading | ready | failed
  let mlcEngine = null;
  let initPromise = null;
  const listeners = new Set();

  function notify(stateInfo) {
    for (const fn of listeners) {
      try {
        fn(stateInfo);
      } catch (_) {}
    }
  }

  function getCustomConfig() {
    try {
      const raw = localStorage.getItem(CUSTOM_CFG_KEY);
      if (!raw) return null;
      const cfg = JSON.parse(raw);
      if (cfg && cfg.baseUrl && cfg.model) return cfg;
    } catch (_) {}
    return null;
  }

  function saveCustomConfig(cfg) {
    try {
      if (!cfg || !cfg.baseUrl) {
        localStorage.removeItem(CUSTOM_CFG_KEY);
      } else {
        localStorage.setItem(CUSTOM_CFG_KEY, JSON.stringify(cfg));
      }
      notify(getStatus());
    } catch (_) {}
  }

  function isMobileDevice() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  }

  async function initWebGPU(onProgress) {
    if (engineState === "ready" && mlcEngine) return true;
    if (initPromise) return initPromise;

    engineState = "loading";
    notify({ state: "loading", text: "Checking WebGPU…" });

    initPromise = (async () => {
      try {
        if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU GPU adapter found.");

        const modelId = isMobileDevice()
          ? "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
          : "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

        const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.88");
        const worker = new Worker("static/llm_worker.js", { type: "module" });

        mlcEngine = await webllm.CreateWebWorkerMLCEngine(worker, modelId, {
          initProgressCallback: (report) => {
            if (!report) return;
            const pct = typeof report.progress === "number" ? Math.round(report.progress * 100) : null;
            const txt = pct !== null && pct > 0 && pct < 100 ? `Loading Neural AI (${pct}%)…` : report.text || "Loading Neural AI…";
            if (onProgress) onProgress(txt);
            notify({ state: "loading", progress: pct, text: txt });
          },
        });
        engineState = "ready";
        notify(getStatus());
        return true;
      } catch (err) {
        engineState = "failed";
        initPromise = null;
        notify({ state: "failed", text: err && err.message ? err.message : "WebGPU unavailable" });
        return false;
      }
    })();

    return initPromise;
  }

  // ==================== CANVAS-ENHANCED DOCUMENT & PHOTO OCR ====================
  const scriptCache = {};
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

  async function preprocessImageToCanvas(fileOrCanvas) {
    if (fileOrCanvas instanceof HTMLCanvasElement) return fileOrCanvas;
    const url = URL.createObjectURL(fileOrCanvas);
    try {
      const img = await new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = url;
      });
      const scale = img.width < 1100 ? Math.min(2.2, 1400 / Math.max(1, img.width)) : 1.0;
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);

      // Contrast-boosted grayscale for cleaner handwriting & printed paper OCR
      const imgData = ctx.getImageData(0, 0, w, h);
      const d = imgData.data;
      for (let i = 0; i < d.length; i += 4) {
        const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        const stretched = Math.max(0, Math.min(255, (lum - 128) * 1.38 + 132));
        d[i] = d[i + 1] = d[i + 2] = stretched;
      }
      ctx.putImageData(imgData, 0, 0);
      return canvas;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function ocrImageSource(fileOrCanvas, onProgress) {
    await loadScriptOnce("https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js");
    if (!global.Tesseract) throw new Error("Tesseract unavailable");
    const canvas = await preprocessImageToCanvas(fileOrCanvas);
    const res = await global.Tesseract.recognize(canvas, "eng", {
      logger: (m) => {
        if (onProgress && m && m.status === "recognizing text" && typeof m.progress === "number") {
          onProgress(`Reading text from photo (${Math.round(m.progress * 100)}%)…`);
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
        const viewport = page.getViewport({ scale: 1.8 });
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
        if (onProgress) onProgress("Enhancing & reading uploaded image…");
        return await ocrImageSource(file, onProgress);
      }
      return (await file.text()).trim();
    } catch (err) {
      console.warn("Document extraction error:", err);
      return "";
    }
  }

  // ==================== REAL STREAMING GENERATION ====================
  async function generateViaCustomEndpoint(cfg, messages, opts) {
    const { maxTokens = 650, temperature = 0.2, onToken = null } = opts;
    const base = cfg.baseUrl.replace(/\/+$/, "");
    const headers = { "Content-Type": "application/json" };
    if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;

    const resp = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: cfg.model,
        messages,
        max_tokens: maxTokens,
        temperature,
        stream: true,
      }),
    });
    if (!resp.ok || !resp.body) return null;

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = "";
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const ln of lines) {
        const trimmed = ln.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const j = JSON.parse(payload);
          const delta = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
          if (delta) {
            full += delta;
            if (onToken) onToken(full, delta);
          }
        } catch (_) {}
      }
    }
    return full.trim() || null;
  }

  async function generate(messages, opts = {}) {
    const customCfg = getCustomConfig();
    if (customCfg) {
      try {
        const res = await generateViaCustomEndpoint(customCfg, messages, opts);
        if (res) return res;
      } catch (err) {
        console.warn("Custom LLM endpoint fallback:", err);
      }
    }

    if (engineState !== "ready" || !mlcEngine) {
      return null;
    }

    const { maxTokens = 520, temperature = 0.2, onProgress = null, onToken = null, timeoutMs = 12000 } = opts;
    if (onProgress) onProgress("Generating explanation…");

    const runStream = (async () => {
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
    })();

    const timeout = new Promise((resolve) => setTimeout(() => resolve(null), timeoutMs));
    try {
      return await Promise.race([runStream, timeout]);
    } catch (_) {
      return null;
    }
  }

  function getStatus() {
    const customCfg = getCustomConfig();
    return {
      state: customCfg ? "custom_ready" : engineState,
      backend: customCfg ? "custom_api" : engineState === "ready" ? "webllm" : "harony_rag",
      hasWebGPU: typeof navigator !== "undefined" && Boolean(navigator.gpu),
      customModel: customCfg ? customCfg.model : null,
    };
  }

  global.HaroonyNeuralAI = {
    isReady: () => Boolean(getCustomConfig()) || engineState === "ready",
    getStatus,
    onStatusChange: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    enableWebGPU: initWebGPU,
    getCustomConfig,
    saveCustomConfig,
    generate,
    extractDocumentText,
  };
})(typeof window !== "undefined" ? window : globalThis);
