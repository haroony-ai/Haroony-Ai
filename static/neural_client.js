/* Haroony AI — Fast In-Browser Vision OCR/PDF + Optional Cached WebGPU Neural Enhancer
 * 1. Instant-first architecture: NEVER blocks chat or grading waiting for multi-hundred-MB downloads.
 * 2. Optional WebGPU Neural Engine (@mlc-ai/web-llm) warms quietly ONLY when WebGPU is supported;
 *    only used when already cached & ready, with a strict timeout.
 * 3. Browser Document & Vision Reader: PDF.js + Tesseract.js v5 (loaded on demand for uploads).
 */
(function (global) {
  "use strict";

  let engineState = "idle"; // idle | loading | ready | failed
  let mlcEngine = null;
  let initPromise = null;

  function isMobileDevice() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  }

  async function tryInitWebLLM() {
    if (!navigator.gpu || isMobileDevice()) {
      engineState = "failed";
      return false;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      engineState = "failed";
      return false;
    }

    const maxBuf = (adapter.limits && adapter.limits.maxStorageBufferBindingSize) || 0;
    const modelId =
      maxBuf > 0 && maxBuf < 268435456
        ? "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
        : "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";

    const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.88");
    const worker = new Worker("static/llm_worker.js", { type: "module" });

    mlcEngine = await webllm.CreateWebWorkerMLCEngine(worker, modelId, {});
    engineState = "ready";
    return true;
  }

  function warmInBackground() {
    if (initPromise || engineState === "ready" || engineState === "failed") return;
    engineState = "loading";
    initPromise = tryInitWebLLM().catch(() => {
      engineState = "failed";
      return false;
    });
  }

  // ==================== DOCUMENT & PHOTO OCR IN BROWSER ====================
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

  // ==================== NON-BLOCKING FAST GENERATION ====================
  async function generate(messages, opts = {}) {
    // NEVER block the student waiting for a download; only generate when WebGPU is already ready!
    if (engineState !== "ready" || !mlcEngine) {
      warmInBackground();
      return null;
    }

    const { maxTokens = 480, temperature = 0.2, onProgress = null, onToken = null, timeoutMs = 6000 } = opts;
    if (onProgress) onProgress("Writing clear step-by-step explanation…");

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

  if (typeof window !== "undefined") {
    window.addEventListener("load", () => {
      setTimeout(warmInBackground, 1200);
    });
  }

  global.HaroonyNeuralAI = {
    isReady: () => engineState === "ready",
    getStatus: () => ({ state: engineState, backend: engineState === "ready" ? "webllm" : "instant" }),
    generate,
    extractDocumentText,
  };
})(typeof window !== "undefined" ? window : globalThis);
