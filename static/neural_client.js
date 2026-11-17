/* Haroony AI — Enhanced Multi-Pass Document/Handwriting OCR + Optional WebGPU Engine
 * 1. Adaptive Threshold + Contrast-Boosted Upscaled Canvas OCR (Tesseract.js worker + PDF.js 2.2x render).
 * 2. Automatic 90-degree orientation fallback for sideways phone photos.
 * 3. Instant-first: never freezes or blocks waiting on large downloads; optional WebGPU Qwen2.5-1.5B/0.5B.
 * 4. Strictly branded as Haroony AI (no third-party provider names or keys in frontend).
 */
(function (global) {
  "use strict";

  let engineState = "idle"; // idle | loading | ready | failed
  let mlcEngine = null;
  let initPromise = null;
  let tesseractWorkerPromise = null;
  const listeners = new Set();

  function notify(stateInfo) {
    for (const fn of listeners) {
      try {
        fn(stateInfo);
      } catch (_) {}
    }
  }

  function isMobileDevice() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  }

  async function initWebGPU(onProgress) {
    if (engineState === "ready" && mlcEngine) return true;
    if (initPromise) return initPromise;

    engineState = "loading";
    notify({ state: "loading", text: "Preparing browser acceleration…" });

    initPromise = (async () => {
      try {
        if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No WebGPU GPU adapter found.");

        const modelId = isMobileDevice()
          ? "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
          : "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";

        const webllm = await import("https://esm.run/@mlc-ai/web-llm@0.2.88");
        const worker = new Worker("static/llm_worker.js", { type: "module" });

        mlcEngine = await webllm.CreateWebWorkerMLCEngine(worker, modelId, {
          initProgressCallback: (report) => {
            if (!report) return;
            const pct = typeof report.progress === "number" ? Math.round(report.progress * 100) : null;
            const txt =
              pct !== null && pct > 0 && pct < 100
                ? `Loading Haroony Neural AI (${pct}%)…`
                : report.text || "Loading Haroony Neural AI…";
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

  // ==================== ENHANCED MULTI-PASS CANVAS OCR ====================
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

  async function getTesseractWorker() {
    if (!tesseractWorkerPromise) {
      tesseractWorkerPromise = (async () => {
        await loadScriptOnce("https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js");
        if (!global.Tesseract) throw new Error("Tesseract unavailable");
        const worker = await global.Tesseract.createWorker("eng", 1, {
          errorHandler: () => {},
        });
        await worker.setParameters({
          tessedit_pageseg_mode: "6", // Assume uniform block of text/math lines
          preserve_interword_spaces: "1",
        });
        return worker;
      })().catch((err) => {
        tesseractWorkerPromise = null;
        throw err;
      });
    }
    return tesseractWorkerPromise;
  }

  function cleanPhysicsOcrText(text) {
    return String(text || "")
      .replace(/\u2009/g, " ")
      .replace(/×/g, "x")
      .replace(/−/g, "-")
      .replace(/\bm\s*\/\s*s\s*2\b/gi, "m/s2")
      .replace(/\bkg\s*\/\s*m\s*3\b/gi, "kg/m3")
      .replace(/\b(\d+)\s*[oO]hms?\b/g, "$1 ohms")
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  async function loadImageElement(fileOrCanvas) {
    if (fileOrCanvas instanceof HTMLCanvasElement) return fileOrCanvas;
    const url = URL.createObjectURL(fileOrCanvas);
    try {
      return await new Promise((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = url;
      });
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function buildScaledCanvas(source, rotateDeg = 0) {
    const rawW = source.width || source.naturalWidth || 800;
    const rawH = source.height || source.naturalHeight || 600;
    const maxDim = Math.max(rawW, rawH);
    const scale = maxDim < 1400 ? Math.min(2.6, 1500 / Math.max(1, maxDim)) : maxDim > 2600 ? 2400 / maxDim : 1.0;
    const w = Math.round(rawW * scale);
    const h = Math.round(rawH * scale);

    const canvas = document.createElement("canvas");
    if (rotateDeg === 90 || rotateDeg === 270) {
      canvas.width = h;
      canvas.height = w;
    } else {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    if (rotateDeg === 90) {
      ctx.translate(h, 0);
      ctx.rotate(Math.PI / 2);
    } else if (rotateDeg === 270) {
      ctx.translate(0, w);
      ctx.rotate(-Math.PI / 2);
    }
    ctx.drawImage(source, 0, 0, w, h);
    ctx.restore();
    return canvas;
  }

  function makeContrastGrayCanvas(baseCanvas) {
    const w = baseCanvas.width;
    const h = baseCanvas.height;
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    ctx.drawImage(baseCanvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;

    // Percentile-like min/max stretch for crisp gray contrast
    let minL = 255;
    let maxL = 0;
    const lums = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      lums[p] = lum;
      if (lum < minL) minL = lum;
      if (lum > maxL) maxL = lum;
    }
    const span = Math.max(30, maxL - minL);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const norm = ((lums[p] - minL) * 255) / span;
      const v = Math.max(0, Math.min(255, (norm - 128) * 1.32 + 134));
      d[i] = d[i + 1] = d[i + 2] = v;
    }
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  function makeAdaptiveBinaryCanvas(baseCanvas) {
    // Bradley-Roth integral-image adaptive threshold: eliminates uneven shadows & lined paper tint
    const w = baseCanvas.width;
    const h = baseCanvas.height;
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    ctx.drawImage(baseCanvas, 0, 0);
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;

    const gray = new Uint8Array(w * h);
    const integral = new Float64Array((w + 1) * (h + 1));
    for (let y = 0; y < h; y++) {
      let rowSum = 0;
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const g = 0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2];
        gray[y * w + x] = g;
        rowSum += g;
        integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + rowSum;
      }
    }

    const radius = Math.max(12, Math.round(Math.min(w, h) / 28));
    const tFactor = 0.88; // pixel < 88% of local mean => dark ink
    for (let y = 0; y < h; y++) {
      const y1 = Math.max(0, y - radius);
      const y2 = Math.min(h - 1, y + radius);
      for (let x = 0; x < w; x++) {
        const x1 = Math.max(0, x - radius);
        const x2 = Math.min(w - 1, x + radius);
        const count = (x2 - x1 + 1) * (y2 - y1 + 1);
        const sum =
          integral[(y2 + 1) * (w + 1) + (x2 + 1)] -
          integral[y1 * (w + 1) + (x2 + 1)] -
          integral[(y2 + 1) * (w + 1) + x1] +
          integral[y1 * (w + 1) + x1];
        const val = gray[y * w + x] * count < sum * tFactor ? 0 : 255;
        const idx = (y * w + x) * 4;
        d[idx] = d[idx + 1] = d[idx + 2] = val;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  async function runWorkerOcr(worker, canvas) {
    const res = await worker.recognize(canvas);
    const txt = cleanPhysicsOcrText(res && res.data ? res.data.text : "");
    const conf = res && res.data && typeof res.data.confidence === "number" ? res.data.confidence : 0;
    return { text: txt, confidence: conf };
  }

  async function ocrImageSource(fileOrCanvas, onProgress) {
    if (onProgress) onProgress("Enhancing image contrast & reading text…");
    const worker = await getTesseractWorker();
    const imgSource = await loadImageElement(fileOrCanvas);

    const base = buildScaledCanvas(imgSource, 0);
    const contrastCanvas = makeContrastGrayCanvas(base);
    let best = await runWorkerOcr(worker, contrastCanvas);

    // If initial pass looks short or noisy (shadowed phone photo / faint handwriting), try adaptive binarization!
    if (best.text.length < 55 || best.confidence < 72) {
      if (onProgress) onProgress("Running adaptive lighting cleanup pass…");
      const binCanvas = makeAdaptiveBinaryCanvas(base);
      const binRes = await runWorkerOcr(worker, binCanvas);
      const scoreA = best.text.length * Math.max(25, best.confidence);
      const scoreB = binRes.text.length * Math.max(25, binRes.confidence);
      if (scoreB > scoreA * 1.1) {
        best = binRes;
      }
    }

    // Sideways photo fallback (90° / 270°)
    if (best.text.length < 25 || best.confidence < 58) {
      for (const deg of [90, 270]) {
        const rotCanvas = makeContrastGrayCanvas(buildScaledCanvas(imgSource, deg));
        const rotRes = await runWorkerOcr(worker, rotCanvas);
        if (rotRes.text.length > best.text.length * 1.35 && rotRes.confidence >= best.confidence - 5) {
          best = rotRes;
          break;
        }
      }
    }

    return best.text;
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
      const pageTxt = cleanPhysicsOcrText(
        (tc.items || [])
          .map((it) => it.str || "")
          .join(" ")
          .replace(/\s+/g, " ")
      );

      if (pageTxt.length >= 25) {
        pagesOut.push(pageTxt);
      } else if (p <= 4) {
        const viewport = page.getViewport({ scale: 2.2 });
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
        return await ocrImageSource(file, onProgress);
      }
      return (await file.text()).trim();
    } catch (err) {
      console.warn("Document extraction error:", err);
      return "";
    }
  }

  async function generate(messages, opts = {}) {
    if (engineState !== "ready" || !mlcEngine) {
      return null;
    }

    const { maxTokens = 560, temperature = 0.2, onProgress = null, onToken = null, timeoutMs = 15000 } = opts;
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
    return {
      state: engineState,
      backend: engineState === "ready" ? "webllm" : "harony_rag",
      hasWebGPU: typeof navigator !== "undefined" && Boolean(navigator.gpu),
    };
  }

  global.HaroonyNeuralAI = {
    isReady: () => engineState === "ready",
    getStatus,
    onStatusChange: (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    enableWebGPU: initWebGPU,
    generate,
    extractDocumentText,
  };
})(typeof window !== "undefined" ? window : globalThis);
