/* Haroony AI — High-Accuracy Client-Side Document & Screenshot OCR
 * Features:
 * 1. Dark-mode screenshot auto-detection & inversion (white-on-dark -> crisp black-on-white).
 * 2. Local illumination / shadow flattening + contrast boost + adaptive binarization.
 * 3. Multi-PSM recognition (PSM 6 uniform block + PSM 3/11 sparse diagram labels).
 * 4. 90° / 180° / 270° orientation fallback + Physics math/symbol cleanup.
 * Runs strictly on-demand when a student attaches an image or PDF (zero large downloads).
 */
(function (global) {
  "use strict";

  let tesseractWorkerPromise = null;
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
          tessedit_pageseg_mode: "6",
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
      .replace(/\u2009|\u00a0/g, " ")
      .replace(/×/g, "x")
      .replace(/[−–]/g, "-")
      .replace(/Ω/g, " ohms")
      .replace(/[µμ]/g, "u")
      .replace(/\bm\s*\/\s*s\s*2\b/gi, "m/s2")
      .replace(/\bm\s*s\s*-?\s*2\b/gi, "m/s2")
      .replace(/\bkg\s*\/\s*m\s*3\b/gi, "kg/m3")
      .replace(/\bg\s*\/\s*cm\s*3\b/gi, "g/cm3")
      .replace(/\b(\d+(?:\.\d+)?)\s*[°o0]\s*C\b/g, "$1 °C")
      .replace(/\b(\d+(?:\.\d+)?)\s*[oO]hms?\b/g, "$1 ohms")
      .replace(/(\d+(?:\.\d+)?)\s*[xX*]\s*10\s*\^?\s*([+-]?\d+)/g, "$1 x 10^$2")
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
    const scale = maxDim < 1600 ? Math.min(3.0, 1750 / Math.max(1, maxDim)) : maxDim > 2600 ? 2500 / maxDim : 1.0;
    const w = Math.round(rawW * scale);
    const h = Math.round(rawH * scale);
    const pad = 24;

    const canvas = document.createElement("canvas");
    if (rotateDeg === 90 || rotateDeg === 270) {
      canvas.width = h + pad * 2;
      canvas.height = w + pad * 2;
    } else {
      canvas.width = w + pad * 2;
      canvas.height = h + pad * 2;
    }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    if (rotateDeg === 90) {
      ctx.translate(h + pad, pad);
      ctx.rotate(Math.PI / 2);
    } else if (rotateDeg === 180) {
      ctx.translate(w + pad, h + pad);
      ctx.rotate(Math.PI);
    } else if (rotateDeg === 270) {
      ctx.translate(pad, w + pad);
      ctx.rotate(-Math.PI / 2);
    } else {
      ctx.translate(pad, pad);
    }
    ctx.drawImage(source, 0, 0, w, h);
    ctx.restore();
    return canvas;
  }

  function getLumaWithDarkInvert(baseCanvas) {
    const w = baseCanvas.width;
    const h = baseCanvas.height;
    const ctx = baseCanvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;
    const lums = new Uint8Array(w * h);
    const hist = new Uint32Array(256);

    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) | 0;
      lums[p] = lum;
      hist[lum]++;
    }
    // Find median luminance to detect dark-mode screenshots
    let count = 0;
    let median = 255;
    const half = (w * h) >> 1;
    for (let v = 0; v < 256; v++) {
      count += hist[v];
      if (count >= half) {
        median = v;
        break;
      }
    }
    if (median < 115) {
      for (let p = 0; p < lums.length; p++) {
        lums[p] = 255 - lums[p];
      }
    }
    return { w, h, lums };
  }

  function makeShadowNormalizedGrayCanvas(baseCanvas) {
    const { w, h, lums } = getLumaWithDarkInvert(baseCanvas);
    // Integral image for fast box-blur background illumination estimation
    const integral = new Float64Array((w + 1) * (h + 1));
    for (let y = 0; y < h; y++) {
      let rowSum = 0;
      for (let x = 0; x < w; x++) {
        const g = lums[y * w + x];
        rowSum += g;
        integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + rowSum;
      }
    }

    const bgRadius = Math.max(24, Math.round(Math.min(w, h) / 22));
    const normGray = new Uint8Array(w * h);
    let minL = 255;
    let maxL = 0;
    for (let y = 0; y < h; y++) {
      const y1 = Math.max(0, y - bgRadius);
      const y2 = Math.min(h - 1, y + bgRadius);
      for (let x = 0; x < w; x++) {
        const x1 = Math.max(0, x - bgRadius);
        const x2 = Math.min(w - 1, x + bgRadius);
        const cnt = (x2 - x1 + 1) * (y2 - y1 + 1);
        const sum =
          integral[(y2 + 1) * (w + 1) + (x2 + 1)] -
          integral[y1 * (w + 1) + (x2 + 1)] -
          integral[(y2 + 1) * (w + 1) + x1] +
          integral[y1 * (w + 1) + x1];
        const bg = Math.max(20, sum / cnt);
        const v = Math.max(0, Math.min(255, Math.round((lums[y * w + x] / bg) * 238)));
        normGray[y * w + x] = v;
        if (v < minL) minL = v;
        if (v > maxL) maxL = v;
      }
    }

    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    const imgData = ctx.createImageData(w, h);
    const d = imgData.data;
    const span = Math.max(30, maxL - minL);
    for (let p = 0, i = 0; p < normGray.length; p++, i += 4) {
      const stretched = ((normGray[p] - minL) * 255) / span;
      const v = Math.max(0, Math.min(255, (stretched - 128) * 1.38 + 136));
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    return { canvas: out, w, h, gray: normGray };
  }

  function makeAdaptiveBinaryCanvas(normInfo) {
    const { w, h, gray } = normInfo;
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    const imgData = ctx.createImageData(w, h);
    const d = imgData.data;

    const integral = new Float64Array((w + 1) * (h + 1));
    for (let y = 0; y < h; y++) {
      let rowSum = 0;
      for (let x = 0; x < w; x++) {
        rowSum += gray[y * w + x];
        integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + rowSum;
      }
    }

    const radius = Math.max(12, Math.round(Math.min(w, h) / 28));
    const tFactor = 0.87;
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
        d[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    return out;
  }

  function scoreOcrResult(txt, conf) {
    const alnum = (txt.match(/[a-zA-Z0-9]/g) || []).length;
    return alnum * (0.45 + Math.max(0.15, (conf || 0) / 100));
  }

  async function runWorkerOcr(worker, canvas, psm = "6") {
    await worker.setParameters({ tessedit_pageseg_mode: psm, preserve_interword_spaces: "1" });
    const res = await worker.recognize(canvas);
    const txt = cleanPhysicsOcrText(res && res.data ? res.data.text : "");
    const conf = res && res.data && typeof res.data.confidence === "number" ? res.data.confidence : 0;
    return { text: txt, confidence: conf, score: scoreOcrResult(txt, conf) };
  }

  async function ocrImageSource(fileOrCanvas, onProgress) {
    if (onProgress) onProgress("Enhancing lighting, dark-mode & sharpness to read text…");
    const worker = await getTesseractWorker();
    const imgSource = await loadImageElement(fileOrCanvas);

    const base = buildScaledCanvas(imgSource, 0);
    const normInfo = makeShadowNormalizedGrayCanvas(base);
    let best = await runWorkerOcr(worker, normInfo.canvas, "6");

    if (best.text.length < 65 || best.confidence < 76) {
      if (onProgress) onProgress("Running adaptive shadow cleanup & layout pass…");
      const binCanvas = makeAdaptiveBinaryCanvas(normInfo);
      const binRes = await runWorkerOcr(worker, binCanvas, "6");
      if (binRes.score > best.score * 1.08) best = binRes;

      // Sparse / multi-column / diagram label layout fallback
      if (best.text.length < 55 || best.confidence < 72) {
        const autoRes = await runWorkerOcr(worker, normInfo.canvas, "3");
        if (autoRes.score > best.score * 1.08) best = autoRes;
      }
    }

    if (best.text.length < 28 || best.confidence < 62) {
      for (const deg of [90, 180, 270]) {
        const rotNorm = makeShadowNormalizedGrayCanvas(buildScaledCanvas(imgSource, deg));
        const rotRes = await runWorkerOcr(worker, rotNorm.canvas, "6");
        if (rotRes.score > best.score * 1.25 && rotRes.confidence >= best.confidence - 6) {
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
        const viewport = page.getViewport({ scale: 2.5 });
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

  global.HaroonyNeuralAI = {
    isReady: () => false,
    isLoading: () => false,
    getStatus: () => ({ state: "instant", backend: "harony_instant", modelId: null, hasWebGPU: false }),
    onStatusChange: () => () => {},
    generate: async () => null,
    extractDocumentText,
  };
})(typeof window !== "undefined" ? window : globalThis);
