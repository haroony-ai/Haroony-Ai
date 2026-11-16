/* Haroony AI — Clean Real-Time Frontend (FastAPI Backend + Standalone GitHub Pages Engine) */
(function () {
  "use strict";

  const AI_AVATAR = "static/assets/harony_avatar.webp";

  let serverAvailable = null;
  let chatHistory = [];
  let chatAttachedFile = null;
  let graderStudentFiles = [];
  let graderMsFile = null;
  let activeMsMode = "auto";
  let currentBookQuestions = [];

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function normalizeAssetUrl(u) {
    if (!u) return "";
    return String(u).replace(/^\/+static\//, "static/");
  }

  function renderMarkdown(md) {
    if (!md) return "";
    const lines = String(md).split("\n");
    const html = [];
    let inList = false;

    function closeList() {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
    }

    function inlineFormat(s) {
      let out = escapeHtml(s);
      out = out.replace(/10\^([+-]?\d+)/g, "10<sup>$1</sup>");
      out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
      out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
      out = out.replace(
        /(https:\/\/youtu\.be\/[A-Za-z0-9_\-]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer" class="video-pill-link">🎥 Watch Video</a>'
      );
      return out;
    }

    for (const raw of lines) {
      const t = raw.trim();
      if (!t) {
        closeList();
        continue;
      }
      if (t === "---") {
        closeList();
        html.push("<hr />");
        continue;
      }
      if (t.startsWith("### ")) {
        closeList();
        html.push("<h3>" + inlineFormat(t.slice(4)) + "</h3>");
        continue;
      }
      if (t.startsWith("## ")) {
        closeList();
        html.push("<h3>" + inlineFormat(t.slice(3)) + "</h3>");
        continue;
      }
      if (t.startsWith("> ")) {
        closeList();
        html.push("<blockquote>" + inlineFormat(t.slice(2)) + "</blockquote>");
        continue;
      }
      if (t.startsWith("- ") || t.startsWith("* ") || /^\d+\.\s+/.test(t)) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        const itemTxt = t.replace(/^([-*]|\d+\.)\s+/, "");
        html.push("<li>" + inlineFormat(itemTxt) + "</li>");
        continue;
      }
      closeList();
      html.push("<p>" + inlineFormat(t) + "</p>");
    }
    closeList();
    return html.join("");
  }

  // ==================== LIGHTBOX & MODALS ====================
  const lightboxModal = document.getElementById("lightboxModal");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxCaption = document.getElementById("lightboxCaption");
  const aiSettingsModal = document.getElementById("aiSettingsModal");

  function openLightbox(src, caption) {
    if (!src) return;
    lightboxImg.src = normalizeAssetUrl(src);
    lightboxCaption.textContent = caption || "";
    lightboxModal.classList.remove("hidden");
  }

  function closeLightbox() {
    lightboxModal.classList.add("hidden");
    lightboxImg.src = "";
  }

  function closeSettingsModal() {
    if (aiSettingsModal) aiSettingsModal.classList.add("hidden");
  }

  document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
  document.getElementById("lightboxBackdrop").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      closeLightbox();
      closeSettingsModal();
    }
  });

  document.body.addEventListener("click", function (e) {
    const target = e.target.closest("[data-lightbox]");
    if (target) {
      openLightbox(target.getAttribute("data-lightbox"), target.getAttribute("data-caption") || target.alt || "");
    }
  });

  // ==================== AI ENGINE SETTINGS MODAL ====================
  const aiSettingsBtn = document.getElementById("aiSettingsBtn");
  const webgpuStatusText = document.getElementById("webgpuStatusText");
  const enableWebgpuBtn = document.getElementById("enableWebgpuBtn");
  const customBaseUrl = document.getElementById("customBaseUrl");
  const customModelName = document.getElementById("customModelName");
  const customApiKey = document.getElementById("customApiKey");

  function refreshSettingsModalState() {
    if (!window.HaroonyNeuralAI) return;
    const cfg = window.HaroonyNeuralAI.getCustomConfig();
    if (customBaseUrl) customBaseUrl.value = cfg.baseUrl || "";
    if (customModelName) customModelName.value = cfg.model || "";
    if (customApiKey) customApiKey.value = cfg.apiKey || "";
    const status = window.HaroonyNeuralAI.getStatus ? window.HaroonyNeuralAI.getStatus() : {};
    if (webgpuStatusText) {
      if (status.webgpuReady) {
        webgpuStatusText.textContent = "Status: WebGPU Neural AI active";
      } else if (status.webgpuSupported) {
        webgpuStatusText.textContent = "Status: WebGPU supported (click to load on demand)";
      } else {
        webgpuStatusText.textContent = "Status: WebGPU not detected in this browser";
      }
    }
    if (enableWebgpuBtn) {
      enableWebgpuBtn.disabled = Boolean(status.webgpuReady || !status.webgpuSupported);
      if (status.webgpuReady) enableWebgpuBtn.textContent = "Loaded";
    }
  }

  if (aiSettingsBtn && aiSettingsModal) {
    aiSettingsBtn.addEventListener("click", function () {
      refreshSettingsModalState();
      aiSettingsModal.classList.remove("hidden");
    });
    document.getElementById("aiSettingsClose").addEventListener("click", closeSettingsModal);
    document.getElementById("aiSettingsBackdrop").addEventListener("click", closeSettingsModal);
  }

  if (enableWebgpuBtn) {
    enableWebgpuBtn.addEventListener("click", async function () {
      if (!window.HaroonyNeuralAI) return;
      enableWebgpuBtn.disabled = true;
      enableWebgpuBtn.textContent = "Loading…";
      const ok = await window.HaroonyNeuralAI.enableWebGPU((msg) => {
        if (webgpuStatusText) webgpuStatusText.textContent = msg;
      });
      if (ok) {
        enableWebgpuBtn.textContent = "Loaded";
        if (webgpuStatusText) webgpuStatusText.textContent = "Status: WebGPU Neural AI active!";
      } else {
        enableWebgpuBtn.disabled = false;
        enableWebgpuBtn.textContent = "Retry WebGPU";
      }
    });
  }

  const saveCustomCfgBtn = document.getElementById("saveCustomCfgBtn");
  if (saveCustomCfgBtn) {
    saveCustomCfgBtn.addEventListener("click", function () {
      if (!window.HaroonyNeuralAI) return;
      window.HaroonyNeuralAI.saveCustomConfig({
        baseUrl: customBaseUrl.value.trim(),
        model: customModelName.value.trim(),
        apiKey: customApiKey.value.trim(),
      });
      closeSettingsModal();
    });
  }

  const clearCustomCfgBtn = document.getElementById("clearCustomCfgBtn");
  if (clearCustomCfgBtn) {
    clearCustomCfgBtn.addEventListener("click", function () {
      if (!window.HaroonyNeuralAI) return;
      window.HaroonyNeuralAI.saveCustomConfig({ baseUrl: "", model: "", apiKey: "" });
      refreshSettingsModalState();
    });
  }

  // ==================== TAB NAVIGATION ====================
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanels = document.querySelectorAll(".tab-panel");

  function switchTab(tabName) {
    navTabs.forEach((btn) => {
      const active = btn.dataset.tab === tabName;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === "tab-" + tabName);
    });
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("brandHome").addEventListener("click", () => switchTab("tutor"));

  // ==================== QUESTION PICKER FOR HARONY MS ====================
  const msLibrarySelect = document.getElementById("msLibrarySelect");
  const questionSearchInput = document.getElementById("questionSearchInput");
  const questionSelect = document.getElementById("questionSelect");

  function renderQuestionOptions(filterText) {
    if (!questionSelect) return;
    const query = String(filterText || "").trim().toLowerCase();
    questionSelect.innerHTML = "";

    const autoOpt = document.createElement("option");
    autoOpt.value = "";
    autoOpt.textContent = query
      ? `Auto-match best question for "${filterText.trim()}"`
      : "Auto-detect best matching question (or pick one below)";
    questionSelect.appendChild(autoOpt);

    const filtered = currentBookQuestions.filter((q) => {
      if (!query) return true;
      const hay = `${q.question_id || ""} ${q.title || ""} ${q.chapter || ""}`.toLowerCase();
      return query.split(/\s+/).every((tok) => hay.includes(tok));
    });

    filtered.slice(0, 150).forEach((q, idx) => {
      const opt = document.createElement("option");
      opt.value = q.question_id;
      const vidMark = q.video_url ? " 🎥" : "";
      const titleStr = (q.title || q.question_id || "").slice(0, 62);
      opt.textContent = `${idx + 1}. ${titleStr} [${q.max_marks || 1}m]${vidMark}`;
      questionSelect.appendChild(opt);
    });
  }

  async function loadQuestionsForBook(msId) {
    currentBookQuestions = [];
    if (!msId) {
      renderQuestionOptions("");
      return;
    }
    try {
      if (serverAvailable) {
        const resp = await fetch(`api/mark-schemes/${encodeURIComponent(msId)}`);
        if (resp.ok) {
          const data = await resp.json();
          const qs = data && (data.questions || (data.mark_scheme && data.mark_scheme.questions));
          if (Array.isArray(qs)) {
            currentBookQuestions = qs;
          }
        }
      } else if (window.HaroonyBrowserEngine && window.HaroonyBrowserEngine.getMarkSchemeQuestions) {
        currentBookQuestions = await window.HaroonyBrowserEngine.getMarkSchemeQuestions(msId);
      }
    } catch (_) {}
    renderQuestionOptions(questionSearchInput ? questionSearchInput.value : "");
  }

  if (msLibrarySelect) {
    msLibrarySelect.addEventListener("change", () => loadQuestionsForBook(msLibrarySelect.value));
  }
  if (questionSearchInput) {
    questionSearchInput.addEventListener("input", () => renderQuestionOptions(questionSearchInput.value));
  }

  // ==================== INITIALIZE STATUS & MS LIST ====================
  async function loadSystemStatus() {
    let data = null;
    try {
      const resp = await fetch("api/status", { headers: { Accept: "application/json" } });
      if (resp.ok) {
        const ct = resp.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          data = await resp.json();
          if (data && data.ok) {
            serverAvailable = true;
          }
        }
      }
    } catch (_) {}

    if (!data || !data.ok) {
      serverAvailable = false;
      if (window.HaroonyBrowserEngine) {
        try {
          data = await window.HaroonyBrowserEngine.getStatus();
        } catch (err) {
          console.error("Browser engine init error:", err);
          return;
        }
      } else {
        return;
      }
    }

    const kbBadge = document.getElementById("kbBadge");
    if (kbBadge && data.notes_chunks_count) {
      kbBadge.textContent = `${data.notes_chunks_count} Chunks • ${data.video_solutions_count || 0} Videos`;
    }

    if (msLibrarySelect) {
      msLibrarySelect.innerHTML = "";
      (data.mark_schemes || []).forEach((ms) => {
        const opt = document.createElement("option");
        opt.value = ms.id;
        opt.textContent = `${ms.title} (${ms.question_count} Qs)`;
        msLibrarySelect.appendChild(opt);
      });
      if (msLibrarySelect.value) {
        await loadQuestionsForBook(msLibrarySelect.value);
      }
    }
  }

  // ==================== ASK AI CHAT ====================
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");
  const chatFileInput = document.getElementById("chatFileInput");
  const chatAttachmentBar = document.getElementById("chatAttachmentBar");
  const chatAttachmentName = document.getElementById("chatAttachmentName");
  const chatAttachmentThumb = document.getElementById("chatAttachmentThumb");

  function autoResizeChatInput() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
  }
  chatInput.addEventListener("input", autoResizeChatInput);

  chatFileInput.addEventListener("change", function () {
    if (chatFileInput.files && chatFileInput.files[0]) {
      chatAttachedFile = chatFileInput.files[0];
      chatAttachmentName.textContent = chatAttachedFile.name;
      if (chatAttachmentThumb) {
        if (chatAttachedFile.type && chatAttachedFile.type.startsWith("image/")) {
          chatAttachmentThumb.src = URL.createObjectURL(chatAttachedFile);
          chatAttachmentThumb.classList.remove("hidden");
        } else {
          chatAttachmentThumb.classList.add("hidden");
          chatAttachmentThumb.src = "";
        }
      }
      chatAttachmentBar.classList.remove("hidden");
    }
  });

  document.getElementById("removeChatAttachment").addEventListener("click", function () {
    chatAttachedFile = null;
    chatFileInput.value = "";
    if (chatAttachmentThumb) {
      chatAttachmentThumb.classList.add("hidden");
      chatAttachmentThumb.src = "";
    }
    chatAttachmentBar.classList.add("hidden");
  });

  document.getElementById("clearChatBtn").addEventListener("click", function () {
    chatHistory = [];
    chatMessages.innerHTML = `
      <div class="msg msg-ai">
        <img src="${AI_AVATAR}" alt="Harony Physics Clinic" class="msg-avatar" />
        <div class="msg-bubble">
          <div class="msg-content">
            <p>New chat ready! Ask me any <strong>Physics</strong> question, request <strong>YouTube video solutions</strong>, or attach a photo or PDF of a problem.</p>
          </div>
        </div>
      </div>`;
  });

  chatInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  function scrollChatToBottom() {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
  }

  function appendUserMessage(text, fileObj) {
    const div = document.createElement("div");
    div.className = "msg msg-user";
    let imgPreviewHtml = "";
    if (fileObj && fileObj.type && fileObj.type.startsWith("image/")) {
      const objUrl = URL.createObjectURL(fileObj);
      imgPreviewHtml = `<img src="${objUrl}" alt="Uploaded" class="msg-uploaded-img" data-lightbox="${objUrl}" data-caption="${escapeHtml(fileObj.name)}" />`;
    } else if (fileObj) {
      imgPreviewHtml = `<div style="margin-top:6px;font-size:0.82rem;opacity:0.9;">📄 ${escapeHtml(fileObj.name)}</div>`;
    }
    div.innerHTML = `
      <div class="msg-user-avatar">You</div>
      <div class="msg-bubble">
        <div class="msg-content">
          ${text ? `<p>${escapeHtml(text)}</p>` : ""}
          ${imgPreviewHtml}
        </div>
      </div>`;
    chatMessages.appendChild(div);
    scrollChatToBottom();
  }

  function createLiveAiBubble(hasFile) {
    const div = document.createElement("div");
    div.className = "msg msg-ai";
    const step1 = hasFile ? "Reading your uploaded file…" : "Solving & checking Harony Notes…";
    div.innerHTML = `
      <img src="${AI_AVATAR}" alt="Harony Physics Clinic" class="msg-avatar" />
      <div class="msg-bubble">
        <div class="thinking-box">
          <div class="thinking-dots" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
          <span class="thinking-text">${step1}</span>
        </div>
        <div class="msg-content hidden"></div>
        <div class="msg-extras"></div>
      </div>`;
    chatMessages.appendChild(div);
    scrollChatToBottom();

    const thinkingBox = div.querySelector(".thinking-box");
    const textEl = div.querySelector(".thinking-text");
    const contentEl = div.querySelector(".msg-content");
    const extrasEl = div.querySelector(".msg-extras");

    return {
      setStatus(msg) {
        if (!msg || !textEl || !div.isConnected) return;
        textEl.textContent = msg;
      },
      streamPreview(partialMd) {
        if (!div.isConnected) return;
        if (!thinkingBox.classList.contains("hidden")) {
          thinkingBox.classList.add("hidden");
          contentEl.classList.remove("hidden");
        }
        contentEl.innerHTML = renderMarkdown(partialMd);
        scrollChatToBottom();
      },
      finalize(res) {
        if (!div.isConnected) return;
        if (res.refused) div.classList.add("msg-refused");
        thinkingBox.classList.add("hidden");
        contentEl.classList.remove("hidden");
        contentEl.innerHTML = renderMarkdown(res.reply);

        let videosHtml = "";
        const vids = res.video_links || [];
        if (!res.refused && vids.length > 0) {
          videosHtml = `
            <div class="video-pills">
              ${vids
                .map(
                  (v) =>
                    `<a href="${escapeHtml(v.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">
                      🎥 Video Solution: ${escapeHtml((v.label || v.chapter || "Watch").slice(0, 52))}
                    </a>`
                )
                .join("")}
            </div>`;
        }

        let cardsHtml = "";
        const cardsWithImages = (res.notes_cards || []).filter((c) => c.image_url);
        if (!res.refused && cardsWithImages.length > 0) {
          const items = cardsWithImages
            .slice(0, 2)
            .map((c) => {
              const imgUrl = normalizeAssetUrl(c.image_url);
              return `
                <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Page ${c.page || ""} — ${escapeHtml(c.topic)}">
                  <img src="${escapeHtml(imgUrl)}" alt="Notes Page ${c.page || ""}" loading="lazy" />
                  <div class="mini-note-card-body">
                    <strong>Page ${c.page || ""} &bull; ${escapeHtml(c.topic)}</strong>
                    <span>Click to zoom</span>
                  </div>
                </div>`;
            })
            .join("");
          cardsHtml = `
            <div class="msg-notes-strip">
              <div class="msg-notes-label">Related Harony Notes</div>
              <div class="msg-notes-cards">${items}</div>
            </div>`;
        }

        extrasEl.innerHTML = videosHtml + cardsHtml;
        scrollChatToBottom();
      },
    };
  }

  async function enhanceServerReplyWithNeural(userText, data, liveBubble) {
    if (!window.HaroonyNeuralAI || !window.HaroonyNeuralAI.isReady || !window.HaroonyNeuralAI.isReady()) return data;
    if (data.refused || data.category !== "physics" || data.neural_generated) return data;
    try {
      const sysPrompt =
        "You are Haroony AI, Mr. Haroony's expert, friendly IGCSE & A-Level Physics tutor from Harony Physics Clinic.\n" +
        "Answer ONLY Physics questions directly, clearly, and accurately. Bold key Cambridge IGCSE 0625 marking phrases.";

      const msgs = [{ role: "system", content: sysPrompt }];
      for (const h of chatHistory.slice(-4)) {
        msgs.push({ role: h.role, content: String(h.content).slice(0, 700) });
      }
      let usr = userText || "Please solve/explain this uploaded Physics problem.";
      if (data.ocr_text) usr += `\n\nExtracted text from uploaded file:\n${data.ocr_text.slice(0, 1400)}`;
      if (data.rag_context) usr += `\n\n---\nReference Context:\n${data.rag_context.slice(0, 1800)}`;
      msgs.push({ role: "user", content: usr });

      const genText = await window.HaroonyNeuralAI.generate(msgs, {
        maxTokens: 560,
        temperature: 0.2,
        onProgress: (st) => liveBubble.setStatus(st),
        onToken: (full) => liveBubble.streamPreview(full),
      });
      if (genText) {
        let finalMd = genText;
        if (data.ocr_text && !finalMd.includes("Read from")) {
          const prev = data.ocr_text.slice(0, 240) + (data.ocr_text.length > 240 ? "…" : "");
          finalMd = `**📷 Read from your uploaded file:**\n> ${prev.replace(/\n/g, "\n> ")}\n\n${finalMd}`;
        }
        data.reply = finalMd;
      }
    } catch (_) {}
    return data;
  }

  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    const fileToSend = chatAttachedFile;
    if (!text && !fileToSend) return;

    appendUserMessage(text, fileToSend);
    chatInput.value = "";
    chatInput.style.height = "auto";
    chatAttachedFile = null;
    chatFileInput.value = "";
    if (chatAttachmentThumb) {
      chatAttachmentThumb.classList.add("hidden");
      chatAttachmentThumb.src = "";
    }
    chatAttachmentBar.classList.add("hidden");

    chatSendBtn.disabled = true;
    const liveBubble = createLiveAiBubble(Boolean(fileToSend));

    try {
      if (serverAvailable === null) await loadSystemStatus();

      let data;
      if (serverAvailable) {
        const fd = new FormData();
        fd.append("message", text);
        fd.append("history", JSON.stringify(chatHistory.slice(-6)));
        if (fileToSend) fd.append("file", fileToSend);

        const resp = await fetch("api/chat", { method: "POST", body: fd });
        const raw = await resp.json();
        if (!resp.ok) {
          data = {
            refused: true,
            reply: raw.detail || "Something went wrong. Please try again.",
            notes_cards: [],
          };
        } else {
          data = await enhanceServerReplyWithNeural(text, raw, liveBubble);
        }
      } else {
        data = await window.HaroonyBrowserEngine.chat(
          text,
          chatHistory.slice(-6),
          fileToSend,
          {
            onProgress: (st) => liveBubble.setStatus(st),
            onToken: (full) => liveBubble.streamPreview(full),
          }
        );
      }

      liveBubble.finalize(data);

      if (!data.refused) {
        const userHist = data.ocr_text
          ? `${text}\n[Uploaded: ${data.ocr_text.slice(0, 300)}]`.trim()
          : text || "[Uploaded Physics file]";
        chatHistory.push({ role: "user", content: userHist });
        chatHistory.push({ role: "assistant", content: data.reply });
      }
    } catch (err) {
      liveBubble.finalize({
        refused: true,
        reply: "Could not process your question right now. Please try again.",
        notes_cards: [],
      });
    } finally {
      chatSendBtn.disabled = false;
    }
  });

  // ==================== CHECK ANSWER & MARK SCHEME ====================
  const studentFilesInput = document.getElementById("studentFilesInput");
  const studentFilesList = document.getElementById("studentFilesList");
  const studentDropzone = document.getElementById("studentDropzone");
  const msFileInput = document.getElementById("msFileInput");
  const msFilePill = document.getElementById("msFilePill");
  const clearGraderBtn = document.getElementById("clearGraderBtn");

  function renderStudentFilePills() {
    studentFilesList.innerHTML = graderStudentFiles
      .map((f, i) => {
        if (f.type && f.type.startsWith("image/")) {
          const url = URL.createObjectURL(f);
          return `<span class="file-thumb-pill"><img src="${url}" alt="" data-lightbox="${url}" data-caption="${escapeHtml(f.name)}" /><span>${escapeHtml(f.name)}</span><button type="button" class="btn-icon" data-rm-stu="${i}" aria-label="Remove">&times;</button></span>`;
        }
        return `<span class="file-pill">📄 ${escapeHtml(f.name)} <button type="button" class="btn-icon" data-rm-stu="${i}" aria-label="Remove">&times;</button></span>`;
      })
      .join("");
  }

  if (clearGraderBtn) {
    clearGraderBtn.addEventListener("click", function () {
      graderStudentFiles = [];
      studentFilesInput.value = "";
      renderStudentFilePills();
      document.getElementById("studentTextInput").value = "";
      const area = document.getElementById("gradeResultArea");
      area.classList.add("hidden");
      area.innerHTML = "";
    });
  }

  studentFilesList.addEventListener("click", function (e) {
    const btn = e.target.closest("[data-rm-stu]");
    if (!btn) return;
    graderStudentFiles.splice(Number(btn.dataset.rmStu), 1);
    renderStudentFilePills();
  });

  studentFilesInput.addEventListener("change", function () {
    if (studentFilesInput.files && studentFilesInput.files.length) {
      for (const f of studentFilesInput.files) {
        graderStudentFiles.push(f);
      }
      renderStudentFilePills();
    }
  });

  ["dragenter", "dragover"].forEach((ev) => {
    studentDropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      studentDropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    studentDropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      studentDropzone.classList.remove("dragover");
    });
  });
  studentDropzone.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files) {
      for (const f of e.dataTransfer.files) {
        graderStudentFiles.push(f);
      }
      renderStudentFilePills();
    }
  });

  msFileInput.addEventListener("change", function () {
    if (msFileInput.files && msFileInput.files[0]) {
      graderMsFile = msFileInput.files[0];
      msFilePill.innerHTML = `<span class="file-pill">📄 ${escapeHtml(graderMsFile.name)}</span>`;
    }
  });

  document.querySelectorAll(".ms-mode-btn").forEach((btn) => {
    btn.addEventListener("click", function () {
      activeMsMode = btn.dataset.msMode;
      document.querySelectorAll(".ms-mode-btn").forEach((b) => b.classList.toggle("active", b === btn));
      ["auto", "library", "upload", "paste"].forEach((m) => {
        document.getElementById("ms-pane-" + m).classList.toggle("hidden", m !== activeMsMode);
      });
    });
  });

  function showGradingLoader() {
    const area = document.getElementById("gradeResultArea");
    area.classList.remove("hidden");
    area.innerHTML = `
      <div class="grading-loader-card">
        <div class="thinking-box" style="justify-content:center;">
          <div class="thinking-dots" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
          <span class="thinking-text" id="gradeLoaderText">Reading your Physics answer…</span>
        </div>
        <div class="loader-bar-track">
          <div class="loader-bar-fill"></div>
        </div>
      </div>`;
    area.scrollIntoView({ behavior: "smooth", block: "nearest" });

    return {
      setStatus(msg) {
        const el = document.getElementById("gradeLoaderText");
        if (el && msg) el.textContent = msg;
      },
      stop() {},
    };
  }

  function gradeBadgeClass(grade) {
    const g = String(grade || "").toUpperCase();
    if (g.startsWith("A")) return "grade-badge-green";
    if (g === "B" || g === "C") return "grade-badge-blue";
    if (g === "D" || g === "E") return "grade-badge-amber";
    return "grade-badge-red";
  }

  function renderGradeReport(res) {
    const area = document.getElementById("gradeResultArea");
    area.classList.remove("hidden");

    if (!res.ok) {
      area.innerHTML = `
        <div class="panel-card" style="border-color:#fecaca;background:#fef2f2;animation:fadeSlideUp 0.25s var(--ease-out);">
          <h3 style="color:#b91c1c;margin-bottom:6px;">Need More Context to Grade</h3>
          <div>${renderMarkdown(res.error || res.detail || "Please provide a Physics answer.")}</div>
          ${
            res.extracted_student_text
              ? `<details class="ocr-details" style="margin-top:10px;"><summary>View what was read from your submission</summary><pre style="white-space:pre-wrap;background:#fff;padding:10px;border-radius:8px;border:1px solid #fecaca;font-size:0.82rem;margin:8px 0 0;">${escapeHtml(res.extracted_student_text)}</pre></details>`
              : ""
          }
        </div>`;
      area.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }

    const feedbackHtml = res.examiner_feedback
      ? `<div class="examiner-feedback-card"><strong>AI Examiner Feedback:</strong> ${renderMarkdown(res.examiner_feedback)}</div>`
      : "";

    const questionsHtml = (res.questions || [])
      .map((q) => {
        const rows = (q.points || [])
          .map(
            (pt) => `
            <div class="mark-row ${escapeHtml(pt.status)}">
              <strong>[${escapeHtml(pt.code)}]</strong> ${renderMarkdown(pt.comment)}
            </div>`
          )
          .join("");
        const vidBtn = q.video_url
          ? `<a href="${escapeHtml(q.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">🎥 Watch Video Solution</a>`
          : "";
        return `
          <div class="q-breakdown-card">
            <div class="q-breakdown-header">
              <strong>${escapeHtml(q.title || "Question " + q.question_id)}</strong>
              <span class="q-score-pill">${q.awarded_marks} / ${q.max_marks} marks</span>
            </div>
            ${rows}
            ${
              q.model_answer
                ? `<div style="margin-top:10px;font-size:0.84rem;color:#475569;">
                    <strong>Model Answer:</strong> <code>${escapeHtml(q.model_answer)}</code>
                   </div>`
                : ""
            }
            ${vidBtn ? `<div class="video-pills">${vidBtn}</div>` : ""}
          </div>`;
      })
      .join("");

    const extraVideos = (res.related_videos || []).filter(
      (v) => v && v.video_url && !(res.questions || []).some((q) => q.video_url === v.video_url)
    );
    const videosBlockHtml =
      extraVideos.length > 0
        ? `<div class="panel-card" style="margin-top:14px;animation:fadeSlideUp 0.3s var(--ease-out);">
            <h3 style="margin-bottom:8px;">🎥 Related Harony YouTube Video Solutions</h3>
            <div class="video-pills">
              ${extraVideos
                .map(
                  (v) =>
                    `<a href="${escapeHtml(v.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">
                      🎥 ${escapeHtml((v.label || v.chapter || "Watch Video Solution").slice(0, 56))}
                    </a>`
                )
                .join("")}
            </div>
          </div>`
        : "";

    const noteRefs = res.notes_references || [];
    let notesHtml = "";
    if (noteRefs.length > 0) {
      notesHtml = `
        <div class="panel-card revision-notes-card" style="margin-top:14px;animation:fadeSlideUp 0.32s var(--ease-out);">
          <h3 style="margin-bottom:10px;">📖 Part to Revise from Harony Notes</h3>
          <div class="revision-list">
            ${noteRefs
              .map((n) => {
                const pts = (n.revision_points || []).filter(Boolean);
                const ptsHtml = pts.length
                  ? `<ul class="revision-bullets">${pts.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`
                  : n.snippet
                  ? `<p class="revision-snippet">${escapeHtml(n.snippet)}</p>`
                  : "";
                const imgHtml = n.image_url
                  ? (() => {
                      const imgUrl = normalizeAssetUrl(n.image_url);
                      return `
                      <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Page ${n.page || ""} — ${escapeHtml(n.topic)}">
                        <img src="${escapeHtml(imgUrl)}" alt="Page ${n.page || ""}" />
                        <div class="mini-note-card-body">
                          <strong>Page ${n.page || ""} &bull; ${escapeHtml(n.topic)}</strong>
                          <span>Click to zoom slide</span>
                        </div>
                      </div>`;
                    })()
                  : "";
                return `
                  <div class="revision-item">
                    <div class="revision-item-text">
                      <div class="revision-badge">${escapeHtml(n.unit || "Harony Notes")} &bull; Page ${n.page || "?"}</div>
                      <h4>${escapeHtml(n.topic || "Physics Revision")}</h4>
                      ${ptsHtml}
                    </div>
                    ${imgHtml}
                  </div>`;
              })
              .join("")}
          </div>
        </div>`;
    }

    area.innerHTML = `
      <div class="score-summary-card">
        <div>
          <div style="font-size:0.78rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.03em;">
            ${escapeHtml(res.mark_scheme_title)}
          </div>
          <h3 style="margin:3px 0 0;">Marking Result</h3>
        </div>
        <div class="score-pill-big">
          <span class="score-numbers">${res.awarded_marks} / ${res.max_marks}</span>
          <span style="font-weight:600;color:#64748b;">(${res.percentage}%)</span>
          <span class="score-grade-badge ${gradeBadgeClass(res.grade)}">Grade ${escapeHtml(res.grade)}</span>
        </div>
      </div>

      ${feedbackHtml}
      ${questionsHtml}
      ${videosBlockHtml}
      ${notesHtml}

      <details class="ocr-details">
        <summary>View what was read from your submission</summary>
        <pre style="white-space:pre-wrap;background:#f8fafc;padding:10px;border-radius:8px;border:1px solid #e2e8f0;font-size:0.82rem;margin:10px 0 0;">${escapeHtml(res.extracted_student_text || "")}</pre>
      </details>
    `;
    area.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("gradeForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const btn = document.getElementById("gradeSubmitBtn");
    btn.disabled = true;
    btn.textContent = "Checking…";

    const loader = showGradingLoader();

    let stuText = document.getElementById("studentTextInput").value.trim();
    const msId = activeMsMode === "library" && msLibrarySelect ? msLibrarySelect.value : "";
    const pickedQId = questionSelect ? questionSelect.value.trim() : "";
    const searchTxt = questionSearchInput ? questionSearchInput.value.trim() : "";
    const qId =
      activeMsMode === "library"
        ? pickedQId || searchTxt
        : activeMsMode === "upload"
        ? document.getElementById("questionIdUploadInput").value.trim()
        : "";
    const msText = activeMsMode === "paste" ? document.getElementById("msTextInput").value.trim() : "";

    try {
      if (serverAvailable === null) await loadSystemStatus();

      let data;
      if (serverAvailable) {
        const fd = new FormData();
        fd.append("student_text", stuText);
        graderStudentFiles.forEach((f) => fd.append("student_files", f));
        if (activeMsMode === "library") {
          fd.append("ms_id", msId);
          fd.append("question_id", qId);
        } else if (activeMsMode === "upload") {
          if (graderMsFile) fd.append("ms_file", graderMsFile);
          fd.append("question_id", qId);
        } else if (activeMsMode === "paste") {
          fd.append("ms_text", msText);
        }
        const resp = await fetch("api/grade", { method: "POST", body: fd });
        data = await resp.json();
      } else {
        if (graderStudentFiles.length > 0 && window.HaroonyNeuralAI) {
          const extractedParts = [stuText];
          for (const f of graderStudentFiles) {
            const txt = await window.HaroonyNeuralAI.extractDocumentText(f, (st) => loader.setStatus(st));
            if (txt) extractedParts.push(txt);
          }
          stuText = extractedParts.filter(Boolean).join("\n\n");
        }

        data = await window.HaroonyBrowserEngine.grade({
          studentText: stuText,
          studentFile: null,
          msId: msId,
          msText: msText,
          msFile: activeMsMode === "upload" ? graderMsFile : null,
          questionId: qId,
          onProgress: (st) => loader.setStatus(st),
        });
      }

      loader.stop();
      renderGradeReport(data);
    } catch (err) {
      loader.stop();
      renderGradeReport({ ok: false, error: "Error checking submission. Please try again." });
    } finally {
      btn.disabled = false;
      btn.textContent = "Check My Answer";
    }
  });

  loadSystemStatus();
})();
