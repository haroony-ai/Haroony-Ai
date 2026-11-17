/* Haroony AI — Clean Dual-Mode Frontend (Ask AI + Mark My Answer, Server + Standalone GitHub Pages) */
(function () {
  "use strict";

  const AI_AVATAR = "static/assets/harony_avatar.webp";

  let serverAvailable = null;
  let chatHistory = [];
  let chatAttachedFile = null;
  let activeTab = "tutor";

  // Grader state
  let studentFiles = [];
  let msFile = null;
  let activeMsMode = "ai"; // ai | saved | upload | paste
  let loadedMarkSchemes = [];
  let currentQuestionOptions = [];

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

  function extractYoutubeId(url) {
    const s = String(url || "");
    const m = s.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/))([A-Za-z0-9_-]{11})/);
    return m ? m[1] : "";
  }

  function toWatchUrl(url) {
    const id = extractYoutubeId(url);
    return id ? `https://www.youtube.com/watch?v=${id}` : String(url || "");
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
        /(https:\/\/(?:youtu\.be\/|www\.youtube\.com\/watch\?v=)[A-Za-z0-9_\-]+)/g,
        function (m) {
          const watch = toWatchUrl(m);
          return `<a href="${watch}" target="_blank" rel="noopener noreferrer" class="video-pill-link" data-video-url="${watch}" data-video-title="Harony Video Solution">🎥 Watch Video</a>`;
        }
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

  // ==================== NAVIGATION TABS ====================
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanels = document.querySelectorAll(".tab-panel");

  function switchTab(tabName) {
    activeTab = tabName;
    navTabs.forEach((b) => b.classList.toggle("active", b.getAttribute("data-tab") === tabName));
    tabPanels.forEach((p) => p.classList.toggle("active", p.id === `tab-${tabName}`));
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab")));
  });
  document.getElementById("brandHome").addEventListener("click", () => switchTab("tutor"));

  // ==================== LIGHTBOX & VIDEO PLAYER MODAL ====================
  const lightboxModal = document.getElementById("lightboxModal");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxCaption = document.getElementById("lightboxCaption");
  const videoModal = document.getElementById("videoModal");
  const videoIframe = document.getElementById("videoIframe");
  const videoModalTitle = document.getElementById("videoModalTitle");
  const openVideoExternalLink = document.getElementById("openVideoExternalLink");
  const copyVideoLinkBtn = document.getElementById("copyVideoLinkBtn");
  let activeVideoUrl = "";

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

  function openVideoModal(url, title) {
    const watch = toWatchUrl(url);
    const id = extractYoutubeId(watch);
    activeVideoUrl = watch;
    if (videoModalTitle) videoModalTitle.textContent = title || "Harony Video Solution";
    if (openVideoExternalLink) openVideoExternalLink.href = watch;
    if (copyVideoLinkBtn) copyVideoLinkBtn.textContent = "Copy YouTube Link";
    if (videoIframe) {
      videoIframe.src = id ? `https://www.youtube.com/embed/${id}?autoplay=1&rel=0` : watch;
    }
    if (videoModal) videoModal.classList.remove("hidden");
  }

  function closeVideoModal() {
    if (videoModal) videoModal.classList.add("hidden");
    if (videoIframe) videoIframe.src = "";
  }

  document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
  document.getElementById("lightboxBackdrop").addEventListener("click", closeLightbox);
  if (videoModal) {
    document.getElementById("videoModalClose").addEventListener("click", closeVideoModal);
    document.getElementById("videoBackdrop").addEventListener("click", closeVideoModal);
  }
  if (copyVideoLinkBtn) {
    copyVideoLinkBtn.addEventListener("click", async function () {
      if (!activeVideoUrl) return;
      try {
        await navigator.clipboard.writeText(activeVideoUrl);
        copyVideoLinkBtn.textContent = "Copied!";
      } catch (_) {
        copyVideoLinkBtn.textContent = activeVideoUrl;
      }
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      closeLightbox();
      closeVideoModal();
    }
  });

  document.body.addEventListener("click", function (e) {
    const target = e.target.closest("[data-lightbox]");
    if (target) {
      openLightbox(target.getAttribute("data-lightbox"), target.getAttribute("data-caption") || target.alt || "");
      return;
    }
    const vidLink = e.target.closest("a.video-pill-link");
    if (vidLink && !vidLink.closest("#videoModal")) {
      const url = vidLink.getAttribute("data-video-url") || vidLink.getAttribute("href");
      const title = vidLink.getAttribute("data-video-title") || vidLink.textContent.replace(/^🎥\s*/, "").trim();
      const inIframe = window.self !== window.top;
      if (inIframe) {
        e.preventDefault();
        openVideoModal(url, title);
      }
    }
    const quickChip = e.target.closest(".quick-chip");
    if (quickChip) {
      const prompt = quickChip.getAttribute("data-prompt");
      if (prompt && chatInput) {
        chatInput.value = prompt;
        autoResizeChatInput();
        chatForm.requestSubmit();
      }
    }
  });

  // ==================== INITIALIZE STATUS & MARK SCHEMES ====================
  async function loadSystemStatus() {
    let statusData = null;
    try {
      const resp = await fetch("api/status", { headers: { Accept: "application/json" } });
      if (resp.ok) {
        const ct = resp.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const data = await resp.json();
          if (data && data.ok) {
            serverAvailable = true;
            statusData = data;
          }
        }
      }
    } catch (_) {}

    if (!statusData) {
      serverAvailable = false;
      if (window.HaroonyBrowserEngine) {
        try {
          statusData = await window.HaroonyBrowserEngine.getStatus();
        } catch (err) {
          console.error("Browser engine init error:", err);
        }
      }
    }

    if (statusData && Array.isArray(statusData.mark_schemes)) {
      loadedMarkSchemes = statusData.mark_schemes;
      populateSavedMarkSchemes(loadedMarkSchemes);
    }
  }

  // ==================== ASK AI CHAT + CTRL+V SCREENSHOT PASTE ====================
  const chatCard = document.getElementById("chatCard");
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");
  const chatFileInput = document.getElementById("chatFileInput");
  const chatAttachmentBar = document.getElementById("chatAttachmentBar");
  const chatAttachmentName = document.getElementById("chatAttachmentName");
  const chatAttachmentThumb = document.getElementById("chatAttachmentThumb");

  function attachFileToChat(fileObj) {
    if (!fileObj) return;
    chatAttachedFile = fileObj;
    chatAttachmentName.textContent = fileObj.name || "screenshot.png";
    if (chatAttachmentThumb) {
      if (fileObj.type && fileObj.type.startsWith("image/")) {
        chatAttachmentThumb.src = URL.createObjectURL(fileObj);
        chatAttachmentThumb.classList.remove("hidden");
      } else {
        chatAttachmentThumb.classList.add("hidden");
        chatAttachmentThumb.src = "";
      }
    }
    chatAttachmentBar.classList.remove("hidden");
  }

  function clearAttachedChatFile() {
    chatAttachedFile = null;
    chatFileInput.value = "";
    if (chatAttachmentThumb) {
      chatAttachmentThumb.classList.add("hidden");
      chatAttachmentThumb.src = "";
    }
    chatAttachmentBar.classList.add("hidden");
  }

  function autoResizeChatInput() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
  }
  chatInput.addEventListener("input", autoResizeChatInput);

  chatFileInput.addEventListener("change", function () {
    if (chatFileInput.files && chatFileInput.files[0]) {
      attachFileToChat(chatFileInput.files[0]);
    }
  });

  document.getElementById("removeChatAttachment").addEventListener("click", clearAttachedChatFile);

  // Allow Ctrl+V / Cmd+V pasting screenshots directly into active tab (Ask AI OR Mark My Answer)
  function extractClipboardImageOrPdf(e) {
    const cd = e.clipboardData || (window.clipboardData && window.clipboardData);
    if (!cd) return null;
    if (cd.items && cd.items.length) {
      for (const item of cd.items) {
        if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) {
            const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
            return new File([blob], `screenshot_${Date.now()}.${ext}`, { type: blob.type });
          }
        }
      }
    }
    if (cd.files && cd.files.length) {
      const f = cd.files[0];
      if (f && ((f.type && f.type.startsWith("image/")) || /\.pdf$/i.test(f.name || ""))) {
        return f;
      }
    }
    return null;
  }

  function handleGlobalPaste(e) {
    const file = extractClipboardImageOrPdf(e);
    if (!file) return;
    e.preventDefault();
    if (activeTab === "grader") {
      addStudentFiles([file]);
    } else {
      attachFileToChat(file);
      chatInput.focus();
    }
  }

  chatInput.addEventListener("paste", function (e) {
    const file = extractClipboardImageOrPdf(e);
    if (file) {
      e.preventDefault();
      attachFileToChat(file);
    }
  });
  document.addEventListener("paste", function (e) {
    if (e.target === chatInput) return;
    handleGlobalPaste(e);
  });

  if (chatCard) {
    ["dragenter", "dragover"].forEach((ev) => {
      chatCard.addEventListener(ev, (e) => {
        e.preventDefault();
        chatCard.classList.add("chat-dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      chatCard.addEventListener(ev, (e) => {
        e.preventDefault();
        chatCard.classList.remove("chat-dragover");
      });
    });
    chatCard.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        attachFileToChat(e.dataTransfer.files[0]);
        chatInput.focus();
      }
    });
  }

  document.getElementById("clearChatBtn").addEventListener("click", function () {
    chatHistory = [];
    clearAttachedChatFile();
    chatMessages.innerHTML = `
      <div class="msg msg-ai">
        <img src="${AI_AVATAR}" alt="Harony Physics Clinic" class="msg-avatar" />
        <div class="msg-bubble">
          <div class="msg-content">
            <p>New chat ready! Ask me any <strong>Physics</strong> question, request <strong>YouTube video solutions</strong>, or press <strong><code>Ctrl+V</code></strong> to paste a screenshot.</p>
            <div class="quick-prompts">
              <button type="button" class="quick-chip" data-prompt="Explain motion graphs (distance-time and speed-time)">Motion Graphs</button>
              <button type="button" class="quick-chip" data-prompt="Explain Ohm's law and filament lamp I-V graph">Ohm's Law</button>
              <button type="button" class="quick-chip" data-prompt="Explain convection in fluids">Convection</button>
              <button type="button" class="quick-chip" data-prompt="Explain A.C. generator 1st and 2nd half cycle">A.C. Generator</button>
              <button type="button" class="quick-chip" data-prompt="YouTube videos for diffraction and waves">Wave Videos</button>
            </div>
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
    const step1 = hasFile ? "Reading your attached image/PDF…" : "Thinking through the physics…";
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
                .map((v) => {
                  const watch = toWatchUrl(v.video_url);
                  const lbl = (v.label || v.chapter || "Watch").slice(0, 56);
                  return `<a href="${escapeHtml(watch)}" target="_blank" rel="noopener noreferrer" class="video-pill-link" data-video-url="${escapeHtml(watch)}" data-video-title="${escapeHtml(lbl)}">
                      🎥 Video Solution: ${escapeHtml(lbl)}
                    </a>`;
                })
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

  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    const fileToSend = chatAttachedFile;
    if (!text && !fileToSend) return;

    appendUserMessage(text, fileToSend);
    chatInput.value = "";
    chatInput.style.height = "auto";
    clearAttachedChatFile();

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
        data = !resp.ok
          ? { refused: true, reply: raw.detail || "Something went wrong. Please try again.", notes_cards: [] }
          : raw;
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

  // ==================== TAB 2: MARK MY ANSWER ====================
  const studentDropzone = document.getElementById("studentDropzone");
  const studentFileInput = document.getElementById("studentFileInput");
  const studentFilePills = document.getElementById("studentFilePills");
  const studentTextInput = document.getElementById("studentTextInput");
  const clearGraderBtn = document.getElementById("clearGraderBtn");

  const msModeBtns = document.querySelectorAll(".ms-mode-btn");
  const savedMsSelect = document.getElementById("savedMsSelect");
  const questionSearchInput = document.getElementById("questionSearchInput");
  const questionSelect = document.getElementById("questionSelect");
  const msDropzone = document.getElementById("msDropzone");
  const msFileInput = document.getElementById("msFileInput");
  const msFilePill = document.getElementById("msFilePill");
  const msTextInput = document.getElementById("msTextInput");
  const gradeSubmitBtn = document.getElementById("gradeSubmitBtn");
  const gradeResultArea = document.getElementById("gradeResultArea");

  function addStudentFiles(fileList) {
    for (const f of fileList) {
      if (f && studentFiles.length < 6) {
        studentFiles.push(f);
      }
    }
    renderStudentFilePills();
  }

  function renderStudentFilePills() {
    if (!studentFilePills) return;
    studentFilePills.innerHTML = studentFiles
      .map((f, idx) => {
        if (f.type && f.type.startsWith("image/")) {
          const url = URL.createObjectURL(f);
          return `<span class="file-thumb-pill">
              <img src="${url}" alt="${escapeHtml(f.name)}" data-lightbox="${url}" data-caption="${escapeHtml(f.name)}" />
              <span>${escapeHtml(f.name)}</span>
              <button type="button" class="btn-icon" data-rm-stu="${idx}" aria-label="Remove">&times;</button>
            </span>`;
        }
        return `<span class="file-pill">📄 ${escapeHtml(f.name)} <button type="button" class="btn-icon" data-rm-stu="${idx}" aria-label="Remove">&times;</button></span>`;
      })
      .join("");
  }

  if (studentFilePills) {
    studentFilePills.addEventListener("click", function (e) {
      const rm = e.target.closest("[data-rm-stu]");
      if (rm) {
        const idx = Number(rm.getAttribute("data-rm-stu"));
        studentFiles.splice(idx, 1);
        renderStudentFilePills();
      }
    });
  }

  if (studentFileInput) {
    studentFileInput.addEventListener("change", function () {
      if (studentFileInput.files && studentFileInput.files.length) {
        addStudentFiles(studentFileInput.files);
        studentFileInput.value = "";
      }
    });
  }

  if (studentDropzone) {
    ["dragenter", "dragover"].forEach((ev) =>
      studentDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        studentDropzone.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      studentDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        studentDropzone.classList.remove("dragover");
      })
    );
    studentDropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files) {
        addStudentFiles(e.dataTransfer.files);
      }
    });
  }

  if (clearGraderBtn) {
    clearGraderBtn.addEventListener("click", function () {
      studentFiles = [];
      renderStudentFilePills();
      if (studentTextInput) studentTextInput.value = "";
      if (gradeResultArea) {
        gradeResultArea.innerHTML = "";
        gradeResultArea.classList.add("hidden");
      }
    });
  }

  msModeBtns.forEach((btn) => {
    btn.addEventListener("click", function () {
      activeMsMode = btn.getAttribute("data-ms-mode") || "ai";
      msModeBtns.forEach((b) => b.classList.toggle("active", b === btn));
      ["ai", "saved", "upload", "paste"].forEach((m) => {
        const pane = document.getElementById(`ms-pane-${m}`);
        if (pane) pane.classList.toggle("hidden", m !== activeMsMode);
      });
    });
  });

  function populateSavedMarkSchemes(list) {
    if (!savedMsSelect) return;
    if (!list || !list.length) {
      savedMsSelect.innerHTML = '<option value="">No mark schemes loaded</option>';
      return;
    }
    savedMsSelect.innerHTML = list
      .map((m) => {
        const count = m.question_count || (m.question_ids ? m.question_ids.length : 0);
        return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.title)} (${count} parts)</option>`;
      })
      .join("");
    loadQuestionsForBook(savedMsSelect.value);
  }

  function renderQuestionDropdown(filterTerm) {
    if (!questionSelect) return;
    const term = (filterTerm || "").trim().toLowerCase();
    const filtered = term
      ? currentQuestionOptions.filter((q) => q.label.toLowerCase().includes(term))
      : currentQuestionOptions;

    const opts = ['<option value="">Auto-detect best matching question</option>'];
    for (const q of filtered.slice(0, 250)) {
      opts.push(`<option value="${escapeHtml(q.id)}">${escapeHtml(q.label)}</option>`);
    }
    questionSelect.innerHTML = opts.join("");
  }

  async function loadQuestionsForBook(msId) {
    if (!msId || !questionSelect) return;
    currentQuestionOptions = [];
    questionSelect.innerHTML = '<option value="">Loading questions…</option>';

    try {
      let msObj = null;
      if (serverAvailable) {
        const r = await fetch(`api/mark-schemes/${encodeURIComponent(msId)}`);
        if (r.ok) {
          const d = await r.json();
          msObj = d.mark_scheme;
        }
      } else if (window.HaroonyBrowserEngine && window.HaroonyBrowserEngine.getMarkSchemeById) {
        msObj = await window.HaroonyBrowserEngine.getMarkSchemeById(msId);
      }

      if (msObj && Array.isArray(msObj.questions)) {
        currentQuestionOptions = msObj.questions.map((q) => {
          const qid = q.question_id || "";
          const topic = q.topic || q.chapter || "";
          const marks = q.total_marks || 0;
          const preview = (q.summary_answer || "").slice(0, 48);
          return {
            id: qid,
            label: `${qid}${topic ? " — " + topic : ""} [${marks}m]${preview ? ": " + preview : ""}`,
          };
        });
      }
    } catch (_) {}

    renderQuestionDropdown(questionSearchInput ? questionSearchInput.value : "");
  }

  if (savedMsSelect) {
    savedMsSelect.addEventListener("change", () => loadQuestionsForBook(savedMsSelect.value));
  }
  if (questionSearchInput) {
    questionSearchInput.addEventListener("input", () => renderQuestionDropdown(questionSearchInput.value));
  }

  function renderMsFilePill() {
    if (!msFilePill) return;
    msFilePill.innerHTML = msFile
      ? `<span class="file-pill">📄 ${escapeHtml(msFile.name)} <button type="button" class="btn-icon" id="rmMsFileBtn" aria-label="Remove">&times;</button></span>`
      : "";
    const rmBtn = document.getElementById("rmMsFileBtn");
    if (rmBtn) {
      rmBtn.addEventListener("click", () => {
        msFile = null;
        if (msFileInput) msFileInput.value = "";
        renderMsFilePill();
      });
    }
  }

  if (msFileInput) {
    msFileInput.addEventListener("change", function () {
      if (msFileInput.files && msFileInput.files[0]) {
        msFile = msFileInput.files[0];
        renderMsFilePill();
      }
    });
  }
  if (msDropzone) {
    ["dragenter", "dragover"].forEach((ev) =>
      msDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        msDropzone.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      msDropzone.addEventListener(ev, (e) => {
        e.preventDefault();
        msDropzone.classList.remove("dragover");
      })
    );
    msDropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        msFile = e.dataTransfer.files[0];
        renderMsFilePill();
      }
    });
  }

  function gradeBadgeClass(pct) {
    if (pct >= 80) return "grade-badge-green";
    if (pct >= 60) return "grade-badge-blue";
    if (pct >= 40) return "grade-badge-amber";
    return "grade-badge-red";
  }

  function renderGradeResult(res) {
    if (!gradeResultArea) return;
    gradeResultArea.classList.remove("hidden");

    if (!res || !res.ok || res.refused) {
      const msg = (res && (res.error || res.detail)) || "Could not grade this submission yet.";
      gradeResultArea.innerHTML = `
        <div class="q-breakdown-card" style="border-color:#fecaca;background:#fff8f8;">
          <h3 style="margin-top:0;color:#b91c1c;">Could Not Complete Marking</h3>
          <div>${renderMarkdown(msg)}</div>
          ${
            res && res.extracted_student_text
              ? `<details class="ocr-details"><summary>Text read from your submission</summary><pre style="white-space:pre-wrap;margin:8px 0 0;">${escapeHtml(res.extracted_student_text)}</pre></details>`
              : ""
          }
        </div>`;
      gradeResultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }

    const pct = Number(res.percentage || 0);
    const badgeCls = gradeBadgeClass(pct);
    const qs = res.questions || [];

    const qsHtml = qs
      .map((q) => {
        const ptsHtml = (q.points || [])
          .map((pt) => {
            const st = pt.status || (pt.marks_awarded > 0 ? "awarded" : "missed");
            return `<div class="mark-row ${escapeHtml(st)}">
              <strong>[${escapeHtml(pt.code || "B1")}] (${pt.marks_awarded || 0}/${pt.max_marks || 1})</strong> — ${renderMarkdown(pt.comment || pt.expected || "")}
            </div>`;
          })
          .join("");

        const vidBtn = q.video_url
          ? `<a href="${escapeHtml(toWatchUrl(q.video_url))}" target="_blank" rel="noopener noreferrer" class="video-pill-link" data-video-url="${escapeHtml(toWatchUrl(q.video_url))}" data-video-title="${escapeHtml(q.question_id || "Video Solution")}">🎥 Watch Video Solution</a>`
          : "";

        return `<div class="q-breakdown-card">
          <div class="q-breakdown-header">
            <div>
              <strong>${escapeHtml(q.question_id || "Question")}</strong>
              ${q.title ? `<span class="muted-tag"> &bull; ${escapeHtml(q.title)}</span>` : ""}
            </div>
            <span class="q-score-pill">${q.awarded_marks || 0} / ${q.max_marks || 0} marks</span>
          </div>
          ${ptsHtml}
          ${q.model_answer ? `<div style="margin-top:10px;font-size:0.86rem;"><strong>Model Answer:</strong> ${renderMarkdown(q.model_answer)}</div>` : ""}
          ${vidBtn ? `<div class="video-pills">${vidBtn}</div>` : ""}
        </div>`;
      })
      .join("");

    const extraVids = (res.related_videos || []).slice(0, 3);
    const extraVidsHtml =
      extraVids.length && !qs.some((q) => q.video_url)
        ? `<div class="q-breakdown-card">
            <strong>Related Harony Video Walkthroughs</strong>
            <div class="video-pills">${extraVids
              .map((v) => {
                const w = toWatchUrl(v.video_url);
                const lbl = (v.label || v.chapter || "Video Solution").slice(0, 56);
                return `<a href="${escapeHtml(w)}" target="_blank" rel="noopener noreferrer" class="video-pill-link" data-video-url="${escapeHtml(w)}" data-video-title="${escapeHtml(lbl)}">🎥 ${escapeHtml(lbl)}</a>`;
              })
              .join("")}</div>
          </div>`
        : "";

    const notesRefs = (res.notes_references || []).filter((n) => n.image_url);
    const notesHtml = notesRefs.length
      ? `<div class="q-breakdown-card revision-notes-card">
          <div class="q-breakdown-header"><strong>Recommended Harony Notes Revision</strong></div>
          <div class="revision-list">
            ${notesRefs
              .slice(0, 2)
              .map((n) => {
                const imgUrl = normalizeAssetUrl(n.image_url);
                const bullets = Array.isArray(n.bullets) && n.bullets.length
                  ? `<ul class="revision-bullets">${n.bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul>`
                  : `<p class="revision-snippet">${escapeHtml(n.snippet || "")}</p>`;
                return `<div class="revision-item">
                  <div class="revision-item-text">
                    <span class="revision-badge">Page ${n.page || ""} &bull; ${escapeHtml(n.unit || "Harony Notes")}</span>
                    <h4>${escapeHtml(n.topic || "")}</h4>
                    ${bullets}
                  </div>
                  <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Page ${n.page || ""} — ${escapeHtml(n.topic || "")}">
                    <img src="${escapeHtml(imgUrl)}" alt="Page ${n.page || ""}" loading="lazy" />
                    <div class="mini-note-card-body"><strong>Page ${n.page || ""}</strong><span>Click to zoom</span></div>
                  </div>
                </div>`;
              })
              .join("")}
          </div>
        </div>`
      : "";

    const ocrDetailsHtml = res.extracted_student_text
      ? `<details class="ocr-details"><summary>View text read from your submission</summary><pre style="white-space:pre-wrap;margin:8px 0 0;">${escapeHtml(res.extracted_student_text)}</pre></details>`
      : "";

    gradeResultArea.innerHTML = `
      <div class="score-summary-card">
        <div class="score-pill-big">
          <span class="score-numbers">${res.awarded_marks || 0} / ${res.max_marks || 0} Marks</span>
          <span class="score-grade-badge ${badgeCls}">${pct}% &bull; Grade ${escapeHtml(res.grade || "-")}</span>
        </div>
        <div class="muted-tag" style="font-size:0.86rem;">${escapeHtml(res.mark_scheme_title || "Cambridge 0625 Check")}</div>
      </div>
      ${res.examiner_feedback ? `<div class="examiner-feedback-card">${renderMarkdown(res.examiner_feedback)}</div>` : ""}
      ${qsHtml}
      ${extraVidsHtml}
      ${notesHtml}
      ${ocrDetailsHtml}`;

    gradeResultArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  if (gradeSubmitBtn) {
    gradeSubmitBtn.addEventListener("click", async function () {
      const stuTxt = studentTextInput ? studentTextInput.value.trim() : "";
      if (!stuTxt && studentFiles.length === 0) {
        renderGradeResult({
          ok: false,
          error: "Please upload a photo/PDF of your Physics answer or type your answer in Step 1 first.",
        });
        return;
      }

      if (serverAvailable === null) await loadSystemStatus();

      gradeSubmitBtn.disabled = true;
      gradeResultArea.classList.remove("hidden");
      gradeResultArea.innerHTML = `
        <div class="grading-loader-card">
          <strong id="gradingLoaderMsg">Reading your Physics submission &amp; checking Mark Scheme…</strong>
          <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
        </div>`;

      const setLoaderMsg = (m) => {
        const el = document.getElementById("gradingLoaderMsg");
        if (el && m) el.textContent = m;
      };

      try {
        let result;
        const chosenMsId = activeMsMode === "saved" && savedMsSelect ? savedMsSelect.value : "";
        const chosenQid = activeMsMode === "saved" && questionSelect ? questionSelect.value : "";
        const pastedMs = activeMsMode === "paste" && msTextInput ? msTextInput.value.trim() : "";
        const uploadedMs = activeMsMode === "upload" ? msFile : null;

        if (serverAvailable) {
          const fd = new FormData();
          fd.append("student_text", stuTxt);
          fd.append("ms_id", chosenMsId);
          fd.append("question_id", chosenQid);
          fd.append("ms_text", pastedMs);
          for (const sf of studentFiles) {
            fd.append("student_files", sf);
          }
          if (uploadedMs) fd.append("ms_file", uploadedMs);

          const resp = await fetch("api/grade", { method: "POST", body: fd });
          const raw = await resp.json();
          if (!resp.ok) {
            result = { ok: false, error: raw.detail || "Could not grade submission." };
          } else {
            result = raw;
          }
        } else {
          result = await window.HaroonyBrowserEngine.grade(
            {
              studentText: stuTxt,
              studentFiles: studentFiles,
              msMode: activeMsMode,
              msId: chosenMsId,
              questionId: chosenQid,
              msText: pastedMs,
              msFile: uploadedMs,
            },
            setLoaderMsg
          );
        }

        renderGradeResult(result);
      } catch (err) {
        renderGradeResult({
          ok: false,
          error: "Something went wrong while marking your answer. Please try again.",
        });
      } finally {
        gradeSubmitBtn.disabled = false;
      }
    });
  }

  loadSystemStatus();
})();
