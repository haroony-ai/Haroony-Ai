/* Haroony AI — Minimal, Smooth Dual-Mode Frontend (Server API + Static Browser Engine) */
(function () {
  "use strict";

  const MIN_THINK_MS = 2200; // Minimum 2.2s smooth thinking animation

  let serverAvailable = null;
  let chatHistory = [];
  let chatAttachedFile = null;
  let graderStudentFiles = [];
  let graderMsFile = null;
  let activeMsMode = "library";

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

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
    let lines = String(md).split("\n");
    let html = [];
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

    for (let raw of lines) {
      let t = raw.trim();
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
      if (t.startsWith("> ")) {
        closeList();
        html.push("<blockquote>" + inlineFormat(t.slice(2)) + "</blockquote>");
        continue;
      }
      if (t.startsWith("- ") || t.startsWith("* ")) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push("<li>" + inlineFormat(t.slice(2)) + "</li>");
        continue;
      }
      closeList();
      html.push("<p>" + inlineFormat(t) + "</p>");
    }
    closeList();
    return html.join("");
  }

  // ==================== LIGHTBOX ====================
  const lightboxModal = document.getElementById("lightboxModal");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxCaption = document.getElementById("lightboxCaption");

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

  document.getElementById("lightboxClose").addEventListener("click", closeLightbox);
  document.getElementById("lightboxBackdrop").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeLightbox();
  });

  document.body.addEventListener("click", function (e) {
    const target = e.target.closest("[data-lightbox]");
    if (target) {
      openLightbox(target.getAttribute("data-lightbox"), target.getAttribute("data-caption") || target.alt || "");
    }
  });

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

    const msSelect = document.getElementById("msLibrarySelect");
    if (msSelect) {
      msSelect.innerHTML = "";
      (data.mark_schemes || []).forEach((ms) => {
        const opt = document.createElement("option");
        opt.value = ms.id;
        opt.textContent = ms.title;
        msSelect.appendChild(opt);
      });
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
  const starterChips = document.getElementById("starterChips");

  function autoResizeChatInput() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 130) + "px";
  }
  chatInput.addEventListener("input", autoResizeChatInput);

  chatFileInput.addEventListener("change", function () {
    if (chatFileInput.files && chatFileInput.files[0]) {
      chatAttachedFile = chatFileInput.files[0];
      chatAttachmentName.textContent = chatAttachedFile.name;
      chatAttachmentBar.classList.remove("hidden");
    }
  });

  document.getElementById("removeChatAttachment").addEventListener("click", function () {
    chatAttachedFile = null;
    chatFileInput.value = "";
    chatAttachmentBar.classList.add("hidden");
  });

  document.getElementById("clearChatBtn").addEventListener("click", function () {
    chatHistory = [];
    starterChips.classList.remove("hidden");
    chatMessages.innerHTML = `
      <div class="msg msg-ai">
        <img src="static/assets/einstein_mascot.webp" alt="Haroony AI" class="msg-avatar" />
        <div class="msg-bubble">
          <div class="msg-content">
            <p>Hi! Ask me any <strong>Physics</strong> question, or attach a photo or PDF of a problem you want solved step-by-step.</p>
          </div>
        </div>
      </div>`;
  });

  starterChips.addEventListener("click", function (e) {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    chatInput.value = chip.dataset.prompt || "";
    autoResizeChatInput();
    chatForm.requestSubmit();
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
    starterChips.classList.add("hidden");
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

  function createThinkingBubble(hasFile) {
    const div = document.createElement("div");
    div.className = "msg msg-ai";
    const step1 = hasFile ? "Reading your attached file…" : "Thinking through the physics…";
    const step2 = "Checking formulas & notes…";
    div.innerHTML = `
      <img src="static/assets/einstein_mascot.webp" alt="Haroony AI" class="msg-avatar" />
      <div class="msg-bubble">
        <div class="thinking-box">
          <div class="thinking-dots" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
          <span class="thinking-text">${step1}</span>
        </div>
      </div>`;
    chatMessages.appendChild(div);
    scrollChatToBottom();

    const textEl = div.querySelector(".thinking-text");
    const timer = setTimeout(() => {
      if (textEl && div.isConnected) {
        textEl.textContent = step2;
      }
    }, 1050);

    return {
      remove() {
        clearTimeout(timer);
        if (div.isConnected) div.remove();
      },
    };
  }

  function appendAiMessage(res) {
    const div = document.createElement("div");
    div.className = "msg msg-ai" + (res.refused ? " msg-refused" : "");

    let videosHtml = "";
    const vids = res.video_links || [];
    if (!res.refused && vids.length > 0) {
      videosHtml = `
        <div class="video-pills">
          ${vids
            .map(
              (v) =>
                `<a href="${escapeHtml(v.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">
                  🎥 Video Solution: ${escapeHtml((v.label || v.chapter || "Watch").slice(0, 48))}
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
          <div class="msg-notes-label">Related Notes</div>
          <div class="msg-notes-cards">${items}</div>
        </div>`;
    }

    div.innerHTML = `
      <img src="static/assets/einstein_mascot.webp" alt="Haroony AI" class="msg-avatar" />
      <div class="msg-bubble">
        <div class="msg-content">${renderMarkdown(res.reply)}</div>
        ${videosHtml}
        ${cardsHtml}
      </div>`;
    chatMessages.appendChild(div);
    scrollChatToBottom();
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
    chatAttachmentBar.classList.add("hidden");

    chatSendBtn.disabled = true;
    const thinking = createThinkingBubble(Boolean(fileToSend));

    try {
      if (serverAvailable === null) await loadSystemStatus();

      const workPromise = (async () => {
        if (serverAvailable) {
          const fd = new FormData();
          fd.append("message", text);
          fd.append("history", JSON.stringify(chatHistory.slice(-6)));
          if (fileToSend) fd.append("file", fileToSend);

          const resp = await fetch("api/chat", { method: "POST", body: fd });
          const data = await resp.json();
          if (!resp.ok) {
            return {
              refused: true,
              reply: data.detail || "Something went wrong. Please try again.",
              notes_cards: [],
            };
          }
          return data;
        }
        return window.HaroonyBrowserEngine.chat(text, chatHistory.slice(-6), fileToSend);
      })();

      // Run AI calculation/retrieval in parallel with smooth >= 2.2s thinking animation
      const [data] = await Promise.all([workPromise, sleep(MIN_THINK_MS)]);
      thinking.remove();
      appendAiMessage(data);

      if (!data.refused) {
        chatHistory.push({ role: "user", content: text || "[Uploaded Physics file]" });
        chatHistory.push({ role: "assistant", content: data.reply });
      }
    } catch (err) {
      thinking.remove();
      appendAiMessage({
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

  function renderStudentFilePills() {
    studentFilesList.innerHTML = graderStudentFiles
      .map(
        (f, i) =>
          `<span class="file-pill">📎 ${escapeHtml(f.name)} <button type="button" class="btn-icon" data-rm-stu="${i}" aria-label="Remove">&times;</button></span>`
      )
      .join("");
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
      ["library", "upload", "paste", "auto"].forEach((m) => {
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
          <span class="thinking-text" id="gradeLoaderText">Reading your answer…</span>
        </div>
        <div class="loader-bar-track">
          <div class="loader-bar-fill"></div>
        </div>
      </div>`;
    area.scrollIntoView({ behavior: "smooth", block: "nearest" });

    return setTimeout(() => {
      const el = document.getElementById("gradeLoaderText");
      if (el) el.textContent = "Checking against Mark Scheme criteria…";
    }, 1050);
  }

  function renderGradeReport(res) {
    const area = document.getElementById("gradeResultArea");
    area.classList.remove("hidden");

    if (!res.ok) {
      area.innerHTML = `
        <div class="panel-card" style="border-color:#fecaca;background:#fef2f2;animation:fadeSlideUp 0.35s var(--ease-out);">
          <h3 style="color:#b91c1c;margin-bottom:6px;">Unable to Check Answer</h3>
          <div>${renderMarkdown(res.error || res.detail || "Please provide a Physics answer.")}</div>
        </div>`;
      area.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }

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
            <div style="margin-top:10px;font-size:0.83rem;color:#64748b;">
              <strong>Model Key:</strong> <code>${escapeHtml(q.model_answer)}</code>
            </div>
            ${vidBtn ? `<div class="video-pills">${vidBtn}</div>` : ""}
          </div>`;
      })
      .join("");

    const refsWithImg = (res.notes_references || []).filter((n) => n.image_url);
    let notesHtml = "";
    if (refsWithImg.length > 0) {
      notesHtml = `
        <div class="panel-card" style="margin-top:14px;animation:fadeSlideUp 0.44s var(--ease-out);">
          <h3 style="margin-bottom:10px;">Related Notes</h3>
          <div class="msg-notes-cards">
            ${refsWithImg
              .map((n) => {
                const imgUrl = normalizeAssetUrl(n.image_url);
                return `
                <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Page ${n.page} — ${escapeHtml(n.topic)}">
                  <img src="${escapeHtml(imgUrl)}" alt="Page ${n.page}" />
                  <div class="mini-note-card-body">
                    <strong>Page ${n.page} &bull; ${escapeHtml(n.topic)}</strong>
                    <span>Click to zoom</span>
                  </div>
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
          <span class="score-grade-badge">Grade ${escapeHtml(res.grade)}</span>
        </div>
      </div>

      ${questionsHtml}

      <details class="ocr-details">
        <summary>View extracted answer text</summary>
        <pre style="white-space:pre-wrap;background:#f8fafc;padding:10px;border-radius:8px;border:1px solid #e2e8f0;font-size:0.82rem;margin:10px 0 0;">${escapeHtml(res.extracted_student_text || "")}</pre>
      </details>

      ${notesHtml}
    `;
    area.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("gradeForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const btn = document.getElementById("gradeSubmitBtn");
    btn.disabled = true;
    btn.textContent = "Checking…";

    const loaderTimer = showGradingLoader();

    const stuText = document.getElementById("studentTextInput").value.trim();
    const msId = activeMsMode === "library" ? document.getElementById("msLibrarySelect").value : "";
    const qId =
      activeMsMode === "library"
        ? document.getElementById("questionIdInput").value.trim()
        : activeMsMode === "upload"
        ? document.getElementById("questionIdUploadInput").value.trim()
        : "";
    const msText = activeMsMode === "paste" ? document.getElementById("msTextInput").value.trim() : "";

    try {
      if (serverAvailable === null) await loadSystemStatus();

      const gradePromise = (async () => {
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
          return resp.json();
        }
        return window.HaroonyBrowserEngine.grade({
          studentText: stuText,
          msId: msId,
          msText: msText,
          questionId: qId,
        });
      })();

      const [data] = await Promise.all([gradePromise, sleep(MIN_THINK_MS)]);
      clearTimeout(loaderTimer);
      renderGradeReport(data);
    } catch (err) {
      clearTimeout(loaderTimer);
      renderGradeReport({ ok: false, error: "Error checking submission. Please try again." });
    } finally {
      btn.disabled = false;
      btn.textContent = "Check My Answer";
    }
  });

  loadSystemStatus();
})();
