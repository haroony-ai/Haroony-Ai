/* Haroony AI — Dual-Mode Frontend Application (Server API + GitHub Pages Serverless Engine)
 * Uses strictly relative URLs so it runs identically on:
 *   1. FastAPI Live Server / Hugging Face Spaces / Docker (root `/`)
 *   2. GitHub Pages (`https://toolautomation2-droid.github.io/Haroony-Ai/`)
 */
(function () {
  "use strict";

  let serverAvailable = null; // null = probing, true = FastAPI server, false = GitHub Pages browser engine
  let chatHistory = [];
  let chatAttachedFile = null;
  let graderStudentFiles = [];
  let graderMsFile = null;
  let activeMsMode = "library";
  let useSampleDemo = "";
  let unlockedTeacherPin = "";

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
      // Convert bare https://youtu.be/... links into clickable video links
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

  // Lightbox
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

  // Tab Navigation
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
    if (tabName === "notes" && !document.getElementById("notesGrid").children.length) {
      loadNotesExplorer();
    }
  }

  navTabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("brandHome").addEventListener("click", () => switchTab("tutor"));

  // Probe server or fall back to GitHub Pages Browser Engine
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
    msSelect.innerHTML = "";
    (data.mark_schemes || []).forEach((ms) => {
      const opt = document.createElement("option");
      opt.value = ms.id;
      opt.textContent = `${ms.title} (${ms.question_count} Qs • ${ms.total_marks} Marks)`;
      msSelect.appendChild(opt);
    });

    const unitSelect = document.getElementById("notesUnitFilter");
    const currentVal = unitSelect.value;
    unitSelect.innerHTML = '<option value="">All 7 IGCSE Physics Units</option>';
    (data.curriculum || []).forEach((u) => {
      const opt = document.createElement("option");
      opt.value = u.unit;
      opt.textContent = u.unit;
      unitSelect.appendChild(opt);
    });
    unitSelect.value = currentVal;

    document.getElementById("metricNotesCount").textContent = `${data.notes_chunks_count} Chunks (243 Slides)`;
    const totalQs = (data.mark_schemes || []).reduce((acc, m) => acc + (m.question_count || 0), 0);
    const vidCnt = data.video_solutions_count || (data.metrics && data.metrics.video_solutions_indexed) || 2085;
    document.getElementById("metricMsCount").textContent = `${totalQs.toLocaleString()} Qs • ${vidCnt.toLocaleString()} Videos`;
    if (data.metrics && data.metrics.validation_accuracy) {
      document.getElementById("metricAccuracy").textContent = `${(data.metrics.validation_accuracy * 100).toFixed(1)}%`;
    }
    const modeEl = document.getElementById("metricEngineMode");
    if (modeEl) {
      modeEl.textContent = serverAvailable ? "Live Server • Unlimited" : "GitHub Pages • Unlimited";
    }
  }

  // ==================== TUTOR CHAT ====================
  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");
  const chatFileInput = document.getElementById("chatFileInput");
  const chatAttachmentBar = document.getElementById("chatAttachmentBar");
  const chatAttachmentName = document.getElementById("chatAttachmentName");

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
    chatMessages.innerHTML = `
      <div class="msg msg-ai">
        <img src="static/assets/einstein_mascot.webp" alt="Haroony AI" class="msg-avatar" />
        <div class="msg-bubble">
          <div class="msg-meta">Haroony AI &bull; Harony Physics Clinic</div>
          <div class="msg-content"><p>Chat cleared! Ask me any Physics question or attach a photo of a Physics problem.</p></div>
        </div>
      </div>`;
  });

  document.getElementById("starterChips").addEventListener("click", function (e) {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    chatInput.value = chip.dataset.prompt || "";
    chatForm.requestSubmit();
  });

  chatInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  function appendUserMessage(text, fileObj) {
    const div = document.createElement("div");
    div.className = "msg msg-user";
    let imgPreviewHtml = "";
    if (fileObj && fileObj.type && fileObj.type.startsWith("image/")) {
      const objUrl = URL.createObjectURL(fileObj);
      imgPreviewHtml = `<img src="${objUrl}" alt="Uploaded" class="msg-uploaded-img" data-lightbox="${objUrl}" data-caption="${escapeHtml(fileObj.name)}" />`;
    } else if (fileObj) {
      imgPreviewHtml = `<div style="margin-top:6px;font-size:0.8rem;">📄 ${escapeHtml(fileObj.name)}</div>`;
    }
    div.innerHTML = `
      <div class="msg-user-avatar">You</div>
      <div class="msg-bubble">
        <div class="msg-meta">Student</div>
        <div class="msg-content">
          ${text ? `<p>${escapeHtml(text)}</p>` : ""}
          ${imgPreviewHtml}
        </div>
      </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
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
                  🎥 Mr. Haroony Video: ${escapeHtml((v.label || v.chapter || "Watch Solution").slice(0, 52))}
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
            <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Harony Notes p.${c.page || ""} — ${escapeHtml(c.unit)}: ${escapeHtml(c.topic)}">
              <img src="${escapeHtml(imgUrl)}" alt="Harony Notes Page ${c.page || ""}" loading="lazy" />
              <div class="mini-note-card-body">
                <strong>Page ${c.page || ""} &bull; ${escapeHtml(c.topic)}</strong>
                <span>${escapeHtml(c.unit)} (Click to enlarge)</span>
              </div>
            </div>`;
        })
        .join("");
      cardsHtml = `
        <div class="msg-notes-strip">
          <div class="msg-notes-label">📘 Illustrated Pages from Harony Notes 2026 (Click to Zoom):</div>
          <div class="msg-notes-cards">${items}</div>
        </div>`;
    }

    div.innerHTML = `
      <img src="static/assets/einstein_mascot.webp" alt="Haroony AI" class="msg-avatar" />
      <div class="msg-bubble">
        <div class="msg-meta">${escapeHtml(res.engine || "Haroony AI")}</div>
        <div class="msg-content">${renderMarkdown(res.reply)}</div>
        ${videosHtml}
        ${cardsHtml}
      </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  chatForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    const fileToSend = chatAttachedFile;
    if (!text && !fileToSend) return;

    appendUserMessage(text, fileToSend);
    chatInput.value = "";
    chatAttachedFile = null;
    chatFileInput.value = "";
    chatAttachmentBar.classList.add("hidden");

    chatSendBtn.disabled = true;
    chatSendBtn.textContent = "Thinking...";

    try {
      if (serverAvailable === null) await loadSystemStatus();
      if (serverAvailable) {
        const fd = new FormData();
        fd.append("message", text);
        fd.append("history", JSON.stringify(chatHistory.slice(-6)));
        if (fileToSend) fd.append("file", fileToSend);

        const resp = await fetch("api/chat", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok) {
          appendAiMessage({
            refused: true,
            engine: "Haroony AI",
            reply: data.detail || "Something went wrong. Please try again.",
            notes_cards: [],
          });
        } else {
          appendAiMessage(data);
          if (!data.refused) {
            chatHistory.push({ role: "user", content: text || "[Uploaded Physics image/PDF]" });
            chatHistory.push({ role: "assistant", content: data.reply });
          }
        }
      } else {
        const data = await window.HaroonyBrowserEngine.chat(text, chatHistory.slice(-6), fileToSend);
        appendAiMessage(data);
        if (!data.refused) {
          chatHistory.push({ role: "user", content: text || "[Physics query]" });
          chatHistory.push({ role: "assistant", content: data.reply });
        }
      }
    } catch (err) {
      appendAiMessage({
        refused: true,
        engine: "Haroony AI",
        reply: "Error processing your Physics query. Please try again.",
        notes_cards: [],
      });
    } finally {
      chatSendBtn.disabled = false;
      chatSendBtn.innerHTML = "Ask AI &rarr;";
    }
  });

  // ==================== MARK SCHEME GRADER ====================
  const studentFilesInput = document.getElementById("studentFilesInput");
  const studentFilesList = document.getElementById("studentFilesList");
  const studentDropzone = document.getElementById("studentDropzone");
  const msFileInput = document.getElementById("msFileInput");
  const msFilePill = document.getElementById("msFilePill");
  const samplePreviewBox = document.getElementById("samplePreviewBox");

  function renderStudentFilePills() {
    studentFilesList.innerHTML = graderStudentFiles
      .map(
        (f, i) =>
          `<span class="file-pill">📎 ${escapeHtml(f.name)} <button type="button" class="btn-icon" data-rm-stu="${i}">&times;</button></span>`
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
      useSampleDemo = "";
      samplePreviewBox.classList.add("hidden");
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
      useSampleDemo = "";
      samplePreviewBox.classList.add("hidden");
      for (const f of e.dataTransfer.files) {
        graderStudentFiles.push(f);
      }
      renderStudentFilePills();
    }
  });

  msFileInput.addEventListener("change", function () {
    if (msFileInput.files && msFileInput.files[0]) {
      graderMsFile = msFileInput.files[0];
      msFilePill.innerHTML = `<span class="file-pill">📑 ${escapeHtml(graderMsFile.name)}</span>`;
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

  // 1-Click Harony ClassWork 2027 Sample Demo
  document.getElementById("runSampleDemoBtn").addEventListener("click", function () {
    useSampleDemo = "cw_u1_density";
    graderStudentFiles = [];
    renderStudentFilePills();
    document.getElementById("studentTextInput").value =
      "2023 March P4 V2 Q02 a - Density:\n" +
      "The ship is not solid steel because there are air spaces inside the ship, " +
      "so the overall average density of the ship is less than the density of the water.";
    document.getElementById("msLibrarySelect").value = "harony-cw-2027-u1";
    document.getElementById("questionIdInput").value = "2023-March-P4-V2-Q02-a";
    samplePreviewBox.classList.remove("hidden");
    document.querySelector('[data-ms-mode="library"]').click();
    document.getElementById("gradeForm").requestSubmit();
  });

  document.getElementById("clearSampleDemoBtn").addEventListener("click", function () {
    useSampleDemo = "";
    samplePreviewBox.classList.add("hidden");
  });

  function renderGradeReport(res) {
    const area = document.getElementById("gradeResultArea");
    area.classList.remove("hidden");

    if (!res.ok) {
      area.innerHTML = `
        <div class="panel-card" style="border-color:#fecaca;background:#fef2f2;">
          <h3 style="color:#b91c1c;margin-bottom:8px;">🛡️ Cannot Grade This Submission</h3>
          <div>${renderMarkdown(res.error || res.detail || "Unable to grade submission.")}</div>
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
          ? `<a href="${escapeHtml(q.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">🎥 Watch Mr. Haroony's Video Solution</a>`
          : "";
        return `
          <div class="q-breakdown-card">
            <div class="q-breakdown-header">
              <strong>${escapeHtml(q.title || "Question " + q.question_id)}</strong>
              <span class="badge badge-free">${q.awarded_marks} / ${q.max_marks} marks</span>
            </div>
            ${rows}
            <div style="margin-top:8px;font-size:0.82rem;color:#475569;">
              <strong>Official Model Key:</strong> <code>${escapeHtml(q.model_answer)}</code>
            </div>
            ${vidBtn ? `<div class="video-pills">${vidBtn}</div>` : ""}
          </div>`;
      })
      .join("");

    const refsWithImg = (res.notes_references || []).filter((n) => n.image_url);
    let notesHtml = "";
    if (refsWithImg.length > 0) {
      notesHtml = `
        <div class="panel-card" style="margin-top:14px;">
          <h3 style="margin-bottom:10px;">📘 Revise This Topic in Harony Notes 2026</h3>
          <div class="msg-notes-cards">
            ${refsWithImg
              .map((n) => {
                const imgUrl = normalizeAssetUrl(n.image_url);
                return `
                <div class="mini-note-card" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Harony Notes p.${n.page} — ${escapeHtml(n.topic)}">
                  <img src="${escapeHtml(imgUrl)}" alt="Page ${n.page}" />
                  <div class="mini-note-card-body">
                    <strong>Page ${n.page} &bull; ${escapeHtml(n.topic)}</strong>
                    <span>${escapeHtml(n.unit)} (Click to enlarge)</span>
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
          <div style="font-size:0.82rem;font-weight:700;color:#475569;text-transform:uppercase;">
            ${escapeHtml(res.mark_scheme_title)}
          </div>
          <h3 style="margin:4px 0 0;">Cambridge Examiner Marking Report</h3>
        </div>
        <div class="score-pill-big">
          <span class="score-numbers">${res.awarded_marks} / ${res.max_marks}</span>
          <span style="font-weight:700;color:#475569;">(${res.percentage}%)</span>
          <span class="score-grade-badge">Grade ${escapeHtml(res.grade)}</span>
        </div>
      </div>

      ${questionsHtml}

      <details class="ocr-details">
        <summary>🔍 View / Edit Extracted Answer Text &amp; Re-Grade</summary>
        <p class="panel-help" style="margin-top:8px;">You can copy or refine the extracted text in Step 1 above and click Grade again.</p>
        <pre style="white-space:pre-wrap;background:#f8fafc;padding:10px;border-radius:8px;border:1px solid #e2e8f0;font-size:0.82rem;">${escapeHtml(res.extracted_student_text || "")}</pre>
      </details>

      ${notesHtml}
    `;
    area.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  document.getElementById("gradeForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const btn = document.getElementById("gradeSubmitBtn");
    btn.disabled = true;
    btn.textContent = "Reading & Marking with Cambridge Rules...";

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
      if (serverAvailable) {
        const fd = new FormData();
        fd.append("student_text", stuText);
        fd.append("use_sample_demo", useSampleDemo);
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
        const data = await resp.json();
        renderGradeReport(data);
      } else {
        const data = await window.HaroonyBrowserEngine.grade({
          studentText: stuText,
          msId: msId,
          msText: msText,
          questionId: qId,
          useSampleDemo: useSampleDemo,
        });
        renderGradeReport(data);
      }
    } catch (err) {
      renderGradeReport({ ok: false, error: "Error grading submission." });
    } finally {
      btn.disabled = false;
      btn.textContent = "✅ Correct & Grade My Physics Answer";
    }
  });

  // ==================== HARONY NOTES EXPLORER ====================
  async function loadNotesExplorer() {
    const q = document.getElementById("notesSearchInput").value.trim();
    const unit = document.getElementById("notesUnitFilter").value;
    const grid = document.getElementById("notesGrid");
    grid.innerHTML = `<p style="color:#475569;">Loading illustrated notes...</p>`;

    try {
      if (serverAvailable === null) await loadSystemStatus();
      let data;
      if (serverAvailable) {
        const params = new URLSearchParams();
        if (q) params.set("q", q);
        if (unit) params.set("unit", unit);
        const resp = await fetch("api/notes?" + params.toString());
        data = await resp.json();
      } else {
        data = await window.HaroonyBrowserEngine.searchNotes(q, unit);
      }

      const items = (data.results || []).filter((c) => c.image_url && c.page !== 1 && c.page !== 2 && c.page !== 3);
      if (!items.length) {
        grid.innerHTML = `<p style="color:#475569;">No matching pages found. Try another Physics keyword.</p>`;
        return;
      }
      grid.innerHTML = items
        .map((c) => {
          const imgUrl = normalizeAssetUrl(c.image_url);
          return `
          <div class="note-card">
            <img src="${escapeHtml(imgUrl)}" alt="Harony Notes Page ${c.page}" loading="lazy" data-lightbox="${escapeHtml(imgUrl)}" data-caption="Harony Notes p.${c.page} — ${escapeHtml(c.unit)}: ${escapeHtml(c.topic)}" />
            <div class="note-card-body">
              <div class="note-card-unit">${escapeHtml(c.unit)} ${c.page ? "&bull; Page " + c.page : ""}</div>
              <div class="note-card-title">${escapeHtml(c.topic)}</div>
              <div class="note-card-snippet">${escapeHtml((c.text || "").slice(0, 180))}</div>
            </div>
          </div>`;
        })
        .join("");
    } catch (err) {
      grid.innerHTML = `<p style="color:#dc2626;">Could not load notes.</p>`;
    }
  }

  document.getElementById("notesSearchBtn").addEventListener("click", loadNotesExplorer);
  document.getElementById("notesUnitFilter").addEventListener("change", loadNotesExplorer);
  document.getElementById("notesSearchInput").addEventListener("keydown", function (e) {
    if (e.key === "Enter") loadNotesExplorer();
  });

  // ==================== TEACHER TRAINING STUDIO ====================
  function showTeacherToast(msg, isError) {
    const el = document.getElementById("teacherStatusMsg");
    el.textContent = msg;
    el.style.background = isError ? "#fef2f2" : "#f0fdf4";
    el.style.color = isError ? "#b91c1c" : "#15803d";
    el.style.borderColor = isError ? "#fecaca" : "#86efac";
    el.classList.remove("hidden");
  }

  document.getElementById("teacherPinForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const pin = document.getElementById("teacherPinInput").value.trim();
    if (serverAvailable) {
      const fd = new FormData();
      fd.append("pin", pin);
      const resp = await fetch("api/teacher/verify", { method: "POST", body: fd });
      if (!resp.ok) {
        alert("Incorrect Teacher PIN (Default is 0625).");
        return;
      }
      const data = await resp.json();
      unlockedTeacherPin = pin;
      document.getElementById("teacherLockCard").classList.add("hidden");
      document.getElementById("teacherStudioContent").classList.remove("hidden");
      if (data.settings) {
        document.getElementById("ollamaEnabledChk").checked = Boolean(data.settings.ollama_enabled);
        document.getElementById("ollamaUrlInput").value = data.settings.ollama_base_url || "http://127.0.0.1:11434";
        document.getElementById("ollamaModelInput").value = data.settings.ollama_model || "qwen2.5-vl:7b";
      }
    } else {
      if (pin !== "0625") {
        alert("Incorrect Teacher PIN (Default is 0625).");
        return;
      }
      unlockedTeacherPin = pin;
      document.getElementById("teacherLockCard").classList.add("hidden");
      document.getElementById("teacherStudioContent").classList.remove("hidden");
    }
  });

  document.getElementById("uploadNotesForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    if (serverAvailable) {
      const fd = new FormData(e.target);
      fd.append("pin", unlockedTeacherPin);
      const resp = await fetch("api/teacher/upload-material", { method: "POST", body: fd });
      const data = await resp.json();
      if (resp.ok && data.ok) {
        showTeacherToast(`Added ${data.chunks_added} Physics blocks & re-trained Haroony AI!`, false);
        e.target.reset();
        loadSystemStatus();
      } else {
        showTeacherToast(data.detail || "Failed to upload material.", true);
      }
    } else {
      const fd = new FormData(e.target);
      const txt = String(fd.get("text_content") || "").trim();
      if (!txt) {
        showTeacherToast("On static GitHub Pages mode, paste notes text directly (or run the Python/Docker server for PDF ingestion).", true);
        return;
      }
      const saved = JSON.parse(localStorage.getItem("haroony_custom_chunks") || "[]");
      saved.push({
        id: "browser-" + Date.now(),
        source: String(fd.get("title") || "Teacher Notes"),
        unit: String(fd.get("unit") || "Teacher Notes"),
        topic: String(fd.get("topic") || "Physics"),
        text: txt,
        formulas: [],
        image_url: null,
      });
      localStorage.setItem("haroony_custom_chunks", JSON.stringify(saved));
      showTeacherToast("Saved Physics notes to browser knowledge base!", false);
      e.target.reset();
    }
  });

  document.getElementById("uploadMsForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    if (serverAvailable) {
      const fd = new FormData(e.target);
      fd.append("pin", unlockedTeacherPin);
      const resp = await fetch("api/teacher/upload-ms", { method: "POST", body: fd });
      const data = await resp.json();
      if (resp.ok && data.ok) {
        showTeacherToast(
          `Added "${data.mark_scheme.title}" (${data.mark_scheme.question_count} parts, ${data.mark_scheme.total_marks} marks) & re-trained!`,
          false
        );
        e.target.reset();
        loadSystemStatus();
      } else {
        showTeacherToast(data.detail || "Failed to parse Mark Scheme.", true);
      }
    } else {
      showTeacherToast("Use Paste MS in Correct Answer & MS on static GitHub Pages, or run Docker/Spaces for full PDF MS upload.", false);
    }
  });

  document.getElementById("retrainNowBtn").addEventListener("click", async function () {
    if (!serverAvailable) {
      showTeacherToast("Browser Engine is up-to-date with Harony 2026/2027 models!", false);
      return;
    }
    const fd = new FormData();
    fd.append("pin", unlockedTeacherPin);
    const resp = await fetch("api/teacher/retrain", { method: "POST", body: fd });
    const data = await resp.json();
    if (resp.ok && data.ok) {
      showTeacherToast(
        `Re-trained Haroony AI! Guardrail Accuracy: ${(data.metrics.validation_accuracy * 100).toFixed(1)}%, Indexed Chunks: ${data.metrics.indexed_notes_chunks}`,
        false
      );
      loadSystemStatus();
    }
  });

  document.getElementById("neuralSettingsForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!serverAvailable) {
      showTeacherToast("Ollama settings apply when running the self-hosted Python/Docker server.", false);
      return;
    }
    const fd = new FormData();
    fd.append("pin", unlockedTeacherPin);
    fd.append("ollama_enabled", document.getElementById("ollamaEnabledChk").checked ? "true" : "false");
    fd.append("ollama_base_url", document.getElementById("ollamaUrlInput").value.trim());
    fd.append("ollama_model", document.getElementById("ollamaModelInput").value.trim());
    const resp = await fetch("api/teacher/settings", { method: "POST", body: fd });
    if (resp.ok) {
      showTeacherToast("Saved self-hosted neural engine settings!", false);
      loadSystemStatus();
    }
  });

  loadSystemStatus();
})();
