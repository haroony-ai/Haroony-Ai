/* Haroony AI — Clean Real-Time Chat Frontend (FastAPI Backend + Standalone GitHub Pages Engine) */
(function () {
  "use strict";

  const AI_AVATAR = "static/assets/harony_avatar.webp";

  let serverAvailable = null;
  let chatHistory = [];
  let chatAttachedFile = null;

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

  // ==================== INITIALIZE STATUS ====================
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
      kbBadge.textContent = `${data.notes_chunks_count} Notes & MS • ${data.video_solutions_count || 0} Videos`;
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

  // Allow Ctrl+V / Cmd+V pasting screenshots or images directly in the text bar or anywhere in the chat!
  function handleClipboardPaste(e) {
    const cd = e.clipboardData || (window.clipboardData && window.clipboardData);
    if (!cd) return;

    // 1. Check clipboard items (standard screenshot / copied image)
    if (cd.items && cd.items.length) {
      for (const item of cd.items) {
        if (item.kind === "file" && item.type && item.type.startsWith("image/")) {
          const blob = item.getAsFile();
          if (blob) {
            e.preventDefault();
            const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
            const file = new File([blob], `screenshot.${ext}`, { type: blob.type });
            attachFileToChat(file);
            chatInput.focus();
            return;
          }
        }
      }
    }

    // 2. Fallback: cd.files
    if (cd.files && cd.files.length) {
      const f = cd.files[0];
      if (f && ((f.type && f.type.startsWith("image/")) || /\.pdf$/i.test(f.name || ""))) {
        e.preventDefault();
        attachFileToChat(f);
        chatInput.focus();
      }
    }
  }

  chatInput.addEventListener("paste", handleClipboardPaste);
  document.addEventListener("paste", function (e) {
    if (e.target === chatInput) return;
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    handleClipboardPaste(e);
  });

  // Allow Drag & Drop onto the chat card too
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
                .map(
                  (v) =>
                    `<a href="${escapeHtml(v.video_url)}" target="_blank" rel="noopener noreferrer" class="video-pill-link">
                      🎥 Video Solution: ${escapeHtml((v.label || v.chapter || "Watch").slice(0, 56))}
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
        "Answer ONLY Physics questions directly, clearly, and accurately. Stay strictly on the asked topic. Bold key Cambridge IGCSE 0625 marking phrases.";

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

  loadSystemStatus();
})();
