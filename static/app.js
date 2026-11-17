/* Haroony AI — Instant, Fast Physics Chat Frontend (Server + Standalone GitHub Pages) */
(function () {
  "use strict";

  const AI_AVATAR = "static/assets/harony_avatar.webp";

  // Optional external FastAPI backend override for GitHub Pages (`?api=https://your-backend.example`)
  const API_BASE = (() => {
    try {
      const p = new URLSearchParams(window.location.search).get("api");
      if (p !== null) {
        const clean = p.trim().replace(/\/+$/, "");
        if (clean) localStorage.setItem("harony_api_base", clean);
        else localStorage.removeItem("harony_api_base");
      }
      return (localStorage.getItem("harony_api_base") || "").replace(/\/+$/, "");
    } catch (_) {
      return "";
    }
  })();

  function apiUrl(path) {
    const clean = String(path || "").replace(/^\/+/, "");
    return API_BASE ? `${API_BASE}/${clean}` : clean;
  }

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

  // ==================== INITIALIZE STATUS ====================
  async function loadSystemStatus() {
    try {
      const resp = await fetch(apiUrl("api/status"), { headers: { Accept: "application/json" } });
      if (resp.ok) {
        const ct = resp.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const data = await resp.json();
          if (data && data.ok) {
            serverAvailable = true;
            return;
          }
        }
      }
    } catch (_) {}

    serverAvailable = false;
    if (window.HaroonyBrowserEngine) {
      try {
        await window.HaroonyBrowserEngine.getStatus();
      } catch (err) {
        console.error("Browser engine init error:", err);
      }
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

  chatInput.addEventListener("paste", function (e) {
    const file = extractClipboardImageOrPdf(e);
    if (file) {
      e.preventDefault();
      attachFileToChat(file);
    }
  });
  document.addEventListener("paste", function (e) {
    if (e.target === chatInput) return;
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    const file = extractClipboardImageOrPdf(e);
    if (file) {
      e.preventDefault();
      attachFileToChat(file);
      chatInput.focus();
    }
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
              <button type="button" class="quick-chip" data-prompt="Give me a practice question on moments">Practice Question</button>
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

        let feedbackHtml = "";
        if (!res.refused && res.category !== "greeting" && res.interaction_id) {
          feedbackHtml = `
            <div class="msg-feedback-row" data-interaction-id="${escapeHtml(res.interaction_id)}" data-unit="${escapeHtml(res.unit || "")}" data-query="${escapeHtml(res._userQuery || "")}">
              <span class="msg-feedback-label">Was this helpful?</span>
              <button type="button" class="fb-btn" data-helpful="true" title="Helpful — teach Haroony AI this worked well">👍 Helpful</button>
              <button type="button" class="fb-btn" data-helpful="false" title="Needs improvement">👎 Needs detail</button>
            </div>`;
        }

        extrasEl.innerHTML = videosHtml + cardsHtml + feedbackHtml;
        scrollChatToBottom();
      },
    };
  }

  document.body.addEventListener("click", async function (e) {
    const fbBtn = e.target.closest(".fb-btn");
    if (!fbBtn) return;
    const row = fbBtn.closest(".msg-feedback-row");
    if (!row || row.dataset.voted === "1") return;
    row.dataset.voted = "1";
    const helpful = fbBtn.getAttribute("data-helpful") === "true";
    const iid = row.getAttribute("data-interaction-id") || "";
    const unit = row.getAttribute("data-unit") || "";
    const query = row.getAttribute("data-query") || "";

    row.querySelectorAll(".fb-btn").forEach((b) => {
      b.disabled = true;
      if (b === fbBtn) b.classList.add("fb-btn-selected");
    });
    const lbl = row.querySelector(".msg-feedback-label");
    if (lbl) lbl.textContent = helpful ? "Thanks! Haroony AI learned from this." : "Thanks! Noted for improvement.";

    try {
      if (serverAvailable) {
        const fd = new FormData();
        fd.append("interaction_id", iid);
        fd.append("helpful", helpful ? "true" : "false");
        fd.append("query", query);
        fd.append("unit", unit);
        await fetch(apiUrl("api/feedback"), { method: "POST", body: fd });
      } else if (window.HaroonyBrowserEngine && window.HaroonyBrowserEngine.recordFeedback) {
        window.HaroonyBrowserEngine.recordFeedback(iid, helpful, query, unit);
      }
    } catch (_) {}
  });

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

        const resp = await fetch(apiUrl("api/chat"), { method: "POST", body: fd });
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

      if (data && !data.refused) data._userQuery = text;
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
