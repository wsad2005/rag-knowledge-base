/* =========================================================================
 * RAG Knowledge Base — 前端逻辑（原生 JS，无框架、无构建）
 * 负责：健康状态、文档上传/列表/删除、RAG 问答渲染、引用跳转、检索调试
 * ========================================================================= */

(() => {
  "use strict";

  const MAX_UPLOAD_MB = 20;

  const $ = (selector, root = document) => root.querySelector(selector);
  const state = { documents: [], health: null, sending: false };

  // ------------------------------------------------------------------ 工具
  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
  }

  function formatDate(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const pad = (n) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function toast(message, kind = "info", timeout = 4200) {
    const stack = $("#toasts");
    const node = document.createElement("div");
    node.className = `toast toast-${kind}`;
    node.textContent = message;
    stack.appendChild(node);
    setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      setTimeout(() => node.remove(), 200);
    }, timeout);
  }

  class ApiError extends Error {
    constructor(message, code, status) {
      super(message);
      this.code = code;
      this.status = status;
    }
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new ApiError("无法连接后端服务，请确认服务已启动", "network_error", 0);
    }
    if (response.status === 204) return null;

    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    if (!response.ok) {
      const detail = payload?.error?.detail;
      let message = payload?.error?.message || `请求失败（HTTP ${response.status}）`;
      if (detail?.fields?.length) {
        message += `：${detail.fields.map((f) => `${f.field} ${f.message}`).join("；")}`;
      }
      throw new ApiError(message, payload?.error?.code || `http_${response.status}`, response.status);
    }
    return payload;
  }

  // ------------------------------------------------------------------ 健康状态
  function renderHealth(health) {
    const strip = $("#status-strip");
    const llm = health.llm || {};
    const embedding = health.embedding || {};
    const vectors = health.vector_store?.vectors ?? 0;
    const providerLabel = {
      hashing: "哈希（离线）",
      "sentence-transformers": "本地模型",
      "openai-compatible": "远程 API",
    }[embedding.provider] || embedding.provider || "未知";

    const pills = [
      health.status === "ok"
        ? `<span class="pill pill-ok">服务正常 · v${escapeHtml(health.version)}</span>`
        : `<span class="pill pill-warn">服务降级</span>`,
      `<span class="pill">向量 ${vectors}</span>`,
      `<span class="pill">Embedding ${escapeHtml(providerLabel)}${embedding.dimension ? ` · ${embedding.dimension}d` : ""}</span>`,
      llm.configured
        ? `<span class="pill pill-ok">大模型 ${escapeHtml(llm.model || "已配置")}</span>`
        : `<span class="pill pill-warn">未配置大模型 · 仅检索</span>`,
    ];
    strip.innerHTML = pills.join("");
  }

  async function loadHealth() {
    try {
      state.health = await api("/api/health");
      renderHealth(state.health);
      if (state.health.embedding?.provider === "hashing") {
        $("#chat-subtitle").textContent =
          "当前使用离线哈希 Embedding（字面匹配）；配置本地或远程模型可提升语义召回";
      }
    } catch (error) {
      $("#status-strip").innerHTML = `<span class="pill pill-warn">后端未连接</span>`;
      throw error;
    }
  }

  // ------------------------------------------------------------------ 文档列表
  const STATUS_LABEL = { ready: "可检索", processing: "处理中", pending: "待处理", failed: "失败" };
  const STATUS_CLASS = { ready: "badge-ready", processing: "badge-processing", pending: "badge-processing", failed: "badge-failed" };

  function renderDocuments() {
    const list = $("#doc-list");
    const keyword = $("#doc-search").value.trim().toLowerCase();
    const items = keyword
      ? state.documents.filter((doc) => doc.filename.toLowerCase().includes(keyword))
      : state.documents;

    $("#doc-counter").textContent = String(state.documents.length);

    if (!items.length) {
      list.innerHTML = state.documents.length
        ? `<p class="empty">没有匹配「${escapeHtml(keyword)}」的文档</p>`
        : `<p class="empty">知识库还是空的<br />先上传一份 PDF / DOCX / TXT / Markdown 试试</p>`;
      return;
    }

    list.innerHTML = items
      .map((doc) => {
        const status = doc.status || "ready";
        const meta = [
          `<span>${formatBytes(doc.file_size)}</span>`,
          `<span>${doc.chunk_count} 块</span>`,
          doc.page_count ? `<span>${doc.page_count} 页</span>` : "",
          `<span>${formatDate(doc.created_at)}</span>`,
        ]
          .filter(Boolean)
          .join("");
        const error = status === "failed" && doc.error_message
          ? `<div class="doc-error">解析失败：${escapeHtml(doc.error_message)}</div>`
          : "";
        return `
          <article class="doc-item" data-id="${escapeHtml(doc.id)}">
            <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
            <button class="delete-button" type="button" title="删除文档" data-delete="${escapeHtml(doc.id)}">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" />
              </svg>
            </button>
            <div class="doc-meta">
              <span class="badge ${STATUS_CLASS[status] || ""}">${STATUS_LABEL[status] || status}</span>
              ${meta}
            </div>
            ${error}
          </article>`;
      })
      .join("");
  }

  async function loadDocuments() {
    const result = await api("/api/documents?limit=100");
    state.documents = result.items || [];
    renderDocuments();
  }

  // ------------------------------------------------------------------ 上传
  function uploadOne(file) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file, file.name);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/documents/upload");
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 90);
          updateQueueItem(file.name, percent);
        }
      });
      xhr.addEventListener("load", () => {
        let payload = null;
        try {
          payload = JSON.parse(xhr.responseText || "{}");
        } catch {
          payload = null;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          updateQueueItem(file.name, 100, "ok", "已入库");
          resolve(payload);
        } else {
          const message = payload?.error?.message || `上传失败（HTTP ${xhr.status}）`;
          const code = payload?.error?.code || `http_${xhr.status}`;
          updateQueueItem(file.name, 100, "err", message);
          reject(new ApiError(message, code, xhr.status));
        }
      });
      xhr.addEventListener("error", () => {
        updateQueueItem(file.name, 100, "err", "网络错误");
        reject(new ApiError("上传失败：网络错误", "network_error", 0));
      });
      xhr.send(form);
    });
  }

  function updateQueueItem(name, percent, status, message) {
    const queue = $("#upload-queue");
    let item = [...queue.children].find((node) => node.dataset.name === name);
    if (!item) {
      item = document.createElement("li");
      item.className = "upload-item";
      item.dataset.name = name;
      item.innerHTML = `
        <span class="u-name"></span>
        <span class="u-state">上传中…</span>
        <span class="upload-bar"><i></i></span>`;
      item.querySelector(".u-name").textContent = name;
      queue.prepend(item);
    }
    item.querySelector(".u-state").textContent =
      message || (percent < 100 ? `上传中 ${percent}%` : "处理中…");
    item.querySelector(".upload-bar > i").style.width = `${percent}%`;
    if (status) item.classList.add(status);
  }

  async function handleFiles(fileList) {
    const files = [...fileList];
    if (!files.length) return;

    for (const file of files) {
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        updateQueueItem(file.name, 0, "err", `超过 ${MAX_UPLOAD_MB}MB 限制`);
        toast(`《${file.name}》超过 ${MAX_UPLOAD_MB}MB 限制`, "err");
        continue;
      }
      try {
        const result = await uploadOne(file);
        toast(`《${result.document.filename}》已入库，${result.document.chunk_count} 个片段`, "ok");
      } catch (error) {
        if (error.code === "duplicate_document") {
          toast(`《${file.name}》内容已存在，未重复入库`, "info");
        } else {
          toast(`《${file.name}》：${error.message}`, "err", 6000);
        }
      }
    }

    await Promise.all([loadDocuments().catch(() => {}), loadHealth().catch(() => {})]);
    setTimeout(() => {
      $("#upload-queue").innerHTML = "";
    }, 3200);
  }

  // ------------------------------------------------------------------ 删除
  async function deleteDocument(id) {
    const doc = state.documents.find((item) => item.id === id);
    const name = doc ? doc.filename : "该文档";
    if (!window.confirm(`确定删除《${name}》？\n对应的文本块与向量索引会一并清理。`)) return;

    try {
      await api(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
      state.documents = state.documents.filter((item) => item.id !== id);
      renderDocuments();
      toast(`已删除《${name}》`, "ok");
      loadHealth().catch(() => {});
    } catch (error) {
      toast(`删除失败：${error.message}`, "err", 6000);
    }
  }

  // ------------------------------------------------------------------ 问答渲染
  function renderRichText(text, sourceCount) {
    const escaped = escapeHtml(text);
    const withCites = escaped.replace(/\[(\d{1,2})\]/g, (match, number) => {
      if (Number(number) < 1 || Number(number) > sourceCount) return match;
      return `<span class="cite" data-cite="${number}">[${number}]</span>`;
    });

    const lines = withCites.split(/\n/);
    const html = [];
    let listType = null;

    const closeList = () => {
      if (listType) {
        html.push(`</${listType}>`);
        listType = null;
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        closeList();
        continue;
      }
      const bullet = line.match(/^[-*•]\s+(.*)$/);
      const numbered = line.match(/^\d+[.、)]\s+(.*)$/);
      if (bullet) {
        if (listType !== "ul") {
          closeList();
          html.push("<ul>");
          listType = "ul";
        }
        html.push(`<li>${bullet[1]}</li>`);
      } else if (numbered) {
        if (listType !== "ol") {
          closeList();
          html.push("<ol>");
          listType = "ol";
        }
        html.push(`<li>${numbered[1]}</li>`);
      } else {
        closeList();
        html.push(`<p>${line}</p>`);
      }
    }
    closeList();
    return html.join("");
  }

  function renderSources(sources) {
    if (!sources?.length) return "";
    const cards = sources
      .map((source) => {
        const location = [
          source.page_number ? `第 ${source.page_number} 页` : "",
          source.section ? escapeHtml(source.section) : "",
          `块 #${source.chunk_index}`,
        ]
          .filter(Boolean)
          .join(" · ");
        return `
          <div class="source-card" data-source="${source.index}">
            <div class="source-head" data-toggle="${source.index}">
              <span class="source-index">[${source.index}]</span>
              <span class="source-name" title="${escapeHtml(source.document_name)}">${escapeHtml(source.document_name)}</span>
              <span class="source-score">${location} · 相似度 ${Number(source.score).toFixed(3)}</span>
            </div>
            <div class="source-body" data-body="${source.index}">${escapeHtml(source.snippet)}</div>
          </div>`;
      })
      .join("");
    return `<div class="sources">${cards}</div>`;
  }

  function renderDebug(debug) {
    if (!debug) return "";
    const rows = [
      ["召回条数", debug.retrieved],
      ["进入上下文", debug.used_in_context],
      ["上下文长度", `${debug.context_chars} 字`],
      ["耗时", `${debug.latency_ms} ms`],
      ["Embedding", `${debug.embedding_model} · ${debug.embedding_dimension}d`],
      ["生成模型", debug.llm_model || "未使用"],
    ];
    return `
      <details class="debug">
        <summary>检索调试</summary>
        <div class="debug-grid">
          ${rows.map(([key, value]) => `<span><b>${key}</b><br />${escapeHtml(value)}</span>`).join("")}
        </div>
      </details>`;
  }

  function scrollToBottom() {
    const box = $("#messages");
    box.scrollTop = box.scrollHeight;
  }

  function appendUserMessage(text) {
    $("#welcome")?.remove();
    const node = document.createElement("div");
    node.className = "msg msg-user";
    node.innerHTML = `<div class="msg-role">你</div><div class="bubble"></div>`;
    node.querySelector(".bubble").textContent = text;
    $("#messages").appendChild(node);
    scrollToBottom();
  }

  function appendPendingMessage() {
    const node = document.createElement("div");
    node.className = "msg msg-assistant";
    node.innerHTML = `
      <div class="msg-role">知识库助手</div>
      <div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>`;
    $("#messages").appendChild(node);
    scrollToBottom();
    return node;
  }

  function fillAssistantMessage(node, response) {
    const bubble = node.querySelector(".bubble");
    const modeLabel = { llm: "大模型生成", extractive: "仅检索（未配置模型）", no_context: "无相关资料" }[response.answer_mode] || response.answer_mode;
    const notice = response.notice
      ? `<div class="notice ${response.answer_mode === "llm" ? "notice-ok" : ""}">${escapeHtml(response.notice)}</div>`
      : "";
    bubble.innerHTML = `
      <div class="answer-text">${renderRichText(response.answer, response.sources.length)}</div>
      ${notice}
      ${renderSources(response.sources)}
      ${renderDebug(response.debug)}
      <div class="msg-role" style="margin-top:10px">${modeLabel}</div>`;
    scrollToBottom();
  }

  function fillErrorMessage(node, error) {
    node.querySelector(".bubble").innerHTML = `
      <div class="notice">请求失败：${escapeHtml(error.message)}</div>
      <div class="msg-role" style="margin-top:8px">错误码 ${escapeHtml(error.code || "unknown")}</div>`;
    scrollToBottom();
  }

  // ------------------------------------------------------------------ 提问
  async function ask(question) {
    if (state.sending) return;
    const trimmed = question.trim();
    if (!trimmed) return;

    state.sending = true;
    $("#send").disabled = true;
    appendUserMessage(trimmed);
    const pending = appendPendingMessage();
    const topK = Number($("#top-k").value) || 5;

    try {
      const response = await api("/api/knowledge/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, top_k: topK }),
      });
      fillAssistantMessage(pending, response);
    } catch (error) {
      fillErrorMessage(pending, error);
      toast(error.message, "err", 6000);
    } finally {
      state.sending = false;
      $("#send").disabled = false;
      $("#question").focus();
    }
  }

  // ------------------------------------------------------------------ 事件绑定
  function bindEvents() {
    const dropzone = $("#dropzone");
    const input = $("#file-input");

    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      handleFiles(input.files);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach((type) => {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((type) => {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      if (event.dataTransfer?.files?.length) handleFiles(event.dataTransfer.files);
    });

    $("#doc-list").addEventListener("click", (event) => {
      const button = event.target.closest("[data-delete]");
      if (button) deleteDocument(button.dataset.delete);
    });

    $("#doc-search").addEventListener("input", renderDocuments);
    $("#refresh-docs").addEventListener("click", () => {
      loadDocuments()
        .then(() => toast("列表已刷新", "ok", 1600))
        .catch((error) => toast(error.message, "err"));
    });

    $("#composer").addEventListener("submit", (event) => {
      event.preventDefault();
      const value = $("#question").value;
      if (!value.trim()) return;
      $("#question").value = "";
      $("#question").style.height = "auto";
      ask(value);
    });

    const textarea = $("#question");
    textarea.addEventListener("input", () => {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        $("#composer").requestSubmit();
      }
    });

    $("#messages").addEventListener("click", (event) => {
      const cite = event.target.closest("[data-cite]");
      if (cite) {
        const wrapper = cite.closest(".bubble");
        const card = wrapper?.querySelector(`[data-source="${cite.dataset.cite}"]`);
        if (card) {
          const body = card.querySelector(".source-body");
          body?.classList.add("open");
          card.classList.add("is-target");
          card.scrollIntoView({ behavior: "smooth", block: "center" });
          setTimeout(() => card.classList.remove("is-target"), 1600);
        }
        return;
      }

      const toggle = event.target.closest("[data-toggle]");
      if (toggle) {
        const body = toggle.parentElement.querySelector(".source-body");
        body?.classList.toggle("open");
      }
    });

    $("#clear-chat").addEventListener("click", () => {
      $("#messages").innerHTML = `
        <div class="welcome" id="welcome">
          <h3>把文档变成可以追问的知识</h3>
          <p>上传制度、手册、笔记或论文，然后直接提问。系统会先检索最相关的片段，再基于片段作答。</p>
          <ul>
            <li>答案里的 <span class="cite-demo">[1]</span> 可点击，直接跳到对应原文</li>
            <li>展开「检索调试」能看到召回条数、上下文长度与耗时</li>
            <li>未配置大模型时，会自动降级为原文片段展示，不会编造答案</li>
          </ul>
        </div>`;
      $("#question").focus();
    });

    $("#theme-toggle").addEventListener("click", () => {
      const root = document.documentElement;
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("ragkb-theme", next);
      } catch {
        /* 隐私模式下忽略 */
      }
    });
  }

  // ------------------------------------------------------------------ 启动
  async function boot() {
    try {
      const saved = localStorage.getItem("ragkb-theme");
      if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
    } catch {
      /* 忽略 */
    }

    $("#max-size").textContent = String(MAX_UPLOAD_MB);
    bindEvents();

    try {
      await loadHealth();
    } catch {
      toast("无法连接后端服务，请确认已运行 uvicorn app.main:app", "err", 8000);
    }
    try {
      await loadDocuments();
    } catch (error) {
      $("#doc-list").innerHTML = `<p class="empty">加载失败：${escapeHtml(error.message)}</p>`;
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
