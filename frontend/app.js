const API_BASE_URL = "http://127.0.0.1:8000";

const statusEl = document.querySelector("#backend-status");
const uploadEl = document.querySelector("#pdf-upload");
const uploadMessageEl = document.querySelector("#upload-message");
const documentListEl = document.querySelector("#document-list");
const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#question-input");
const newChatEl = document.querySelector("#new-chat");

let conversationId = crypto.randomUUID();

const documents = [];

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(2) : "Not available";
}

function verificationLabel(data) {
  const verdict = String(data.verification_verdict || "").toUpperCase();

  if (verdict === "NOT_REQUIRED") {
    return "Not required";
  }

  if (verdict === "SUPPORTED" || data.verified) {
    return "Triggered → Supported";
  }

  if (verdict === "CORRECTED") {
    return "Triggered → Corrected";
  }

  if (verdict === "UNSUPPORTED") {
    return "Triggered → Unsupported";
  }

  return "Not required";
}

function setBackendStatus(connected) {
  statusEl.className = `status ${connected ? "connected" : "unavailable"}`;
  statusEl.innerHTML = `<span></span>Backend ${connected ? "connected" : "unavailable"}`;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    setBackendStatus(response.ok);
  } catch {
    setBackendStatus(false);
  }
}

function clearEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function addMessage(role, text) {
  clearEmptyState();

  const article = document.createElement("article");
  article.className = `message ${role.toLowerCase()}`;
  article.innerHTML = `
    <div class="role">${role}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return article;
}

function addAssistantResponse(data) {
  const article = addMessage("Assistant", data.answer);
  const metadata = document.createElement("div");
  metadata.className = "metadata";
  metadata.innerHTML = `
    <div class="metric"><span>Confidence</span>${formatNumber(data.confidence)}</div>
    <div class="metric"><span>Retrieval</span>${formatNumber(data.retrieval_confidence)}</div>
    <div class="metric"><span>Generation</span>${formatNumber(data.generation_confidence)}</div>
    <div class="metric"><span>Evidence</span>${formatNumber(data.evidence_confidence)}</div>
    <div class="metric"><span>Verification status</span>${verificationLabel(data)}</div>
    <div class="metric"><span>Retrieved chunks</span>${data.retrieved_chunks ?? "Not available"}</div>
  `;
  article.appendChild(metadata);

  if (data.verification_reason) {
    const reason = document.createElement("div");
    reason.className = "source";
    reason.innerHTML = `
      <div class="source-title">Routing / verification reason</div>
      ${escapeHtml(data.verification_reason)}
    `;
    article.appendChild(reason);
  }

  if (Array.isArray(data.sources) && data.sources.length > 0) {
    const sources = document.createElement("section");
    sources.className = "sources";
    sources.innerHTML = `<div class="role">Sources</div>`;

    data.sources.forEach((source, index) => {
      const sourceEl = document.createElement("div");
      sourceEl.className = "source";
      sourceEl.innerHTML = `
        <div class="source-title">[${index + 1}] ${escapeHtml(source.source || "Indexed document")} · score ${formatNumber(source.score)}</div>
        ${escapeHtml(source.text)}
      `;
      sources.appendChild(sourceEl);
    });

    article.appendChild(sources);
  }
}

function renderDocuments() {
  documentListEl.innerHTML = documents
    .map((doc) => `<li>${escapeHtml(doc.filename)}<br /><small>${doc.chunks} chunks</small></li>`)
    .join("");
}

function friendlyError(response, fallback) {
  if (response.status === 503) {
    return "RAG pipeline is not initialized. Please check the backend.";
  }
  if (response.status >= 500) {
    return "Something went wrong while generating the answer.";
  }
  return fallback;
}

async function uploadPdf(file) {
  uploadMessageEl.className = "upload-message";
  uploadMessageEl.textContent = "Indexing document...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      uploadMessageEl.className = "upload-message error";
      uploadMessageEl.textContent = friendlyError(response, "Could not upload this PDF.");
      return;
    }

    const data = await response.json();
    documents.push({ filename: data.filename, chunks: data.chunks });
    renderDocuments();
    uploadMessageEl.textContent = `Document indexed successfully: ${data.filename}, ${data.chunks} chunks`;
  } catch {
    uploadMessageEl.className = "upload-message error";
    uploadMessageEl.textContent =
      "Cannot connect to HARC-RAG backend. Make sure FastAPI is running on port 8000.";
  }
}

async function askQuestion(question) {
  addMessage("User", question);
  const loading = addMessage("Assistant", "Retrieving evidence...");

  window.setTimeout(() => {
    const bubble = loading.querySelector(".bubble");
    if (bubble) {
      bubble.textContent = "Generating answer...";
    }
  }, 900);

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
      }),
    });

    loading.remove();

    if (!response.ok) {
      addMessage("Assistant", friendlyError(response, "Could not generate an answer."));
      return;
    }

    addAssistantResponse(await response.json());
  } catch {
    loading.remove();
    addMessage(
      "Assistant",
      "Cannot connect to HARC-RAG backend. Make sure FastAPI is running on port 8000."
    );
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

uploadEl.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    uploadPdf(file);
  }
});

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) {
    return;
  }
  inputEl.value = "";
  askQuestion(question);
});

newChatEl.addEventListener("click", () => {
  conversationId = crypto.randomUUID();

  messagesEl.innerHTML = `
    <article class="empty-state">
      <h3>Ask a question after indexing a PDF.</h3>
      <p>The answer will come from the local HARC-RAG pipeline and Ollama.</p>
    </article>
  `;
});

checkHealth();
window.setInterval(checkHealth, 10000);
