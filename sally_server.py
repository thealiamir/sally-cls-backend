import os
import uuid
import math
from datetime import datetime
from typing import List, Dict, Any

import requests
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import openai

# =========================
#   CONFIG
# =========================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment variables.")

openai.api_key = OPENAI_API_KEY

# Excel file with your CLS Health URLs (MUST exist in the same folder)
URLS_EXCEL_PATH = "urls.xlsx"  # make sure to upload this file too

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # can switch to "gpt-4o" if you want

# Put your CLS logo URL on your backend widget (optional)
CLS_LOGO_URL = "https://your-domain.com/path/to/cls-logo.png"  # not used by Wix, just for /sally-widget

COLORS = {
    "teal": "#008577",
    "navy": "#003A52",
    "orange": "#FF6A00",
    "light_bg": "#F7F9FB",
}

SYSTEM_PROMPT = """
You are Sally, the CLS Health virtual assistant.

Identity & tone:
- You are professional, warm, concise, and reassuring.
- You sound like a knowledgeable clinic front-desk specialist.
- You represent CLS Health and say "we" when referring to the clinic.

Knowledge:
- You ONLY use the context passages provided to you (from CLS Health webpages) to answer questions.
- If the answer is not clearly supported by the context, say you’re not completely sure and direct the user to call CLS Health or visit the website.
- Do not invent services, locations, prices, or clinical advice.

Safety & boundaries:
- You do NOT provide medical diagnosis or treatment plans.
- For clinical questions, provide general information only, and say:
  "I can provide general information, but this is not medical advice. Please consult a CLS Health provider for medical decisions."
- You are HIPAA-aware:
  - Never ask for full SSN, full credit card numbers, or other highly sensitive identifiers.
  - If the user shares sensitive information, do not repeat it back in full.

Style:
- Be clear and concise: 2–5 sentences for most replies.
- When helpful, provide direct links to CLS Health pages from the context you used.
- Use bullet points only for complex answers.
"""

# =========================
#   FASTAPI APP
# =========================

app = FastAPI(title="Sally - CLS Health Chatbot")

# In-memory knowledge index: list of dicts with "embedding", "url", "text"
KNOWLEDGE_INDEX: List[Dict[str, Any]] = []


# =========================
#   TEXT & EMBEDDING UTILS
# =========================

def clean_text(html: str) -> str:
    """Strip HTML down to readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API."""
    resp = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / norm_a / norm_b


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top_k most relevant chunks for a user query."""
    if not KNOWLEDGE_INDEX:
        return []

    q_emb = get_embeddings([query])[0]

    scored = []
    for doc in KNOWLEDGE_INDEX:
        score = cosine_similarity(q_emb, doc["embedding"])
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for (s, d) in scored[:top_k] if s > 0.2]  # basic threshold to avoid noise


# =========================
#   LOADING URLS & BUILDING INDEX
# =========================

def load_urls_from_excel(path: str) -> List[str]:
    """
    Load URLs from Excel.
    - If there is a 'url' column, use it.
    - Otherwise, use the first column in the sheet (e.g. 'Unnamed: 0').
    This way, the app won't crash just because the column name is different.
    """
    if not os.path.exists(path):
        print(f"[Sally] WARNING: Excel file not found at: {path}")
        return []

    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[Sally] ERROR reading Excel file: {e}")
        return []

    if df.empty:
        print("[Sally] WARNING: Excel file is empty.")
        return []

    if "url" in df.columns:
        col = "url"
    else:
        # fall back to the first column (your file uses 'Unnamed: 0')
        col = df.columns[0]
        print(f"[Sally] INFO: Using first column '{col}' for URLs.")

    urls = df[col].dropna().astype(str).unique().tolist()
    print(f"[Sally] Loaded {len(urls)} URLs from Excel.")
    return urls



def crawl_and_build_index() -> None:
    """Load URLs from Excel, crawl them, and fill KNOWLEDGE_INDEX."""
    global KNOWLEDGE_INDEX
    KNOWLEDGE_INDEX = []

    urls = load_urls_from_excel(URLS_EXCEL_PATH)
    print(f"[Sally] Found {len(urls)} URLs in {URLS_EXCEL_PATH}")

    for url in urls:
        try:
            print(f"[Sally] Fetching {url}")
            res = requests.get(url, timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"[Sally] Failed to fetch {url}: {e}")
            continue

        text = clean_text(res.text)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)

        timestamp = datetime.utcnow().isoformat()
        for chunk_text_val, emb in zip(chunks, embeddings):
            KNOWLEDGE_INDEX.append(
                {
                    "id": str(uuid.uuid4()),
                    "url": url,
                    "text": chunk_text_val,
                    "embedding": emb,
                    "last_crawled_at": timestamp,
                }
            )

    print(f"[Sally] Index built with {len(KNOWLEDGE_INDEX)} chunks.")


# =========================
#   REQUEST / RESPONSE MODELS
# =========================

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: List[str]


# =========================
#   STARTUP
# =========================

@app.on_event("startup")
def on_startup():
    print("[Sally] Building knowledge index from Excel URLs...")
    crawl_and_build_index()
    print("[Sally] Ready.")


# =========================
#   CHAT ENDPOINT
# =========================

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    relevant = retrieve_relevant_chunks(user_message, top_k=5)

    if not relevant:
        fallback = (
            "I’m not completely sure based on the information I have. "
            "Please call CLS Health directly or visit our website for more details."
        )
        return ChatResponse(reply=fallback, sources=[])

    context_snippets = []
    used_urls = []
    for doc in relevant:
        url = doc["url"]
        if url not in used_urls:
            used_urls.append(url)
        snippet = f"(url: {url})\n{doc['text'][:1000]}"
        context_snippets.append(snippet)

    context_text = "\n\n".join(context_snippets)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": (
                "You have access to the following CLS Health reference snippets:\n\n"
                f"{context_text}\n\n"
                "Use ONLY this information to answer the user."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    completion = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    reply = completion.choices[0].message.content.strip()
    return ChatResponse(reply=reply, sources=used_urls)


# =========================
#   SIMPLE WIDGET HTML (optional for preview)
# =========================

@app.get("/sally-widget", response_class=HTMLResponse)
def sally_widget():
    """Embeddable widget for testing (not needed for Wix)."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Sally - CLS Health Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: {COLORS['light_bg']};
    }}
    .chat-container {{
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}
    .chat-header {{
      background: {COLORS['navy']};
      color: #fff;
      padding: 10px 14px;
      display: flex;
      align-items: center;
    }}
    .chat-header img {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      margin-right: 10px;
    }}
    .chat-body {{
      flex: 1;
      padding: 10px;
      overflow-y: auto;
      background: {COLORS['light_bg']};
    }}
    .msg {{
      max-width: 80%;
      padding: 8px 12px;
      border-radius: 12px;
      margin-bottom: 8px;
      font-size: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .msg-user {{
      margin-left: auto;
      background: {COLORS['teal']};
      color: #fff;
    }}
    .msg-sally {{
      margin-right: auto;
      background: #fff;
      color: #111;
    }}
    .chat-input {{
      display: flex;
      padding: 8px 10px;
      border-top: 1px solid #eee;
      background: #fff;
    }}
    .chat-input input {{
      flex: 1;
      border-radius: 999px;
      border: 1px solid #ddd;
      padding: 6px 10px;
      font-size: 14px;
      outline: none;
    }}
    .chat-input button {{
      margin-left: 8px;
      border-radius: 999px;
      border: none;
      padding: 6px 12px;
      background: {COLORS['orange']};
      color: #fff;
      cursor: pointer;
      font-size: 14px;
    }}
    .footer {{
      font-size: 10px;
      text-align: center;
      padding: 4px;
      background: #f9fafb;
      color: #777;
    }}
  </style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <img src="{CLS_LOGO_URL}" alt="CLS Health" />
    <div>
      <div style="font-weight: 600;">Sally</div>
      <div style="font-size: 12px; opacity: 0.8;">CLS Health virtual assistant</div>
    </div>
  </div>
  <div id="chat-body" class="chat-body">
    <div class="msg msg-sally">
      Hi, I’m Sally, your CLS Health virtual assistant. How can I help you today?
    </div>
  </div>
  <div class="chat-input">
    <input id="chat-input" placeholder="Type your question…" />
    <button onclick="sendMessage()">Send</button>
  </div>
  <div class="footer">
    Powered by CLS Health
  </div>
</div>

<script>
  const sessionId = "sess-" + Math.random().toString(36).slice(2);
  const chatBody = document.getElementById("chat-body");
  const inputEl = document.getElementById("chat-input");

  function appendMessage(text, from) {{
    const div = document.createElement("div");
    div.className = "msg " + (from === "user" ? "msg-user" : "msg-sally");
    div.textContent = text;
    chatBody.appendChild(div);
    chatBody.scrollTop = chatBody.scrollHeight;
  }}

  async function sendMessage() {{
    const text = inputEl.value.trim();
    if (!text) return;
    appendMessage(text, "user");
    inputEl.value = "";

    try {{
      const res = await fetch("/chat", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ message: text, session_id: sessionId }})
      }});
      const data = await res.json();
      appendMessage(data.reply, "sally");
    }} catch (e) {{
      appendMessage(
        "Something went wrong. Please try again or visit the CLS Health website.",
        "sally"
      );
    }}
  }}

  inputEl.addEventListener("keydown", function(e) {{
    if (e.key === "Enter") {{
      sendMessage();
    }}
  }});
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)

