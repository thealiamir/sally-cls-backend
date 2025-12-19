import os
import uuid
import math
import asyncio
import requests
import pandas as pd
import openai
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# ========================= #
#   CONFIG                  #
# ========================= #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in environment variables.")
openai.api_key = OPENAI_API_KEY

URLS_EXCEL_PATH = "urls.xlsx"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

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
- You ONLY use the context passages provided to you to answer questions.
- If the answer is not supported by context, direct the user to call CLS Health.
"""

# ========================= #
#   FASTAPI APP & LIFESPAN  #
# ========================= #

# In-memory knowledge index
KNOWLEDGE_INDEX: List[Dict[str, Any]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block allows the server to open its port immediately
    # while the heavy indexing runs in the background.
    print("[Sally] Server starting... launching background indexer.")
    asyncio.create_task(asyncio.to_thread(crawl_and_build_index))
    yield
    print("[Sally] Server shutting down.")

app = FastAPI(title="Sally - CLS Health Chatbot", lifespan=lifespan)

# ========================= #
#   TEXT & EMBEDDING UTILS  #
# ========================= #

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
def load_urls_from_excel(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return []
    try:
        df = pd.read_excel(path)
        col = "url" if "url" in df.columns else df.columns[0]
        return df[col].dropna().astype(str).unique().tolist()
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def get_embeddings(texts: List[str]) -> List[List[float]]:
    # This checks for empty text so OpenAI doesn't error out
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return []
        
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=valid_texts)
    return [d.embedding for d in resp.data]
def crawl_and_build_index() -> None:
    global KNOWLEDGE_INDEX
    urls = load_urls_from_excel(URLS_EXCEL_PATH)
    print(f"[Sally] Crawling {len(urls)} URLs...")
    new_index = []
    
    for url in urls:
        try:
            res = requests.get(url, timeout=15)
            text = clean_text(res.text)
            
            if not text:
                continue
                
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)
            
            if not embeddings:
                continue
                
            for ch, emb in zip(chunks, embeddings):
                new_index.append({
                    "id": str(uuid.uuid4()),
                    "url": url,
                    "text": ch,
                    "embedding": emb
                })
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    KNOWLEDGE_INDEX = new_index
    print(f"[Sally] Indexing complete: {len(KNOWLEDGE_INDEX)} chunks ready.")
