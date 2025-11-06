from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
from sentence_transformers import SentenceTransformer
import faiss
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are Parikshit’s personal AI assistant. "
    "You are logical, accurate, and answer clearly in simple language."
)

# Optional vector memory folder
DB_PATH = "knowledge"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_FILE = os.path.join(DB_PATH, "index.faiss")
META_FILE = os.path.join(DB_PATH, "meta.json")

# Load memory index if present
if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r") as f:
        texts = json.load(f)
else:
    index, texts = None, []

@app.get("/")
def home():
    return {"status": "AI backend running!"}


@app.post("/chat")
async def chat(message: str = Form(...)):
    # retrieve context from knowledge memory if available
    context = ""
    if index is not None and len(texts) > 0:
        query_vec = EMBED_MODEL.encode([message])
        D, I = index.search(query_vec, 3)
        context = "\n".join([texts[i] for i in I[0] if i < len(texts)])

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser: {message}\nAI:"

    if not shutil.which("ollama"):
        return {"reply": "⚠️ Ollama not found. Please install it from https://ollama.com."}

    cmd = ["ollama", "run", "mistral", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    reply = result.stdout.strip() if result.stdout else "No response."
    return {"reply": reply}


@app.post("/train")
async def train(text: str = Form(...)):
    """Add text data to your knowledge memory"""
    os.makedirs(DB_PATH, exist_ok=True)
    global index, texts

    texts.append(text)
    vec = EMBED_MODEL.encode([text])

    if index is None:
        index = faiss.IndexFlatL2(vec.shape[1])

    index.add(vec)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(texts, f)

    return {"status": "Data added to memory", "total_docs": len(texts)}
