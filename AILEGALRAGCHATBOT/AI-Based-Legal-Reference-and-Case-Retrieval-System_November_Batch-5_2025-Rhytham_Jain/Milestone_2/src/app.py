from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import get_rag_response

load_dotenv()

app = FastAPI(title="Legal AI Backend")

# ✅ Allow frontend (React) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class ChatRequest(BaseModel):
    query: str


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Legal AI Backend is live 🚀"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    answer = get_rag_response(req.query)
    return {"answer": answer}
