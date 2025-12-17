#!/usr/bin/env python3
"""
Upsert documents into Pinecone (new API) with Gemini -> local fallback embeddings.
FIXED: Uses hyphens for index names to comply with Pinecone naming rules.
"""

import os
import json
import time
import re
import datetime
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# === ENV CONFIG ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# Ensure base index name is lowercase
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ai-legal-index").lower()

INPUT_JSON = os.getenv("INPUT_JSON", "./loaded_documents.json")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
SAMPLE_QUERY = os.getenv("SAMPLE_QUERY", "right to privacy landmark case summary")
OUTPUT_MANIFEST = os.getenv("OUTPUT_MANIFEST", "./pinecone_manifest.json")

if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in your .env file")


# === DEPENDENCIES ===
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

# LangChain splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # compatibility

# Embeddings
from sentence_transformers import SentenceTransformer

# Optional Gemini
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# PDF/HTML parsing
try:
    import fitz
    HAS_PYMUPDF = True
except:
    HAS_PYMUPDF = False

from pdfminer.high_level import extract_text as pdfminer_extract_text
from bs4 import BeautifulSoup


# === TEXT CLEANING ===
def clean_text(s: str) -> str:
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"-\n(\w)", lambda m: m.group(1), s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


# === FILE LOADERS ===
def extract_pdf(path: Path) -> str:
    text = ""
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(path))
            pages = [p.get_text() for p in doc]
            text = "\n".join(pages)
        except:
            pass
    if not text:
        try:
            text = pdfminer_extract_text(str(path))
        except:
            text = ""
    return clean_text(text)


def extract_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    main = soup.find(id="content") or soup.find("main") or soup
    return clean_text(main.get_text(separator="\n"))


def load_from_json(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    docs = []
    for entry in data:
        text = entry.get("text") or entry.get("content") or ""
        docs.append({
            "filename": entry.get("filename"),
            "path": entry.get("path") or entry.get("source_location"),
            "type": entry.get("type"),
            "text": text
        })
    return docs


def load_from_folders(data_dir: Path):
    docs = []
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        return []
        
    pdfs = (data_dir / "pdfs").glob("*.pdf")
    htmls = (data_dir / "html").glob("*.html")

    for p in pdfs:
        docs.append({"filename": p.name, "path": str(p), "type": "pdf", "text": extract_pdf(p)})
    for h in htmls:
        docs.append({"filename": h.name, "path": str(h), "type": "html", "text": extract_html(h)})

    return docs


# === CHUNKING ===
def chunk_docs(docs: List[Dict[str, Any]]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for d in docs:
        pieces = splitter.split_text(d["text"])
        for i, p in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{Path(d['filename']).stem}_c{i}",
                "doc_filename": d["filename"],
                "doc_path": d["path"],
                "text": p,
                "chunk_index": i,
                "chunk_len": len(p)
            })
    return chunks


# === EMBEDDINGS: Gemini + fallback ===
_local_model = None

def get_local_model():
    global _local_model
    if _local_model is None:
        _local_model = SentenceTransformer(EMBED_MODEL_LOCAL)
    return _local_model


def embed_with_fallback(texts):
    # Try Gemini if key exists
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            gem_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Using Gemini embeddings...")
            return np.array(gem_emb.embed_documents(texts))
        except Exception as e:
            # Check specifically for quota errors to give a clearer message
            if "429" in str(e) or "Quota" in str(e):
                 print("⚠️ Gemini Quota Exceeded. Switching to local embeddings permanently for this run.")
            else:
                 print("Gemini failed -> falling back to local:", e)

    print("Using local MiniLM embeddings (Dimensions: 384)...")
    model = get_local_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# === PINECONE (new API) ===
def init_pinecone_new():
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index(pc: Pinecone, index_name: str, dim: int):
    # Normalize name to lowercase just in case
    index_name = index_name.lower()
    
    try:
        existing = [i["name"] for i in pc.list_indexes()]
    except:
        # Fallback for older client versions or different response structures
        existing = [i.name for i in pc.list_indexes()]

    if index_name in existing:
        info = pc.describe_index(index_name)
        if info.dimension == dim:
            print(f"Using existing index: {index_name}")
            return pc.Index(index_name)
        else:
            # FIX: Use hyphens instead of underscores for compatibility
            # FIX: Use timezone-aware datetime
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
            new_name = f"{index_name}-{dim}d-{ts}"
            print(f"Dimension mismatch (Index: {info.dimension}, Vectors: {dim}) -> Creating new index: {new_name}")
            
            pc.create_index(
                name=new_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1") # Defaulting to us-east-1 if env var missing
            )
            # Wait for index to be ready
            while not pc.describe_index(new_name).status['ready']:
                time.sleep(1)
            return pc.Index(new_name)

    # Index doesn't exist -> create
    print(f"Creating new index: {index_name} with dimension {dim}")
    try:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    except Exception as e:
        print(f"Error creating index: {e}")
        # Try creating with a unique timestamp if the name is taken or invalid
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        fallback_name = f"legal-docs-{dim}d-{ts}"
        print(f"Retrying with fallback name: {fallback_name}")
        pc.create_index(
            name=fallback_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(fallback_name).status['ready']:
            time.sleep(1)
        return pc.Index(fallback_name)
        
    return pc.Index(index_name)


# === UPSERT ===
def upsert_chunks(index, chunks, vectors, batch_size):
    total = len(chunks)
    print(f"Starting upsert of {total} vectors...")
    
    for start in range(0, total, batch_size):
        end = start + batch_size
        batch = chunks[start:end]
        vecs = vectors[start:end]

        to_upsert = []
        for c, v in zip(batch, vecs):
            to_upsert.append({
                "id": c["chunk_id"],
                "values": v.tolist(),
                "metadata": {
                    "doc_filename": c["doc_filename"],
                    "doc_path": c["doc_path"],
                    "chunk_index": c["chunk_index"],
                    "chunk_len": c["chunk_len"],
                    "text": c["text"] # Storing text in metadata for easier retrieval
                }
            })

        try:
            index.upsert(vectors=to_upsert)
            print(f"Upserted batch {start}-{end} ({len(to_upsert)} items)")
        except Exception as e:
            print(f"Error upserting batch {start}: {e}")


# === SAMPLE QUERY ===
def sample_query(index, query: str):
    print(f"\nRunning sample query: '{query}'")
    # Must use the SAME model for query as for documents
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY") and False: # Force local for now since quota is dead
         pass 
    
    # We are using local model because of the fallback earlier
    model = get_local_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    res = index.query(
        vector=vec.tolist()[0],
        top_k=3,
        include_metadata=True
    )
    
    print("\n=== Sample Query Results ===")
    for match in res["matches"]:
        score = match["score"]
        meta = match["metadata"]
        print(f"\nScore: {score:.4f}")
        print(f"Source: {meta.get('doc_filename')}")
        print(f"Text Snippet: {meta.get('text', '')[:200]}...")


# === MAIN ===
def main():
    # 1. Load docs
    if Path(INPUT_JSON).exists():
        docs = load_from_json(Path(INPUT_JSON))
    else:
        print(f"JSON not found at {INPUT_JSON}, scanning folders...")
        docs = load_from_folders(DATA_DIR)

    if not docs:
        print("No documents found. Exiting.")
        return

    print(f"Loaded {len(docs)} documents.")

    # 2. Chunk
    chunks = chunk_docs(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embeddings
    texts = [c["text"] for c in chunks]
    vectors = embed_with_fallback(texts)
    dim = vectors.shape[1]
    print("Embedding dimension =", dim)

    # 4. Pinecone (new client)
    pc = init_pinecone_new()
    index = ensure_index(pc, PINECONE_INDEX, dim)

    # 5. Upsert
    upsert_chunks(index, chunks, vectors, BATCH_SIZE)

    # 6. Query
    sample_query(index, SAMPLE_QUERY)

    # 7. Manifest
    try:
        Path(OUTPUT_MANIFEST).write_text(
            json.dumps(chunks, indent=2),
            encoding="utf-8"
        )
        print("Manifest saved:", OUTPUT_MANIFEST)
    except Exception as e:
        print(f"Could not save manifest (non-critical): {e}")


if __name__ == "__main__":
    main()