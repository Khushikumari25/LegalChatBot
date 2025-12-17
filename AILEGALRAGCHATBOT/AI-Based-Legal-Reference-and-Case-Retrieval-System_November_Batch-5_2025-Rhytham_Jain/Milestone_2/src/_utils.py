# Milestone 2/src/_utils.py
"""
Shared utilities for Milestone 2
Integrates with Milestone 1's Pinecone setup
"""

import os
import re
from dotenv import load_dotenv
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = re.sub(r'[^a-z0-9\-]', '-', str(os.getenv("PINECONE_INDEX", "legal-docs-index")).strip().lower())
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
OUTPUT_MANIFEST = os.getenv("OUTPUT_MANIFEST", "./pinecone_manifest.json")

# Safe imports for Pinecone
try:
    from pinecone import Pinecone
except Exception:
    import pinecone as pinecone_client
    Pinecone = pinecone_client.Pinecone

import numpy as np
from sentence_transformers import SentenceTransformer

# Load local sentence-transformers model once
_local_model = None

def _ensure_local_model():
    """Load and cache the local embedding model"""
    global _local_model
    if _local_model is None:
        print(f"Loading embedding model: {EMBED_MODEL}")
        _local_model = SentenceTransformer(EMBED_MODEL)
    return _local_model

def embed_text(text):
    """
    Generate embedding for given text
    
    Args:
        text: Input text string
    
    Returns:
        numpy array of embeddings (normalized float32)
    """
    model = _ensure_local_model()
    arr = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def convert_vector(vec_np):
    """
    Convert numpy vector to Python list of floats
    Validates no NaN/Inf values
    
    Args:
        vec_np: numpy array
    
    Returns:
        List of floats
    """
    if not np.issubdtype(vec_np.dtype, np.number):
        raise ValueError("Embedding dtype is not numeric")
    if not np.all(np.isfinite(vec_np)):
        raise ValueError("Embedding contains NaN/Inf")
    return [float(x) for x in vec_np.tolist()]

def init_pinecone():
    """
    Initialize Pinecone client
    
    Returns:
        Pinecone client instance
    """
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not found in .env file")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def safe_list_index_names(pc):
    """
    Get list of index names (handles different Pinecone API versions)
    
    Args:
        pc: Pinecone client
    
    Returns:
        List of index names
    """
    raw = pc.list_indexes()
    names = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and "name" in item:
            names.append(item["name"])
        else:
            nm = getattr(item, "name", None)
            if nm:
                names.append(nm)
    return names

def get_index_obj(pc):
    """
    Get Pinecone index object
    Finds the latest index matching PINECONE_INDEX pattern
    
    Args:
        pc: Pinecone client
    
    Returns:
        Tuple of (index_object, index_name)
    """
    names = safe_list_index_names(pc)
    
    # Look for exact match or timestamped versions
    matches = [n for n in names if n == PINECONE_INDEX or n.startswith(PINECONE_INDEX + "-")]
    
    if not matches:
        raise RuntimeError(
            f"No Pinecone index found matching '{PINECONE_INDEX}'. "
            f"Available indexes: {names}\n"
            f"Please run the Milestone 1 upsert_to_pinecone.py script first."
        )
    
    # Sort and get latest (in case of multiple timestamped versions)
    matches.sort(reverse=True)
    idx_name = matches[0]
    
    print(f"✅ Connected to Pinecone index: {idx_name}")
    return pc.Index(idx_name), idx_name

def query_index_with_fallback(index_obj, q_list, top_k=5):
    """
    Query Pinecone index with fallback for different API versions
    
    Args:
        index_obj: Pinecone index object
        q_list: Query vector as list
        top_k: Number of results to return
    
    Returns:
        Query results
    """
    last_exc = None
    
    # Try different query formats for compatibility
    try:
        return index_obj.query(
            vector=q_list, 
            top_k=top_k, 
            include_metadata=True, 
            include_values=False
        )
    except Exception as e:
        last_exc = e
    
    try:
        return index_obj.query(
            queries=[q_list], 
            top_k=top_k, 
            include_metadata=True, 
            include_values=False
        )
    except Exception as e:
        last_exc = e
    
    try:
        return index_obj.query(
            queries=[{"values": q_list}], 
            top_k=top_k, 
            include_metadata=True, 
            include_values=False
        )
    except Exception as e:
        last_exc = e
    
    raise last_exc