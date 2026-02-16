import os
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# We use a standard, small, efficient model hosted by Hugging Face
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL_ID}"

# Get the token from Environment Variable
hf_token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {hf_token}"}

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

def query_hf_api(texts):
    """
    Sends text to Hugging Face and returns embeddings.
    """
    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        if response.status_code != 200:
            logger.error(f"HF API Error: {response.text}")
            raise Exception(f"Hugging Face API Error: {response.status_code}")
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise e

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    try:
        # 1. Get Embeddings for Query AND Docs in one batch call (faster)
        # We combine them into one list: [query, doc1, doc2, doc3...]
        all_texts = [request.query] + request.docs
        
        embeddings = query_hf_api(all_texts)
        
        # Check if HF is loading the model (it sometimes returns a list of errors)
        if isinstance(embeddings, dict) and "error" in embeddings:
             raise HTTPException(status_code=503, detail="Model is loading, please try again in 10 seconds.")

        # 2. Separate Query Embedding from Doc Embeddings
        query_emb = np.array(embeddings[0])
        doc_embs = [np.array(e) for e in embeddings[1:]]

        # 3. Calculate Scores
        scores = []
        for doc_text, doc_vec in zip(request.docs, doc_embs):
            score = cosine_similarity(query_emb, doc_vec)
            scores.append((doc_text, score))

        # 4. Sort and Return Top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = [doc for doc, score in scores[:3]]
        
        return SimilarityResponse(matches=top_matches)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "active", "provider": "HuggingFace Free Tier"}
