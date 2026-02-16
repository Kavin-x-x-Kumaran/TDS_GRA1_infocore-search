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
# Switching to BAAI/bge-small-en-v1.5 which is currently active on free tier
# and using the direct /models/ endpoint which is more stable.
HF_MODEL_ID = "BAAI/bge-small-en-v1.5"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Get the token from Environment Variable
hf_token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {hf_token}"}

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

def query_hf_api(payload):
    """
    Sends text to Hugging Face and returns embeddings.
    """
    try:
        response = requests.post(
            HF_API_URL, 
            headers=headers, 
            json=payload
        )
        return response.json()
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise e

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    try:
        # 1. Prepare inputs
        # The /models/ endpoint expects a specific structure:
        # { "inputs": { "source_sentence": "query", "sentences": ["doc1", "doc2"] } }
        # BUT for feature-extraction models, it usually just takes a list of strings.
        # We will send them individually if batching fails, but let's try batch first.
        
        all_texts = [request.query] + request.docs
        
        # Request embeddings
        data = query_hf_api({"inputs": all_texts, "options": {"wait_for_model": True}})

        # Error Handling for HF specific errors
        if isinstance(data, dict) and "error" in data:
            raise HTTPException(status_code=500, detail=f"HF Error: {data['error']}")
            
        # 2. Parse Embeddings
        # The BGE model might return a slightly different shape, but usually it's a list of lists.
        # If it returns a 410/404, we catch it in the exception.
        
        embeddings = np.array(data)
        
        # Ensure we got valid arrays
        if embeddings.ndim != 2:
             raise HTTPException(status_code=500, detail=f"Unexpected API response format: {str(data)[:100]}")

        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

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
    return {"status": "active", "model": HF_MODEL_ID}
