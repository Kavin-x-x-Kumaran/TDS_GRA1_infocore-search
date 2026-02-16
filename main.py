import os
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

@app.get("/")
async def home():
    return {"status": "online", "method": "direct_httpx"}

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    # --- 1. Get Settings ---
    api_key = os.getenv("AIPIPE_KEY")
    # AI Pipe standard endpoint for embeddings
    url = "https://api.aipipe.org/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": [request.query] + request.docs,
        "model": "openai/text-embedding-3-small"
    }

    # --- 2. Call API Directly ---
    try:
        # We use a standard HTTP client to avoid "Connection Error" bugs in the OpenAI SDK
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"AI Pipe Error: {response.text}")
            
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")

    # --- 3. Math (Cosine Similarity) ---
    query_vec = np.array(embeddings[0])
    doc_vecs = [np.array(e) for e in embeddings[1:]]

    scores = []
    for doc_text, doc_vec in zip(request.docs, doc_vecs):
        norm = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        # Avoid division by zero
        similarity = np.dot(query_vec, doc_vec) / norm if norm > 0 else 0
        scores.append((doc_text, float(similarity)))

    # --- 4. Rank and Return ---
    scores.sort(key=lambda x: x[1], reverse=True)
    return {"matches": [doc for doc, score in scores[:3]]}
