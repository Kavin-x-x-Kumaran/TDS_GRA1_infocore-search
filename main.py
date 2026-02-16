import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI Client Setup ---
# Using the standard AI Pipe v1 endpoint
client = OpenAI(
    api_key=os.getenv("AIPIPE_KEY"), 
    base_url="https://api.aipipe.org/v1" 
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

# --- Health Check Route ---
# Open your URL in a browser; if you see this, the app is working!
@app.get("/")
async def home():
    return {"status": "online", "model": "text-embedding-3-small"}

# --- Main Similarity Logic ---
@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    if not request.docs or not request.query:
        raise HTTPException(status_code=400, detail="Missing docs or query")

    try:
        # 1. Get Embeddings (Batch query + docs for efficiency)
        all_texts = [request.query] + request.docs
        
        response = client.embeddings.create(
            input=all_texts,
            model="openai/text-embedding-3-small"
        )
        
        # 2. Extract and Convert to Numpy Arrays
        embeddings = [item.embedding for item in response.data]
        query_vec = np.array(embeddings[0])
        doc_vecs = [np.array(e) for e in embeddings[1:]]

        # 3. Calculate Cosine Similarity
        # Formula: (A . B) / (||A|| * ||B||)
        scores = []
        for doc_text, doc_vec in zip(request.docs, doc_vecs):
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            if norm_product == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vec, doc_vec) / norm_product
            scores.append((doc_text, float(similarity)))

        # 4. Sort by highest similarity and take top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = [doc for doc, score in scores[:3]]

        return {"matches": top_matches}

    except Exception as e:
        # This will show up in your Vercel Logs if something fails
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
