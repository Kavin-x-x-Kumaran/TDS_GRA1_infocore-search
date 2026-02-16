import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# --- CORS (Required) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# Note: AI Pipe usually prefers the 'openai/' prefix in the model name 
# to ensure it routes to the correct provider.
client = OpenAI(
    api_key=os.getenv("AIPIPE_KEY"), 
    base_url="https://api.aipipe.org/openai/v1" 
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    try:
        # 1. Get Embeddings using the exact model requested
        # We combine query + docs to get all embeddings in one API call (cheaper/faster)
        all_texts = [request.query] + request.docs
        
        response = client.embeddings.create(
            input=all_texts,
            model="openai/text-embedding-3-small" # <--- Added 'openai/' prefix
        )
        
        # 2. Extract vectors
        embeddings = [item.embedding for item in response.data]
        query_vec = np.array(embeddings[0])
        doc_vecs = [np.array(e) for e in embeddings[1:]]

        # 3. Calculate Cosine Similarity
        scores = []
        for doc_text, doc_vec in zip(request.docs, doc_vecs):
            # The math: (A dot B) / (||A|| * ||B||)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            similarity = np.dot(query_vec, doc_vec) / norm_product
            scores.append((doc_text, similarity))

        # 4. Rank and return top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = [doc for doc, score in scores[:3]]

        return {"matches": top_matches}

    except Exception as e:
        # If AI Pipe fails, we want to see the EXACT error message in Vercel logs
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
