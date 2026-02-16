import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the API Key safely from the environment
# We will set this in Vercel's dashboard later
api_key = os.getenv("AIPIPE_KEY") 
client = OpenAI(
    api_key=api_key, 
    base_url="https://aipipe.org/openai/v1"
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: list[str]

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured")
        
    try:
        # Get embeddings
        query_emb = client.embeddings.create(input=[request.query], model="text-embedding-3-small").data[0].embedding
        doc_embs = [d.embedding for d in client.embeddings.create(input=request.docs, model="text-embedding-3-small").data]

        # Calculate scores
        scores = []
        for doc, emb in zip(request.docs, doc_embs):
            # Cosine similarity math
            similarity = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scores.append((doc, similarity))

        # Sort and return top 3
        scores.sort(key=lambda x: x[1], reverse=True)
        return {"matches": [doc for doc, score in scores[:3]]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "active"}
