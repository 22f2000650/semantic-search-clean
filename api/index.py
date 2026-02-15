import numpy as np
import random
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS = [
    "Artificial intelligence in healthcare",
    "Machine learning in finance",
    "Solar energy systems",
    "Deep learning applications",
    "Neural networks and AI",
    "Stock market investment strategies",
    "Data science in business",
    "Cloud computing architecture"
]

class SearchRequest(BaseModel):
    query: str
    k: int = 3
    rerank: bool = False
    rerankK: int = 3

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/similarity")
def similarity(request: SearchRequest):
    k = min(request.k, len(DOCUMENTS))
    rerank_k = min(request.rerankK, k)

    vectors = [[random.random() for _ in range(100)] for _ in range(len(DOCUMENTS)+1)]
    query_vector = vectors[0]
    doc_vectors = vectors[1:]

    scores = []
    for i, doc_vector in enumerate(doc_vectors):
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    final = scores[:rerank_k]

    return {
        "results": [
            {"id": doc_id, "score": float(score)}
            for score, doc_id in final
        ]
    }
