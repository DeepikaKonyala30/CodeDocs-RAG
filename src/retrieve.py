"""
Phase 2a: Retrieval.
Given a question, embed it and find the most similar chunks
in our vector store using cosine similarity -- no vector DB needed.
"""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"


def load_vector_store():
    with open(VECTOR_STORE_PATH, "r") as f:
        return json.load(f)


def embed_query(text):
    """Embed the user's question the same way we embedded chunks."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


def cosine_similarity(vec_a, vec_b):
    """How similar are two vectors? 1.0 = identical direction, 0 = unrelated."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(question, k=4):
    """Return the k chunks most similar to the question."""
    store = load_vector_store()
    query_embedding = embed_query(question)

    scored = []
    for chunk in store:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = []
    for score, chunk in scored[:k]:
        chunk_with_score = {**chunk, "similarity_score": round(score, 4)}
        top_chunks.append(chunk_with_score)

    return top_chunks


if __name__ == "__main__":
    # Quick manual test: run this file directly to sanity-check retrieval
    test_question = "How does the app authenticate users?"
    results = retrieve_top_k(test_question)
    for r in results:
        print(f"\n--- {r['source']} (lines {r['start_line']}-{r['end_line']}) ---")
        print(r["text"][:200])