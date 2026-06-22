# CodeDocs RAG

A retrieval-augmented generation system that answers questions about a codebase by retrieving the most relevant code chunks and grounding the LLM's answer in them — built from scratch with **no RAG frameworks** (no LangChain, no Chroma, no FAISS).

## Why no frameworks

Frameworks like LangChain are great in production, but they hide the actual mechanics of RAG behind abstractions. This project implements the pipeline directly — chunking, embedding, cosine similarity search, and prompt construction — using only the Gemini API and `numpy`. The goal was to understand exactly what's happening at each step, not just call a `.run()` method.

## How it works

**Phase 1 — Indexing (offline, run once per codebase):**
1. Walk the target repo and collect source files
2. Split each file into overlapping line-based chunks
3. Embed every chunk with Gemini's embedding model
4. Save chunks + embeddings to a local JSON file (`data/vector_store.json`)

**Phase 2 — Querying (runs on every question):**
1. Embed the user's question with the same embedding model
2. Compute cosine similarity between the question and every stored chunk
3. Take the top-k most similar chunks
4. Build a prompt that includes those chunks as context
5. Ask Gemini to answer using only that context, with citations

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your real Gemini API key
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Usage

```bash
# Step 1: index a codebase (point this at any local repo)
python src/ingest.py /path/to/SmartLedger

# Step 2: ask questions about it
python src/ask.py "How does the budget tracking feature work?"
```

## Project structure

```
src/
  ingest.py    - chunking + embedding pipeline
  retrieve.py  - cosine similarity search over stored embeddings
  ask.py       - prompt construction + answer generation
data/
  vector_store.json - generated after running ingest.py (not committed)
```

## Current limitations / planned upgrades

- Chunking is line-based (fixed 40-line windows with overlap), not AST-aware. This means a function can occasionally be split across two chunks. Function/class-level chunking using Python's `ast` module or `tree-sitter` is a planned upgrade.
- Retrieval is pure vector similarity. No hybrid (keyword + vector) search yet.
- No reranking step after initial retrieval.
- No evaluation set yet to measure retrieval quality.

## Stack

Python, Gemini API (`gemini-embedding-001` for embeddings, `gemini-2.5-flash` for generation), numpy for vector math.
