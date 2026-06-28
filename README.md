# CodeDocs RAG

A retrieval-augmented generation system that answers questions about a codebase by retrieving the most relevant code chunks and grounding the LLM's answer in them — built from scratch with **no RAG frameworks** (no LangChain, no Chroma, no FAISS).

## Why no frameworks

Frameworks like LangChain are great in production, but they hide the actual mechanics of RAG behind abstractions. This project implements the pipeline directly — chunking, embedding, cosine similarity search, and prompt construction — using only the Gemini API and `numpy`. The goal was to understand exactly what's happening at each step, not just call a `.run()` method.

## How it works

**Phase 1 — Indexing (offline, run once per codebase):**
1. Walk the target repo and collect source files
2. For JS/TS files: parse with Babel's AST parser and extract each function, class, and method as a complete, structurally-correct chunk (see `src/ast_chunk.js`). For other file types, or if AST parsing fails on a file, fall back to overlapping line-based chunks.
3. Embed every chunk with Gemini's embedding model
4. Save chunks + embeddings to a local JSON file (`data/vector_store.json`), with resume support so an interrupted run doesn't re-embed (and re-spend API quota on) chunks already processed

**Phase 2 — Querying (runs on every question):**
1. Embed the user's question with the same embedding model
2. Compute cosine similarity between the question and every stored chunk
3. Take the top-k most similar chunks
4. Build a prompt that includes those chunks as context, labeled by function/class name where available
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

- AST chunking covers function declarations, arrow-function/expression assignments (including `exports.fn =` and `module.exports.fn =` patterns), classes, and class methods. Top-level imports and standalone object/config literals aren't captured as chunks.
- **Known gap, confirmed by testing:** very long functions (100+ lines handling several sequential responsibilities, e.g. validation + calculation + DB query + threshold check in one function) produce a single large chunk. Because a chunk's embedding represents the average meaning of everything inside it, long multi-responsibility functions retrieve with noticeably weaker, less discriminating similarity scores than short, single-purpose functions. Confirmed directly on `budgetCheckMiddleware.js` (122 lines, one function) — top-4 retrieval scores for a related query clustered tightly around 0.62-0.64 rather than showing a clear best match. Sub-function chunking (splitting one long function into logical blocks) is a planned upgrade to address this.
- Retrieval is pure vector similarity. No hybrid (keyword + vector) search yet.
- No reranking step after initial retrieval.
- No evaluation set yet to measure retrieval quality.
- AST chunking currently supports JS/TS (via Babel). Python files still use line-based chunking.

## Stack

Python, Node.js (for AST parsing only), Gemini API (`gemini-embedding-001` for embeddings, `gemini-2.5-flash` for generation), `@babel/parser` + `@babel/traverse` for AST-aware JS/TS chunking, numpy for vector math.