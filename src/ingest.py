"""
Phase 1: Ingestion pipeline.
Walks a target codebase, splits files into chunks, embeds each chunk
using the Gemini embedding API, and saves everything to a local JSON
file that acts as our (very simple) vector store.

Run this once whenever the target codebase changes:
    python src/ingest.py /path/to/SmartLedger
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Only look at files with these extensions -- skips images, locks, etc.
VALID_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".py", ".md"}

# Skip noisy folders that aren't real source code
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".venv"}

# How many lines per chunk. Simple line-based chunking for v1 --
# AST-aware (function-level) chunking is a planned upgrade.
CHUNK_LINES = 40
CHUNK_OVERLAP = 5


def find_source_files(root_dir):
    """Walk the repo and return paths to every file we want to index."""
    root = Path(root_dir)
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in VALID_EXTENSIONS:
            if not any(skip in path.parts for skip in SKIP_DIRS):
                files.append(path)
    return files


def chunk_file(filepath):
    """Split one file into overlapping line-based chunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        chunk_text = "".join(lines[start:end])
        if chunk_text.strip():  # skip empty chunks
            chunks.append(
                {
                    "text": chunk_text,
                    "source": str(filepath),
                    "start_line": start + 1,
                    "end_line": end,
                }
            )
        if end == len(lines):
            break
        start += CHUNK_LINES - CHUNK_OVERLAP

    return chunks


def embed_chunk(text):
    """Call Gemini's embedding model on a single chunk of text."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/ingest.py /path/to/target/repo")
        sys.exit(1)

    target_repo = sys.argv[1]
    print(f"Scanning {target_repo} ...")

    files = find_source_files(target_repo)
    print(f"Found {len(files)} source files")

    all_chunks = []
    for filepath in files:
        all_chunks.extend(chunk_file(filepath))

    print(f"Created {len(all_chunks)} chunks. Embedding now (this calls the API)...")

    output_path = Path(__file__).parent.parent / "data" / "vector_store.json"

    # Resume support: if a previous run already saved progress, pick up from there
    # instead of re-embedding (and re-spending API calls on) chunks we already have.
    vector_store = []
    already_done = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            vector_store = json.load(f)
        already_done = {(c["source"], c["start_line"]) for c in vector_store}
        print(f"Found existing progress: {len(vector_store)} chunks already embedded. Resuming...")

    SAVE_EVERY = 10  # write progress to disk periodically, not just at the end

    for i, chunk in enumerate(all_chunks):
        key = (chunk["source"], chunk["start_line"])
        if key in already_done:
            continue  # already embedded in a previous run

        try:
            embedding = embed_chunk(chunk["text"])
        except Exception as e:
            print(f"  Skipped a chunk from {chunk['source']}: {e}")
            continue

        vector_store.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "embedding": embedding,
            }
        )
        print(f"  Embedded chunk {i + 1}/{len(all_chunks)}")

        if len(vector_store) % SAVE_EVERY == 0:
            with open(output_path, "w") as f:
                json.dump(vector_store, f)
            print(f"  -- progress saved ({len(vector_store)} chunks so far) --")

        time.sleep(0.2)  # gentle pacing to avoid rate limits

    with open(output_path, "w") as f:
        json.dump(vector_store, f)

    print(f"\nSaved {len(vector_store)} embedded chunks to {output_path}")


if __name__ == "__main__":
    main()