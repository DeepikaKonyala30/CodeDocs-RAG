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
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Only look at files with these extensions -- skips images, locks, etc.
VALID_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".py", ".md"}

# Skip noisy folders that aren't real source code
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".venv"}

# How many lines per chunk. Line-based chunking is the fallback for
# file types the AST chunker doesn't handle (.py, .md), or if AST
# parsing fails on a malformed/unusual JS file.
CHUNK_LINES = 40
CHUNK_OVERLAP = 5

# File types the AST chunker (Babel) understands.
AST_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}

AST_CHUNKER_PATH = Path(__file__).parent / "ast_chunk.js"


def find_source_files(root_dir):
    """Walk the repo and return paths to every file we want to index."""
    root = Path(root_dir)
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in VALID_EXTENSIONS:
            if not any(skip in path.parts for skip in SKIP_DIRS):
                files.append(path)
    return files


def ast_chunk_file(filepath):
    """
    Chunk a JS/TS file by function/class using ast_chunk.js (Babel).
    Returns None if parsing fails, so the caller can fall back to
    line-based chunking instead of losing the file entirely.
    """
    try:
        result = subprocess.run(
            ["node", str(AST_CHUNKER_PATH), str(filepath)],
            capture_output=True,
            text=True,
            encoding="utf-8",       # ✅ force UTF-8
            errors="ignore",        # ✅ skip bad characters
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  AST chunker unavailable for {filepath}: {e}")
        return None

    if result.returncode != 0 or not result.stdout:
        # ast_chunk.js prints PARSE_ERROR to stderr and exits 1 on bad syntax
        print(f"  AST parse failed for {filepath}, falling back to line chunks")
        return None

    try:
        raw_chunks = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"  JSON decode error for {filepath}: {e}")
        return None

    chunks = []
    for c in raw_chunks:
        chunks.append(
            {
                "text": c["text"],
                "source": str(filepath),
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "label": c.get("label"),
            }
        )
    return chunks


def line_chunk_file(filepath):
    """Split one file into overlapping line-based chunks. Fallback strategy."""
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
                    "label": None,
                }
            )
        if end == len(lines):
            break
        start += CHUNK_LINES - CHUNK_OVERLAP

    return chunks


def chunk_file(filepath):
    """
    Pick the right chunking strategy for this file.
    JS/TS files get AST-aware (function/class-level) chunking.
    Everything else, or any file where AST parsing fails, gets
    line-based chunking as a safe fallback.
    """
    if filepath.suffix in AST_EXTENSIONS:
        chunks = ast_chunk_file(filepath)
        if chunks is not None:
            return chunks
        # fall through to line-based if AST chunking returned None

    return line_chunk_file(filepath)


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
        already_done = {(c["source"], c["start_line"], c.get("label")) for c in vector_store}
        print(f"Found existing progress: {len(vector_store)} chunks already embedded. Resuming...")

    SAVE_EVERY = 10  # write progress to disk periodically, not just at the end

    for i, chunk in enumerate(all_chunks):
        key = (chunk["source"], chunk["start_line"], chunk.get("label"))
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
                "label": chunk.get("label"),
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
