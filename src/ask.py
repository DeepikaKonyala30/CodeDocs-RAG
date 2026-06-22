"""
Phase 2b: Generation.
Takes a question, retrieves relevant code chunks, builds a grounded
prompt, and asks Gemini to answer using only that context.

Usage:
    python src/ask.py "How does the budget tracking feature work?"
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from retrieve import retrieve_top_k

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_prompt(question, chunks):
    """Stuff retrieved chunks into a prompt that grounds the model's answer."""
    context_blocks = []
    for chunk in chunks:
        block = (
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code context below to answer. If the context doesn't contain
the answer, say so honestly instead of guessing.

Always cite which file(s) your answer is based on.

CODE CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def ask(question, k=4):
    print(f"Retrieving top {k} relevant chunks...")
    chunks = retrieve_top_k(question, k=k)

    prompt = build_prompt(question, chunks)

    print("Generating answer...\n")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text, chunks


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/ask.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    answer, sources_used = ask(question)

    print("=" * 60)
    print("ANSWER:")
    print(answer)
    print("=" * 60)
    print("\nSources retrieved:")
    for s in sources_used:
        print(f"  - {s['source']} (lines {s['start_line']}-{s['end_line']})")


if __name__ == "__main__":
    main()
