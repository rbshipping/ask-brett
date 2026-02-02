"""
Ask Brett (Semantic Search) - Uses TF-IDF embeddings for better search.
Run with: python ask_brett_semantic.py

First run: python build_index.py (one time only)
"""

import os
import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INDEX_FILE = OUTPUT_DIR / "search_index.pkl"

MAX_CHUNKS_TO_SEND = 8
MAX_CONTEXT_WORDS = 6000


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def load_index():
    """Load the pre-built TF-IDF index"""
    if not INDEX_FILE.exists():
        return None

    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def semantic_search(query, index_data, top_k=20):
    """
    Search using TF-IDF cosine similarity.
    Returns list of (score, chunk) tuples sorted by relevance.
    """
    vectorizer = index_data["vectorizer"]
    tfidf_matrix = index_data["tfidf_matrix"]
    chunks = index_data["chunks"]

    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity with all chunks
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top results
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.05:  # Minimum relevance threshold
            results.append((score, chunks[idx]))

    return results


def build_context(query, index_data, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    """
    Search and build context for Claude.
    Returns context string and list of sources.
    """
    results = semantic_search(query, index_data)

    if not results:
        return None, []

    context_parts = []
    sources = []
    total_words = 0

    for score, chunk in results:
        if len(sources) >= max_chunks:
            break

        content = chunk["full_text"]
        word_count = len(content.split())

        if total_words + word_count > max_words:
            continue

        context_parts.append(
            f"--- Source: {chunk['source']} (Topic: {chunk['topic']}, Relevance: {score:.0%}) ---\n{content}"
        )
        sources.append({
            "file": chunk["source"],
            "topic": chunk["topic"],
            "score": score
        })
        total_words += word_count

    if not context_parts:
        return None, []

    return "\n\n".join(context_parts), sources


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(client, question, context, sources):
    """Send question and context to Claude"""

    source_list = "\n".join([f"- {s['file']} ({s['topic']}) - {s['score']:.0%} match" for s in sources])

    system_prompt = """You are "Ask Brett", a helpful assistant that answers questions based on Brett Blundy's business knowledge, training materials, and documented conversations.

Your role:
- Answer questions using ONLY the provided context documents
- Quote or reference specific advice when relevant
- Be direct and practical, matching Brett's communication style
- If the context doesn't contain enough information to answer, say so clearly
- Always cite which source document(s) your answer comes from

Keep answers concise but complete. Use bullet points for actionable advice."""

    user_prompt = f"""Based on the following documents from Brett's knowledge base, please answer my question.

DOCUMENTS:
{context}

SOURCES (ranked by relevance):
{source_list}

MY QUESTION: {question}

Please provide a helpful answer based on the documents above, citing your sources."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  ASK BRETT - Semantic Search Edition")
    print("=" * 60)
    print("\nType your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Load environment
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key == "sk-ant-your-key-here":
        print("ERROR: Please add your Anthropic API key to the .env file")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Load index
    print("Loading search index...")
    index_data = load_index()

    if index_data is None:
        print("\nERROR: Search index not found!")
        print("Please run: python build_index.py")
        return

    print(f"Loaded {len(index_data['chunks'])} chunks.")
    print("Semantic search ready.\n")

    # Main loop
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        # Search and build context
        print("\nSearching (semantic)...")
        context, sources = build_context(question, index_data)

        if not context:
            print("\nNo relevant documents found.")
            print("Try rephrasing your question.\n")
            continue

        print(f"Found {len(sources)} relevant sources. Generating answer...\n")

        # Get answer
        try:
            answer = ask_claude(client, question, context, sources)
            print("-" * 60)
            print(f"\nBrett's Knowledge Base:\n")
            print(answer)
            print("\n" + "-" * 60)
            print("\nTop sources:")
            for s in sources[:5]:
                print(f"  - {s['file']} ({s['topic']}) [{s['score']:.0%}]")
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
