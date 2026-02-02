"""
Ask Brett - Conversational interface to search and query your knowledge base.
Run with: python ask_brett.py
"""

import os
import csv
from pathlib import Path

from dotenv import load_dotenv
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
INDEX_PATH = OUTPUT_DIR / "index.csv"

# Search settings
MAX_CHUNKS_TO_SEND = 8  # Maximum chunks to include in context
MAX_CONTEXT_WORDS = 6000  # Maximum words to send to Claude


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def load_index():
    """Load the index.csv into memory"""
    index = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append(row)
    return index


def search_chunks(query, index):
    """
    Search for relevant chunks based on the query.
    Returns list of matching index entries, scored by relevance.
    """
    # Clean and expand query words
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) >= 3]

    # Common word stems to help matching
    scored_results = []

    for entry in index:
        score = 0

        # Search metadata
        metadata_text = (
            entry["topic"].lower() + " " +
            entry["summary"].lower() + " " +
            entry["keywords"].lower() + " " +
            entry["source_file"].lower()
        )

        # Also load and search actual chunk content for better matching
        chunk_content = ""
        chunk_path = CHUNKS_DIR / entry["chunk_file"]
        if chunk_path.exists():
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    chunk_content = f.read().lower()
            except:
                pass

        full_text = metadata_text + " " + chunk_content

        # Score based on word matches (partial matching)
        for word in query_words:
            # Exact word match
            if word in full_text:
                score += 1
            # Check for word stem matches (e.g., "negotiate" matches "negotiating", "negotiation")
            word_stem = word[:5] if len(word) > 5 else word[:3]
            if word_stem in full_text:
                score += 0.5
            # Bonus for matches in topic or keywords
            if word in entry["topic"].lower():
                score += 3
            if word in entry["keywords"].lower():
                score += 3
            if word_stem in entry["topic"].lower():
                score += 1
            if word_stem in entry["keywords"].lower():
                score += 1

        # Bonus for matching multiple query words
        if score > len(query_words):
            score += 2

        if score > 0:
            scored_results.append((score, entry))

    # Sort by score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Return just the entries (without scores)
    return [entry for score, entry in scored_results]


def load_chunk_content(chunk_file):
    """Load the full text content of a chunk file"""
    chunk_path = CHUNKS_DIR / chunk_file
    if chunk_path.exists():
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def build_context(query, index, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    """
    Search for relevant chunks and build context for Claude.
    Returns the context string and list of sources used.
    """
    matches = search_chunks(query, index)

    if not matches:
        return None, []

    context_parts = []
    sources = []
    total_words = 0

    for entry in matches[:max_chunks * 2]:  # Get extra in case some are too long
        if len(sources) >= max_chunks:
            break

        content = load_chunk_content(entry["chunk_file"])
        if not content:
            continue

        word_count = len(content.split())
        if total_words + word_count > max_words:
            continue

        # Add to context
        context_parts.append(f"--- Source: {entry['source_file']} (Topic: {entry['topic']}) ---\n{content}")
        sources.append({
            "file": entry["source_file"],
            "topic": entry["topic"],
            "chunk": entry["chunk_file"]
        })
        total_words += word_count

    if not context_parts:
        return None, []

    context = "\n\n".join(context_parts)
    return context, sources


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(client, question, context, sources):
    """Send question and context to Claude, get response"""

    source_list = "\n".join([f"- {s['file']} ({s['topic']})" for s in sources])

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

SOURCES AVAILABLE:
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
    print("  ASK BRETT - Your Business Knowledge Assistant")
    print("=" * 60)
    print("\nType your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Load environment and initialize
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key == "sk-ant-your-key-here":
        print("ERROR: Please add your Anthropic API key to the .env file")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Load index
    print("Loading knowledge base...")
    index = load_index()
    print(f"Loaded {len(index)} chunks from {len(set(e['source_file'] for e in index))} documents.\n")

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
        print("\nSearching knowledge base...")
        context, sources = build_context(question, index)

        if not context:
            print("\nNo relevant documents found for your question.")
            print("Try rephrasing or using different keywords.\n")
            continue

        print(f"Found {len(sources)} relevant sources. Generating answer...\n")

        # Get answer from Claude
        try:
            answer = ask_claude(client, question, context, sources)
            print("-" * 60)
            print(f"\nBrett's Knowledge Base:\n")
            print(answer)
            print("\n" + "-" * 60)
            print(f"\nSources used: {', '.join(s['file'] for s in sources)}\n")

        except Exception as e:
            print(f"\nError getting response: {e}\n")


if __name__ == "__main__":
    main()
