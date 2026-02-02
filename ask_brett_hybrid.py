"""
Ask Brett (Hybrid Search) - Combines keyword + semantic search with conversation memory.
Run with: python ask_brett_hybrid.py

First run: python build_index.py (one time only)

Features:
- Hybrid search using Reciprocal Rank Fusion (keyword + semantic)
- Conversation memory (last 3 exchanges)
- Content flag system (exclude/include chunks)

Commands:
- /clear           - Clear conversation history
- /exclude <id>    - Exclude a chunk from search results
- /include <id>    - Re-include a previously excluded chunk
- /list-excluded   - Show all excluded chunks
- /sources         - Show chunk IDs from last search
- quit/exit        - Exit the program
"""

import os
import csv
import json
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
# Try index_new.csv first (has Kotter updates), fall back to index.csv
CSV_INDEX_PATH = OUTPUT_DIR / "index_new.csv" if (OUTPUT_DIR / "index_new.csv").exists() else OUTPUT_DIR / "index.csv"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
EXCLUDED_FILE = OUTPUT_DIR / "excluded_chunks.json"

MAX_CHUNKS_TO_SEND = 8
MAX_CONTEXT_WORDS = 6000
MAX_HISTORY_EXCHANGES = 3
RRF_K = 30  # Reciprocal Rank Fusion constant (lower = more weight to top results)

# Adaptive weighting thresholds
KEYWORD_HIGH_CONFIDENCE = 15  # If top keyword score exceeds this, boost keyword weight
KEYWORD_WEIGHT_HIGH = 0.80    # Weight when keyword has strong match (80/20)
KEYWORD_WEIGHT_DEFAULT = 0.50 # Default balanced weight


# =============================================================================
# EXCLUSION MANAGEMENT
# =============================================================================

def load_exclusions():
    """Load excluded chunks from JSON file"""
    if not EXCLUDED_FILE.exists():
        return {"excluded": [], "reason": {}}

    with open(EXCLUDED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_exclusions(data):
    """Save exclusions to JSON file"""
    with open(EXCLUDED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def add_exclusion(chunk_id, reason=""):
    """Add a chunk to the exclusion list"""
    data = load_exclusions()
    if chunk_id not in data["excluded"]:
        data["excluded"].append(chunk_id)
        if reason:
            data["reason"][chunk_id] = reason
        save_exclusions(data)
        return True
    return False


def remove_exclusion(chunk_id):
    """Remove a chunk from the exclusion list"""
    data = load_exclusions()
    if chunk_id in data["excluded"]:
        data["excluded"].remove(chunk_id)
        data["reason"].pop(chunk_id, None)
        save_exclusions(data)
        return True
    return False


def is_excluded(chunk_id):
    """Check if a chunk is excluded"""
    data = load_exclusions()
    return chunk_id in data["excluded"]


# =============================================================================
# INDEX LOADING
# =============================================================================

def load_tfidf_index():
    """Load the pre-built TF-IDF index"""
    if not INDEX_FILE.exists():
        return None

    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_csv_index():
    """Load the CSV index for keyword search"""
    if not CSV_INDEX_PATH.exists():
        return []

    index = []
    with open(CSV_INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append(row)
    return index


def load_chunk_content(chunk_file):
    """Load the full text content of a chunk file"""
    chunk_path = CHUNKS_DIR / chunk_file
    if chunk_path.exists():
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def semantic_search(query, index_data, top_k=20):
    """
    Search using TF-IDF cosine similarity.
    Returns list of (rank, chunk_id, score, chunk) tuples.
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
    rank = 1
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.01:  # Lower threshold for hybrid
            chunk = chunks[idx]
            chunk_id = chunk.get("chunk_id", chunk.get("source", "unknown"))
            results.append((rank, chunk_id, score, chunk))
            rank += 1

    return results


def keyword_search(query, csv_index, top_k=20):
    """
    Search using keyword matching on metadata and content.
    Returns list of (rank, chunk_id, score, chunk) tuples.
    """
    # Common words to skip (they match too broadly)
    STOP_WORDS = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
                  'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'what',
                  'when', 'who', 'how', 'why', 'this', 'that', 'with', 'from', 'they',
                  'will', 'would', 'there', 'their', 'about', 'which', 'could', 'other',
                  'more', 'some', 'into', 'than', 'then', 'them', 'these', 'only'}

    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) >= 3 and w not in STOP_WORDS]

    if not query_words:
        return []

    scored_results = []

    for entry in csv_index:
        score = 0

        # Search metadata
        metadata_text = (
            entry["topic"].lower() + " " +
            entry["summary"].lower() + " " +
            entry["keywords"].lower() + " " +
            entry["source_file"].lower()
        )

        # Load chunk content
        chunk_content = ""
        chunk_path = CHUNKS_DIR / entry["chunk_file"]
        if chunk_path.exists():
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    chunk_content = f.read().lower()
            except:
                pass

        full_text = metadata_text + " " + chunk_content

        # Score based on word matches
        source_lower = entry["source_file"].lower()
        for word in query_words:
            if word in full_text:
                score += 1
            word_stem = word[:5] if len(word) > 5 else word[:3]
            if word_stem in full_text:
                score += 0.5
            # Boost matches in topic, keywords, and source filename
            if word in entry["topic"].lower():
                score += 3
            if word in entry["keywords"].lower():
                score += 3
            if word in source_lower:
                score += 10  # Very strong boost for source filename matches
            if word_stem in entry["topic"].lower():
                score += 1
            if word_stem in entry["keywords"].lower():
                score += 1
            if word_stem in source_lower:
                score += 2  # Boost for partial source filename matches

        if score > len(query_words):
            score += 2

        if score > 0:
            # Build chunk dict compatible with semantic search format
            content = load_chunk_content(entry["chunk_file"]) or ""
            chunk = {
                "chunk_id": entry["chunk_file"].replace(".txt", ""),
                "source": entry["source_file"],
                "topic": entry["topic"],
                "full_text": content
            }
            scored_results.append((score, entry["chunk_file"].replace(".txt", ""), chunk))

    # Sort by score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Convert to ranked format
    results = []
    for rank, (score, chunk_id, chunk) in enumerate(scored_results[:top_k], 1):
        results.append((rank, chunk_id, score, chunk))

    return results


def reciprocal_rank_fusion(semantic_results, keyword_results, k=RRF_K,
                           keyword_weight=0.5, semantic_weight=0.5):
    """
    Combine semantic and keyword search results using Reciprocal Rank Fusion.
    RRF score = weight * sum(1 / (k + rank)) for each list where the item appears.
    """
    rrf_scores = {}
    chunk_data = {}

    # Process semantic results with weight
    for rank, chunk_id, score, chunk in semantic_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += semantic_weight * (1.0 / (k + rank))

    # Process keyword results with weight
    for rank, chunk_id, score, chunk in keyword_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += keyword_weight * (1.0 / (k + rank))

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Return as list of (rrf_score, chunk_id, chunk)
    return [(score, chunk_id, chunk_data[chunk_id]) for chunk_id, score in sorted_results]


def hybrid_search(query, index_data, csv_index, top_k=20):
    """
    Perform adaptive hybrid search combining semantic and keyword search.
    Automatically adjusts weights based on keyword match confidence.
    Returns list of (score, chunk_id, chunk) tuples and the mode used.
    """
    semantic_results = semantic_search(query, index_data, top_k=top_k)
    keyword_results = keyword_search(query, csv_index, top_k=top_k)

    # Determine adaptive weights based on top keyword score
    top_keyword_score = keyword_results[0][2] if keyword_results else 0

    if top_keyword_score >= KEYWORD_HIGH_CONFIDENCE:
        # Strong keyword match - favor keyword search
        keyword_weight = KEYWORD_WEIGHT_HIGH
        semantic_weight = 1.0 - KEYWORD_WEIGHT_HIGH
        mode = "keyword-boosted"
    else:
        # Balanced approach
        keyword_weight = KEYWORD_WEIGHT_DEFAULT
        semantic_weight = 1.0 - KEYWORD_WEIGHT_DEFAULT
        mode = "balanced"

    results = reciprocal_rank_fusion(
        semantic_results, keyword_results,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight
    )

    return results, mode


def build_context(query, index_data, csv_index, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    """
    Search and build context for Claude.
    Returns context string, list of sources, list of chunk IDs, and search mode.
    """
    results, mode = hybrid_search(query, index_data, csv_index)

    if not results:
        return None, [], [], "none"

    # Filter out excluded chunks
    results = [(score, chunk_id, chunk) for score, chunk_id, chunk in results
               if not is_excluded(chunk_id)]

    if not results:
        return None, [], [], mode

    context_parts = []
    sources = []
    chunk_ids = []
    total_words = 0

    for score, chunk_id, chunk in results:
        if len(sources) >= max_chunks:
            break

        content = chunk.get("full_text", "")
        if not content:
            continue

        word_count = len(content.split())

        if total_words + word_count > max_words:
            continue

        context_parts.append(
            f"--- Source: {chunk['source']} (Topic: {chunk['topic']}, Relevance: {score:.4f}) ---\n{content}"
        )
        sources.append({
            "file": chunk["source"],
            "topic": chunk["topic"],
            "score": score,
            "chunk_id": chunk_id
        })
        chunk_ids.append(chunk_id)
        total_words += word_count

    if not context_parts:
        return None, [], [], mode

    return "\n\n".join(context_parts), sources, chunk_ids, mode


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(client, question, context, sources, conversation_history):
    """Send question and context to Claude with conversation history"""

    source_list = "\n".join([f"- {s['file']} ({s['topic']}) - relevance {s['score']:.4f}" for s in sources])

    # Build conversation history section
    history_text = ""
    if conversation_history:
        history_parts = []
        for q, a in conversation_history:
            history_parts.append(f"User: {q}\nBrett: {a}")
        history_text = "\n\n".join(history_parts)

    system_prompt = """You are "Ask Brett", a helpful assistant that answers questions based on Brett Blundy's business knowledge, training materials, and documented conversations.

Your role:
- Answer questions using ONLY the provided context documents
- Quote or reference specific advice when relevant
- Be direct and practical, matching Brett's communication style
- If the context doesn't contain enough information to answer, say so clearly
- Always cite which source document(s) your answer comes from
- If there's conversation history, use it to understand follow-up questions

Keep answers concise but complete. Use bullet points for actionable advice."""

    # Build user prompt with history
    if history_text:
        user_prompt = f"""Based on the following documents from Brett's knowledge base, please answer my question.

CONVERSATION HISTORY:
{history_text}

DOCUMENTS:
{context}

SOURCES (ranked by relevance):
{source_list}

CURRENT QUESTION: {question}

Please provide a helpful answer based on the documents above, citing your sources. Consider the conversation history for context."""
    else:
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
# COMMAND HANDLERS
# =============================================================================

def handle_clear(conversation_history):
    """Clear conversation history"""
    conversation_history.clear()
    print("\nConversation history cleared.\n")


def handle_exclude(args):
    """Exclude a chunk from search results"""
    if not args:
        print("\nUsage: /exclude <chunk_id> [reason]")
        print("Example: /exclude 040519_chunk003 personal content\n")
        return

    parts = args.split(maxsplit=1)
    chunk_id = parts[0]
    reason = parts[1] if len(parts) > 1 else ""

    if add_exclusion(chunk_id, reason):
        print(f"\nExcluded chunk: {chunk_id}")
        if reason:
            print(f"Reason: {reason}")
    else:
        print(f"\nChunk '{chunk_id}' is already excluded.")
    print()


def handle_include(args):
    """Re-include a previously excluded chunk"""
    if not args:
        print("\nUsage: /include <chunk_id>")
        print("Example: /include 040519_chunk003\n")
        return

    chunk_id = args.strip()

    if remove_exclusion(chunk_id):
        print(f"\nRe-included chunk: {chunk_id}\n")
    else:
        print(f"\nChunk '{chunk_id}' was not in the exclusion list.\n")


def handle_list_excluded():
    """Show all excluded chunks"""
    data = load_exclusions()

    if not data["excluded"]:
        print("\nNo chunks are currently excluded.\n")
        return

    print(f"\nExcluded chunks ({len(data['excluded'])}):")
    for chunk_id in data["excluded"]:
        reason = data["reason"].get(chunk_id, "")
        if reason:
            print(f"  - {chunk_id}: {reason}")
        else:
            print(f"  - {chunk_id}")
    print()


def handle_sources(last_chunk_ids):
    """Show chunk IDs from last search"""
    if not last_chunk_ids:
        print("\nNo search results yet. Ask a question first.\n")
        return

    print(f"\nChunk IDs from last search ({len(last_chunk_ids)}):")
    for i, chunk_id in enumerate(last_chunk_ids, 1):
        excluded = " [EXCLUDED]" if is_excluded(chunk_id) else ""
        print(f"  {i}. {chunk_id}{excluded}")
    print("\nUse '/exclude <chunk_id>' to exclude a chunk from future searches.\n")


def print_help():
    """Print available commands"""
    print("""
Available commands:
  /clear           - Clear conversation history
  /exclude <id>    - Exclude a chunk from search results
  /include <id>    - Re-include a previously excluded chunk
  /list-excluded   - Show all excluded chunks
  /sources         - Show chunk IDs from last search
  /help            - Show this help message
  quit/exit        - Exit the program
""")


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  ASK BRETT - Hybrid Search Edition")
    print("=" * 60)
    print("\nType your question and press Enter.")
    print("Type '/help' for commands, 'quit' or 'exit' to stop.\n")

    # Load environment
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key or api_key == "sk-ant-your-key-here":
        print("ERROR: Please add your Anthropic API key to the .env file")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Load indexes
    print("Loading search indexes...")
    index_data = load_tfidf_index()
    csv_index = load_csv_index()

    if index_data is None:
        print("\nERROR: TF-IDF index not found!")
        print("Please run: python build_index.py")
        return

    if not csv_index:
        print("\nWARNING: CSV index not found, keyword search disabled.")

    print(f"Loaded {len(index_data['chunks'])} chunks (semantic).")
    if csv_index:
        print(f"Loaded {len(csv_index)} chunks (keyword).")

    # Load exclusions info
    exclusions = load_exclusions()
    if exclusions["excluded"]:
        print(f"Excluding {len(exclusions['excluded'])} chunks.")

    print("Hybrid search ready.\n")

    # State
    conversation_history = []  # List of (question, answer) tuples
    last_chunk_ids = []

    # Main loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            args = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd == "/clear":
                handle_clear(conversation_history)
            elif cmd == "/exclude":
                handle_exclude(args)
            elif cmd == "/include":
                handle_include(args)
            elif cmd == "/list-excluded":
                handle_list_excluded()
            elif cmd == "/sources":
                handle_sources(last_chunk_ids)
            elif cmd == "/help":
                print_help()
            else:
                print(f"\nUnknown command: {cmd}")
                print("Type '/help' for available commands.\n")
            continue

        # Regular question - search and respond
        print("\nSearching (adaptive hybrid)...")
        context, sources, chunk_ids, search_mode = build_context(user_input, index_data, csv_index)
        last_chunk_ids = chunk_ids

        if not context:
            print("\nNo relevant documents found.")
            print("Try rephrasing your question.\n")
            continue

        mode_display = "keyword-boosted (80/20)" if search_mode == "keyword-boosted" else "balanced (50/50)"
        print(f"Found {len(sources)} sources using {mode_display} mode. Generating answer...\n")

        # Get answer
        try:
            answer = ask_claude(client, user_input, context, sources, conversation_history)

            # Update conversation history
            conversation_history.append((user_input, answer))
            if len(conversation_history) > MAX_HISTORY_EXCHANGES:
                conversation_history.pop(0)

            # Display answer
            print("-" * 60)
            print(f"\nBrett's Knowledge Base:\n")
            print(answer)
            print("\n" + "-" * 60)
            print("\nTop sources:")
            for s in sources[:5]:
                print(f"  - {s['file']} ({s['topic']}) [{s['chunk_id']}]")
            print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
