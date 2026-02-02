"""
Search Comparison Tool - Compare keyword, semantic, and hybrid search results.
Run with: python compare_search.py

Tests the same question across all search methods and lets you rate the results.
"""

import os
import csv
import json
import pickle
from pathlib import Path
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INDEX_FILE = OUTPUT_DIR / "search_index.pkl"
CSV_INDEX_PATH = OUTPUT_DIR / "index.csv"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
COMPARISON_LOG = SCRIPT_DIR / "comparison_results.json"

MAX_CHUNKS_TO_SEND = 8
MAX_CONTEXT_WORDS = 6000
RRF_K = 60


# =============================================================================
# INDEX LOADING
# =============================================================================

def load_tfidf_index():
    if not INDEX_FILE.exists():
        return None
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_csv_index():
    if not CSV_INDEX_PATH.exists():
        return []
    index = []
    with open(CSV_INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append(row)
    return index


def load_chunk_content(chunk_file):
    chunk_path = CHUNKS_DIR / chunk_file
    if chunk_path.exists():
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# =============================================================================
# SEARCH METHODS
# =============================================================================

def keyword_search(query, csv_index, top_k=20):
    """Original keyword-based search from ask_brett.py"""
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) >= 3]

    if not query_words:
        return []

    scored_results = []

    for entry in csv_index:
        score = 0
        metadata_text = (
            entry["topic"].lower() + " " +
            entry["summary"].lower() + " " +
            entry["keywords"].lower() + " " +
            entry["source_file"].lower()
        )

        chunk_content = ""
        chunk_path = CHUNKS_DIR / entry["chunk_file"]
        if chunk_path.exists():
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    chunk_content = f.read().lower()
            except:
                pass

        full_text = metadata_text + " " + chunk_content

        for word in query_words:
            if word in full_text:
                score += 1
            word_stem = word[:5] if len(word) > 5 else word[:3]
            if word_stem in full_text:
                score += 0.5
            if word in entry["topic"].lower():
                score += 3
            if word in entry["keywords"].lower():
                score += 3
            if word_stem in entry["topic"].lower():
                score += 1
            if word_stem in entry["keywords"].lower():
                score += 1

        if score > len(query_words):
            score += 2

        if score > 0:
            content = load_chunk_content(entry["chunk_file"]) or ""
            result = {
                "score": score,
                "chunk_id": entry["chunk_file"].replace(".txt", ""),
                "source": entry["source_file"],
                "topic": entry["topic"],
                "full_text": content
            }
            scored_results.append(result)

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]


def semantic_search(query, index_data, top_k=20):
    """TF-IDF semantic search from ask_brett_semantic.py"""
    vectorizer = index_data["vectorizer"]
    tfidf_matrix = index_data["tfidf_matrix"]
    chunks = index_data["chunks"]

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.01:
            chunk = chunks[idx]
            results.append({
                "score": float(score),
                "chunk_id": chunk.get("chunk_id", chunk.get("source", "unknown")),
                "source": chunk["source"],
                "topic": chunk["topic"],
                "full_text": chunk.get("full_text", "")
            })
    return results


def hybrid_search(query, index_data, csv_index, top_k=20, keyword_weight=1.0, semantic_weight=1.0):
    """Hybrid search with adjustable weights"""
    # Get results from both methods
    kw_results = keyword_search(query, csv_index, top_k=top_k)
    sem_results = semantic_search(query, index_data, top_k=top_k)

    # RRF fusion with weights
    rrf_scores = {}
    chunk_data = {}

    for rank, result in enumerate(sem_results, 1):
        chunk_id = result["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = result
        rrf_scores[chunk_id] += semantic_weight * (1.0 / (RRF_K + rank))

    for rank, result in enumerate(kw_results, 1):
        chunk_id = result["chunk_id"]
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = result
        rrf_scores[chunk_id] += keyword_weight * (1.0 / (RRF_K + rank))

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_id, score in sorted_results[:top_k]:
        data = chunk_data[chunk_id]
        data["score"] = score
        results.append(data)

    return results


# =============================================================================
# CONTEXT BUILDING
# =============================================================================

def build_context_from_results(results, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    """Build context string from search results"""
    context_parts = []
    sources = []
    total_words = 0

    for result in results:
        if len(sources) >= max_chunks:
            break

        content = result.get("full_text", "")
        if not content:
            continue

        word_count = len(content.split())
        if total_words + word_count > max_words:
            continue

        context_parts.append(
            f"--- Source: {result['source']} (Topic: {result['topic']}) ---\n{content}"
        )
        sources.append({
            "file": result["source"],
            "topic": result["topic"],
            "score": result["score"],
            "chunk_id": result["chunk_id"]
        })
        total_words += word_count

    if not context_parts:
        return None, []

    return "\n\n".join(context_parts), sources


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(client, question, context, sources):
    """Get answer from Claude"""
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

SOURCES:
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
# COMPARISON LOGGING
# =============================================================================

def load_comparison_log():
    if COMPARISON_LOG.exists():
        with open(COMPARISON_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"comparisons": []}


def save_comparison(question, results, winner, notes=""):
    log = load_comparison_log()
    log["comparisons"].append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "winner": winner,
        "notes": notes,
        "sources": {
            "keyword": [s["chunk_id"] for s in results["keyword"]["sources"][:5]],
            "semantic": [s["chunk_id"] for s in results["semantic"]["sources"][:5]],
            "hybrid": [s["chunk_id"] for s in results["hybrid"]["sources"][:5]]
        }
    })
    with open(COMPARISON_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def show_comparison_stats():
    log = load_comparison_log()
    if not log["comparisons"]:
        print("\nNo comparisons logged yet.\n")
        return

    total = len(log["comparisons"])
    winners = {"keyword": 0, "semantic": 0, "hybrid": 0, "tie": 0}
    for c in log["comparisons"]:
        w = c.get("winner", "").lower()
        if w in winners:
            winners[w] += 1

    print(f"\n{'='*60}")
    print("COMPARISON STATISTICS")
    print(f"{'='*60}")
    print(f"Total comparisons: {total}")
    print(f"\nWins by method:")
    for method, count in winners.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {method:10} : {count:3} ({pct:5.1f}%) {bar}")
    print()


# =============================================================================
# MAIN COMPARISON INTERFACE
# =============================================================================

def compare_question(question, client, index_data, csv_index, show_answers=True):
    """Run a question through all three search methods and compare"""

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    results = {}

    # Keyword search
    print("\n[1] KEYWORD SEARCH")
    print("-" * 40)
    kw_results = keyword_search(question, csv_index)
    kw_context, kw_sources = build_context_from_results(kw_results)
    results["keyword"] = {"sources": kw_sources}

    print(f"Found {len(kw_sources)} sources:")
    for i, s in enumerate(kw_sources[:5], 1):
        print(f"  {i}. {s['file']} ({s['topic']}) - score: {s['score']:.2f}")

    # Semantic search
    print("\n[2] SEMANTIC SEARCH")
    print("-" * 40)
    sem_results = semantic_search(question, index_data)
    sem_context, sem_sources = build_context_from_results(sem_results)
    results["semantic"] = {"sources": sem_sources}

    print(f"Found {len(sem_sources)} sources:")
    for i, s in enumerate(sem_sources[:5], 1):
        print(f"  {i}. {s['file']} ({s['topic']}) - score: {s['score']:.3f}")

    # Hybrid search
    print("\n[3] HYBRID SEARCH")
    print("-" * 40)
    hyb_results = hybrid_search(question, index_data, csv_index)
    hyb_context, hyb_sources = build_context_from_results(hyb_results)
    results["hybrid"] = {"sources": hyb_sources}

    print(f"Found {len(hyb_sources)} sources:")
    for i, s in enumerate(hyb_sources[:5], 1):
        print(f"  {i}. {s['file']} ({s['topic']}) - score: {s['score']:.4f}")

    # Show unique sources in each
    kw_ids = set(s["chunk_id"] for s in kw_sources[:5])
    sem_ids = set(s["chunk_id"] for s in sem_sources[:5])
    hyb_ids = set(s["chunk_id"] for s in hyb_sources[:5])

    print("\n[SOURCE COMPARISON]")
    print("-" * 40)
    print(f"Only in Keyword:  {kw_ids - sem_ids - hyb_ids or 'None'}")
    print(f"Only in Semantic: {sem_ids - kw_ids - hyb_ids or 'None'}")
    print(f"In all three:     {kw_ids & sem_ids & hyb_ids or 'None'}")

    # Get answers if requested
    if show_answers and client:
        print("\n" + "=" * 60)
        print("GENERATING ANSWERS (this may take a moment)...")
        print("=" * 60)

        if kw_context:
            print("\n[KEYWORD ANSWER]")
            print("-" * 40)
            kw_answer = ask_claude(client, question, kw_context, kw_sources)
            results["keyword"]["answer"] = kw_answer
            print(kw_answer[:1000] + "..." if len(kw_answer) > 1000 else kw_answer)

        if sem_context:
            print("\n[SEMANTIC ANSWER]")
            print("-" * 40)
            sem_answer = ask_claude(client, question, sem_context, sem_sources)
            results["semantic"]["answer"] = sem_answer
            print(sem_answer[:1000] + "..." if len(sem_answer) > 1000 else sem_answer)

        if hyb_context:
            print("\n[HYBRID ANSWER]")
            print("-" * 40)
            hyb_answer = ask_claude(client, question, hyb_context, hyb_sources)
            results["hybrid"]["answer"] = hyb_answer
            print(hyb_answer[:1000] + "..." if len(hyb_answer) > 1000 else hyb_answer)

    return results


def main():
    print("\n" + "=" * 60)
    print("  SEARCH COMPARISON TOOL")
    print("=" * 60)
    print("\nCommands:")
    print("  [question]     - Compare search methods for a question")
    print("  /sources       - Compare sources only (no Claude answers)")
    print("  /stats         - Show comparison statistics")
    print("  /batch         - Run batch comparison from file")
    print("  quit           - Exit")
    print()

    # Load environment
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = None

    if api_key and api_key != "sk-ant-your-key-here":
        client = anthropic.Anthropic(api_key=api_key)
    else:
        print("WARNING: No API key found. Will compare sources only.\n")

    # Load indexes
    print("Loading indexes...")
    index_data = load_tfidf_index()
    csv_index = load_csv_index()

    if index_data is None:
        print("ERROR: TF-IDF index not found. Run: python build_index.py")
        return

    print(f"Loaded {len(index_data['chunks'])} chunks (semantic)")
    print(f"Loaded {len(csv_index)} entries (keyword)")
    print()

    sources_only = False

    while True:
        try:
            user_input = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if user_input == "/sources":
            sources_only = True
            print("Mode: Sources only (no Claude answers)")
            continue

        if user_input == "/answers":
            sources_only = False
            print("Mode: Full comparison with Claude answers")
            continue

        if user_input == "/stats":
            show_comparison_stats()
            continue

        if user_input == "/batch":
            print("Enter questions (one per line, empty line to finish):")
            questions = []
            while True:
                q = input("  > ").strip()
                if not q:
                    break
                questions.append(q)

            if questions:
                for q in questions:
                    compare_question(q, client if not sources_only else None, index_data, csv_index, show_answers=not sources_only)
            continue

        # Regular question - run comparison
        results = compare_question(
            user_input,
            client if not sources_only else None,
            index_data,
            csv_index,
            show_answers=not sources_only
        )

        # Ask for rating
        print("\n" + "-" * 40)
        print("Which method gave the best results?")
        print("  1 = Keyword")
        print("  2 = Semantic")
        print("  3 = Hybrid")
        print("  t = Tie")
        print("  s = Skip (don't log)")

        rating = input("Your choice: ").strip().lower()

        if rating == "1":
            save_comparison(user_input, results, "keyword")
            print("Logged: Keyword wins")
        elif rating == "2":
            save_comparison(user_input, results, "semantic")
            print("Logged: Semantic wins")
        elif rating == "3":
            save_comparison(user_input, results, "hybrid")
            print("Logged: Hybrid wins")
        elif rating == "t":
            save_comparison(user_input, results, "tie")
            print("Logged: Tie")
        else:
            print("Skipped")

        print()


if __name__ == "__main__":
    main()
