"""
Systematic search test for Ask Brett
Tests search quality and performance
"""

import time
import pickle
import csv
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INDEX_FILE = OUTPUT_DIR / "search_index.pkl"
CSV_INDEX_PATH = OUTPUT_DIR / "index_new.csv" if (OUTPUT_DIR / "index_new.csv").exists() else OUTPUT_DIR / "index.csv"
CHUNKS_DIR = OUTPUT_DIR / "chunks"

# Test queries with expected topics/keywords in results
TEST_QUERIES = [
    ("customer focus", ["customer", "focus"]),
    ("inventory management", ["inventory"]),
    ("leadership", ["leadership", "leader"]),
    ("culture", ["culture"]),
    ("kotter change management", ["kotter", "change"]),
    ("CEO role responsibilities", ["ceo", "chief"]),
    ("store visit", ["store", "visit"]),
    ("retail strategy", ["retail", "strategy"]),
    ("team building", ["team"]),
    ("goal setting", ["goal"]),
]


def load_tfidf_index():
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_csv_index():
    index = []
    with open(CSV_INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append(row)
    return index


def load_all_chunks():
    chunk_cache = {}
    for chunk_file in CHUNKS_DIR.glob("*.txt"):
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_cache[chunk_file.name] = f.read()
        except:
            pass
    return chunk_cache


def semantic_search(query, index_data, top_k=20):
    vectorizer = index_data["vectorizer"]
    tfidf_matrix = index_data["tfidf_matrix"]
    chunks = index_data["chunks"]

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    rank = 1
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.01:
            chunk = chunks[idx]
            chunk_id = chunk.get("chunk_id", chunk.get("source", "unknown"))
            results.append((rank, chunk_id, score, chunk))
            rank += 1
    return results


def keyword_search(query, csv_index, chunk_cache, top_k=20):
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
        metadata_text = (
            entry["topic"].lower() + " " +
            entry["summary"].lower() + " " +
            entry["keywords"].lower() + " " +
            entry["source_file"].lower()
        )

        chunk_content = chunk_cache.get(entry["chunk_file"], "").lower()
        full_text = metadata_text + " " + chunk_content

        source_lower = entry["source_file"].lower()
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
            if word in source_lower:
                score += 10
            if word_stem in entry["topic"].lower():
                score += 1
            if word_stem in entry["keywords"].lower():
                score += 1
            if word_stem in source_lower:
                score += 2

        if score > len(query_words):
            score += 2

        if score > 0:
            content = chunk_cache.get(entry["chunk_file"], "")
            chunk = {
                "chunk_id": entry["chunk_file"].replace(".txt", ""),
                "source": entry["source_file"],
                "topic": entry["topic"],
                "full_text": content
            }
            scored_results.append((score, entry["chunk_file"].replace(".txt", ""), chunk))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, chunk_id, chunk) in enumerate(scored_results[:top_k], 1):
        results.append((rank, chunk_id, score, chunk))
    return results


def hybrid_search(query, index_data, csv_index, chunk_cache, top_k=20):
    RRF_K = 30
    KEYWORD_HIGH_CONFIDENCE = 15
    KEYWORD_WEIGHT_HIGH = 0.80
    KEYWORD_WEIGHT_DEFAULT = 0.50

    semantic_results = semantic_search(query, index_data, top_k=top_k)
    keyword_results = keyword_search(query, csv_index, chunk_cache, top_k=top_k)

    top_keyword_score = keyword_results[0][2] if keyword_results else 0

    if top_keyword_score >= KEYWORD_HIGH_CONFIDENCE:
        keyword_weight = KEYWORD_WEIGHT_HIGH
        semantic_weight = 1.0 - KEYWORD_WEIGHT_HIGH
        mode = "keyword-boosted"
    else:
        keyword_weight = KEYWORD_WEIGHT_DEFAULT
        semantic_weight = 1.0 - KEYWORD_WEIGHT_DEFAULT
        mode = "balanced"

    # RRF fusion
    rrf_scores = {}
    chunk_data = {}

    for rank, chunk_id, score, chunk in semantic_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += semantic_weight * (1.0 / (RRF_K + rank))

    for rank, chunk_id, score, chunk in keyword_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += keyword_weight * (1.0 / (RRF_K + rank))

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = [(score, chunk_id, chunk_data[chunk_id]) for chunk_id, score in sorted_results]

    return results[:top_k], mode


def check_relevance(results, expected_keywords):
    """Check if top results contain expected keywords"""
    if not results:
        return False, "No results"

    top_3 = results[:3]
    for score, chunk_id, chunk in top_3:
        text = (chunk.get("topic", "") + " " + chunk.get("source", "") + " " + chunk.get("full_text", "")[:500]).lower()
        for kw in expected_keywords:
            if kw.lower() in text:
                return True, chunk.get("topic", "Unknown")

    return False, results[0][2].get("topic", "Unknown") if results else "No results"


def main():
    print("=" * 70)
    print("  ASK BRETT - SYSTEMATIC SEARCH TEST")
    print("=" * 70)

    # Load indexes
    print("\nLoading indexes...")
    load_start = time.time()
    index_data = load_tfidf_index()
    csv_index = load_csv_index()
    chunk_cache = load_all_chunks()
    load_time = time.time() - load_start

    print(f"  Loaded {len(index_data['chunks'])} semantic chunks")
    print(f"  Loaded {len(csv_index)} keyword entries")
    print(f"  Cached {len(chunk_cache)} chunk files")
    print(f"  Load time: {load_time:.2f}s")

    # Run tests
    print("\n" + "-" * 70)
    print("SEARCH QUALITY TESTS")
    print("-" * 70)

    passed = 0
    failed = 0
    total_search_time = 0

    for query, expected_keywords in TEST_QUERIES:
        start = time.time()
        results, mode = hybrid_search(query, index_data, csv_index, chunk_cache)
        search_time = time.time() - start
        total_search_time += search_time

        relevant, top_topic = check_relevance(results, expected_keywords)

        status = "PASS" if relevant else "FAIL"
        if relevant:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] \"{query}\"")
        print(f"       Mode: {mode} | Time: {search_time*1000:.1f}ms | Top: {top_topic[:50]}")
        if not relevant:
            print(f"       Expected keywords: {expected_keywords}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tests passed: {passed}/{len(TEST_QUERIES)}")
    print(f"  Tests failed: {failed}/{len(TEST_QUERIES)}")
    print(f"  Average search time: {(total_search_time/len(TEST_QUERIES))*1000:.1f}ms")
    print(f"  Total test time: {total_search_time:.2f}s")

    # Performance benchmark
    print("\n" + "-" * 70)
    print("PERFORMANCE BENCHMARK (10 iterations)")
    print("-" * 70)

    benchmark_query = "customer focus strategy"
    times = []
    for i in range(10):
        start = time.time()
        results, mode = hybrid_search(benchmark_query, index_data, csv_index, chunk_cache)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Query: \"{benchmark_query}\"")
    print(f"  Avg: {avg_time*1000:.1f}ms | Min: {min_time*1000:.1f}ms | Max: {max_time*1000:.1f}ms")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
