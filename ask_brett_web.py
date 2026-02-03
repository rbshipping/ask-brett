"""
Ask Brett - Web Interface (Streamlit)
Run with: streamlit run ask_brett_web.py

Then expose with: ngrok http 8501
"""

import os
import csv
import json
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st
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
QUERY_LOG_FILE = OUTPUT_DIR / "query_log.csv"

MAX_CHUNKS_TO_SEND = 8
MAX_CONTEXT_WORDS = 6000
MAX_HISTORY_EXCHANGES = 3
RRF_K = 30  # Lower = more weight to top results

# Adaptive weighting thresholds
KEYWORD_HIGH_CONFIDENCE = 15
KEYWORD_WEIGHT_HIGH = 0.80    # 80/20 when keyword has strong match
KEYWORD_WEIGHT_DEFAULT = 0.50


# =============================================================================
# EXCLUSION MANAGEMENT
# =============================================================================

def load_exclusions():
    if not EXCLUDED_FILE.exists():
        return {"excluded": [], "reason": {}}
    with open(EXCLUDED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_excluded(chunk_id):
    data = load_exclusions()
    return chunk_id in data["excluded"]


# =============================================================================
# QUERY LOGGING
# =============================================================================

def log_query(user_name, question, search_mode, num_sources, top_source=None, response_text=None, response_time_ms=None):
    """Log a query to CSV file for usage tracking."""
    file_exists = QUERY_LOG_FILE.exists()

    with open(QUERY_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "user", "question", "search_mode", "num_sources", "top_source", "response_time_ms", "response_preview"])

        # Truncate response for log (first 200 chars)
        response_preview = (response_text[:200] + "...") if response_text and len(response_text) > 200 else (response_text or "")

        writer.writerow([
            datetime.now().isoformat(),
            user_name or "anonymous",
            question,
            search_mode,
            num_sources,
            top_source or "",
            response_time_ms or "",
            response_preview
        ])


# =============================================================================
# INDEX LOADING (cached)
# =============================================================================

@st.cache_resource
def load_tfidf_index():
    if not INDEX_FILE.exists():
        return None
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_csv_index():
    if not CSV_INDEX_PATH.exists():
        return []
    index = []
    with open(CSV_INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index.append(row)
    return index


@st.cache_resource
def load_all_chunks():
    """Load all chunk content into memory at startup for fast search."""
    chunk_cache = {}
    if CHUNKS_DIR.exists():
        for chunk_file in CHUNKS_DIR.glob("*.txt"):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_cache[chunk_file.name] = f.read()
            except:
                pass
    return chunk_cache


def load_chunk_content(chunk_file, chunk_cache=None):
    if chunk_cache and chunk_file in chunk_cache:
        return chunk_cache[chunk_file]
    chunk_path = CHUNKS_DIR / chunk_file
    if chunk_path.exists():
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

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
        metadata_text = (
            entry["topic"].lower() + " " +
            entry["summary"].lower() + " " +
            entry["keywords"].lower() + " " +
            entry["source_file"].lower()
        )

        # Use cached chunk content instead of reading from disk
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


def reciprocal_rank_fusion(semantic_results, keyword_results, k=RRF_K,
                           keyword_weight=0.5, semantic_weight=0.5):
    rrf_scores = {}
    chunk_data = {}

    for rank, chunk_id, score, chunk in semantic_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += semantic_weight * (1.0 / (k + rank))

    for rank, chunk_id, score, chunk in keyword_results:
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0
            chunk_data[chunk_id] = chunk
        rrf_scores[chunk_id] += keyword_weight * (1.0 / (k + rank))

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, chunk_id, chunk_data[chunk_id]) for chunk_id, score in sorted_results]


def hybrid_search(query, index_data, csv_index, chunk_cache, top_k=20):
    semantic_results = semantic_search(query, index_data, top_k=top_k)
    keyword_results = keyword_search(query, csv_index, chunk_cache, top_k=top_k)

    # Adaptive weighting based on keyword confidence
    top_keyword_score = keyword_results[0][2] if keyword_results else 0

    if top_keyword_score >= KEYWORD_HIGH_CONFIDENCE:
        keyword_weight = KEYWORD_WEIGHT_HIGH
        semantic_weight = 1.0 - KEYWORD_WEIGHT_HIGH
        mode = "keyword-boosted"
    else:
        keyword_weight = KEYWORD_WEIGHT_DEFAULT
        semantic_weight = 1.0 - KEYWORD_WEIGHT_DEFAULT
        mode = "balanced"

    results = reciprocal_rank_fusion(
        semantic_results, keyword_results,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight
    )
    return results, mode


def build_context(query, index_data, csv_index, chunk_cache, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    results, mode = hybrid_search(query, index_data, csv_index, chunk_cache)

    if not results:
        return None, [], mode

    results = [(score, chunk_id, chunk) for score, chunk_id, chunk in results
               if not is_excluded(chunk_id)]

    if not results:
        return None, [], mode

    context_parts = []
    sources = []
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
            f"--- Source: {chunk['source']} (Topic: {chunk['topic']}) ---\n{content}"
        )
        sources.append({
            "file": chunk["source"],
            "topic": chunk["topic"],
            "score": score,
            "chunk_id": chunk_id
        })
        total_words += word_count

    if not context_parts:
        return None, [], mode

    return "\n\n".join(context_parts), sources, mode


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(client, question, context, sources, conversation_history):
    source_list = "\n".join([f"- {s['file']} ({s['topic']})" for s in sources])

    history_text = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Brett"
            history_parts.append(f"{role}: {msg['content']}")
        history_text = "\n\n".join(history_parts)

    system_prompt = """You are "Ask Brett", a conversational assistant that helps users explore Brett Blundy's business knowledge, training materials, and documented wisdom.

Your role:
- Answer questions using ONLY the provided context documents
- Quote or reference specific advice when relevant
- Be direct and practical, matching Brett's communication style
- If the context doesn't contain enough information to answer, say so clearly
- Always cite which source document(s) your answer comes from
- If there's conversation history, use it to understand follow-up questions

IMPORTANT - Be conversational:
- After providing your answer, ask a relevant follow-up question to help the user explore deeper
- Tailor follow-up questions to understand their specific situation (e.g., "What type of difficulty are you experiencing?" or "Is this about a performance issue or a interpersonal conflict?")
- Keep follow-up questions focused and helpful, not interrogating
- If the user's question is already very specific, you can offer related topics to explore instead

Keep answers concise but complete. Use bullet points for actionable advice. End with a follow-up question or suggestion for deeper exploration."""

    if history_text:
        user_prompt = f"""Based on the following documents from Brett's knowledge base, please answer my question.

CONVERSATION HISTORY:
{history_text}

DOCUMENTS:
{context}

SOURCES:
{source_list}

CURRENT QUESTION: {question}

Please provide a helpful answer based on the documents above, citing your sources."""
    else:
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
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Ask Brett",
        page_icon="💼",
        layout="centered"
    )

    st.title("💼 Ask Brett")
    st.caption("Search Brett's business knowledge base")

    # Load environment (supports both local .env and Streamlit Cloud secrets)
    load_dotenv()

    # Get secrets - try Streamlit secrets first, then fall back to env vars
    def get_secret(key, default=None):
        try:
            return st.secrets[key]
        except (KeyError, FileNotFoundError):
            return os.getenv(key, default)

    # Password protection
    app_password = get_secret("APP_PASSWORD", "askbrett2025")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    if not st.session_state.authenticated:
        password = st.text_input("Enter password to access", type="password")
        if st.button("Login"):
            if password == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

    # Ask for user name if not set
    if not st.session_state.user_name:
        st.markdown("""
**Retail isn't complicated—it's just demanding.**

I've spent 45 years building businesses from a single record store in Pakenham to global brands. This is what I've learned.

**Culture eats strategy for breakfast.** The customer is always the boss. Speed matters more than perfection. Costs are the enemy. These aren't theories—they're the fundamentals that separate winning from losing.

Use this to get better. That's what continuous improvement means—never arriving, always doing more. If you're not improving, you're going backwards.

You're the customer here. Let's get started—what's your name?
        """)
        user_name_input = st.text_input("Your name")
        if st.button("Let's Go"):
            if user_name_input.strip():
                st.session_state.user_name = user_name_input.strip()
                st.rerun()
            else:
                st.error("Please enter your name")
        st.stop()

    api_key = get_secret("ANTHROPIC_API_KEY")

    if not api_key or api_key == "sk-ant-your-key-here":
        st.error("Please add your Anthropic API key to secrets (Streamlit Cloud) or .env file (local)")
        return

    # Load indexes and chunk cache
    index_data = load_tfidf_index()
    csv_index = load_csv_index()
    chunk_cache = load_all_chunks()

    if index_data is None:
        st.error("Search index not found. Please run: python build_index.py")
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "client" not in st.session_state:
        st.session_state.client = anthropic.Anthropic(api_key=api_key)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write(f"**{len(index_data['chunks'])}** chunks indexed")
        st.write(f"**{len(csv_index)}** entries in keyword index")
        st.write(f"**{len(chunk_cache)}** chunks cached in memory")

        exclusions = load_exclusions()
        if exclusions["excluded"]:
            st.write(f"**{len(exclusions['excluded'])}** chunks excluded")

        st.divider()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_name = None
            st.session_state.messages = []
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for s in message["sources"]:
                        st.write(f"- {s['file']} ({s['topic']})")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            import time
            start_time = time.time()

            with st.spinner("Searching..."):
                context, sources, search_mode = build_context(prompt, index_data, csv_index, chunk_cache)

            top_source = sources[0]["file"] if sources else None

            if not context:
                response = "I couldn't find any relevant documents for your question. Try rephrasing or using different keywords."
                sources = []
                search_mode = "none"
            else:
                with st.spinner("Generating answer..."):
                    # Get recent history for context (last 3 exchanges = 6 messages)
                    recent_history = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []

                    response = ask_claude(
                        st.session_state.client,
                        prompt,
                        context,
                        sources,
                        recent_history
                    )

            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)

            # Log the query with all details
            log_query(
                user_name=st.session_state.user_name,
                question=prompt,
                search_mode=search_mode,
                num_sources=len(sources),
                top_source=top_source,
                response_text=response,
                response_time_ms=response_time_ms
            )

            st.markdown(response)

            if sources:
                mode_label = "keyword-boosted 80/20" if search_mode == "keyword-boosted" else "balanced 50/50"
                with st.expander(f"Sources ({mode_label})"):
                    for s in sources:
                        st.write(f"- {s['file']} ({s['topic']})")

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
