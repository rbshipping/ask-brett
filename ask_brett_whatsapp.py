"""
Ask Brett - WhatsApp Interface (Twilio)
Run with: python ask_brett_whatsapp.py

Then expose with ngrok: ngrok http 5000
Configure Twilio webhook to: https://your-ngrok-url/webhook
"""

import os
import csv
import json
import pickle
from pathlib import Path
from datetime import datetime

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INDEX_FILE = OUTPUT_DIR / "search_index.pkl"
CSV_INDEX_PATH = OUTPUT_DIR / "index_new.csv" if (OUTPUT_DIR / "index_new.csv").exists() else OUTPUT_DIR / "index.csv"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
EXCLUDED_FILE = OUTPUT_DIR / "excluded_chunks.json"
QUERY_LOG_FILE = OUTPUT_DIR / "query_log_whatsapp.csv"

MAX_CHUNKS_TO_SEND = 6  # Slightly fewer for WhatsApp (shorter responses)
MAX_CONTEXT_WORDS = 4000
RRF_K = 30

KEYWORD_HIGH_CONFIDENCE = 15
KEYWORD_WEIGHT_HIGH = 0.80
KEYWORD_WEIGHT_DEFAULT = 0.50

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = Flask(__name__)

# Store conversation history per phone number
conversation_history = {}


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


def load_all_chunks():
    chunk_cache = {}
    if CHUNKS_DIR.exists():
        for chunk_file in CHUNKS_DIR.glob("*.txt"):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_cache[chunk_file.name] = f.read()
            except:
                pass
    return chunk_cache


def load_exclusions():
    if not EXCLUDED_FILE.exists():
        return {"excluded": [], "reason": {}}
    with open(EXCLUDED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_excluded(chunk_id):
    data = load_exclusions()
    return chunk_id in data["excluded"]


# Load indexes at startup
print("Loading indexes...")
index_data = load_tfidf_index()
csv_index = load_csv_index()
chunk_cache = load_all_chunks()
print(f"Loaded {len(chunk_cache)} chunks")


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def semantic_search(query, top_k=20):
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


def keyword_search(query, top_k=20):
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


def hybrid_search(query, top_k=20):
    semantic_results = semantic_search(query, top_k=top_k)
    keyword_results = keyword_search(query, top_k=top_k)

    top_keyword_score = keyword_results[0][2] if keyword_results else 0

    if top_keyword_score >= KEYWORD_HIGH_CONFIDENCE:
        keyword_weight = KEYWORD_WEIGHT_HIGH
        semantic_weight = 1.0 - KEYWORD_WEIGHT_HIGH
    else:
        keyword_weight = KEYWORD_WEIGHT_DEFAULT
        semantic_weight = 1.0 - KEYWORD_WEIGHT_DEFAULT

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
    return [(score, chunk_id, chunk_data[chunk_id]) for chunk_id, score in sorted_results]


def build_context(query, max_chunks=MAX_CHUNKS_TO_SEND, max_words=MAX_CONTEXT_WORDS):
    results = hybrid_search(query)

    if not results:
        return None, []

    results = [(score, chunk_id, chunk) for score, chunk_id, chunk in results
               if not is_excluded(chunk_id)]

    if not results:
        return None, []

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
        return None, []

    return "\n\n".join(context_parts), sources


# =============================================================================
# CLAUDE INTERACTION
# =============================================================================

def ask_claude(question, context, sources, history):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    source_list = "\n".join([f"- {s['file']} ({s['topic']})" for s in sources])

    history_text = ""
    if history:
        history_parts = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Brett"
            history_parts.append(f"{role}: {msg['content']}")
        history_text = "\n\n".join(history_parts)

    system_prompt = """You are "Ask Brett", a conversational assistant on WhatsApp that helps users explore Brett Blundy's business knowledge.

Your role:
- Answer questions using ONLY the provided context documents
- Be direct and practical, matching Brett's communication style
- Keep responses concise (WhatsApp format - max 2-3 short paragraphs)
- Always mention which source your answer comes from
- If there's conversation history, use it to understand follow-up questions

IMPORTANT - Be conversational:
- After providing your answer, ask a relevant follow-up question
- Tailor questions to understand their specific situation
- Keep it friendly and helpful

Format for WhatsApp:
- Use short paragraphs
- Use emoji sparingly for warmth 👍
- No markdown formatting (no ** or ##)"""

    if history_text:
        user_prompt = f"""CONVERSATION HISTORY:
{history_text}

DOCUMENTS:
{context}

SOURCES:
{source_list}

CURRENT QUESTION: {question}

Provide a helpful, concise answer for WhatsApp."""
    else:
        user_prompt = f"""DOCUMENTS:
{context}

SOURCES:
{source_list}

QUESTION: {question}

Provide a helpful, concise answer for WhatsApp."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,  # Shorter for WhatsApp
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


# =============================================================================
# LOGGING
# =============================================================================

def log_query(phone_number, question, num_sources, response_text=None, response_time_ms=None):
    file_exists = QUERY_LOG_FILE.exists()

    # Mask phone number for privacy (show last 4 digits)
    masked_phone = f"***{phone_number[-4:]}" if len(phone_number) >= 4 else "****"

    # Truncate response for log (first 200 chars)
    response_preview = (response_text[:200] + "...") if response_text and len(response_text) > 200 else (response_text or "")

    with open(QUERY_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "phone_masked", "question", "num_sources", "response_time_ms", "response_preview"])
        writer.writerow([
            datetime.now().isoformat(),
            masked_phone,
            question,
            num_sources,
            response_time_ms or "",
            response_preview
        ])


# =============================================================================
# WHATSAPP WEBHOOK
# =============================================================================

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")

    print(f"Message from {from_number}: {incoming_msg}")

    # Initialize conversation history for this number
    if from_number not in conversation_history:
        conversation_history[from_number] = []

    # Handle special commands
    if incoming_msg.lower() == "/clear":
        conversation_history[from_number] = []
        resp = MessagingResponse()
        resp.message("Conversation cleared! Ask me anything about Brett's business knowledge.")
        return str(resp)

    if incoming_msg.lower() == "/help":
        resp = MessagingResponse()
        resp.message("Ask Brett - WhatsApp Edition\n\nJust send me a question about business, leadership, culture, or retail strategy.\n\nCommands:\n/clear - Reset conversation\n/help - Show this message")
        return str(resp)

    # Search and generate response
    import time
    start_time = time.time()

    context, sources = build_context(incoming_msg)

    if not context:
        response_time_ms = int((time.time() - start_time) * 1000)
        log_query(from_number, incoming_msg, 0, "No relevant information found", response_time_ms)
        resp = MessagingResponse()
        resp.message("I couldn't find relevant information for that question. Try rephrasing or ask about leadership, culture, inventory, customer focus, or retail strategy.")
        return str(resp)

    # Add user message to history
    conversation_history[from_number].append({"role": "user", "content": incoming_msg})

    # Get Claude's response
    answer = ask_claude(incoming_msg, context, sources, conversation_history[from_number])

    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)

    # Log the query with full details
    log_query(from_number, incoming_msg, len(sources), answer, response_time_ms)

    # Add assistant response to history
    conversation_history[from_number].append({"role": "assistant", "content": answer})

    # Keep history manageable (last 6 messages = 3 exchanges)
    if len(conversation_history[from_number]) > 6:
        conversation_history[from_number] = conversation_history[from_number][-6:]

    # Send response
    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "chunks_loaded": len(chunk_cache)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set in .env file")
        exit(1)

    print("\n" + "=" * 50)
    print("  ASK BRETT - WhatsApp Edition")
    print("=" * 50)
    print(f"\nLoaded {len(chunk_cache)} chunks")
    print("\nStarting server on http://localhost:5000")
    print("\nNext steps:")
    print("1. Run: ngrok http 5000")
    print("2. Copy the ngrok URL")
    print("3. In Twilio Console, set webhook to: <ngrok-url>/webhook")
    print("\n" + "=" * 50 + "\n")

    app.run(debug=True, port=5000)
