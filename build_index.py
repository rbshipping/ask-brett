"""
Build Search Index - Creates TF-IDF embeddings for semantic search.
Run with: python build_index.py

This only needs to be run once, or when you add new documents.
"""

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
INDEX_FILE = OUTPUT_DIR / "search_index.pkl"


# =============================================================================
# BUILD INDEX
# =============================================================================

def load_all_chunks():
    """Load all chunk files and their content"""
    chunks = []

    for chunk_file in sorted(CHUNKS_DIR.glob("*.txt")):
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse metadata from header
            lines = content.split("\n")
            metadata = {}
            content_start = 0

            for i, line in enumerate(lines):
                if line.strip() == "---":
                    content_start = i + 1
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()

            # Get the actual content (after ---)
            text_content = "\n".join(lines[content_start:]).strip()

            chunks.append({
                "filename": chunk_file.name,
                "source": metadata.get("source", ""),
                "topic": metadata.get("topic", ""),
                "summary": metadata.get("summary", ""),
                "keywords": metadata.get("keywords", ""),
                "content": text_content,
                "full_text": content  # Keep full file for retrieval
            })

        except Exception as e:
            print(f"Warning: Could not load {chunk_file.name}: {e}")

    return chunks


def build_tfidf_index(chunks):
    """Build TF-IDF vectorizer and transform all chunks"""

    # Combine all searchable text for each chunk
    documents = []
    for chunk in chunks:
        # Include source filename (important for context like "CEO INDUCTION TRAINING")
        # Weight topic, keywords, and source more by repeating them
        source_name = chunk["source"].replace(".", " ").replace("_", " ").replace("-", " ")
        searchable = (
            source_name + " " + source_name + " " +  # Source filename for context
            chunk["topic"] + " " + chunk["topic"] + " " +
            chunk["keywords"] + " " + chunk["keywords"] + " " +
            chunk["summary"] + " " +
            chunk["content"]
        )
        documents.append(searchable)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,      # Limit vocabulary size
        stop_words="english",    # Remove common words
        ngram_range=(1, 2),      # Include word pairs
        min_df=2,                # Word must appear in at least 2 docs
        max_df=0.95              # Ignore words in >95% of docs
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)

    return vectorizer, tfidf_matrix


def main():
    print("\n" + "=" * 60)
    print("  BUILD SEARCH INDEX")
    print("=" * 60 + "\n")

    # Load chunks
    print("Loading chunks...")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    if not chunks:
        print("No chunks found. Run process_docs.py first.")
        return

    # Build index
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} terms")
    print(f"Matrix shape: {tfidf_matrix.shape}\n")

    # Save index
    print("Saving index...")
    index_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": chunks
    }

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index_data, f)

    print(f"Index saved to: {INDEX_FILE}")
    print(f"Index size: {INDEX_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    print("\n" + "=" * 60)
    print("  INDEX BUILD COMPLETE")
    print("=" * 60)
    print("\nYou can now use: python ask_brett.py")


if __name__ == "__main__":
    main()
