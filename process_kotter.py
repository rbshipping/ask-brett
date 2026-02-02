"""
Process Kotter Resources - Chunks the fetched Kotter text files and adds to knowledge base.
Run with: python process_kotter.py

After running, rebuild the index:
    python build_index.py
"""

import os
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
KOTTER_DIR = SCRIPT_DIR / "kotter_knowledge"
OUTPUT_DIR = SCRIPT_DIR / "output"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
INDEX_PATH = OUTPUT_DIR / "index.csv"

CHUNK_MIN_WORDS = 800
CHUNK_MAX_WORDS = 1000
CHUNK_OVERLAP_WORDS = 50

SUBFOLDER_NAME = "Kotter Change Management"


# =============================================================================
# CHUNKING
# =============================================================================

def chunk_text(text, min_words=CHUNK_MIN_WORDS, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    if len(words) <= max_words:
        return [text]

    start = 0
    while start < len(words):
        end = start + max_words

        # Find a good break point (end of sentence)
        if end < len(words):
            # Look for sentence end near the target
            for i in range(end, max(start + min_words, end - 100), -1):
                if words[i - 1].endswith(('.', '!', '?', ':', '"')):
                    end = i
                    break

        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))

        # Move start with overlap
        start = end - overlap
        if start >= len(words) - overlap:
            break

    return chunks


# =============================================================================
# CLAUDE METADATA GENERATION
# =============================================================================

def generate_metadata(client, content, source_file):
    """Use Claude to generate topic, summary, and keywords for a chunk"""

    prompt = f"""Analyze this text excerpt from "{source_file}" and provide metadata.

TEXT:
{content[:3000]}

Respond with exactly this format (no extra text):
TOPIC: [2-5 word topic description]
SUMMARY: [1-2 sentence summary of key points]
KEYWORDS: [5-8 comma-separated keywords]"""

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Parse response
        topic = ""
        summary = ""
        keywords = ""

        for line in text.split('\n'):
            if line.startswith('TOPIC:'):
                topic = line.replace('TOPIC:', '').strip()
            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('KEYWORDS:'):
                keywords = line.replace('KEYWORDS:', '').strip()

        return topic, summary, keywords

    except Exception as e:
        print(f"    Error generating metadata: {e}")
        return "Kotter Change Management", "Content about change management.", "change, management, leadership, kotter"


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

def load_existing_index():
    """Load existing index.csv"""
    entries = []
    if INDEX_PATH.exists():
        with open(INDEX_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            entries = list(reader)
    return entries


def save_index(entries):
    """Save index.csv"""
    if not entries:
        return

    fieldnames = ['chunk_file', 'source_file', 'subfolder', 'topic', 'summary', 'keywords']

    with open(INDEX_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def get_next_chunk_number(source_name, existing_entries):
    """Find the next available chunk number for a source"""
    pattern = source_name.replace('.txt', '')
    max_num = 0

    for entry in existing_entries:
        if entry['chunk_file'].startswith(pattern):
            try:
                num = int(entry['chunk_file'].split('_chunk')[-1].replace('.txt', ''))
                max_num = max(max_num, num)
            except:
                pass

    return max_num + 1


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_kotter_files():
    print("\n" + "=" * 60)
    print("  KOTTER RESOURCES PROCESSOR")
    print("=" * 60)

    # Check directories
    if not KOTTER_DIR.exists():
        print(f"\nERROR: Kotter directory not found: {KOTTER_DIR}")
        print("Run fetch_kotter_resources.py first.")
        return

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Load environment
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not found in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Get text files
    txt_files = list(KOTTER_DIR.glob("*.txt"))
    print(f"\nFound {len(txt_files)} text files to process")

    # Load existing index
    index_entries = load_existing_index()
    existing_sources = {e['source_file'] for e in index_entries}

    print(f"Existing index has {len(index_entries)} entries")

    # Process each file
    new_chunks = 0
    skipped = 0

    for i, txt_file in enumerate(txt_files, 1):
        source_name = txt_file.name

        # Skip if already processed
        if source_name in existing_sources:
            print(f"\n[{i}/{len(txt_files)}] Skipping (already in index): {source_name}")
            skipped += 1
            continue

        print(f"\n[{i}/{len(txt_files)}] Processing: {source_name}")

        # Read content
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) < 100:
            print(f"    Skipping: too little content ({len(content)} chars)")
            continue

        # Chunk the content
        chunks = chunk_text(content)
        print(f"    Created {len(chunks)} chunks")

        # Process each chunk
        chunk_start_num = get_next_chunk_number(source_name, index_entries)

        for j, chunk_content in enumerate(chunks):
            chunk_num = chunk_start_num + j
            chunk_filename = f"{source_name.replace('.txt', '')}_chunk{chunk_num:03d}.txt"

            # Generate metadata
            print(f"    Generating metadata for chunk {j+1}/{len(chunks)}...")
            topic, summary, keywords = generate_metadata(client, chunk_content, source_name)

            # Save chunk file
            chunk_path = CHUNKS_DIR / chunk_filename
            full_content = f"""SOURCE: {source_name}
SUBFOLDER: {SUBFOLDER_NAME}
TOPIC: {topic}
SUMMARY: {summary}
KEYWORDS: {keywords}
---

{chunk_content}"""

            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(full_content)

            # Add to index
            index_entries.append({
                'chunk_file': chunk_filename,
                'source_file': source_name,
                'subfolder': SUBFOLDER_NAME,
                'topic': topic,
                'summary': summary,
                'keywords': keywords
            })

            new_chunks += 1

    # Save updated index
    save_index(index_entries)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(txt_files) - skipped}")
    print(f"Files skipped (already indexed): {skipped}")
    print(f"New chunks created: {new_chunks}")
    print(f"Total index entries: {len(index_entries)}")

    print("\n" + "-" * 60)
    print("NEXT STEP:")
    print("-" * 60)
    print("Rebuild the search index:")
    print("    python build_index.py")
    print()


if __name__ == "__main__":
    process_kotter_files()
