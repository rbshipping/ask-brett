# Ask Brett - Build Log

## Project Overview
A RAG-based chatbot that searches Brett Blundy's business knowledge base and generates answers using Claude.

---

## Phase 1: Foundation (Completed Previously)
- Document processing pipeline (process_docs.py)
- Chunking with metadata extraction via Claude
- Basic keyword search (ask_brett.py)
- TF-IDF semantic search (ask_brett_semantic.py)
- Index building (build_index.py)

---

## Phase 2: Hybrid Search & Features (Completed Jan 28, 2025)

### Features Implemented

#### 1. Hybrid Search with RRF
- **File:** ask_brett_hybrid.py
- Combines keyword + semantic search using Reciprocal Rank Fusion
- Adaptive weighting: 80/20 keyword-boosted when keyword confidence > 15, else 50/50
- RRF k=30 (lower = more weight to top results)

#### 2. Conversation Memory
- Stores last 3 question/answer exchanges
- Included in Claude prompt for follow-up context
- /clear command to reset

#### 3. Content Flag System
- **File:** output/excluded_chunks.json
- Commands: /exclude, /include, /list-excluded, /sources
- Excluded chunks filtered before building context

#### 4. Web Interface
- **File:** ask_brett_web.py
- Streamlit-based chat UI
- Shows sources in expandable section
- Displays search mode (keyword-boosted vs balanced)

#### 5. Comparison Tool
- **File:** compare_search.py
- Tests queries across keyword, semantic, and hybrid search
- Logs comparison results for tuning
- /stats command shows win rates

### Search Improvements

#### Stop Word Filtering
Added to keyword search to prevent common words (the, are, what, etc.) from dominating results.

#### Source Filename Boosting
- TF-IDF index now includes source filename in searchable text
- Keyword search gives +10 boost when query words match source filename
- Fixes issue where "CEO INDUCTION TRAINING" wasn't found for CEO queries

### Configuration (Current Settings)
```
RRF_K = 30                      # Lower = more weight to top results
KEYWORD_HIGH_CONFIDENCE = 15    # Threshold to trigger keyword boost
KEYWORD_WEIGHT_HIGH = 0.80      # 80% keyword when boosted
KEYWORD_WEIGHT_DEFAULT = 0.50   # 50/50 balanced
MAX_CHUNKS_TO_SEND = 8
MAX_CONTEXT_WORDS = 6000
MAX_HISTORY_EXCHANGES = 3
```

---

## Phase 2.5: Kotter Knowledge Integration (Completed Jan 28, 2025)

### Scripts Created
- **fetch_kotter_resources.py** - Fetches web pages and PDFs from Kotter URLs
- **process_kotter.py** - Chunks text files and generates metadata

### Content Added
- 21 Kotter resources fetched (web pages + PDFs)
- 53 new chunks created
- Topics: 8 Steps for Change, Change Principles, Leadership, Accelerate methodology

### Issues Resolved
- Removed invalid nul files causing OneDrive sync failures
- Created index_new.csv as workaround for locked index file
- Updated scripts to use index_new.csv as fallback

---

## Key Learnings

### 1. Context Loss in Chunking
**Problem:** Documents like "CEO INDUCTION TRAINING" get chunked, but individual chunks may not mention "CEO" - they discuss specific topics like "culture" or "customer focus."

**Solution:**
- Include source filename in TF-IDF searchable text
- Boost keyword matches on source filename (+10 points)
- Stop word filtering to focus on meaningful terms

### 2. Hybrid Search Tuning
**Problem:** Equal weighting (50/50) diluted strong keyword matches.

**Solution:** Adaptive weighting - when keyword search finds high-confidence match (score > 15), boost to 80/20.

### 3. Common Words Dominate
**Problem:** Queries like "what are brett's thoughts on CEO role" matched documents with lots of "what", "are", "the" content.

**Solution:** Stop word filtering in keyword search.

### 4. RRF k Parameter
- k=60 (default) flattens ranking differences too much
- k=30 gives better differentiation between top results

---

## Current File Structure

```
doc_processor/
├── ask_brett.py              # Original keyword search
├── ask_brett_semantic.py     # TF-IDF semantic search
├── ask_brett_hybrid.py       # Hybrid search (main CLI)
├── ask_brett_web.py          # Streamlit web interface
├── build_index.py            # Builds TF-IDF index
├── process_docs.py           # Document processing pipeline
├── compare_search.py         # Search comparison tool
├── fetch_kotter_resources.py # Kotter content fetcher
├── process_kotter.py         # Kotter chunk processor
├── config.yaml               # Processing configuration
├── requirements.txt          # Dependencies
├── kotter_knowledge/         # Fetched Kotter resources
└── output/
    ├── chunks/               # All chunk text files (1394 total)
    ├── index.csv             # Keyword search index
    ├── index_new.csv         # Updated index with Kotter
    ├── search_index.pkl      # TF-IDF vectors (20.8 MB)
    └── excluded_chunks.json  # Exclusion list
```

---

## Future Build Plan

### Phase 3: RAG Enhancements

#### 1. Dense Embeddings (High Priority)
Replace TF-IDF with dense vector embeddings for better semantic matching.

**Benefits:**
- "CEO" matches "chief executive", "leader", "executive"
- "clear inventory" matches "markdown", "liquidate", "sell through"
- Better handling of synonyms and related concepts

**Implementation:**
```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, runs locally
embeddings = model.encode(chunk_texts)
# Store in FAISS or similar vector DB
```

**Estimated effort:** 2-3 hours

#### 2. Contextual Chunk Headers (High Priority)
Prepend document context to each chunk so meaning isn't lost.

**Current:**
```
[Slide 16]
Your Role as CEO
If we want to avoid being crushed...
```

**Proposed:**
```
[Document: CEO INDUCTION TRAINING | Section: Your Role as CEO]

If we want to avoid being crushed...
```

**Implementation:**
- Modify process_docs.py to add headers during chunking
- Or create migration script to update existing chunks
- Rebuild TF-IDF index

**Estimated effort:** 1-2 hours

#### 3. Query Rewriting (Medium Priority)
Use Claude to expand/clarify queries before searching.

#### 4. Reranking (Medium Priority)
After retrieving top 20 results, use Claude to rerank by relevance.

#### 5. Evaluation Test Set (Important)
Create 20-30 test questions with expected sources to measure retrieval quality.

### Phase 4: Deployment
- Streamlit Cloud or local ngrok for tester access
- Authentication if needed
- Usage analytics

---

## Commands Reference

### Running the Chatbot
```
# CLI version
python ask_brett_hybrid.py

# Web version
python -m streamlit run ask_brett_web.py

# Expose via ngrok
ngrok http 8501
```

### Chatbot Commands
```
/clear          - Clear conversation history
/exclude <id>   - Exclude a chunk from results
/include <id>   - Re-include a chunk
/list-excluded  - Show excluded chunks
/sources        - Show chunk IDs from last search
/help           - Show help
quit            - Exit
```

### Adding New Content
```
# For documents (PDF, DOCX, PPTX)
# Place in source folder, then:
python process_docs.py
python build_index.py

# For web content (like Kotter)
python fetch_kotter_resources.py
python process_kotter.py
python build_index.py
```

---

## Session Notes - Jan 28, 2025

### What We Built
1. Hybrid search chatbot with adaptive weighting
2. Web interface with Streamlit
3. Kotter knowledge integration (53 new chunks)
4. Search comparison tool
5. Multiple search quality improvements

### Issues Encountered & Resolved
1. OneDrive file locking - used fallback index file
2. Invalid nul filenames - removed with PowerShell
3. CEO documents not found - added source filename boosting + stop words
4. Inventory question favored wrong sources - tuned RRF k and weights

### Ready for Testing
The system is ready for tester access via ngrok. Current search quality is good but can be improved with dense embeddings (Phase 3).

---

---

## Session Notes - Jan 29, 2025

### Knowledge Base Consolidation

#### Moved Kotter Content to Main Knowledge Base
- **Before:** `doc_processor/kotter_knowledge/` (separate folder with .txt files)
- **After:** `Ask Brett Data/Kotter Change Management/` (consolidated with other content)
- **Rationale:** Single source of truth for all knowledge content

#### Updated config.yaml
- Added `.txt` to `file_types` list
- Now `process_docs.py` handles all content types in one pipeline
- Eliminates need for separate `process_kotter.py` script

### The Brazin Culture Handbook - Processed

#### Content Overview
Brett's first culture bible - the foundational document defining Brazin's corporate culture.

#### Processing Details
| Part | Pages | Chunks | Words Extracted |
|------|-------|--------|-----------------|
| Part 1 | 40 | 9 | ~8,300 |
| Part 2 | 42 | 10 | ~10,000 |
| Part 3 | 2 | 1 | ~500 |
| **Total** | **84** | **20** | **~18,800** |

#### OCR Processing
- Documents were scanned PDFs (no embedded text)
- Tesseract OCR successfully extracted content at 200 DPI
- Lower word-per-page count (~230 avg) due to handbook formatting:
  - Large fonts and headers
  - Chapter title pages
  - Images and whitespace

#### Metadata Updates
Ensured all 20 chunks reference "culture" for searchability:
- 16 chunks already had culture in topic/keywords
- Updated 4 chunks to add "Brazin culture" keyword:
  - Part 1 Chunk 4: Employee Engagement and Customer Satisfaction
  - Part 1 Chunk 5: Sales Strategies
  - Part 1 Chunk 7: Retail Strategy
  - Part 2 Chunk 5: Retail Business Performance Management

### Chunking Improvements (Option A Implementation)

#### Problem Identified
PowerPoint presentations often have topic slides (Slide X) followed by detail slides (Y, Z) that expand on the topic. Standard word-count chunking could split related content, losing context.

#### Solutions Implemented

**1. Contextual Headers**
Each chunk now starts with document context embedded in the searchable text:
```
[Document: Define Customer Focus.pptx]

[Slide 3]
The 5 Principles of Customer Focus
...
```

**2. Increased Overlap**
- Before: 50 words overlap between chunks
- After: 100 words overlap
- Benefit: Better context preservation at chunk boundaries

#### Code Changes
- `config.yaml`: `chunk_overlap_words: 50` → `chunk_overlap_words: 100`
- `process_docs.py`: Added contextual header injection before saving chunks
- `process_docs.py`: Added `extract_text_from_txt()` function for .txt file support

### Additional Content Processed

#### New "additional" Folder Content
234 new files added covering:

| Category | Content |
|----------|---------|
| **HFT (High Focus Topics)** | 2009-2024 presentations, leadership training |
| **Kotter Methodology** | 8 Steps, Leading Change, Accelerate (.txt files) |
| **Jim Collins** | Good to Great materials, 12 Questions |
| **Books** | Wal-Mart Way (79 chunks), Little Book That Builds Wealth (50 chunks) |
| **MBWA** | Management by Walking Around materials |
| **Store Visits** | Effective Store Visit presentations |
| **Meeting Transcripts** | Leah Uka & Brett Blundy sessions |
| **Culture** | Brazin Culture Handbook (.doc), Culture Academy materials |

#### Processing Results
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Documents | 371 | 612 | +241 |
| Chunks | 1,748 | 2,556 | +808 |
| API Cost | $0.55 | $0.81 | +$0.26 |

- 6 duplicates automatically detected and skipped
- OCR used for scanned PDFs
- .txt file support added mid-session

### Updated File Structure

```
Ask Brett Data/                    # Consolidated knowledge base
├── additional/                    # NEW - 234 files
│   ├── Effective Store Visit/
│   ├── HFT/                       # 2009-2024 presentations
│   ├── Jim Collins/
│   ├── Kotter Change Management/
│   ├── Management by Walking Around (MBWA)/
│   ├── The Brazin Culture Handbook/
│   ├── Brazin Culture Handbook.doc
│   ├── Define Customer Focus.pptx
│   └── Inventory & Sales Strategy.docx
├── Brenton - NCS/
├── Brett & Leah/
├── Cody Transcripts/
├── Kotter Change Management/      # Moved from doc_processor/
└── The Brazin Culture Handbook/

doc_processor/
├── ask_brett.py
├── ask_brett_semantic.py
├── ask_brett_hybrid.py
├── ask_brett_web.py
├── build_index.py
├── process_docs.py                # Updated: contextual headers, .txt support, near-duplicate detection
├── compare_search.py
├── fetch_kotter_resources.py      # Can be removed (content moved)
├── process_kotter.py              # Can be removed (use process_docs.py)
├── config.yaml                    # Updated: .txt file type, 100 word overlap
├── requirements.txt
└── output/
    ├── chunks/                    # 1,933 total chunks (after dedup)
    ├── index.csv
    ├── search_index.pkl           # 28.9 MB
    └── excluded_chunks.json
```

### Duplicate Detection & Cleanup

#### Problem Identified
Content duplicates were polluting the knowledge base:
- Same book in multiple folders (Welch's Winning, Wal-Mart Way)
- Same presentation saved as PDF and PPTX
- Autosaved versions alongside originals
- ~157 duplicate chunks detected

#### Cleanup Performed
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Chunks | 2,090 | 1,933 | -157 removed |
| Index size | 31.3 MB | 28.9 MB | -2.4 MB |

#### Near-Duplicate Detection Added to `process_docs.py`
Two-layer duplicate detection now in place:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: File Hash (existing)                              │
│  - MD5 of raw file bytes                                    │
│  - Catches: exact same file in different folders            │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Text Hash (NEW)                                   │
│  - MD5 of first 5000 chars of normalized extracted text     │
│  - Catches: same content in PDF vs DOCX vs PPTX             │
│  - Catches: minor formatting differences                    │
│  - Catches: edited versions with same core content          │
└─────────────────────────────────────────────────────────────┘
```

#### Code Changes
- Added `compute_text_hash()` function for normalized text hashing
- Added `text_hashes` dict to progress.json for tracking
- Near-duplicate check runs after text extraction, before chunking
- Duplicates logged with type ("exact" or "near-duplicate")

### PowerPoint Topic Detection

#### Problem
PowerPoint presentations often have topic slides (e.g., "Customer Focus") followed by detail slides that expand on the topic but don't explicitly mention it. When chunked, detail slides become unsearchable by the main topic.

```
[Slide 1] "Customer Focus"           ← Topic intro
[Slide 2-3] Details...
─────────── CHUNK BOUNDARY ───────────
[Slide 4-5] "Implementation Steps"   ← About Customer Focus, but doesn't say it
```

#### Solution Implemented
Added `detect_presentation_topic()` function that:
1. Analyzes the first ~4000 chars of each PowerPoint
2. Uses Claude Haiku to identify the main topic (2-5 words)
3. Prepends `[Presentation Topic: X]` to every chunk from that presentation

#### Example Output
```
[Document: Define Customer Focus.pptx]
[Presentation Topic: Customer Focus]    ← NEW

[Slide 15]
Implementation Steps
1. Train your team on the 5 principles...
```

Now slides 15+ are searchable for "customer focus" even though they only discuss implementation details.

#### Reprocessing Results
| Metric | Value |
|--------|-------|
| PowerPoints reprocessed | 211 |
| Near-duplicates detected | 121 |
| Unique chunks created | ~300 |
| Additional API cost | ~$0.06 |

#### Sample Topics Detected
- Customer Focus
- Inventory Management
- Company Culture
- Store Visit Management
- Leadership Development
- Retail/Fashion Strategy

### Current Knowledge Base Stats
- **Total documents processed:** 612
- **Total unique chunks:** 1,718
- **Total API cost:** $0.87
- **Duplicates skipped:** 132 (exact + near-duplicate)
- **Index size:** 26.1 MB
- **Vocabulary:** 10,000 terms
- **Content sources:** Business documents, Kotter methodology, Brazin Culture Handbook, HFT presentations, Books, Meeting transcripts

### Systematic Search Testing

#### Test Results: 9/10 Passed

| Query | Result | Score |
|-------|--------|-------|
| customer focus | ✅ PASS - Customer Focus Strategy | 0.541 |
| inventory management | ✅ PASS - Inventory Management | 0.546 |
| brazin culture handbook | ✅ PASS - Business Culture and Leadership | 0.533 |
| kotter change | ✅ PASS - Change Management | 0.436 |
| leadership | ✅ PASS - Values-Based Leadership | 0.394 |
| store visit | ✅ PASS - Store Management Practices | 0.560 |
| goal setting | ⚠️ WEAK - Found Team Building (versions deduplicated) | 0.495 |
| ceo role responsibilities | ✅ PASS - CEO's Role in Product | 0.270 |
| team culture values | ✅ PASS - Organizational Culture | 0.297 |
| retail strategy | ✅ PASS - Retail Strategy | 0.307 |

#### Verification Results
- **PowerPoint topic headers:** 139/139 (100%) have topic headers
- **Duplicate detection:** Working correctly (caught multiple versions of same presentations)
- **Search quality:** Strong relevance for key business topics

#### What's Working Well
- Customer Focus queries find relevant presentations
- Inventory Management content indexed correctly
- Brazin Culture Handbook findable with culture keywords
- Kotter methodology searchable
- CEO/Leadership content accessible
- Store Visit materials indexed
- Topic headers ensure detail slides remain searchable by main topic

---

## Session Notes - Feb 1, 2025

### Performance Optimization: Memory Caching

#### Problem Identified
The keyword search was reading **every chunk file from disk** on each query - approximately 1,718 file reads per search, causing noticeable latency.

#### Solution Implemented
Added in-memory chunk caching to `ask_brett_web.py`:

1. **New function `load_all_chunks()`** - Loads all chunk content into memory at startup
2. **Cached with `@st.cache_resource`** - Chunks persist in memory across requests
3. **Updated `keyword_search()`** - Uses cached content instead of disk reads

#### Performance Results

| Metric | Before | After |
|--------|--------|-------|
| File reads per query | ~1,718 | 0 |
| Search latency | Slow (disk I/O bound) | ~231ms average |
| RAM usage | Low | +~30-50MB |

#### Code Changes
- `ask_brett_web.py`: Added `load_all_chunks()`, updated `keyword_search()`, `hybrid_search()`, and `build_context()` to use chunk cache

### Query Logging Feature

Added usage tracking to monitor what team members are asking.

#### Implementation
- **Log file:** `output/query_log.csv`
- **Function:** `log_query()` in `ask_brett_web.py`

#### Fields Captured

| Column | Description |
|--------|-------------|
| timestamp | ISO format datetime |
| question | Exact question text |
| search_mode | "keyword-boosted" or "balanced" |
| num_sources | Number of sources found |
| top_source | Top-ranked source document |

#### Usage
```bash
# View query log
cat output/query_log.csv

# Open in Excel/Sheets for analysis
```

### Systematic Test Script

Created `test_search.py` for automated search quality and performance testing.

#### Test Queries
- customer focus
- inventory management
- leadership
- culture
- kotter change management
- CEO role responsibilities
- store visit
- retail strategy
- team building
- goal setting

#### Test Results: 10/10 Passed

| Metric | Value |
|--------|-------|
| Tests passed | 10/10 |
| Average search time | 231ms |
| Min/Max search time | 196ms / 298ms |
| Index load time | 1.37s (one-time) |

### Updated File Structure

```
doc_processor/
├── ask_brett_web.py          # Updated: memory caching, query logging
├── test_search.py            # NEW: systematic search tests
└── output/
    ├── query_log.csv         # NEW: usage tracking log
    └── ...
```

### Streamlit Cloud Deployment (Completed)

#### Setup
- **GitHub repo:** `rbshipping/ask-brett` (PUBLIC - required for Streamlit Cloud free tier)
- **Hosting:** Streamlit Community Cloud (free tier)
- **Access:** Password protected

#### Configuration
- Secrets stored in Streamlit Cloud (API key, app password)
- Code updated to support both local `.env` and Streamlit Cloud secrets
- Version fixes: scikit-learn 1.8.0, anthropic 0.76.0

#### Access
- Share the Streamlit app URL with team members
- Provide the password set in Streamlit secrets
- Works for both local network and remote team members

#### Security Model & Decisions

**Why the repo must be public:**
Streamlit Cloud (free tier) requires public repo access. Every time the app reboots or redeploys, it clones the repo from GitHub. Private repos require Streamlit Teams ($250/month).

**What IS protected:**
| Asset | Protection |
|-------|------------|
| Streamlit app | Password required to use |
| API key | Stored in Streamlit secrets (not in repo) |
| App password | Stored in Streamlit secrets (not in repo) |

**What is publicly visible (if someone finds the repo):**
- Source code
- Chunk files containing Brett's business knowledge

**Risk assessment:**
- Repo URL not advertised or linked anywhere
- Username (`rbshipping`) doesn't reveal content
- Someone would need to specifically find and browse the repo
- Actual app usage is fully password protected

**Decision:** Accept low risk of repo discovery in exchange for free, convenient team access via Streamlit Cloud.

**Alternative if higher security needed:** Use ngrok for remote access (repo stays private, but URL changes and requires ngrok running).

#### Managing Secrets
To change the app password or API key:
1. Go to Streamlit Cloud → Manage app → Settings → Secrets
2. Update the values:
   ```toml
   ANTHROPIC_API_KEY = "your-api-key"
   APP_PASSWORD = "your-new-password"
   ```
3. Click Save, then Reboot the app

### Conversational Mode (Added)

Updated the chatbot to be more conversational rather than a simple Q&A tool.

#### Behavior Change
- **Before:** User asks question → System searches → Returns answer
- **After:** User asks question → System searches → Returns answer + asks follow-up question

#### Example
- User: "I need help with a difficult team member"
- Response: Provides guidance from knowledge base, then asks "What type of difficulty are you experiencing - is it performance-related or an interpersonal conflict?"

#### Implementation
Updated system prompt in `ask_brett_web.py` to instruct Claude to:
- Always ask a relevant follow-up question after answering
- Tailor questions to understand the user's specific situation
- Keep conversation flowing naturally

---

## Next Steps

### Phase 3: RAG Enhancements (Not Started)

#### 1. Dense Embeddings (High Priority)
Replace TF-IDF with sentence-transformers for better semantic matching.
- "CEO" would match "chief executive", "leader", "executive"
- "clear inventory" would match "markdown", "liquidate", "sell through"

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, runs locally
```

#### 2. Query Rewriting (Medium Priority)
Use Claude to expand/clarify queries before searching.

#### 3. Reranking (Medium Priority)
After retrieving top 20 results, use Claude to rerank by relevance.

#### 4. Evaluation Test Set (Important)
Expand test_search.py with 20-30 test questions with expected sources.

### Phase 4: Analytics & Enhancements

#### 1. User Identification
Add optional user name field to track who is asking what.

#### 2. Query Analytics Dashboard
Build simple dashboard to visualize:
- Most common questions
- Questions with no/few results (content gaps)
- Usage over time

#### 3. Feedback Mechanism
Allow users to rate answers or flag incorrect responses.

### Monitoring Tasks
1. Review query_log.csv periodically to identify:
   - Common questions (potential FAQ candidates)
   - Failed searches (content gaps to fill)
   - Search patterns (topics of interest)

---

*Last updated: February 2, 2025*
