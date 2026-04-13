# 04/09/2026
## Understand and outline the requirements. 
### Deliverable requirements:
* File upload:
    * Expected formats: native vs. scanned PDFs, email plain text with metadata, JSON or XML
* Chat: 
    * Ask question in English -> get source if exist
* Retrieval of relevant precedent/clause language
    
Technical notes:
* ~ 2 million documents:
    * Mostly PDF contracts:
        * Scanned PDFs (mostly)
        * Some native PDF 
        * Memos (word docs)
        * Regulatory filings (RegTrack)
        * XML 
        * JSON (exports from compliance – can be 200MB+ in size)
        * Spreadsheets (minimal)
* Email archives (800K):
    * Plain text 
    * Metadata:
        * Sender
        * Date
        * Subject 
        * Thread ID 
* Speed: aim for < 5 sec
* Semantic search 
* Accuracy is crucial. Better to not find the document that to provide hallucinated information 

Other requirements(not necessary at the moment but should be implemented later):
* Access control (not needed for implementation but an architecture to implement late) 
* Document versioning handling (updating docs)
* Provide area/field specific search capability (search only on emails/docs)
* Handle ZIP file bundles 

## Tech stack
**Decision**: Python + Elasticsearch + Free Legal Embeddings + FastAPI

**Core Stack**:
- **Language**: Python 
- **Database**: Elasticsearch 
- **Embeddings**: 
- **API Framework**: FastAPI
- **Deployment**: Docker + Docker Compose

**Document Processing**:
- PDFs: pypdf (native), pytesseract (OCR)
- Emails: Python stdlib (email, mailbox)
- JSON: pandas

**Alternatives Considered**:
1. **Database**: PostgreSQL + pgvector (simpler but weaker full-text search)
2. **Embeddings**: 
   - OpenAI text-embedding-3-large ($1,300 for corpus, 86% MLEB score)
   - BAAI/bge-large-en-v1.5 (free general-purpose)
   - Kanon-2 embeddings - specifically trained for legal domain (commercial, high accuracy)
3. **Framework**: Flask

**Reasoning**:
- Zero-cost constraint → all free/open-source components
- Company uses Elasticsearch → leverage existing expertise
- Legal domain → legal-specific embeddings over general models
- Hybrid search essential (BM25 keyword + vector semantic)

**Trade-offs**:
- Free legal embeddings: ~9% less accurate than commercial (Kanon-2)
- Elasticsearch: More complex than Postgres but better full-text search
- Self-hosted models: Slower embedding generation vs API 

**Implementation**

Start with PDF-native files: mostly text-only, with different structures. Add complexity with tables, images and graphs. Potentially add OCR for structure recognition. Since most of the documents are legal, a lot of the time they are separated in sections and subsections. 

Chunking strartegy for large documents: 
- Start with fixed-size chunking:
    - Easiest to implement 
    - 500 words per chunk
    - 100 words of overlap
    - Use a sliding window approach

If time allows, implement a more advanced chunking strategy:
- Semantic segmentation: use a model to understand the content and split based on paragraphs, sentences or clauses. 

## Database setup ##
In order in the future to be able to make a specified search (emails. vs. docs), elasticsearch is good for having multiple indexes and unstructured data.

# 04/10/2026

## Chunking strategy update
Replaced fixed-size `RecursiveCharacterTextSplitter` with a single-pass semantic chunking approach:
- Pages are merged into one continuous text with character offset tracking, eliminating hard page-boundary breaks
- Text is split into paragraphs (`\n\n`), each paragraph embedded once using `BAAI/bge-large-en-v1.5`
- Semantic breaks detected via cosine similarity between adjacent paragraphs (adaptive 90th-percentile threshold)
- Chunk embeddings computed by averaging constituent paragraph embeddings — no second embedding pass
- Page metadata preserved per chunk; cross-page chunks store a list of page numbers

**Trade-off**: variable chunk sizes vs. fixed 500-word chunks. Accepted — topically coherent chunks improve retrieval quality for legal text.

## File identity and versioning
- `file_id`: SHA-256 hash of file bytes. Shared across all chunks of the same file. Used for deduplication — re-uploading identical content is a no-op.
- `version`: auto-incremented integer per `file_name`. Same filename + different content = new version.
- `file_name`: original uploaded filename.

# 04/11/2026

## PDF extraction: switch to Docling
Previous approach (pymupdf4llm) struggled to maintain clean page breaks — headers, footers and cross-page paragraphs were bleeding into each other, polluting chunks with layout artifacts.

Switched to **Docling** + `HybridChunker`:

**Benefits:**
- Structure-aware extraction — respects headings, sections, tables
- `HybridChunker` splits on document structure, not arbitrary character counts
- Heading metadata available per chunk (`chunk.meta.headings`) — prepended to first chunk of each section for better retrieval
- No mid-sentence or mid-table splits

**Drawbacks:**
- Slower than pymupdf4llm (full document understanding pipeline)
- No built-in overlap — implemented manually: 200-char sentence-boundary overlap within sections only, reset at section boundaries

## Email archive assumptions

- **Format**: `.mbox` — Google Takeout exports email archives in mbox format, which is the most likely source for bulk email ingestion. A single `.mbox` file contains the entire archive as a sequence of messages; Python stdlib `mailbox.mbox` iterates over them directly. `.eml` (single-message) is also supported for one-off uploads.
- **Schema**: `email_id`, `thread_id`, `sender`, `date`, `subject`, `body`
- **Threading**: `thread_id` reconstructed from `References` → `In-Reply-To` → `Message-ID` headers. Google Takeout also includes `X-GM-THRID` (the authoritative Gmail thread ID) — can be used as a more reliable fallback if thread grouping accuracy becomes a problem.
- **Body**: plain text part extracted from `multipart/alternative` messages; quoted-printable and base64 transfer encodings decoded transparently by `email.message.Message.get_payload(decode=True)`.
- **Quoted reply content**: stripped before embedding — `>` prefixed lines removed, and content below "On ... wrote:" reply headers truncated. Each message indexes only its new content.
- **Deduplication**: `email_id` = SHA-256 of `Message-ID` header, consistent with PDF file_id approach.
- **No second store**: ES handles metadata filtering (date range, sender, thread), keyword (BM25), and vector search — no SQLite or secondary index needed.
- **"First mention"**: implemented as `date: asc` + `size: 1` on filtered results — not a special pipeline.

# 04/12/2026

## Hybrid search ranking: manual RRF (dev) vs. native RRF (production)

**Context**: The search endpoint combines BM25 keyword results and KNN dense-vector results. These two ranked lists need to be merged into a single ranking. Elasticsearch provides native **Reciprocal Rank Fusion (RRF)** for exactly this, but it requires a Platinum or Enterprise license — unavailable in the free local dev environment (Basic license). Attempting to use `rank: {rrf: {}}` raises a `403 security_exception: current license is non-compliant for [Reciprocal Rank Fusion (RRF)]`.

**Current implementation (dev)**: Manual RRF in Python (`es_service.py: search`).
- BM25 and KNN queries are issued as two separate ES requests.
- Results are merged in Python using the standard RRF formula: `score += 1 / (k + rank)` for each hit across both lists, with `k = 60` (ES default).
- Top `size` hits by combined score are returned.

**Production recommendation**: Replace with native ES RRF (`rank: {rrf: {}}`).

Reasons to prefer native RRF in production:
- **Single round-trip**: one request instead of two, halving ES network overhead at query time.
- **Consistent tie-breaking**: ES applies RRF internally with access to full shard-level ranking; the Python implementation merges only the top-N results from each query, so hits that rank just outside the fetch window are invisible to the merge step.
- **Maintained by Elastic**: behaviour is versioned and tested; the manual approach requires re-validation on ES upgrades.
- **Future features**: native RRF composes with other ES features (e.g. re-rankers, learning-to-rank) more cleanly than a Python post-processing step.

**Migration**: swap the two-query pattern in `es_service.py: search` back to a single `self.es.search(index=..., knn=..., query=..., rank={"rrf": {}}, ...)` call once running on a licensed cluster.

## JSON ingestion: adaptive field classification

Compliance JSON exports have an unknown schema — no sample files are available and the field structure varies by export type.

**Decision**: classify fields at ingestion time based on content length rather than a hardcoded schema.

- **Content fields** (embedded + BM25 indexed): string fields where the value is ≥ 10 words. These drive semantic search. Stored as `"field_name: value"` concatenated into the chunk's `content` field.
- **Metadata fields** (stored, not embedded): short strings, numbers, booleans, dates — anything that isn't substantive prose. Stored as individual fields on the ES document. ES dynamic mapping picks up the types automatically (no mapping change required).

**Threshold**: 10 words is a practical cut-off that separates IDs/codes/statuses from actual descriptive text. Adjustable if real files show different distributions.

**Batching**: records are batched into ~400-word chunks before embedding to avoid near-empty vectors from short records. Per-record metadata is not preserved at the chunk level when batching — a known limitation. If records are large enough to be one-chunk-each, per-record metadata can be added to the Document metadata and passed through `index_chunks`.

**Production note**: once a schema is known, replace the adaptive classifier with explicit field mappings targeting the actual text and metadata fields for better embedding quality.
