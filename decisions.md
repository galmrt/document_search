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
