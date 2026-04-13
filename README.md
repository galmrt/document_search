# Legal Document Search

A prototype for hybrid semantic + keyword search over large legal document corpora (PDFs, emails, JSON compliance exports). Designed to scale — see the scaling section below.

**Loom video demonstration: link**

---

## Running locally

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start everything (Elasticsearch, API, UI)
./start.sh
```

`start.sh` starts Elasticsearch and Kibana via Docker, waits for ES to be healthy, then launches the API and UI. Press `Ctrl+C` to stop all services cleanly.

> **First run:** ~1.3 GB of model weights download automatically on API startup. This is a one-time step.

**Optional — Ollama for JSON schema analysis:**
```bash
ollama pull llama3.2:3b   # ~2 GB, improves JSON field classification
```
Without it the app falls back to generic field classification automatically.

A `.env` file is not required — all settings have working defaults. Copy `.env.example` if you need to customise the Elasticsearch URL or Ollama settings.

---

## What was built

**Ingestion pipeline** — upload endpoint accepts `.pdf`, `.eml`, `.mbox`, `.json`. Each file is processed into chunks, embedded, and bulk-indexed into Elasticsearch.

- **PDFs**: Docling extracts structured text (headings, tables, sections). HybridChunker splits on document structure rather than character counts, with 200-char sentence-boundary overlap within sections.
- **Emails**: Python stdlib parses `.eml` and `.mbox`. Quoted reply content is stripped so only new content is embedded. Thread IDs reconstructed from `References` / `In-Reply-To` headers.
- **JSON**: `ijson` streams large files without loading them into memory. Fields are classified at ingestion time — strings ≥10 words go into the searchable content field; short strings, numbers, and booleans are stored as filterable metadata. A local Ollama LLM (`llama3.2:3b`) optionally identifies noise keys (UUIDs, policy definitions) to skip before indexing.

**Search** — hybrid BM25 + KNN with manual Reciprocal Rank Fusion (RRF). Two ES queries are issued and merged in Python using `score += 1 / (60 + rank)`. Query embeddings use the BGE instruction prefix (`"Represent this sentence for searching relevant passages: "`) for asymmetric retrieval. Results are filterable by document type server-side on both the BM25 and KNN queries.

**API** — FastAPI with lifespan-managed services on `app.state`. All blocking operations (Docling, embeddings, ES I/O) run in a thread pool via `asyncio.to_thread` to keep the event loop free.

**UI** — Streamlit with upload sidebar and search interface. Filters (document type, result count) are set before querying and applied in ES, not post-fetch.

**Embeddings** — `BAAI/bge-large-en-v1.5` (1024-dim, free). ~9% less accurate than domain-specific commercial models (Kanon-2) — an accepted trade-off given the zero-cost constraint.

---

## What was left out and why

| Feature | Reason |
|---|---|
| XML / `.docx` ingestion | Not implemented in the time available. Architecture is the same — add a processor, register the suffix in the upload handler. |
| ZIP bundle ingestion | Requires recursive dispatch; deferred to a later iteration. |
| Access control | Placeholder field (`access_type`) exists in the ES mapping. Full implementation requires an auth layer (OAuth / LDAP) and per-query filter injection — out of scope for the prototype. |
| Per-record metadata in JSON chunks | When JSON records are batched into multi-record chunks, individual record metadata is lost. Acceptable for now; fixable when the actual schema is known. |
| Native ES RRF | Requires Platinum license. Manual Python RRF is used in dev. Swap to `rank: {rrf: {}}` in a single call on a licensed cluster. |

---

## Architectural decisions

**Elasticsearch over pgvector** — the corpus mixes structured metadata (sender, date, doc_type) with full-text and vectors. ES handles all three in one system — BM25, dense KNN, and metadata filtering — without a secondary store. pgvector would require a separate full-text search layer.

**Single index, `doc_type` field** — one `knowledge_base` index holds PDFs, emails, and JSON. A `doc_type` keyword field enables per-type filtering at query time. Simpler to operate than multiple indices; easy to split later if shard sizes demand it.

**Docling over pymupdf4llm** — earlier iterations with pymupdf4llm produced dirty chunks due to header/footer bleed and mid-paragraph page breaks. Docling's document understanding pipeline respects headings and tables, producing structurally coherent chunks that improve retrieval precision.

**Streaming JSON with ijson** — compliance exports can exceed 200 MB. `json.load` would spike RAM for every upload. `ijson` streams key-value pairs so memory usage is proportional to the largest single value, not the full file.

**SHA-256 for deduplication** — file identity is the hash of raw bytes. Re-uploading an identical file is a no-op. Same filename + different content increments a version counter.

**Local embeddings over API** — `sentence-transformers` runs on CPU with no per-token cost. Slower at ingestion time than an API but free and offline. At scale the cost difference becomes significant.

---

## Scaling to 2M documents

**Elasticsearch cluster** — move from 1 shard / 0 replicas (dev) to a 3-node cluster with 3 primary shards and 1 replica each. Target ~50 GB per shard. Add a dedicated coordinating node to offload query fan-out from data nodes.

**Ingestion throughput** — the current synchronous processor-per-request model will be the bottleneck. Replace with a job queue (Celery + Redis or AWS SQS): the upload endpoint enqueues a job and returns immediately; worker processes handle extraction, embedding, and indexing in parallel. Horizontal scaling by adding workers.

**Embedding throughput** — `bge-large-en-v1.5` on CPU is ~5–10 docs/sec. At 2M documents that's weeks. Options in priority order: (1) GPU inference on a single A10 cuts this 20–40×; (2) batch the embedding API calls with larger batch sizes; (3) run multiple embedding workers in parallel.

**KNN index** — with 2M × 1024-dim vectors, the HNSW index will be ~8 GB. ES loads the entire HNSW graph into RAM for ANN search — plan for at least 12 GB heap per data node. If memory is constrained, switch to `int8` quantization (≈75% memory reduction, ~2% accuracy loss).

**Query latency** — the manual two-request RRF pattern doubles ES round-trips per query. Replace with native RRF on a licensed cluster. For very high QPS, cache frequent queries with a short TTL (Redis).

**Document versioning at scale** — the current version counter uses an aggregation query per upload. Under concurrent ingestion this will produce collisions. Replace with an optimistic concurrency control pattern or a dedicated version store.

---

## What I would do differently with more time

**Semantic chunking for emails** — emails are currently one chunk per message. Long email threads with multiple topics would benefit from topic-aware splitting.

**Known JSON schema** — the adaptive 10-word classifier is a heuristic. With even one sample export file, explicit field mappings would produce cleaner embeddings and better retrieval for compliance documents.

**Evaluation-driven iteration** — the RAGAS and hit-rate tests exist but haven't been used to drive chunking decisions. Running them after each parameter change (chunk size, overlap, threshold) would make improvements measurable rather than intuitive.

**Re-ranking** — a cross-encoder re-ranker (e.g. `bge-reranker-large`) applied to the top-20 RRF results before returning top-5 would meaningfully improve precision without changing the index. At 2M documents this matters most for ambiguous queries.

**Unstructured.io for file processing** — the current stack uses Docling (PDFs), Python stdlib (emails), and ijson (JSON) as separate processors. Unstructured.io provides a single unified API across all file types including `.docx`, `.xlsx`, `.pptx`, XML, and HTML, with built-in partitioning and chunking strategies. Consolidating onto it would reduce the ingestion codebase significantly and add format coverage with little extra work.

**Streaming upload responses** — large mbox files (800K emails) block the UI until fully indexed. Server-sent events or WebSocket progress updates would make the experience usable for bulk ingestion.
