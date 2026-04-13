# Decision Cheatsheet

Quick answers to likely questions about technical decisions in this system.

---

## Embeddings

**Why BAAI/bge-large-en-v1.5 over a legal-specific model like Kanon-2?**
Cost. Zero-cost constraint drove all model choices. BGE is free, open-source, and ranks at the top of the MTEB leaderboard for general-purpose English retrieval. Kanon-2 is commercial — no budget allocated at this stage. The 9% gap was judged acceptable for an MVP.

**What does "9% less accurate" mean?**
It is a benchmark score gap on MTEB (Massive Text Embedding Benchmark), not a direct measure of "9% fewer relevant results" in production. Real-world impact depends on query type — factual keyword-heavy queries (case names, statute numbers) are less affected than abstract semantic queries (implied obligations, risk language).

**If we swap the embedding model later, do we need to re-index everything?**
Yes, fully. Embeddings from different models live in incompatible vector spaces — you cannot mix them in one index. A model swap requires: (1) delete the index, (2) re-embed all documents with the new model, (3) re-index. Plan for a full reprocessing pipeline before committing to a model change in production.

**Why not OpenAI embeddings — $1,300 was a one-time cost?**
Two reasons beyond cost: (1) data privacy — sending 2M legal documents to a third-party API is a legal and client confidentiality risk; (2) ongoing dependency — embedding queries in production would also go through the API, adding latency and per-query cost at scale. Self-hosted avoids both.

**Why is the BGE query prefix only applied to queries, not indexed documents?**
BGE retrieval models are trained asymmetrically. The prefix `"Represent this sentence for searching relevant passages: "` is a task instruction that shifts the query embedding toward the retrieval direction in the model's latent space. Document embeddings are produced without it — this asymmetry is intentional and matches how the model was trained (see BGE paper). Applying the prefix to both degrades retrieval quality.

---

## Chunking

**Why did we move away from fixed-size chunking for PDFs?**
Fixed-size chunking (500 words / 100-word overlap) cuts at arbitrary word boundaries — mid-sentence, mid-clause, mid-table. For legal text, a clause split across two chunks means neither chunk is self-contained for retrieval. Docling's HybridChunker splits on document structure (headings, paragraphs, table boundaries), producing chunks that map to actual semantic units.

**What does Docling's HybridChunker do that RecursiveCharacterTextSplitter doesn't?**
`RecursiveCharacterTextSplitter` is text-only — it sees a flat string and splits on character count with separator heuristics. `HybridChunker` operates on Docling's document model, which encodes the document's structure: headings, sections, tables, lists, and layout. It splits on structure boundaries, not character counts. It also provides heading metadata per chunk, which is prepended to the chunk content for better retrieval context.

**Why is overlap reset at section boundaries?**
Overlap carries the tail of one chunk into the head of the next to handle queries that span a boundary. But carrying content from one section into another (e.g. appending a fragment of "Indemnification" into the start of "Governing Law") pollutes the chunk with irrelevant context, hurting both embedding quality and BM25 matching. Resetting at section boundaries keeps chunks topically clean.

**For emails, why semantic chunking — most emails are short?**
Short emails (under ~300 words) are returned as a single chunk unchanged by the SemanticChunker. Semantic chunking only kicks in for long emails with multiple topics — legal opinions, contract reviews, thread summaries — where a fixed word window would cut mid-argument. The cost of running semantic chunking on a short email is one embedding pass; the benefit on long emails is significant retrieval improvement.

**What happens to a 200MB JSON file — does it load into memory?**
No. The JSON processor uses `ijson` for streaming parsing. Records in an array are consumed one at a time; top-level keys in an object are streamed as key-value pairs. Neither approach loads the full file. The schema analysis step peeks at the first 1–2 records only.

---

## Search & Ranking

**Why two ES queries instead of one for hybrid search?**
Native ES RRF (which would allow a single combined query) requires a Platinum or Enterprise license. The current environment uses the free Basic license, which returns a `403` if you attempt `rank: {rrf: {}}`. Two separate queries (BM25 + KNN) are issued and merged in Python as a workaround. In production on a licensed cluster, this collapses to one request.

**What is RRF and why k=60?**
Reciprocal Rank Fusion merges ranked lists by scoring each document as `sum of 1 / (k + rank)` across all lists. A document appearing at rank 1 in both BM25 and KNN scores higher than one appearing at rank 1 in only one list. `k=60` is the Elasticsearch default — it controls how quickly the score decays with rank. Lower k = steeper decay (top ranks dominate more); higher k = flatter (more weight to lower-ranked results). 60 is a well-established empirical default.

**What's the risk of manual RRF missing hits near the fetch window edge?**
Each query fetches `size * 4` candidates (minimum 20). If a document ranks at position 21 in BM25 and position 3 in KNN, but `size * 4 = 20`, it is invisible to the Python merge step even though it would score well. Native ES RRF operates on full shard-level rankings without this truncation. In practice, `size * 4` is a generous buffer — the risk is low for typical query sizes (5–10 results) but increases if `size` is large.

**Is native RRF identical to our manual implementation?**
Not exactly. ES native RRF applies fusion at the shard level before results are merged across shards, which is more accurate for multi-shard setups. Our Python implementation merges only the top-N results from each query after shard-level aggregation. For a single-shard setup (current config), they are equivalent. For a multi-shard production cluster, native RRF is more correct.

**Why fetch `size * 4` candidates?**
After RRF merging, the top `size` results are returned. But a document that ranks poorly in one list and well in the other might not appear in the top `size` of either list individually — it only surfaces after fusion. Fetching more candidates from each list gives fusion more material to work with. `4x` is a conservative multiplier; the minimum floor of 20 handles small result sizes.

---

## Infrastructure

**Why Elasticsearch over PostgreSQL + pgvector?**
Three reasons: (1) the company already runs ES — existing expertise and infrastructure; (2) ES full-text search (BM25 with the English analyser — stemming, stopwords) is significantly stronger than PostgreSQL `tsvector` for legal text; (3) ES scales horizontally for 2M documents without schema changes. pgvector is simpler to operate but weaker on keyword retrieval, which matters for exact legal term matching (statute numbers, case citations).

**Why single shard, zero replicas — what breaks first?**
Zero replicas means no redundancy — if the ES node goes down, search is unavailable until it recovers. Single shard means no horizontal query parallelism. For development and single-node testing this is fine. In production: set replicas ≥ 1 for availability, and increase shards based on index size (ES recommends 10–50GB per shard as a starting point — at ~2M documents expect 50–200GB depending on embedding size).

**Why not a graph database?**
The primary operation is retrieval — given a query, find the most relevant text chunks. ES is purpose-built for this. A graph database (Neo4j, Neptune) adds value when relationships between entities are themselves queryable: citation chains, party-to-contract links, cross-matter entity graphs. The current system models some relationships (thread_id, file_id, version) but does not need to traverse them. If the product evolves to answer questions like "show all contracts involving this party across all matters", a graph layer on top of ES would be the right addition — not a replacement.

**Do emails and PDFs share one index?**
Yes. Everything goes into the `knowledge_base` index. The `doc_type` field (`"pdf"`, `"email"`, `"json"`) enables filtered search. The `POST /query` endpoint accepts an optional `doc_type` parameter that adds a filter to both BM25 and KNN queries, so users can restrict search to emails or documents only.

---

## Data & Scale

**How does deduplication work if the same file is re-uploaded with a different filename?**
It doesn't. `file_id` is a SHA-256 hash of the file bytes. The same content with a different filename gets a different `file_name` but the same `file_id` — and the system currently checks `file_id` existence before indexing PDFs and JSONs. However, the new `file_name` would not match the existing record's `file_name`, so the dedup check would pass and the file would be indexed again. This is a known gap — content-level dedup across filenames is not implemented.

**What happens to old versions — are they still searchable?**
Yes. Old versions remain in the index and are searchable. The `version` field is incremented per filename on each upload; it is stored on every chunk. There is no mechanism to suppress old versions from search results. If this matters (e.g. superseded contracts), a version filter would need to be added to queries.

**At 2M documents, what's the expected index size?**
Rough estimate: each chunk stores a 1024-dimension float32 embedding (4KB), plus content text (~1–2KB), plus metadata. At ~5 chunks per document average and 2M documents: 10M chunks × ~6KB = ~60GB. Add ES overhead (inverted index, doc values): expect 80–120GB. A single node with 128GB RAM and NVMe SSD is the minimum comfortable production setup.

**Has the 10-word threshold for JSON been validated?**
No — it was set based on reasoning (IDs, codes, and statuses are rarely more than 5–6 words; meaningful prose rarely fewer than 10) and tested against the compliance JSON samples in this repo. It has not been validated against the actual production compliance exports. When real files are available, check the word-count distribution of string fields and adjust `CONTENT_MIN_WORDS` in `json_processor.py` accordingly.

**What about scanned PDFs — does Docling handle OCR?**
Docling includes OCR via Tesseract for scanned pages. It detects whether a page is text-based or image-based and applies OCR automatically. Quality depends on scan resolution and language — legal documents scanned at 300 DPI in English work well. Handwritten annotations and low-quality scans will degrade extraction quality. No separate OCR pipeline is needed for standard scanned PDFs.

---

## Accuracy & Evaluation

**How does the system refuse to answer when nothing relevant is found?**
Currently it doesn't — the API returns whatever ES returns, even if the scores are low. The retrieval layer has no confidence threshold. This needs to be implemented before production: add a minimum RRF score threshold and return an empty result (or explicit "not found") if no chunk exceeds it. The score threshold needs calibration against real queries.

**What are the current hit rate and RAGAS scores?**
Hit rate is measured in `tests/test_chunking_hitrate.py` against a Cornell Law bail reform paper — threshold is ≥70% across 12 questions. RAGAS (context precision + recall) is measured in `tests/test_ragas_eval.py` — threshold is ≥0.5 for both metrics. These are development baselines on one document, not production benchmarks. Scores on real legal document corpora have not yet been measured.

**How do we know when recall is good enough to go live?**
Define a golden evaluation set: 50–100 queries with known relevant documents, drawn from representative real documents. Target recall ≥ 0.80 (missing fewer than 1 in 5 relevant chunks). Run this eval after each significant change (chunking strategy, embedding model, index mapping). Recall below 0.75 should block a production release.

**What if a clause spans two chunks and neither is sufficient alone?**
This is handled partially by chunk overlap — the tail of one chunk is repeated at the head of the next, so a boundary-spanning clause appears whole in at least one chunk. For the PDF pipeline, overlap is 200 characters at sentence boundaries within a section. For emails, the SemanticChunker avoids this by keeping topically coherent content together. For very long clauses that genuinely exceed chunk size, neither approach fully solves the problem — the retrieval layer would return both adjacent chunks, and the answer synthesis layer (not yet built) would need to concatenate them.
