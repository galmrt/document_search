import json
import logging
import os
from datetime import datetime, timezone

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document

INDEX_NAME = "knowledge_base"
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "indices_mapping", "knowledge_base.json")


class ESService:
    def __init__(self, es_url: str):
        self.es = Elasticsearch(es_url)
        self.logger = logging.getLogger(__name__)

    def ensure_index(self):
        if not self.es.indices.exists(index=INDEX_NAME):
            with open(MAPPING_PATH) as f:
                mapping = json.load(f)
            self.es.indices.create(index=INDEX_NAME, body=mapping)
            self.logger.info(f"Created index: {INDEX_NAME}")

    def exists(self, field: str, value: str) -> bool:
        result = self.es.count(index=INDEX_NAME, body={"query": {"term": {field: value}}})
        return result["count"] > 0

    def get_next_version(self, filename: str) -> int:
        result = self.es.search(index=INDEX_NAME, body={
            "size": 0,
            "query": {"term": {"file_name": filename}},
            "aggs": {"max_version": {"max": {"field": "version"}}}
        })
        current = result["aggregations"]["max_version"]["value"]
        return int(current) + 1 if current else 1

    def index_chunks(self, filename: str, file_id: str, version: int, chunks: list[Document], embeddings: list[list[float]], doc_type: str = "pdf"):
        timestamp = datetime.now(timezone.utc).isoformat()
        actions = [
            {
                "_index": INDEX_NAME,
                "_source": {
                    "file_id": file_id,
                    "file_name": filename,
                    "doc_type": doc_type,
                    "@timestamp": timestamp,
                    "version": version,
                    "page_number": chunk.metadata.get("page_number", 0),
                    "chunk_index": i,
                    "content": chunk.page_content,
                    "embedding": embedding,
                },
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        indexed_count, errors = bulk(self.es, actions, raise_on_error=False)
        if errors:
            self.logger.error(f"{len(errors)} chunks failed to index for {filename}")
        self.logger.info(f"Successfully indexed {indexed_count}/{len(chunks)} chunks for {filename}")
        return indexed_count
    
    def index_emails(self, filename: str, file_id: str, chunks: list[Document], embeddings: list[list[float]]):
        timestamp = datetime.now(timezone.utc).isoformat()
        actions = [
            {
                "_index": INDEX_NAME,
                "_source": {
                    "file_id": file_id,
                    "file_name": filename,
                    "doc_type": "email",
                    "@timestamp": timestamp,
                    "content": chunk.page_content,
                    "embedding": embedding,
                    "email_id": meta["email_id"],
                    "thread_id": meta["thread_id"],
                    "sender": meta["sender"],
                    "email_date": meta["email_date"],
                    "subject": meta["subject"],
                },
            }
            for chunk, embedding in zip(chunks, embeddings)
            if not self.exists("email_id", (meta := chunk.metadata)["email_id"])
        ]
        indexed_count, errors = bulk(self.es, actions, raise_on_error=False)
        if errors:
            self.logger.error(f"{len(errors)} emails failed to index from {filename}")
        self.logger.info(f"Indexed {indexed_count} emails from {filename}")
        return indexed_count

    def search(self, query_text: str, query_embedding: list[float], size: int = 5):
        fields = ["content", "file_name", "doc_type", "page_number", "sender", "subject", "email_date"]
        fetch_size = max(size * 4, 20)

        bm25_resp = self.es.search(
            index=INDEX_NAME,
            query={"match": {"content": query_text}},
            source=False,
            fields=fields,
            size=fetch_size,
        )
        knn_resp = self.es.search(
            index=INDEX_NAME,
            knn={
                "field": "embedding",
                "query_vector": query_embedding,
                "k": fetch_size,
                "num_candidates": 100,
            },
            source=False,
            fields=fields,
            size=fetch_size,
        )

        # Manual RRF: score = sum of 1 / (k + rank) across result lists
        RRF_K = 60
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, hit in enumerate(bm25_resp["hits"]["hits"], start=1):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            docs[doc_id] = hit["fields"]

        for rank, hit in enumerate(knn_resp["hits"]["hits"], start=1):
            doc_id = hit["_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            docs[doc_id] = hit["fields"]

        top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:size]
        return [
            {k: v[0] if isinstance(v, list) and len(v) == 1 else v
             for k, v in docs[doc_id].items()}
            for doc_id in top_ids
        ]
