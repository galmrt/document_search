import json
import logging
import os
from datetime import datetime, timezone

from elasticsearch import Elasticsearch
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

    def index_chunks(self, filename: str, file_id: str, version: int, chunks: list[Document], embeddings: list[list[float]]):
        indexed_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "file_id": file_id,
                "file_name": filename,
                "doc_type": "pdf",
                "@timestamp": datetime.now(timezone.utc).isoformat(),
                "version": version,
                "page_number": chunk.metadata.get("page_number", 0),
                "chunk_index": i,
                "content": chunk.page_content,
                "embedding": embedding,
            }

            try:
                response = self.es.index(index=INDEX_NAME, document=doc)
                indexed_count += 1
                if i == 0:
                    self.logger.info(f"First chunk indexed with ID: {response['_id']}")
            except Exception as e:
                self.logger.error(f"Error indexing chunk {i} for {filename}: {e}")
                raise

        self.logger.info(f"Successfully indexed {indexed_count}/{len(chunks)} chunks for {filename}")
        return indexed_count
    
    def index_emails(self, filename: str, file_id: str, chunks: list[Document], embeddings: list[list[float]]):
        indexed_count = 0
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk.metadata
            if self.exists("email_id", meta["email_id"]):
                continue
            doc = {
                "file_id": file_id,
                "file_name": filename,
                "doc_type": "email",
                "@timestamp": datetime.now(timezone.utc).isoformat(),
                "content": chunk.page_content,
                "embedding": embedding,
                "email_id": meta["email_id"],
                "thread_id": meta["thread_id"],
                "sender": meta["sender"],
                "email_date": meta["email_date"],
                "subject": meta["subject"],
            }
            try:
                self.es.index(index=INDEX_NAME, document=doc)
                indexed_count += 1
            except Exception as e:
                self.logger.error(f"Error indexing email {meta['email_id']}: {e}")
                raise

        self.logger.info(f"Indexed {indexed_count} emails from {filename}")
        return indexed_count

    def search(self, query_text: str, query_embedding: list[float], size: int = 5):
        knn = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": 100,
        }
        query = {"match": {"content": query_text}}
        fields = ["content", "file_name", "doc_type", "page_number", "sender", "subject", "email_date"]
        response = self.es.search(
            index=INDEX_NAME,
            knn=knn,
            query=query,
            rank={"rrf": {}},
            source=False,
            fields=fields,
            size=size,
        )
        return [
            {k: v[0] if isinstance(v, list) and len(v) == 1 else v
             for k, v in hit["fields"].items()}
            for hit in response["hits"]["hits"]
        ]
