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

    def index_chunks(self, filename: str, chunks: list[Document], embeddings: list[list[float]]):
        indexed_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "doc_id": f"{filename}_{i}",
                "source_file": filename,
                "doc_type": "pdf",
                "@timestamp": datetime.now(timezone.utc).isoformat(),
                "page_number": chunk.metadata.get("page_number", 0),
                "chunk_index": i,
                "content": chunk.page_content,
                "embedding": embedding,
            }

            try:
                response = self.es.index(index=INDEX_NAME, document=doc)
                indexed_count += 1
                if i == 0:  # Log first document for debugging
                    self.logger.info(f"First chunk indexed with ID: {response['_id']}")
            except Exception as e:
                self.logger.error(f"Error indexing chunk {i} for {filename}: {e}")
                raise

        self.logger.info(f"Successfully indexed {indexed_count}/{len(chunks)} chunks for {filename}")
        return indexed_count
