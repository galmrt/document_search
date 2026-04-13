import logging
import ijson
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

from src.utils.embedding_service import EmbeddingService
from src.utils import schema_analyzer

CONTENT_MIN_WORDS = 4
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def _strip_keys(obj, skip_keys: set):
    """Recursively removes keys from dicts at any depth."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, skip_keys) for k, v in obj.items() if k not in skip_keys}
    if isinstance(obj, list):
        return [_strip_keys(item, skip_keys) for item in obj]
    return obj


def _has_content(obj) -> bool:
    """Returns True if obj (or any descendant) contains a string >= CONTENT_MIN_WORDS words."""
    if isinstance(obj, str):
        return len(obj.split()) >= CONTENT_MIN_WORDS
    if isinstance(obj, dict):
        return any(_has_content(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_content(item) for item in obj)
    return False


def _flatten(obj, key: str = "", content_parts: list = None, metadata: dict = None):
    """
    Recursively flattens obj into:
      - content_parts: list of "label: value" strings for long text fields
      - metadata: dict of short/structured fields

    Arrays of objects with no content are skipped entirely to avoid ID noise.
    The label used is the last meaningful key (not the full dotted path).
    """
    if content_parts is None:
        content_parts = []
    if metadata is None:
        metadata = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, k, content_parts, metadata)

    elif isinstance(obj, list):
        if not _has_content(obj):
            return
        for item in obj:
            _flatten(item, key, content_parts, metadata)

    elif isinstance(obj, str) and obj.strip():
        if len(obj.split()) >= CONTENT_MIN_WORDS:
            content_parts.append(obj.strip())
        else:
            if key:
                metadata[key] = obj

    elif obj is not None:
        if key:
            metadata[key] = obj

    return content_parts, metadata


def _chunk_text(content_parts: list[str]) -> list[str]:
    """
    Joins content parts into sliding-window chunks of ~CHUNK_SIZE words
    with CHUNK_OVERLAP word overlap.
    """
    words = "\n\n".join(content_parts).split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _detect_structure(file_obj) -> str:
    """Peeks at the first non-whitespace bytes to determine if the file is a
    top-level array or object. Returns 'array' or 'object'. Seeks back to 0 after."""
    header = file_obj.read(32)
    file_obj.seek(0)
    return "array" if header.lstrip().startswith(b"[") else "object"


class JSONProcessor:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def process(self, file_obj, filename: str) -> tuple[list[Document], list[list[float]]]:
        structure = _detect_structure(file_obj)  # seeks back to 0 internally

        if structure == "array":
            return self._process_array(file_obj, filename)
        else:
            return self._process_object(file_obj, filename)

    def _process_array(self, file_obj, filename: str) -> tuple[list[Document], list[list[float]]]:
        """
        Record-by-record: each array item is flattened and chunked independently.

        Peeks at the first two records with ijson:
        - If they share the same top-level keys (uniform schema), sends the first
          record to the LLM to identify keys worth skipping.
        - If the structure differs between records, skips LLM analysis and falls
          back to generic flattening — a single record's schema wouldn't generalise.
        """
        skip_keys = set()
        records = []

        file_obj.seek(0)
        for record in ijson.items(file_obj, "item"):
            records.append(record)
            if len(records) == 2:
                break

        if not records:
            return [], []

        metadata_keys: set = set()
        if len(records) == 1 or schema_analyzer.consistent_structure(records[0], records[1]):
            schema = schema_analyzer.analyze(records[0])
            if schema:
                skip_keys = set(schema.get("skip_keys", []))
                metadata_keys = set(schema.get("metadata_keys", []))
        else:
            logger.info("Variable record structure detected, skipping LLM schema analysis")

        file_obj.seek(0)
        all_docs = []

        for record in ijson.items(file_obj, "item"):
            cleaned = _strip_keys(record, skip_keys) if skip_keys else record
            content_parts, meta = _flatten(cleaned)
            if not content_parts:
                continue
            json_meta = {k: v for k, v in meta.items() if k in metadata_keys} if metadata_keys else meta
            for chunk_text in _chunk_text(content_parts):
                all_docs.append(Document(
                    page_content=chunk_text,
                    metadata={"file_name": filename, "json_metadata": json_meta},
                ))

        if not all_docs:
            return [], []

        embeddings = self.embedding_service.encode([d.page_content for d in all_docs])
        return all_docs, embeddings

    def _process_object(self, file_obj, filename: str) -> tuple[list[Document], list[list[float]]]:
        """
        Single document: streams top-level key-value pairs with ijson to avoid
        loading the full file into memory. Samples the first 10 keys for LLM
        schema analysis, then streams all keys for content extraction.
        """
        # Collect a small sample for schema analysis without loading the whole file
        file_obj.seek(0)
        sample: dict = {}
        for key, value in ijson.kvitems(file_obj, ""):
            sample[key] = value
            if len(sample) >= 10:
                break

        skip_keys: set = set()
        metadata_keys: set = set()
        if sample:
            schema = schema_analyzer.analyze(sample)
            if schema:
                skip_keys = set(schema.get("skip_keys", []))
                metadata_keys = set(schema.get("metadata_keys", []))

        # Stream all key-value pairs for actual content extraction
        content_parts: list[str] = []
        meta: dict = {}
        file_obj.seek(0)
        for key, value in ijson.kvitems(file_obj, ""):
            if key not in skip_keys:
                _flatten(value, key, content_parts, meta)

        if not content_parts:
            return [], []

        json_meta = {k: v for k, v in meta.items() if k in metadata_keys} if metadata_keys else meta
        docs = [
            Document(page_content=chunk_text, metadata={"file_name": filename, "json_metadata": json_meta})
            for chunk_text in _chunk_text(content_parts)
        ]
        embeddings = self.embedding_service.encode([d.page_content for d in docs])
        return docs, embeddings
