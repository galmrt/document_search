import re

import numpy as np
import pymupdf4llm
from langchain_core.documents import Document

from src.utils.embedding_service import EmbeddingService


class PDFProcessor:
    def __init__(self, embedding_service: EmbeddingService, max_chunk_words: int = 500):
        self.embedding_service = embedding_service
        self.max_chunk_words = max_chunk_words

    def _extract_pages(self, file_path: str) -> list[dict]:
        try:
            return pymupdf4llm.to_markdown(file_path, page_chunks=True)
        except Exception as e:
            print(f"Error extracting markdown from {file_path}: {e}")
            raise

    def _merge_pages(self, page_chunks: list[dict]) -> tuple[str, list[dict]]:
        """Concatenate all page texts, tracking each page's character offsets."""
        pages = []
        parts = []
        offset = 0
        for chunk in page_chunks:
            text = chunk["text"]
            if not text.strip():
                continue
            pages.append({
                "page_number": chunk["metadata"]["page"] + 1,
                "start": offset,
                "end": offset + len(text),
            })
            parts.append(text)
            offset += len(text) + 2  # +2 for "\n\n" separator
        return "\n\n".join(parts), pages

    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """Split merged text into paragraphs, returning (text, start, end) tuples."""
        segments = []
        cursor = 0
        for para in re.split(r'\n\n+', text):
            stripped = para.strip()
            if not stripped:
                cursor += len(para) + 2
                continue
            start = text.index(stripped, cursor)
            end = start + len(stripped)
            segments.append((stripped, start, end))
            cursor = end
        return segments

    def _resolve_pages(self, seg_start: int, seg_end: int, pages: list[dict]) -> int | list[int]:
        overlapping = [
            p["page_number"] for p in pages
            if p["start"] < seg_end and p["end"] > seg_start
        ]
        return overlapping[0] if len(overlapping) == 1 else overlapping

    def get_chunks(self, file_path: str) -> tuple[list[Document], list[list[float]]]:
        page_chunks = self._extract_pages(file_path)
        full_text, pages = self._merge_pages(page_chunks)
        segments = self._split_paragraphs(full_text)

        if not segments:
            return [], []

        seg_texts = [s[0] for s in segments]

        # Single embedding pass for all paragraphs
        seg_embeddings = np.array(self.embedding_service.encode(seg_texts))

        # Cosine similarities between adjacent paragraphs
        norms = np.linalg.norm(seg_embeddings, axis=1, keepdims=True)
        normalized = seg_embeddings / np.maximum(norms, 1e-10)
        similarities = [
            float(np.dot(normalized[i], normalized[i + 1]))
            for i in range(len(normalized) - 1)
        ]

        # Adaptive threshold: split at the top 10% most dissimilar transitions
        if similarities:
            distances = np.array([1 - s for s in similarities])
            breakpoint_threshold = float(np.percentile(distances, 90))
        else:
            breakpoint_threshold = 1.0

        # Group paragraphs into chunks, respecting semantic breaks and word limit
        groups: list[list[int]] = []
        current_group = [0]
        current_words = len(seg_texts[0].split())

        for i, sim in enumerate(similarities):
            next_idx = i + 1
            next_words = len(seg_texts[next_idx].split())
            is_break = (1 - sim) >= breakpoint_threshold
            exceeds_limit = current_words + next_words > self.max_chunk_words

            if is_break or exceeds_limit:
                groups.append(current_group)
                current_group = [next_idx]
                current_words = next_words
            else:
                current_group.append(next_idx)
                current_words += next_words

        groups.append(current_group)

        # Build Documents, averaging sentence embeddings per chunk
        documents = []
        embeddings = []

        for group in groups:
            chunk_text = "\n\n".join(seg_texts[i] for i in group)
            chunk_embedding = seg_embeddings[group].mean(axis=0).tolist()
            page = self._resolve_pages(
                segments[group[0]][1],
                segments[group[-1]][2],
                pages,
            )
            documents.append(Document(
                page_content=chunk_text,
                metadata={"file_name": file_path, "page_number": page},
            ))
            embeddings.append(chunk_embedding)

        return documents, embeddings
