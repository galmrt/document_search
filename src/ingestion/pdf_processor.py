from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from src.utils.embedding_service import EmbeddingService


class PDFProcessor:
    def __init__(self, converter: DocumentConverter, embedding_service: EmbeddingService):
        self.converter = converter
        self.embedding_service = embedding_service
        self.chunker = HybridChunker()

    def get_chunks(self, file_path: str) -> tuple[list[Document], list[list[float]]]:
        result = self.converter.convert(file_path)
        chunks = list(self.chunker.chunk(dl_doc=result.document))

        if not chunks:
            return [], []

        OVERLAP_CHARS = 200

        documents = []
        chunk_texts = []
        prev_text = ""
        prev_heading = ""
        for chunk in chunks:
            pages = sorted({prov.page_no for item in chunk.meta.doc_items for prov in item.prov})
            page = pages[0] if len(pages) == 1 else pages
            heading = " > ".join(chunk.meta.headings) if chunk.meta.headings else ""
            is_new_section = heading != prev_heading
            body = f"{heading}\n\n{chunk.text}" if is_new_section and heading else chunk.text
            if prev_text and not is_new_section:
                overlap = prev_text[-OVERLAP_CHARS:]
                cut = max(overlap.rfind('. '), overlap.rfind('? '), overlap.rfind('! '))
                overlap = overlap[cut + 1:].strip() if cut != -1 else overlap.strip()
                text = f"{overlap}\n\n{body}"
            else:
                text = body
            prev_text = chunk.text
            prev_heading = heading
            chunk_texts.append(text)
            documents.append(Document(
                page_content=text,
                metadata={"file_name": file_path, "page_number": page},
            ))

        embeddings = self.embedding_service.encode(chunk_texts)
        return documents, embeddings
