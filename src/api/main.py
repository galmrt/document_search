import hashlib
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter

from src.utils.es_service import ESService
from src.utils.embedding_service import EmbeddingService
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.email_processor import EmailProcessor

load_dotenv()

es_service: ESService = None
embedding_service: EmbeddingService = None
pdf_processor: PDFProcessor = None
email_processor: EmailProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global es_service, embedding_service, pdf_processor, email_processor
    es_service = ESService(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    es_service.ensure_index()
    embedding_service = EmbeddingService()
    converter = DocumentConverter()
    pdf_processor = PDFProcessor(converter, embedding_service)
    email_processor = EmailProcessor(embedding_service)
    print("Services started")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()
    file_id = hashlib.sha256(contents).hexdigest()
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix == ".pdf":
        if es_service.file_exists(file_id):
            return {"filename": file.filename, "file_id": file_id, "status": "already_indexed"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        version = es_service.get_next_version(file.filename)
        try:
            chunks, embeddings = pdf_processor.get_chunks(tmp_path)
            es_service.index_chunks(file.filename, file_id, version, chunks, embeddings)
        finally:
            os.unlink(tmp_path)

        return {"filename": file.filename, "file_id": file_id, "version": version, "chunks_indexed": len(chunks)}

    elif suffix in (".eml", ".mbox"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            chunks, embeddings, metadata_list = email_processor.process(tmp_path)
            indexed = es_service.index_emails(file.filename, file_id, chunks, embeddings, metadata_list)
        finally:
            os.unlink(tmp_path)

        return {"filename": file.filename, "emails_indexed": indexed}

    else:
        return {"error": f"Unsupported file type: {suffix}. Supported: .pdf, .eml, .mbox"}


@app.post("/query")
async def query(query: str):
    query_embedding = embedding_service.encode([query])[0]
    results = es_service.search(query_embedding, size=5)
    return {"results": results}
