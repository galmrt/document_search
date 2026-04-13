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
from src.ingestion.json_processor import JSONProcessor

load_dotenv()

es_service: ESService = None
embedding_service: EmbeddingService = None
pdf_processor: PDFProcessor = None
email_processor: EmailProcessor = None
json_processor: JSONProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global es_service, embedding_service, pdf_processor, email_processor, json_processor
    es_service = ESService(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    es_service.ensure_index()
    embedding_service = EmbeddingService()
    pdf_processor = PDFProcessor(DocumentConverter(), embedding_service)
    email_processor = EmailProcessor(embedding_service)
    json_processor = JSONProcessor(embedding_service)
    print("Services started")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def upload_file(file: UploadFile):
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix == ".pdf":
        contents = await file.read()
        file_id = hashlib.sha256(contents).hexdigest()
        if es_service.exists("file_id", file_id):
            return {"filename": file.filename, "file_id": file_id, "status": "already_indexed"}
        version = es_service.get_next_version(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            chunks, embeddings = pdf_processor.get_chunks(tmp_path)
            es_service.index_chunks(file.filename, file_id, version, chunks, embeddings)
        finally:
            os.unlink(tmp_path)
        return {"filename": file.filename, "file_id": file_id, "version": version, "chunks_indexed": len(chunks)}

    elif suffix in (".eml", ".mbox"):
        contents = await file.read()
        file_id = hashlib.sha256(contents).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            chunks, embeddings = email_processor.process(tmp_path)
            indexed = es_service.index_emails(file.filename, file_id, chunks, embeddings)
        finally:
            os.unlink(tmp_path)
        return {"filename": file.filename, "emails_indexed": indexed}

    elif suffix == ".json":
        hasher = hashlib.sha256()
        while chunk := await file.read(8192):
            hasher.update(chunk)
        file_id = hasher.hexdigest()
        if es_service.exists("file_id", file_id):
            return {"filename": file.filename, "file_id": file_id, "status": "already_indexed"}
        file.file.seek(0)
        version = es_service.get_next_version(file.filename)
        chunks, embeddings = json_processor.process(file.file, file.filename)
        es_service.index_chunks(file.filename, file_id, version, chunks, embeddings, doc_type="json")
        return {"filename": file.filename, "file_id": file_id, "version": version, "chunks_indexed": len(chunks)}

    else:
        return {"error": f"Unsupported file type: {suffix}. Supported: .pdf, .eml, .mbox, .json"}


@app.post("/query")
async def query(query: str, size: int = 5):
    query_embedding = embedding_service.encode_one(query)
    results = es_service.search(query, query_embedding, size=size)
    return {"results": results}
