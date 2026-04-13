import asyncio
import hashlib
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter

from src.utils.es_service import ESService
from src.utils.embedding_service import EmbeddingService
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.email_processor import EmailProcessor
from src.ingestion.json_processor import JSONProcessor

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.es_service = ESService(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    await asyncio.to_thread(app.state.es_service.ensure_index)
    app.state.embedding_service = EmbeddingService()
    app.state.pdf_processor = PDFProcessor(DocumentConverter(), app.state.embedding_service)
    app.state.email_processor = EmailProcessor(app.state.embedding_service)
    app.state.json_processor = JSONProcessor(app.state.embedding_service)
    print("Services started")
    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    size: int = 5
    doc_type: str | None = None


@app.post("/upload")
async def upload_file(file: UploadFile, request: Request):
    es_service: ESService = request.app.state.es_service
    pdf_processor: PDFProcessor = request.app.state.pdf_processor
    email_processor: EmailProcessor = request.app.state.email_processor
    json_processor: JSONProcessor = request.app.state.json_processor

    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix == ".pdf":
        contents = await file.read()
        file_id = hashlib.sha256(contents).hexdigest()
        if await asyncio.to_thread(es_service.exists, "file_id", file_id):
            return {"filename": file.filename, "file_id": file_id, "status": "already_indexed"}
        version = await asyncio.to_thread(es_service.get_next_version, file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            chunks, embeddings = await asyncio.to_thread(pdf_processor.get_chunks, tmp_path)
            if not chunks:
                return {"filename": file.filename, "error": "No text could be extracted from this PDF"}
            await asyncio.to_thread(es_service.index_chunks, file.filename, file_id, version, chunks, embeddings)
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
            chunks, embeddings = await asyncio.to_thread(email_processor.process, tmp_path)
            if not chunks:
                return {"filename": file.filename, "error": "No emails with extractable text found in this file"}
            indexed = await asyncio.to_thread(es_service.index_emails, file.filename, file_id, chunks, embeddings)
        finally:
            os.unlink(tmp_path)
        return {"filename": file.filename, "emails_indexed": indexed}

    elif suffix == ".json":
        hasher = hashlib.sha256()
        while chunk := await file.read(8192):
            hasher.update(chunk)
        file_id = hasher.hexdigest()
        if await asyncio.to_thread(es_service.exists, "file_id", file_id):
            return {"filename": file.filename, "file_id": file_id, "status": "already_indexed"}
        file.file.seek(0)
        version = await asyncio.to_thread(es_service.get_next_version, file.filename)
        chunks, embeddings = await asyncio.to_thread(json_processor.process, file.file, file.filename)
        if not chunks:
            return {"filename": file.filename, "error": "No indexable text content found in this JSON file"}
        await asyncio.to_thread(es_service.index_chunks, file.filename, file_id, version, chunks, embeddings, "json")
        return {"filename": file.filename, "file_id": file_id, "version": version, "chunks_indexed": len(chunks)}

    else:
        return {"error": f"Unsupported file type: {suffix}. Supported: .pdf, .eml, .mbox, .json"}


@app.post("/query")
async def query(body: QueryRequest, request: Request):
    es_service: ESService = request.app.state.es_service
    embedding_service: EmbeddingService = request.app.state.embedding_service

    query_embedding = await asyncio.to_thread(embedding_service.encode_query, body.query)
    results = await asyncio.to_thread(es_service.search, body.query, query_embedding, body.size, body.doc_type)
    return {"results": results}
