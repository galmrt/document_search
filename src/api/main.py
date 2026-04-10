import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv

from src.utils.es_service import ESService
from src.utils.embedding_service import EmbeddingService
from src.ingestion.pdf_processor import PDFProcessor

load_dotenv()

es_service: ESService = None
embedding_service: EmbeddingService = None
pdf_processor: PDFProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global es_service, embedding_service, pdf_processor
    es_service = ESService(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    es_service.ensure_index()
    embedding_service = EmbeddingService()
    pdf_processor = PDFProcessor()
    print("Services started")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunks = pdf_processor.get_chunks(tmp_path)
        embeddings = embedding_service.encode([c.page_content for c in chunks])
        es_service.index_chunks(file.filename, chunks, embeddings)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_indexed": len(chunks)}


@app.post("/query")
async def query(query: str):
    pass
