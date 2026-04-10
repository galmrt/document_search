import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    def __init__(self):
        self.client = pymupdf4llm
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def extract_markdown(self, file_path: str) -> list[dict]:
        try:
            return self.client.to_markdown(file_path, page_chunks=True)
        except Exception as e:
            print(f"Error extracting markdown from {file_path}: {e}")
            raise e

    def get_chunks(self, file_path: str) -> list[Document]:
        page_chunks = self.extract_markdown(file_path)

        pages = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "source_file": file_path,
                    "page_number": chunk["metadata"]["page"] + 1,
                }
            )
            for chunk in page_chunks
            if chunk["text"].strip()
        ]

        return self.splitter.split_documents(pages)