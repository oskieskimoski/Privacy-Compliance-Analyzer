from pathlib import Path

import fitz
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def load_pdf_documents(data_dir: Path) -> List[Document]:

    documents = []
    files_pdf = list(data_dir.glob("*.pdf"))

    if not files_pdf:
        print(f"No documents in {data_dir}")
        return documents

    for pdf_path in files_pdf:
        try:
            with fitz.open(str(pdf_path.resolve())) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text()

                    if not text.strip():
                        continue

                    langchain_doc = Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num}
                    )
                    documents.append(langchain_doc)
        except Exception as e:
            print(f"Error while loading: {pdf_path.name}: {e}")

    print(f"Loaded {len(documents)} pages")
    return documents

def split_documents(docs: List[Document], chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "  ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Splitted {len(docs)} pages to {len(chunks)} chunks")
    return chunks

