import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR     = Path("data")            # your folder of site pages
INDEX_DIR    = "faiss_index"           # where the FAISS files live
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 500
CHUNK_OVERLAP= 100

def load_all_docs(data_dir: Path) -> List:
    """Load every .txt (or other) file as a Document, tagging its source."""
    docs = []
    for file in data_dir.glob("*.*"):
        # only load .txt here, but you could add PDFLoader, etc.
        if file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            for doc in loader.load():
                # attach metadata so we know which file it came from
                doc.metadata["source"] = file.name
                docs.append(doc)
    return docs

def build_index():
    # 1️⃣ Load & annotate docs
    print(f"Loading docs from {DATA_DIR}…")
    docs = load_all_docs(DATA_DIR)

    # 2️⃣ Split into semantic chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Splitting {len(docs)} documents into chunks…")
    chunks = splitter.split_documents(docs)

    # 3️⃣ Embed & build FAISS (overwrites existing index)
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}…")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    index = FAISS.from_documents(chunks, embedder)

    # 4️⃣ Persist to disk
    index.save_local(INDEX_DIR)
    print(f"[✓] FAISS index built & saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_index()
