from config import CHROMA_DIR, EMBED_MODEL, DATA_DIR
"""
Ingests data/telecom_guide.pdf into the 'guides' Chroma collection.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION = "guides"
PDF_PATH   = str(DATA_DIR / "telecom_guide.pdf")

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    # Tag each chunk so we know it came from the guide
    for i, chunk in enumerate(chunks):
        # The enumerate() function adds a counter to the chunks, 
        # so i is the index and chunk is the current chunk.
        # enumerate() advantage is that we can keep track of the order of chunks, 
        # which can be useful for reconstructing the original document or 
        # for reference purposes.
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
