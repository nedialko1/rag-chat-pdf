# Standalone script to ingest the entire PDF as a single document chunk
# This overwrites the existing vector store with the new content.

import requests
import json
import os
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ----------------------------------------------
# Configuration
# ----------------------------------------------
PDF_FILE_NAME = "NK_CV_ver_7.pdf"
PERSIST_DIR = f"./chroma_db/{PDF_FILE_NAME.replace('.', '_')}"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------------------------------------------
# 1. Custom Class to hit the Local FastAPI Service
# ----------------------------------------------

class LocalEmbeddingService(Embeddings):
    """Custom Embeddings class to interact with our local FastAPI service."""

    def __init__(self, url: str = "http://127.0.0.1:8000/embed", normalize: bool = False):
        self.url = url
        self.normalize = normalize

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"texts": texts, "normalize": self.normalize})

        try:
            response = requests.post(self.url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to the embedding service at {self.url}. "
                "Ensure 'python embedding_service.py' is running in another terminal."
            )
        except Exception as e:
            print(f"An error occurred while calling the embedding service: {e}")
            raise


# ----------------------------------------------
# 2. Main Ingestion Logic (Full Document Load)
# ----------------------------------------------

def ingest_full_document():
    print(f"--- Starting Full Document Ingestion for '{PDF_FILE_NAME}' ---")

    # Check for PDF file existence
    if not os.path.exists(PDF_FILE_NAME):
        print(f"[ERROR] PDF file not found: {PDF_FILE_NAME}")
        print("Please ensure the CV file is in the same directory.")
        return

    # 1. Load the PDF pages
    print("1. Loading and reading PDF content...")
    loader = PyPDFLoader(PDF_FILE_NAME)
    pages = loader.load()

    # 2. Combine all pages into a single document object
    full_content = "\n\n".join(page.page_content for page in pages)

    # Create a single document object to represent the entire CV
    full_document_chunk = Document(
        page_content=full_content,
        metadata={"source": PDF_FILE_NAME, "chunk_type": "full_document"}
    )

    # 3. Initialize Embedding Client
    try:
        embeddings = LocalEmbeddingService()
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding service: {e}")
        return

    # 4. Initialize ChromaDB (This ensures the directory is ready)
    if os.path.exists(PERSIST_DIR):
        print(f"2. WARNING: Existing data found at {PERSIST_DIR}. This data will be overwritten.")
        # To ensure clean overwrite, we explicitly use the client/collection interface

    # Initialize a temporary client to manage the collection
    from chromadb import PersistentClient
    try:
        client = PersistentClient(path=PERSIST_DIR)
        # Using a fixed collection name
        collection_name = "full_load_rag_collection"

        # Delete existing collection if it exists to ensure a clean overwrite
        if collection_name in [c.name for c in client.list_collections()]:
            print(f"   - Deleting existing collection '{collection_name}' for a clean start...")
            client.delete_collection(name=collection_name)

        # Create a new Chroma collection (LangChain's Chroma wrapper will handle creation)
        vectorstore = Chroma.from_documents(
            documents=[full_document_chunk],
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=collection_name
        )
        vectorstore.persist()  # Ensures data is written to disk

        print(f"3. Successfully loaded 1 massive document chunk into the vector store.")
        print(f"4. Vector store ready at: {PERSIST_DIR}")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to save data to ChromaDB: {e}")
        return

    print("--- Ingestion Complete. You can now run 'interactive_rag_qa.py' ---")


if __name__ == "__main__":
    ingest_full_document()
