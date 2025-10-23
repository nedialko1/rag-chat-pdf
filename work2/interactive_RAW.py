import requests
import json
import os
import sys
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time  # Import time for use in backoff/sleep logic, though only needed for embedding service

# --- Configuration ---
METADATA_FILE = "../chroma_db/last_ingested_db.txt"
DEFAULT_DB_FOLDER = "./chroma_db/default_db_full"
EMBEDDING_SERVICE_URL = "http://127.0.0.1:8000/embed"


# NOTE: All Gemini API configurations have been removed to ensure the application is fully standalone and local.
# The application now only performs retrieval and outputs the raw context.


# ----------------------------------------------
# 1. Custom Class to hit the Local FastAPI Embedding Service
# ----------------------------------------------

class LocalEmbeddingService(Embeddings):
    """Custom Embeddings class to interact with our local FastAPI service."""

    def __init__(self, url: str = EMBEDDING_SERVICE_URL, normalize: bool = False):
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
            print(f"An unexpected error occurred during embedding call: {e}")
            raise


# ----------------------------------------------
# 2. Database Loading Logic
# ----------------------------------------------

def get_persist_directory():
    """Reads the last ingested database folder name from the metadata file."""
    if not os.path.exists(METADATA_FILE):
        print("\n[WARNING] Metadata file 'last_ingested_db.txt' not found.")
        print(f"Defaulting to: {DEFAULT_DB_FOLDER}")
        return DEFAULT_DB_FOLDER

    try:
        with open(METADATA_FILE, 'r') as f:
            db_folder_name = f.read().strip()
            if not db_folder_name:
                raise ValueError("Metadata file is empty.")

            # Construct the full persistence directory path
            persist_dir = os.path.join(os.path.dirname(METADATA_FILE), db_folder_name)
            return persist_dir

    except Exception as e:
        print(f"\n[ERROR] Failed to read metadata file: {e}")
        return DEFAULT_DB_FOLDER


def load_vector_store():
    """Loads the Chroma vector store based on the metadata pointer."""

    # Get the directory name from the metadata file
    PERSIST_DIR = get_persist_directory()

    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] Database folder not found at: {PERSIST_DIR}"
            "\n*** Please run 'python ingest_pdf_x0.py' first to create the vector database. ***"
        )

    print(f"\n[INFO] Loading Vector Store from: {PERSIST_DIR}")
    embeddings = LocalEmbeddingService()

    # Load the Chroma database from the determined directory
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore


# ----------------------------------------------
# 3. Main Loop (Retrieval Only)
# ----------------------------------------------

def main():
    print("=" * 60)
    print("--- Interactive RAG Retrieval (Fully Local & Standalone) ---")
    print("--- Outputs the raw context from your PDF ---")
    print("=" * 60)

    try:
        vectorstore = load_vector_store()
    except (FileNotFoundError, ConnectionError, SystemExit) as e:
        print(f"\n[FATAL STARTUP ERROR] {e}")
        return
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] An unknown error occurred: {e}")
        return

    # Define the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("\n[READY] Application is ready. Ask a question about the PDF.")
    print("Type 'exit' or 'quit' to close the application.")
    print("-" * 60)

    while True:
        question = input("Q: ")

        if question.lower() in ['quit', 'exit']:
            print("Exiting RAG Retrieval application. Goodbye!")
            break

        if not question.strip():
            continue

        print("\nSearching database...")

        # 1. Retrieval
        retrieved_docs: List[Document] = retriever.invoke(question)

        if not retrieved_docs:
            print("A: I couldn't find any relevant context in the document.")
            continue

        # 2. Context Formatting (for direct output)
        # We output the raw text found in the document chunks
        formatted_context = "\n\n"
        for i, doc in enumerate(retrieved_docs):
            source = os.path.basename(doc.metadata.get('source', 'Unknown File'))
            page = doc.metadata.get('page', 'Unknown Page') + 1  # 0-indexed to 1-indexed

            formatted_context += f"--- CONTEXT CHUNK {i + 1} (Source: {source}, Page: {page}) ---\n"
            formatted_context += doc.page_content.strip()
            formatted_context += "\n"

        # 3. Output
        print(f"\nA: I found the following relevant information in your document:")
        print(formatted_context)

        print("=" * 60)


if __name__ == "__main__":
    main()
