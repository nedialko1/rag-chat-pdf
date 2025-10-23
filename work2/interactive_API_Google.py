import requests
import json
import os
import argparse
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

# The following imports are not used in this specific file's logic
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
METADATA_FILE = "../chroma_db/last_ingested_db.txt"
DEFAULT_DB_FOLDER = "./chroma_db/default_db_full"
EMBEDDING_SERVICE_URL = "http://127.0.0.1:8000/embed"
# LLM Configuration (for the Gemini API)
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_KEY = ""  # Will be provided by the environment
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
SYSTEM_PROMPT = "You are an expert assistant designed to answer questions based ONLY on the provided context. If the answer is not found in the context, clearly state that the information is not available in the document. Do not use outside knowledge."


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
# 3. LLM Interaction (Gemini API)
# ----------------------------------------------

def gemini_generate_content(context: str, question: str) -> str:
    """Makes a request to the Gemini API with RAG context."""

    # Template the user query to include the retrieved context
    user_query = f"""
    CONTEXT:
    ---
    {context}
    ---
    QUESTION: {question}
    """

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Extract the generated text
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text',
                                                                                              "I received no response from the LLM service."
                                                                                              )
        return text

    except requests.exceptions.HTTPError as e:
        return f"[API Error] HTTP status {e.response.status_code}. Check your API key or model name."
    except requests.exceptions.ConnectionError:
        return "[Connection Error] Could not reach the Gemini API. Check your internet connection."
    except Exception as e:
        return f"[LLM Error] An unexpected error occurred: {e}"


# ----------------------------------------------
# 4. RAG Chain and Main Loop
# ----------------------------------------------

def main():
    print("=" * 60)
    print("--- Interactive RAG Q&A (Powered by Gemini & ChromaDB) ---")
    print("--- Worker 3 (The Analyst) ---")
    print("=" * 60)

    try:
        vectorstore = load_vector_store()
    except (FileNotFoundError, ConnectionError) as e:
        print(f"\n[FATAL STARTUP ERROR] {e}")
        return
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] An unknown error occurred: {e}")
        return

    # Define the retriever
    # We use a simple similarity search retriever, retrieving 4 documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("\n[READY] Application is ready. Ask a question about the PDF.")
    print("Type 'exit' or 'quit' to close the application.")
    print("-" * 60)

    while True:
        question = input("Q: ")

        if question.lower() in ['quit', 'exit']:
            print("Exiting RAG Q&A application. Goodbye!")
            break

        if not question.strip():
            continue

        print("\nThinking...")

        # 1. Retrieval
        retrieved_docs = retriever.get_relevant_documents(question)

        if not retrieved_docs:
            print("A: I couldn't find any relevant context in the document.")
            continue

        # 2. Context Formatting
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

        # 3. Generation
        answer = gemini_generate_content(context, question)

        # 4. Output
        print(f"\nA: {answer}\n")
        print("-" * 60)

        # Show Sources (metadata provides the original file path and page number)
        source_info = set()
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown File')
            page = doc.metadata.get('page', 'Unknown Page')
            # Langchain page numbers are 0-indexed, so we add 1 for user readability
            source_info.add(f"Source: {os.path.basename(source)} (Page {page + 1})")

        print("Sources Used:")
        for info in source_info:
            print(f"  - {info}")
        print("=" * 60)


if __name__ == "__main__":
    main()
