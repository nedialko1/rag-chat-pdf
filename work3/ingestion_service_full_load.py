import requests
import json
import os
import shutil
import time  # Added for sleep functionality
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tkinter as tk
from tkinter import filedialog


# ----------------------------------------------
# 1. Custom Class to hit the Local FastAPI Service
# ----------------------------------------------

class LocalEmbeddingService(Embeddings):
    """Custom Embeddings class to interact with our local FastAPI service."""

    def __init__(self, url: str = "http://127.0.0.1:8000/embed", normalize: bool = False):
        self.url = url
        self.normalize = normalize
        print(f"Embedding client configured to use local service at: {self.url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"texts": texts, "normalize": self.normalize})

        # Retry logic is often needed for external services, but we'll stick to simple request for brevity
        try:
            response = requests.post(self.url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            data = response.json()
            return data["embeddings"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to the embedding service at {self.url}. "
                "Ensure 'python embedding_service.py' is running in another terminal."
            )
        except Exception as e:
            print(f"An unexpected error occurred while calling the embedding service: {e}")
            raise


# ----------------------------------------------
# 2. Main Ingestion Logic
# ----------------------------------------------

# --- GUI FILE SELECTION ---
root = tk.Tk()
root.withdraw()  # Hide the main tk window so only the dialog appears

PDF_FILE_PATH = filedialog.askopenfilename(
    title="Select PDF Document for Ingestion",
    filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
)

if not PDF_FILE_PATH:
    print("\n[INFO] File selection canceled. Exiting ingestion service.")
    exit()

# Define the PDF file path and the corresponding persistence directory
# PDF_FILE_PATH now holds the full path selected by the user
# FILE_BASE_NAME extracts just the filename for use in the DB directory name
FILE_BASE_NAME = os.path.basename(PDF_FILE_PATH)
PERSIST_DIR = f"./chroma_db/{FILE_BASE_NAME.replace('.', '_')}"

print("=" * 60)
print(f"--- RAG Ingestion Service (Full Document Load) ---")
print(f"Target PDF: {PDF_FILE_PATH}")
print(f"Target DB Directory: {PERSIST_DIR}")
print("=" * 60)

try:
    # 1. Initialize the embedding client
    embeddings = LocalEmbeddingService(normalize=False)
    print("Embedding client initialized successfully.")

    # 2. Load the Document
    print(f"\n[STEP 1/3] Loading PDF document: {FILE_BASE_NAME}")
    # Use the full path for the loader
    loader = PyPDFLoader(PDF_FILE_PATH)

    # Load all pages and join them into a single string to represent the full document
    full_document_text = "\n\n".join([page.page_content for page in loader.load()])

    # We must still treat the text as a list of documents for Chroma, but we only have one element
    # Use a dummy text splitter to wrap the full text as a single document object
    # Setting chunk_size to a very large number ensures the text is not split.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000000,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents([full_document_text])

    if not docs:
        print("\n[ERROR] Document loading failed or PDF is empty.")
        exit()

    print(f"Document successfully loaded as a single chunk (Size: {len(full_document_text)} characters).")

    # 3. Clean and Prepare Database
    if os.path.exists(PERSIST_DIR):
        print(f"\n[STEP 2/3] Found existing database. Attemping to remove data to overwrite...")

        # Give the OS a moment to release any transient locks
        time.sleep(1)

        try:
            shutil.rmtree(PERSIST_DIR)
            print("Successfully removed old database files.")
        except OSError as e:
            print("\n" + "=" * 60)
            print("[CRITICAL FILE LOCK ERROR (WinError 32)]")
            print("The existing ChromaDB files are locked.")
            print("Please ensure you have **closed the Q&A application** (Terminal 3) before continuing.")
            print(f"Original OS Error: {e}")
            print("=" * 60 + "\n")
            # Re-raise the error as the process cannot continue without deleting the old files
            raise e

    print(f"Creating new vector store at {PERSIST_DIR}...")

    # 4. Create and Persist the Vector Store
    # This step implicitly uses the LocalEmbeddingService to convert the single large document into a vector
    # and then saves the result to the disk.
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print("\n[SUCCESS] Document ingestion complete!")
    print(f"The entire CV has been saved as a single vector in ChromaDB under the name '{FILE_BASE_NAME}'.")
    print("You can now run 'python interactive_rag_qa.py'.")

except FileNotFoundError:
    print(f"\n[CRITICAL ERROR] The file '{PDF_FILE_PATH}' was not found.")
    print("Please ensure the PDF exists at the selected location.")
except ConnectionError as e:
    print(f"\n[CRITICAL ERROR]: {e}")
    print("Please ensure the embedding service is running in Terminal 1 (python embedding_service.py).")
except Exception as e:
    print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
