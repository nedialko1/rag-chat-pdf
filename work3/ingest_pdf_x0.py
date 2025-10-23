import requests
import json
import os
import shutil
import time
import argparse
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tkinter as tk
from tkinter import filedialog

# ----------------------------------------------
# --- Configuration and Arguments ---
# ----------------------------------------------

# --- ARGPARSE SETUP (for command-line arguments) ---
parser = argparse.ArgumentParser(
    description="RAG Ingestion Service: Selects PDF and loads it into ChromaDB.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '--mode',
    type=str,
    default='chunked',
    choices=['full', 'chunked'],
    help=(
        "Ingestion mode (default: chunked):\n"
        "'full': Load the entire document as a single chunk.\n"
        "'chunked': Split the document into [page] chunks."
    )
)
# Argument for Chunk Size
parser.add_argument(
    '--chunk-size',
    type=int,
    default=1000,
    help='The maximum size of each text chunk (used only in --mode chunked). Default is 1000.'
)
# Argument for Chunk Overlap
parser.add_argument(
    '--chunk-overlap',
    type=int,
    default=200,
    help='The overlap between adjacent chunks (used only in --mode chunked). Default is 200.'
)
args = parser.parse_args()
INGESTION_MODE = args.mode
# Use the parsed values for chunking parameters
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap

# ----------------------------------------------
# 1. Custom Class to get the Embeddings Locally
# ----------------------------------------------

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DEVICE = 'cuda'  # Change to 'cpu' if you don't have a GPU

model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

"""
??? local_embeddings.py
"""

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

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
FILE_BASE_NAME = os.path.basename(PDF_FILE_PATH)
# The database folder name now includes the selected mode (full/chunked)
PERSIST_DIR = f"./chroma_db/{FILE_BASE_NAME.replace('.', '_')}_{INGESTION_MODE}"

print("=" * 60)
print(f"--- RAG Ingestion Service ({INGESTION_MODE.upper()} Document Load) ---")
print(f"Target PDF: {PDF_FILE_PATH}")
print(f"Target DB Directory: {PERSIST_DIR}")
print("=" * 60)

try:
    # 2. Load the Document
    print(f"\n[STEP 1/4] Loading PDF document: {FILE_BASE_NAME}")

    # FIX: Corrected the unclosed parenthesis from the SyntaxError
    loader = PyPDFLoader(PDF_FILE_PATH)
    all_pages = loader.load()

    if not all_pages:
        print("\n[ERROR] Document loading failed or PDF is empty.")
        exit()

    full_document_text_size = sum(len(p.page_content) for p in all_pages)

    # --- MODE SELECTION LOGIC: Full vs. Chunked ---
    if INGESTION_MODE == 'full':
        print(f"Mode: FULL LOAD (Document treated as one large chunk).")
        # Join the pages' content into a single string
        full_document_text = "\n\n".join([page.page_content for page in all_pages])
        # Use a dummy text splitter to wrap the full text as a single document object
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000000,  # Massive size to ensure no split
            chunk_overlap=0
        )
        docs = text_splitter.create_documents([full_document_text])

    else:  # INGESTION_MODE == 'chunked'
        print(f"Mode: CHUNKED LOAD (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")
        # Split the list of Documents (pages) directly
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(all_pages)

    num_chunks = len(docs)
    print(
        f"Document successfully loaded and split into {num_chunks} chunk(s). (Total Size: {full_document_text_size} characters).")

    # 3. Clean and Prepare Database
    if os.path.exists(PERSIST_DIR):
        print(f"\n[STEP 2/4] Found existing database. Attemping to remove data to overwrite...")

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
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    # Removed the redundant and problematic vectorstore.persist() call

    # 5. Write metadata file for the Q&A app
    print("\n[STEP 4/4] Writing metadata for Q&A application...")
    METADATA_FILE = "../chroma_db/last_ingested_db.txt"
    # Ensure the chroma_db folder exists before trying to write the metadata file
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

    # Extract only the folder name relative to chroma_db/
    db_folder_name = os.path.basename(PERSIST_DIR)

    with open(METADATA_FILE, 'w') as f:
        f.write(db_folder_name)

    print(f"Successfully recorded persistence directory in {METADATA_FILE}")

    print("\n[SUCCESS] Document ingestion complete!")
    print(f"The CV has been saved using the {INGESTION_MODE.upper()} strategy in ChromaDB.")
    print("You can now run 'python interactive_rag_qa.py'.")

except FileNotFoundError:
    print(f"\n[CRITICAL ERROR] The file '{PDF_FILE_PATH}' was not found.")
    print("Please ensure the PDF exists at the selected location.")
except Exception as e:
    print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
