# PDF Ingestion Service: Handles document loading, chunking, embedding, and Chroma persistence.
# Run this service independently: uvicorn ingestion_service:app --reload

import requests
import json
import os
import shutil  # For removing directories
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --- 1. FastAPI Setup and Utility Classes ---

app = FastAPI(title="Local RAG Ingestion Service")


# ----------------------------------------------
# 2. Custom Class to hit the Local FastAPI Embedding Service
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

        try:
            # We increase the timeout here since embedding a large document can take time
            response = requests.post(self.url, headers=headers, data=payload, timeout=300)
            response.raise_for_status()

            data = response.json()
            return data["embeddings"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to the embedding service at http://127.0.0.1:8000. "
                "Ensure 'python embedding_service.py' is running in another terminal."
            )
        except Exception as e:
            raise Exception(f"Error calling embedding service: {e}")


# Initialize the embedding service client
try:
    embeddings = LocalEmbeddingService()
except ConnectionError as e:
    print(f"WARNING: Embedding service not running. Ingestion endpoint will not work until it is started. Error: {e}")


# --- 3. Ingestion Endpoint ---

@app.post("/ingest")
async def ingest_document(
        file: UploadFile = File(...),
        mode: str = Form("overwrite")  # 'overwrite' (default) or 'append'
):
    """
    Ingests a PDF file, chunks it, embeds it, and saves it to ChromaDB.
    Mode: 'overwrite' (clears existing store for this file) or 'append' (adds to existing store).
    """
    if not embeddings:
        raise HTTPException(status_code=503,
                            detail="Embedding service is not available. Please start 'embedding_service.py'.")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Save the uploaded file temporarily
    file_path = f"uploaded_data/{file.filename}"
    os.makedirs("../uploaded_data", exist_ok=True)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # 2. Define the persistence directory based on file name (to ensure caching)
    file_base_name = os.path.basename(file.filename).replace('.', '_')
    persist_dir = f"./chroma_db/{file_base_name}"

    # 3. Handle Overwrite/Skip Logic
    if os.path.exists(persist_dir):
        if mode == "overwrite":
            print(f"Mode is 'overwrite'. Removing existing directory: {persist_dir}")
            shutil.rmtree(persist_dir)
            os.makedirs(persist_dir)  # Recreate empty directory
        elif mode == "append":
            # In a true 'append' mode, we could just load the existing store and add to it.
            # However, for simplicity here, we'll only allow skipping or overwriting
            # as LangChain's from_documents usually recreates the store.
            # We'll treat append/skip as one behavior for the first version.
            print(f"Vector store already exists for '{file.filename}' at {persist_dir}. Skipping ingestion.")
            os.remove(file_path)  # Clean up the uploaded file
            return {"status": "skipped",
                    "message": f"Document already ingested and mode is not 'overwrite'. Q&A client can use '{file.filename}'."}
        else:
            raise HTTPException(status_code=400,
                                detail="Invalid mode. Use 'overwrite' or 'append' (which acts as skip if exists).")

    # 4. Load, Chunk, and Embed
    print(f"Starting ingestion for: {file.filename}")
    try:
        # Load Documents
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split Documents into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} page(s) into {len(chunks)} chunks.")

        # Create/Persist Vector Store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print(f"Successfully created vector store at {persist_dir}.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        os.remove(file_path)  # Clean up the uploaded file

    return {
        "status": "success",
        "message": f"Document '{file.filename}' successfully ingested and stored.",
        "chunks_processed": len(chunks)
    }

# --- 4. Running the Service ---
# To run this service, you must use uvicorn:
# uvicorn ingestion_service:app --reload
