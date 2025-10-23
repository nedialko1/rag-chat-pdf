# GPU-Accelerated Embedding Model Initialization (Now using a local service)

import requests
import json
import os
from typing import List
from langchain_core.embeddings import Embeddings  # Import the base class
from langchain_huggingface import HuggingFaceEmbeddings  # Still needed for model name consistency
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


# ----------------------------------------------
# 1. Custom Class to hit the Local FastAPI Service
# ----------------------------------------------

class LocalEmbeddingService(Embeddings):
    """Custom Embeddings class to interact with our local FastAPI service."""

    # FIX APPLIED HERE: Corrected the loopback IP from 127.00.1 to 127.0.0.1
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
            response = requests.post(self.url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            # The service returns {"embeddings": [[...], [...], ...]}
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


# 1. Choose a GPU-friendly embedding model (for logging/metadata only)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Initialize the service client
embeddings = LocalEmbeddingService(
    normalize=False  # Matches your original encode_kwargs
)
print(f"Embedding model initialized using HTTP client.")

# ----------------------------------------------
# 2. Load Vector Store and Define Retriever
# ----------------------------------------------

# Define the PDF and the corresponding persistence directory
PDF_FILE_NAME = "../NK_CV_ver_7.pdf"
# Use a directory specific to the file name to enable unique storage/caching
PERSIST_DIR = f"./chroma_db/{PDF_FILE_NAME.replace('.', '_')}"

# Check if the required vector store directory exists
if not os.path.exists(PERSIST_DIR):
    print("\n--- CRITICAL STARTUP ERROR ---")
    print(f"Vector store for '{PDF_FILE_NAME}' not found at: {PERSIST_DIR}")
    print("Please run the ingestion service first to process the document:")
    print(f"1. Ensure 'embedding_service.py' is running.")
    print("2. Run 'python ingestion_service.py' to process the PDF.")
    exit()

# Load Vector Store
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
print(f"Vector store successfully loaded from {PERSIST_DIR}.")

# Define the Retriever
# Keeping k=5 for better context coverage
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LangChain Integration:

from langchain_ollama import OllamaLLM  # This is the new, recommended class
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# 2. Define Prompt Template
# UPDATED: Softened the prompt to allow for synthesis while remaining grounded.
prompt = ChatPromptTemplate.from_template("""
    You are an accurate data extraction and summarization assistant. 
    Analyze the provided context and directly answer the user's question by extracting or synthesizing factual information from the context.

    If the context does not contain the answer, politely state that the information is unavailable in the document.

    Context: {context}

    Question: {input}
""")

# 3. Build the RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("LLM and RAG chain successfully configured.")
print("\n" + "=" * 50)
print(f"RAG System is Ready. Querying data from: {PDF_FILE_NAME}")
print("=" * 50 + "\n")

# Run the Query[ies] - Execute the final RAG chain in a loop
while True:
    question = input("Q > ")

    if question.lower() in ["/bye!", "exit", "quit"]:
        print("\n--- Goodbye! ---")
        break

    if not question.strip():
        continue

    print(f"\n--- Processing Query ---")
    try:
        response = retrieval_chain.invoke({"input": question})

        print("\n--- RAG Answer ---")
        print(response["answer"].strip())
        print("\n" + "=" * 50 + "\n")
    except ConnectionError as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        print("Please ensure the embedding service is running and try again.")
        break
    except Exception as e:
        print(f"\nAn error occurred during invocation (Is Ollama running?): {e}")
        break  # Exit loop on critical error