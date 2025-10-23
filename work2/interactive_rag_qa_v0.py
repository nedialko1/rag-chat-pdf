# GPU-Accelerated Embedding Model Initialization (Now using a local service)

import requests
import json
from typing import List
from langchain_core.embeddings import Embeddings  # Import the base class
from langchain_huggingface import HuggingFaceEmbeddings  # Still needed for model name consistency


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
# The model configuration is now managed by embedding_service.py
embeddings = LocalEmbeddingService(
    normalize=False  # Matches your original encode_kwargs
)

print(f"Embedding model initialized using HTTP client.")

# Data Loading and Chunking: Standard RAG procedure to prepare your documents.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Documents
# IMPORTANT: Ensure the file "NK_CV_ver_7.pdf" exists in your execution directory.
try:
    loader = PyPDFLoader("../NK_CV_ver_7.pdf")  # Replace with your file
    documents = loader.load()
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Document 'NK_CV_ver_7.pdf' not found. Please ensure the file is in the same directory.")
    exit()

# 2. Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

# Create/Load Vector Store (ChromaDB)
# Use the HTTP-client-accelerated embeddings object to convert your chunks into vectors and store them in ChromaDB.

from langchain_chroma import Chroma

# 1. Create a Vector Store from chunks using the service-based embedding model
# NOTE: If you run this script without running the embedding_service.py first, it will throw a ConnectionError.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,  # Uses the HTTP client instance created above
    persist_directory="./chroma_db"
)

# 2. Define the Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector store created and retriever initialized.")

# LangChain Integration:

from langchain_ollama import OllamaLLM  # This is the new, recommended class

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 1. Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# 2. Define Prompt Template
prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context. 
    If the context does not contain the answer, state that you cannot find the information in the document.
    Context: {context}
    Question: {input}
""")

# 3. Build the RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("LLM and RAG chain successfully configured.")
print("\n" + "=" * 50)
print("RAG System is Ready. Start asking questions about the document.")
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
        print("Please start the embedding service in a separate terminal and try again.")
        break
    except Exception as e:
        print(f"\nAn error occurred during invocation (Is Ollama running?): {e}")
        break  # Exit loop on critical error

# Clean up (optional)
# You might want to include cleanup code here if necessary, but for this script, exiting is sufficient.
