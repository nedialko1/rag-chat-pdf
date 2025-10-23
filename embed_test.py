# embed_test.py

# GPU-Accelerated Embedding Model Initialization
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Choose a GPU-friendly embedding model
# all-MiniLM-L6-v2 is small, fast, and a great default for a modest GPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

from langchain_core.embeddings import Embeddings

import requests
import json
from typing import List

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
            response.raise_for_status()

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


#############################################################

# 2. Configure model to use the GPU
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print(f"Embedding model loaded on device: {embeddings.model_kwargs['device']}")

"""
# Data Loading and Chunking: Standard RAG procedure to prepare your documents.

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Documents
loader = PyPDFLoader("NK_CV_ver_7.pdf") # Replace with your file
documents = loader.load()

# 2. Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")

print(type(documents))      # <class 'list'>
print(type(documents[0]))   # <class 'langchain_core.documents.base.Document'>
print("\n" + "=" * 50 + "\n")
print(type(chunks))         # <class 'list'>
print(type(chunks[0]))      # <class 'langchain_core.documents.base.Document'>
"""

document = Document(
                page_content="The quick brown fox jumped over the lazy dog.", metadata={"source": "TEST"}
            )

documents = [document]
print(documents)
texts = [doc.page_content for doc in documents]

embeddings1 = embeddings.embed_documents(texts)
print(type(embeddings1[0]))

# Now, Generate the local embeddings
MODEL_DEVICE = 'cuda'  # Change to 'cpu' if you don't have a GPU

from sentence_transformers import SentenceTransformer

try:
    normalize: bool = False

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)

    embeddings_local = model.encode(
        texts,
        convert_to_tensor=False,  # Return as standard NumPy array/list
        normalize_embeddings=normalize  # Use the requested normalization
    )

    print(type(embeddings_local))
    embeddings_list = embeddings_local.tolist()

    print(type(embeddings_local[0]))

    # embeddings_list -> data ?

    """
    class EmbeddingResponse(BaseModel):
        # Schema for the embedding response body.
        embeddings: List[List[float]]
    """

    # response = EmbeddingResponse(embeddings=embeddings_list)
    # data = response.json()
    # embeddings2 = data["embeddings"]

    print(type(embeddings_list[0]))

except ConnectionError as e:
    print(f"\n[CRITICAL ERROR]: {e}")

print(len(embeddings1[0]))
print("\n" + "-" * 25 + "\n")
print(len(embeddings_list[0]))

import numpy as np
print(np.sum(np.array(embeddings1[0])-np.array(embeddings_list[0])))