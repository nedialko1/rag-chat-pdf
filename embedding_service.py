import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

# --- Configuration ---
# Must match the model used in interactive_rag_qa.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DEVICE = 'cuda'  # Change to 'cpu' if you don't have a GPU

# --- FastAPI Setup ---
app = FastAPI(title="Local Embedding Service", description="Serves Sentence-Transformer Embeddings via HTTP")

# Load the model globally on startup for performance
print(f"Loading model {EMBEDDING_MODEL_NAME} onto device: {MODEL_DEVICE}...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to CPU if CUDA fails
    MODEL_DEVICE = 'cpu'
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)
    print(f"Failed to load on CUDA, falling back to CPU.")


# --- Request/Response Schemas ---

class EmbeddingRequest(BaseModel):
    """Schema for the embedding request body."""
    texts: List[str]
    normalize: bool = False  # Matches your original encode_kwargs setting


class EmbeddingResponse(BaseModel):
    """Schema for the embedding response body."""
    embeddings: List[List[float]]


# --- Endpoint ---

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generates embeddings for a list of input texts."""

    # Generate embeddings
    embeddings = model.encode(
        request.texts,
        convert_to_tensor=False,  # Return as standard NumPy array/list
        normalize_embeddings=request.normalize  # Use the requested normalization
    )

    # Convert numpy arrays to list of lists for JSON serialization
    embeddings_list = embeddings.tolist()

    return EmbeddingResponse(embeddings=embeddings_list)


# --- Main Entry Point (for running the service) ---

if __name__ == "__main__":
    # To run this service, execute: python embedding_service.py
    # Access it at: http://127.0.0.1:8000/embed
    print("\n--- Starting Uvicorn Server ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)
