# local_embeddings.py

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional


# NOTE: Removed all global model variables. The class is now self-contained.
class LocalEmbeddings(BaseModel, Embeddings):
    """Custom Embeddings class to wrap a local SentenceTransformer model
    and provide the necessary LangChain Embeddings API interface.
    """

    # Store the SentenceTransformer instance here (set in __init__)
    client: Any = Field(default=None)

    # Configuration fields
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: dict[str, Any] = Field(default_factory=lambda: {'normalize_embeddings': False})

    # Pydantic Configuration to allow the SentenceTransformer object
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any):
        """Initialize the SentenceTransformer model and store it in self.client."""
        super().__init__(**kwargs)

        try:
            import sentence_transformers  # Check dependency
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        # CRITICAL FIX: Load the model and store it as a class attribute
        print(f"Loading model {self.model_name}...")
        self.client = SentenceTransformer(
            self.model_name,
            **self.model_kwargs
        )
        print("Model client loaded successfully.")

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings."""
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings."""
        return self._embed(texts)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Helper function to perform the actual embedding via the client."""

        if not self.client:
            raise RuntimeError("Embedding model client is not loaded.")

        # CRITICAL FIX: Use self.client (the stored model instance)
        embeddings_local = self.client.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=self.encode_kwargs.get('normalize_embeddings', False)
        )

        return embeddings_local.tolist()