# local_embeddings.py

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from sentence_transformers import SentenceTransformer
from typing import List, Any, Optional

# NOTE: Using PrivateAttr for the model instance to avoid Pydantic serialization conflicts.

class LocalEmbeddings(BaseModel, Embeddings):
    """Custom Embeddings class to wrap a local SentenceTransformer model
    and provide the necessary LangChain Embeddings API interface.
    """

    # Use PrivateAttr to store the SentenceTransformer instance (not part of the Pydantic model)
    _model: Any = PrivateAttr(default=None)

    # Configuration fields (Pydantic fields)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: dict[str, Any] = Field(default_factory=lambda: {'normalize_embeddings': False})

    # Pydantic Configuration to allow the SentenceTransformer object
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        try:
            import sentence_transformers  # Check dependency
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        # CRITICAL FIX: Load the model into the PrivateAttr instance
        print(f"Loading model {self.model_name}...")
        self._model = SentenceTransformer(
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

        if not self._model:
            raise RuntimeError("Embedding model client is not loaded.")

        # Use the PrivateAttr instance
        embeddings_local = self._model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=self.encode_kwargs.get('normalize_embeddings', False)
        )

        # NOTE: Ensure the output is a List[List[float]]
        return embeddings_local.tolist()
