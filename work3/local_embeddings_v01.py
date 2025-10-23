# local_embeddings.py

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DEVICE = 'cuda'  # Change to 'cpu' if you don't have a GPU

model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=MODEL_DEVICE)

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class LocalEmbeddings(BaseModel, Embeddings):
    """Custom Embeddings class to provide the expected Embeddings API.
    To use, you should have the ``sentence_transformers`` python package installed.
    """

    model_name: str = Field(default=DEFAULT_MODEL_NAME, alias="model")
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    """
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    """
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            msg = (
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg) from exc

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        return self._embed([text], self.encode_kwargs)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the
                `encode` method for the documents of the SentenceTransformer
                encode method.

        Returns:
            List of embeddings, one for each text.

        """
        return self._get_embeddings(texts)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings_local = model.encode(
            texts,
            convert_to_tensor=False,  # Return as standard NumPy array/list
            normalize_embeddings=self.encode_kwargs['normalize_embeddings']  # Use the requested normalization
        )


