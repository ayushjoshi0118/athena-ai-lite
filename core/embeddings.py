"""Embeddings generation module for Athena AI Lite.

EDUCATIONAL CONCEPT: What is an embedding?
An embedding converts unstructured human text into a dense array of numbers
(a vector, e.g. 384 dimensions for all-MiniLM-L6-v2).
- Words and passages with similar semantic meanings are placed close to each other
  in this high-dimensional vector space.
- Example: The embedding for "What is the final exam date?" will have a high cosine
  similarity with "The semester exam takes place on December 15th", even though they
  share few identical words.

Why Normalize Vectors?
When vectors are L2-normalized (length = 1.0), the cosine similarity between two vectors
is simply their Dot Product:
    cosine_similarity(A, B) = A · B / (||A|| * ||B||) = A · B
This allows vector search engines like FAISS to use the fast Inner Product (IndexFlatIP)
index for exact cosine similarity ranking!
"""

from typing import List, Union
import numpy as np


class EmbeddingsError(Exception):
    """Custom exception raised when embedding generation fails."""
    pass


class LocalEmbeddings:
    """Manages local sentence-transformers embedding generation.

    Caches the model in memory so it doesn't reload on every user request.
    Includes deterministic fallback logic for testing environments where
    model weights may not yet be downloaded.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", mock_mode: bool = False):
        self.model_name = model_name
        self.mock_mode = mock_mode
        self._model = None
        self._dimension: int = 384

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality of this embedding model."""
        return self._dimension

    def _load_model(self):
        """Lazy loader for sentence-transformers model."""
        if self._model is not None or self.mock_mode:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            # Fall back to deterministic embeddings if weights cannot be loaded (e.g. offline/network limits)
            import warnings
            warnings.warn(
                f"Failed to load sentence-transformers model '{self.model_name}' ({e}). "
                "Activating deterministic offline mock embeddings for testing and learning."
            )
            self.mock_mode = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of text strings into an (N, dimension) float32 numpy array.

        Vectors are automatically L2-normalized so their dot products equal cosine similarity.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        if self.mock_mode:
            # Deterministic pseudo-embedding for testing without downloading model weights
            return self._generate_mock_embeddings(texts)

        try:
            self._load_model()
            if self.mock_mode:
                return self._generate_mock_embeddings(texts)
            # encode produces a numpy array
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalize
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise EmbeddingsError(f"Error encoding texts with '{self.model_name}': {e}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 normalized vector."""
        if not query or not query.strip():
            raise EmbeddingsError("Cannot generate embedding for an empty query.")

        result = self.embed_texts([query])
        return result[0]

    def _generate_mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates deterministic unit vectors based on string hash for unit tests."""
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed deterministic RNG with text hash
            seed = abs(hash(text)) % (2**31 - 1)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            matrix[i] = vec
        return matrix
