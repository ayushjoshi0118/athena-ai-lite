"""FAISS CPU Vector Store for Athena AI Lite.

EDUCATIONAL CONCEPT: How does FAISS retrieve similar vectors?
1. Vector Indexing:
   FAISS (Facebook AI Similarity Search) is an ultra-fast C++ library for vector indexing.
   We use `IndexFlatIP` (Flat Inner Product). Because all our vectors are L2-normalized
   to length 1.0, the Inner Product (dot product) mathematically equals the Cosine Similarity!
2. The ID Mapping Problem:
   FAISS only stores numerical vectors and internal integer IDs (0, 1, 2, ...).
   FAISS does NOT store text strings or document filenames.
   Therefore, the application must maintain a strict 1-to-1 dictionary or list mapping
   each integer vector ID to the corresponding TextChunk and its metadata.
3. Top-k Retrieval:
   When a user query vector is submitted, FAISS compares it against all indexed vectors
   and returns the top-k highest scoring vector IDs along with their similarity scores.
   We then resolve those IDs back to the actual text chunks and page numbers.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class VectorStoreError(Exception):
    """Custom exception raised for vector store operations and validation errors."""
    pass


@dataclass
class SearchResult:
    """Represents a retrieved chunk with its similarity score and ranking."""
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    score: float  # Cosine similarity score (higher is more similar, max 1.0)
    rank: int     # 1-indexed rank
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FAISSVectorStore:
    """Manages an in-memory or persisted FAISS IndexFlatIP vector store."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384
    ):
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self._index = None
        self._chunks: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Returns the number of vectors currently indexed in FAISS."""
        return len(self._chunks) if self._index is not None else 0

    @property
    def is_empty(self) -> bool:
        """Returns True if no vectors are currently indexed."""
        return self.total_vectors == 0

    def add_chunks(
        self,
        chunks: List[Any],  # List[TextChunk] or dicts with text/metadata
        embeddings: np.ndarray
    ) -> int:
        """Adds text chunks and their normalized embedding vectors into the FAISS index.

        Args:
            chunks: List of TextChunk objects or dict representations.
            embeddings: Float32 numpy array of shape (N, dimension).

        Returns:
            Number of vectors added.

        Raises:
            VectorStoreError: On dimensionality mismatch or invalid inputs.
        """
        if not chunks:
            return 0

        if not isinstance(embeddings, np.ndarray):
            raise VectorStoreError(f"Embeddings must be a numpy array, got {type(embeddings)}")

        if embeddings.ndim != 2:
            raise VectorStoreError(f"Embeddings array must be 2D, got shape {embeddings.shape}")

        num_vectors, dim = embeddings.shape

        if num_vectors != len(chunks):
            raise VectorStoreError(
                f"Mismatch: Received {len(chunks)} chunks but {num_vectors} embeddings."
            )

        if dim != self.dimension:
            raise VectorStoreError(
                f"Dimension mismatch: Store configured for {self.dimension} dimensions, "
                f"but received embeddings with {dim} dimensions."
            )

        try:
            import faiss
        except ImportError:
            raise VectorStoreError("faiss-cpu is not installed. Please install faiss-cpu to enable vector search.")

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Initialize IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to FAISS
        self._index.add(vectors)

        # Store metadata mapping (index ID -> chunk dict)
        for chunk in chunks:
            if hasattr(chunk, "to_dict"):
                self._chunks.append(chunk.to_dict())
            elif isinstance(chunk, dict):
                self._chunks.append(chunk)
            else:
                self._chunks.append({
                    "chunk_id": getattr(chunk, "chunk_id", f"chunk_{len(self._chunks)}"),
                    "doc_name": getattr(chunk, "doc_name", "unknown"),
                    "page_number": getattr(chunk, "page_number", 1),
                    "text": getattr(chunk, "text", str(chunk)),
                    "metadata": getattr(chunk, "metadata", {})
                })

        return num_vectors

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4
    ) -> List[SearchResult]:
        """Searches the index for the top-k most similar chunks.

        Args:
            query_vector: 1D or 2D float32 numpy array.
            top_k: Number of most relevant results to retrieve.

        Returns:
            List of SearchResult objects ordered from highest to lowest similarity.
        """
        if self.is_empty or self._index is None:
            return []

        if top_k <= 0:
            return []

        # Format query vector to (1, dimension)
        if query_vector.ndim == 1:
            q_vec = np.expand_dims(query_vector, axis=0)
        else:
            q_vec = query_vector

        if q_vec.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension ({q_vec.shape[1]}) does not match store dimension ({self.dimension})"
            )

        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        # Limit top_k to total vectors available
        k = min(top_k, self.total_vectors)

        # FAISS search returns (distances/scores, indices)
        scores, indices = self._index.search(q_vec, k)

        results: List[SearchResult] = []
        for rank_idx, (score, vec_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if vec_id == -1 or vec_id >= len(self._chunks):
                continue

            chunk_data = self._chunks[vec_id]
            results.append(
                SearchResult(
                    chunk_id=chunk_data.get("chunk_id", f"chunk_{vec_id}"),
                    doc_name=chunk_data.get("doc_name", "document"),
                    page_number=chunk_data.get("page_number", 1),
                    text=chunk_data.get("text", ""),
                    score=float(score),
                    rank=rank_idx,
                    metadata=chunk_data.get("metadata", {})
                )
            )

        return results

    def save(self, target_dir: Union[str, Path]) -> None:
        """Persists the FAISS index and chunk metadata to disk safely.

        Args:
            target_dir: Directory where index files will be stored.
        """
        if self.is_empty or self._index is None:
            raise VectorStoreError("Cannot save an empty vector store.")

        try:
            import faiss
        except ImportError:
            raise VectorStoreError("faiss-cpu is required to save index files.")

        save_path = Path(target_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        index_file = save_path / "index.faiss"
        meta_file = save_path / "metadata.json"
        config_file = save_path / "config.json"

        # 1. Save FAISS binary index
        faiss.write_index(self._index, str(index_file))

        # 2. Save metadata mapping
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, indent=2)

        # 3. Save configuration for validation during loading
        config = {
            "embedding_model_name": self.embedding_model_name,
            "dimension": self.dimension,
            "total_vectors": self.total_vectors,
            "similarity_metric": "cosine"
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, target_dir: Union[str, Path]) -> None:
        """Loads and validates a persisted FAISS index and metadata.

        Validates embedding model compatibility before loading to prevent
        corrupted or nonsensical distance calculations.
        """
        load_path = Path(target_dir)
        index_file = load_path / "index.faiss"
        meta_file = load_path / "metadata.json"
        config_file = load_path / "config.json"

        if not index_file.exists() or not meta_file.exists() or not config_file.exists():
            raise VectorStoreError(f"Missing required index files in '{load_path}'")

        # 1. Validate configuration
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        saved_model = config.get("embedding_model_name")
        saved_dim = config.get("dimension")

        if saved_model != self.embedding_model_name:
            raise VectorStoreError(
                f"Embedding model mismatch! The index was built with '{saved_model}', "
                f"but the current store is configured for '{self.embedding_model_name}'. "
                "Vectors from different models cannot be compared."
            )

        if saved_dim != self.dimension:
            raise VectorStoreError(
                f"Vector dimension mismatch! Index dimension is {saved_dim}, "
                f"but configured dimension is {self.dimension}."
            )

        try:
            import faiss
        except ImportError:
            raise VectorStoreError("faiss-cpu is required to load index files.")

        # 2. Load FAISS index
        self._index = faiss.read_index(str(index_file))

        # 3. Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

    def clear(self) -> None:
        """Resets the vector index and metadata mapping to empty state."""
        self._index = None
        self._chunks = []
