"""Retriever module connecting Embeddings and FAISS Vector Store.

EDUCATIONAL CONCEPT: What is a Retriever?
In a Retrieval-Augmented Generation (RAG) system, the Retriever acts as the bridge
between user questions and indexed documents:
1. It accepts an arbitrary human language query.
2. It uses the exact same embedding model that was used to index the document
   to convert the query into a high-dimensional vector.
3. It performs top-k semantic search against the FAISS index to find the most relevant chunks.
4. It packages the matching text, page numbers, and similarity scores into an ordered list
   ready for the LLM to inspect.
"""

from typing import List, Optional
from core.chunker import TextChunk, chunk_document
from core.embeddings import LocalEmbeddings
from core.pdf_loader import PDFDocument
from core.vector_store import FAISSVectorStore, SearchResult


class Retriever:
    """Coordinates embedding generation, chunk indexing, and semantic retrieval."""

    def __init__(
        self,
        embeddings: LocalEmbeddings,
        vector_store: FAISSVectorStore
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.active_document_name: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """Returns True if there is at least one vector indexed and ready to search."""
        return not self.vector_store.is_empty

    @property
    def total_chunks(self) -> int:
        """Returns total number of chunks currently in the vector store."""
        return self.vector_store.total_vectors

    def index_document(
        self,
        document: PDFDocument,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[TextChunk]:
        """Splits a PDF document, embeds its chunks, and registers them in the vector store.

        Args:
            document: Extracted PDFDocument.
            chunk_size: Character length per chunk.
            chunk_overlap: Character overlap.

        Returns:
            List of generated TextChunk instances.
        """
        # Reset any existing index for single-document mode
        self.clear()

        if not document.has_extractable_text:
            return []

        chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            return []

        # Extract raw texts for batch embedding
        texts = [chunk.text for chunk in chunks]

        # Generate normalized embedding vectors
        embeddings_matrix = self.embeddings.embed_texts(texts)

        # Store in FAISS
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings_matrix)
        self.active_document_name = document.filename

        return chunks

    def retrieve(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Embeds query and retrieves top-k relevant document passages.

        Args:
            query: User's question or search query.
            top_k: Maximum number of chunks to return.

        Returns:
            Ordered list of SearchResult objects with score and page citations.
        """
        if not self.is_ready:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        query_vector = self.embeddings.embed_query(clean_query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    def clear(self) -> None:
        """Clears the active knowledge base."""
        self.vector_store.clear()
        self.active_document_name = None
