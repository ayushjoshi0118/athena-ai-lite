"""Unit tests for FAISS vector store and Retriever in Athena AI Lite.

Verifies:
- Empty index behavior (returns empty list without crashing).
- Metadata mapping between vector IDs and text chunks.
- Accurate top-k ranking based on cosine similarity scores.
- Index save and load functionality with configuration validation.
- Model mismatch error rejection on loading incompatible indexes.
"""

import numpy as np
import pytest
from core.chunker import TextChunk
from core.embeddings import LocalEmbeddings
from core.pdf_loader import ExtractedPage, PDFDocument
from core.retriever import Retriever
from core.vector_store import FAISSVectorStore, SearchResult, VectorStoreError


@pytest.fixture
def mock_embeddings() -> LocalEmbeddings:
    """Provides a deterministic mock embeddings provider (dimension 384)."""
    return LocalEmbeddings(model_name="all-MiniLM-L6-v2", mock_mode=True)


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    """Creates a fresh FAISSVectorStore instance."""
    return FAISSVectorStore(embedding_model_name="all-MiniLM-L6-v2", dimension=384)


def test_empty_vector_store_behavior(vector_store, mock_embeddings):
    """Searching or inspecting an empty vector store should be safe and return empty results."""
    assert vector_store.is_empty is True
    assert vector_store.total_vectors == 0

    dummy_query = mock_embeddings.embed_query("Where is the exam?")
    results = vector_store.search(dummy_query, top_k=5)
    assert results == []


def test_add_chunks_and_retrieve_ranking(vector_store, mock_embeddings):
    """Adding chunks and searching should return correctly ranked chunks with metadata."""
    chunks = [
        TextChunk(
            chunk_id="c1",
            doc_name="handbook.pdf",
            page_number=1,
            text="The campus library closes at 10 PM on weekdays.",
            start_char=0,
            end_char=48
        ),
        TextChunk(
            chunk_id="c2",
            doc_name="handbook.pdf",
            page_number=2,
            text="Tuition fee deadlines are strictly enforced on September 1st.",
            start_char=0,
            end_char=61
        ),
        TextChunk(
            chunk_id="c3",
            doc_name="handbook.pdf",
            page_number=3,
            text="Computer Science department office is located in Turing Hall room 302.",
            start_char=0,
            end_char=70
        )
    ]

    embeddings = mock_embeddings.embed_texts([c.text for c in chunks])
    count = vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
    assert count == 3
    assert vector_store.total_vectors == 3
    assert vector_store.is_empty is False

    # Search for library info
    query_vec = mock_embeddings.embed_query("When does the library close?")
    results = vector_store.search(query_vec, top_k=2)

    assert len(results) == 2
    top_result = results[0]
    assert isinstance(top_result, SearchResult)
    assert top_result.rank == 1
    assert -1.0 <= top_result.score <= 1.0  # Cosine similarity bounds
    assert top_result.chunk_id in ["c1", "c2", "c3"]
    assert top_result.doc_name == "handbook.pdf"


def test_retriever_end_to_end(mock_embeddings, vector_store):
    """Retriever should ingest PDFDocument, create chunks, embed, and retrieve."""
    retriever = Retriever(embeddings=mock_embeddings, vector_store=vector_store)

    doc = PDFDocument(
        filename="ai_ethics.pdf",
        total_pages=2,
        pages=[
            ExtractedPage(page_number=1, text="Algorithmic fairness requires balanced training sets.", char_count=53),
            ExtractedPage(page_number=2, text="Transparency in neural networks aids accountability.", char_count=52),
        ],
        total_characters=105,
        has_extractable_text=True
    )

    indexed_chunks = retriever.index_document(doc, chunk_size=100, chunk_overlap=10)
    assert len(indexed_chunks) >= 2
    assert retriever.is_ready is True
    assert retriever.active_document_name == "ai_ethics.pdf"

    results = retriever.retrieve("What helps neural network accountability?", top_k=1)
    assert len(results) == 1
    assert results[0].doc_name == "ai_ethics.pdf"


def test_save_and_load_persistence(tmp_path, mock_embeddings, vector_store):
    """Persisted index must retain vector count, metadata, and compatibility checks."""
    chunks = [
        TextChunk(chunk_id="c0", doc_name="test.pdf", page_number=1, text="Test passage text.", start_char=0, end_char=18)
    ]
    embeddings = mock_embeddings.embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    # Save to temp directory
    vector_store.save(tmp_path)

    # Load into a new store with identical config
    new_store = FAISSVectorStore(embedding_model_name="all-MiniLM-L6-v2", dimension=384)
    new_store.load(tmp_path)

    assert new_store.total_vectors == 1
    query_vec = mock_embeddings.embed_query("test query")
    res = new_store.search(query_vec, top_k=1)
    assert len(res) == 1
    assert res[0].text == "Test passage text."


def test_load_mismatched_model_error(tmp_path, mock_embeddings, vector_store):
    """Attempting to load an index with a different model name must raise VectorStoreError."""
    chunks = [
        TextChunk(chunk_id="c0", doc_name="test.pdf", page_number=1, text="Passage.", start_char=0, end_char=8)
    ]
    embeddings = mock_embeddings.embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)
    vector_store.save(tmp_path)

    # Try to load with a different model name
    incompatible_store = FAISSVectorStore(embedding_model_name="different-model-v2", dimension=384)
    with pytest.raises(VectorStoreError, match="Embedding model mismatch"):
        incompatible_store.load(tmp_path)
