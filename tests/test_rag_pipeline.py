"""Unit tests for the RAG pipeline in Athena AI Lite.

Verifies:
- Complete end-to-end RAG execution: Embed query -> Retrieve chunks -> Format prompt -> LLM answer -> Citations.
- Grounding integrity and insufficient evidence detection.
- Propagation of personalization parameters (learning level, answer length).
- Preservation of page citations from vector store metadata.
"""

import pytest
from config import UserPreferences
from core.embeddings import LocalEmbeddings
from core.llm_client import GeminiLLMClient
from core.pdf_loader import ExtractedPage, PDFDocument
from core.rag_pipeline import RAGPipeline, RAGResponse
from core.retriever import Retriever
from core.vector_store import FAISSVectorStore


@pytest.fixture
def rag_pipeline():
    """Initializes a deterministic RAG pipeline with mock components."""
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2", mock_mode=True)
    vector_store = FAISSVectorStore(embedding_model_name="all-MiniLM-L6-v2", dimension=384)
    retriever = Retriever(embeddings=embeddings, vector_store=vector_store)

    doc = PDFDocument(
        filename="distributed_systems.pdf",
        total_pages=2,
        pages=[
            ExtractedPage(page_number=1, text="The Raft consensus algorithm uses leader election and log replication.", char_count=73),
            ExtractedPage(page_number=2, text="Byzantine fault tolerance tolerates arbitrary component failures.", char_count=67),
        ],
        total_characters=140,
        has_extractable_text=True
    )
    retriever.index_document(doc, chunk_size=100, chunk_overlap=15)

    llm_client = GeminiLLMClient(force_mock=True)
    return RAGPipeline(retriever=retriever, llm_client=llm_client)


def test_rag_pipeline_run_success(rag_pipeline):
    """Pipeline should retrieve chunks, synthesize mock answer, and provide citations."""
    prefs = UserPreferences(learning_level="Beginner", answer_length="Short", top_k=2)
    response: RAGResponse = rag_pipeline.run("How does Raft work?", preferences=prefs)

    assert response.question == "How does Raft work?"
    assert response.is_mock is True
    assert len(response.citations) > 0
    assert response.citations[0].doc_name == "distributed_systems.pdf"
    assert response.citations[0].page_number in [1, 2]
    assert response.learning_level == "Beginner"
    assert response.answer_length == "Short"
    assert response.sufficient_evidence is True
    assert response.execution_time_seconds >= 0.0


def test_rag_pipeline_no_active_document():
    """If retriever has no document, pipeline should return clear guidance without error."""
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2", mock_mode=True)
    vector_store = FAISSVectorStore(embedding_model_name="all-MiniLM-L6-v2", dimension=384)
    retriever = Retriever(embeddings=embeddings, vector_store=vector_store)
    llm_client = GeminiLLMClient(force_mock=True)
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

    response = pipeline.run("What is Raft?")
    assert response.sufficient_evidence is False
    assert "upload a university pdf" in response.answer.lower()
    assert len(response.citations) == 0
