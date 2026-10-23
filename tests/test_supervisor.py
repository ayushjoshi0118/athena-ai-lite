"""Unit tests for the Agent Supervisor in Athena AI Lite.

Verifies:
- Supervisor routing decisions based on user preferences and query intent.
- Document-only execution path.
- Multi-agent coordination (Document + Web Agent).
- Citation provenance (distinguishing PDF citations from Web URLs).
- Activity trace tracking.
- Safe handling when evidence is absent.
"""

import pytest
from agents.document_agent import DocumentAgent
from agents.supervisor import AgentSupervisor, SupervisorDecision, SupervisorResult
from agents.web_agent import WebAgent
from config import UserPreferences
from core.chunker import TextChunk
from core.embeddings import LocalEmbeddings
from core.llm_client import GeminiLLMClient
from core.pdf_loader import ExtractedPage, PDFDocument
from core.retriever import Retriever
from core.vector_store import FAISSVectorStore
from core.web_search import WebSearchEngine


@pytest.fixture
def supervisor_setup():
    """Initializes a complete, deterministic, offline multi-agent test environment."""
    embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2", mock_mode=True)
    vector_store = FAISSVectorStore(embedding_model_name="all-MiniLM-L6-v2", dimension=384)
    retriever = Retriever(embeddings=embeddings, vector_store=vector_store)

    # Index a small synthetic document
    doc = PDFDocument(
        filename="university_calendar.pdf",
        total_pages=2,
        pages=[
            ExtractedPage(page_number=1, text="Fall semester final exams begin on December 10th.", char_count=52),
            ExtractedPage(page_number=2, text="Graduation ceremony will be held at Main Auditorium.", char_count=55)
        ],
        total_characters=107,
        has_extractable_text=True
    )
    retriever.index_document(doc, chunk_size=150, chunk_overlap=20)

    doc_agent = DocumentAgent(retriever=retriever)
    web_engine = WebSearchEngine(provider="mock", force_mock=True)
    web_agent = WebAgent(web_engine=web_engine)
    llm_client = GeminiLLMClient(force_mock=True)

    supervisor = AgentSupervisor(
        document_agent=doc_agent,
        web_agent=web_agent,
        llm_client=llm_client
    )

    return supervisor, retriever


def test_supervisor_routing_web_disabled(supervisor_setup):
    """When web search is disabled, supervisor must route solely to the Document Agent."""
    supervisor, _ = supervisor_setup
    prefs = UserPreferences(include_web_search=False)

    decision = supervisor.decide_routing("When do final exams begin?", prefs)
    assert decision.use_document_agent is True
    assert decision.use_web_agent is False
    assert "disabled" in decision.reasoning.lower()


def test_supervisor_routing_web_enabled(supervisor_setup):
    """When web search is enabled, supervisor should coordinate both agents."""
    supervisor, _ = supervisor_setup
    prefs = UserPreferences(include_web_search=True)

    decision = supervisor.decide_routing("What are modern best practices in AI regulation?", prefs)
    assert decision.use_web_agent is True


def test_supervisor_document_only_execution(supervisor_setup):
    """Execution with web search off should produce PDF citations and record activity steps."""
    supervisor, _ = supervisor_setup
    prefs = UserPreferences(include_web_search=False)

    result: SupervisorResult = supervisor.run("When do final exams start?", preferences=prefs)

    assert result.decision.use_document_agent is True
    assert result.decision.use_web_agent is False
    assert len(result.document_citations) > 0
    assert result.document_citations[0].doc_name == "university_calendar.pdf"
    assert len(result.web_citations) == 0

    # Verify Activity Trace has required real steps
    step_names = [step.step_name for step in result.activity_trace]
    assert "Understanding question" in step_names
    assert "Searching uploaded document" in step_names
    assert "Combining evidence" in step_names
    assert "Generating final answer" in step_names


def test_supervisor_multi_agent_execution(supervisor_setup):
    """Execution with web search on should produce both PDF and Web citations."""
    supervisor, _ = supervisor_setup
    prefs = UserPreferences(include_web_search=True)

    result: SupervisorResult = supervisor.run(
        "Compare university exam schedules with national education guidelines.",
        preferences=prefs
    )

    assert result.decision.use_web_agent is True
    assert len(result.document_citations) > 0
    assert len(result.web_citations) > 0
    # Web citations must contain URL and title
    assert "url" in result.web_citations[0]
    assert "title" in result.web_citations[0]


def test_supervisor_no_evidence_handling(supervisor_setup):
    """When no document is indexed and web search is off, report lack of evidence honestly."""
    supervisor, retriever = supervisor_setup
    retriever.clear()  # Empty knowledge base

    prefs = UserPreferences(include_web_search=False)
    result: SupervisorResult = supervisor.run("What is quantum computing?", preferences=prefs)

    assert result.sufficient_evidence is False
    assert "neither" in result.answer.lower() or "not" in result.answer.lower()
    assert len(result.document_citations) == 0
