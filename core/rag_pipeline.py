"""Retrieval-Augmented Generation (RAG) Pipeline for Athena AI Lite.

EDUCATIONAL CONCEPT: The RAG Lifecycle
The classic RAG sequence connects search retrieval to language generation:
1. Query Embedding: The user's question is encoded into a high-dimensional vector.
2. Vector Retrieval: FAISS searches for nearest neighbor chunks by cosine similarity.
3. Context Construction: The application formats retrieved passages into an evidence block.
4. Prompt Augmentation: Context, question, and personalization instructions (level & length)
   are combined into a structured prompt.
5. Grounded LLM Generation: The LLM generates an answer constrained strictly to the evidence.
6. Provenance & Citation: Application code builds explicit citations directly from the
   retrieved chunk metadata, preventing the model from hallucinating sources.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from config import UserPreferences
from core.llm_client import GeminiLLMClient, LLMResponse
from core.retriever import Retriever
from core.vector_store import SearchResult


@dataclass
class Citation:
    """Verifiable source reference grounded in real document metadata."""
    citation_id: str
    doc_name: str
    page_number: int
    score: float
    text_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet
        }


@dataclass
class RAGResponse:
    """Complete output of a RAG pipeline execution."""
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[SearchResult] = field(default_factory=list)
    is_mock: bool = False
    model_name: str = ""
    learning_level: str = "Beginner"
    answer_length: str = "Balanced"
    execution_time_seconds: float = 0.0
    sufficient_evidence: bool = True


class RAGPipeline:
    """Executes the complete RAG sequence with transparent evidence tracking."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: GeminiLLMClient
    ):
        self.retriever = retriever
        self.llm_client = llm_client

    def run(
        self,
        question: str,
        preferences: Optional[UserPreferences] = None
    ) -> RAGResponse:
        """Executes the step-by-step RAG pipeline for a given user question.

        Args:
            question: The user's academic question.
            preferences: Optional user settings (learning level, answer length, top_k).

        Returns:
            RAGResponse containing generated text, citations, and debug info.
        """
        start_time = time.time()
        prefs = preferences or UserPreferences()

        # Step 0: Ensure a document is indexed
        if not self.retriever.is_ready:
            return RAGResponse(
                question=question,
                answer=(
                    "No document is currently active. Please upload a university PDF "
                    "in the sidebar before asking questions."
                ),
                sufficient_evidence=False,
                execution_time_seconds=time.time() - start_time
            )

        # Step 1: Vector Retrieval
        results: List[SearchResult] = self.retriever.retrieve(
            query=question,
            top_k=prefs.top_k
        )

        if not results:
            return RAGResponse(
                question=question,
                answer="The uploaded document does not contain relevant information to answer this question.",
                retrieved_chunks=[],
                sufficient_evidence=False,
                execution_time_seconds=time.time() - start_time
            )

        # Step 2: Context Construction
        context_parts = []
        for i, res in enumerate(results, start=1):
            context_parts.append(
                f"[Passage {i} | Source: {res.doc_name}, Page {res.page_number}]\n{res.text}"
            )
        formatted_context = "\n\n".join(context_parts)

        # Step 3: Personalization & System Instructions
        system_instruction = self._build_system_instruction(prefs)

        prompt = (
            f"You are Athena AI Lite, a helpful and precise university assistant.\n\n"
            f"DOCUMENT CONTEXT:\n"
            f"---------------------\n"
            f"{formatted_context}\n"
            f"---------------------\n\n"
            f"USER QUESTION: {question}\n\n"
            f"GROUNDING INSTRUCTIONS:\n"
            f"1. Answer strictly and factually based on the provided DOCUMENT CONTEXT passages above.\n"
            f"2. If the context does not contain sufficient information to answer the question, state: "
            f"'The uploaded document does not provide sufficient information to answer this question.'\n"
            f"3. Do not invent facts, page numbers, or external sources.\n"
            f"4. Match the requested learning level ({prefs.learning_level}) and answer length ({prefs.answer_length})."
        )

        # Step 4: LLM Generation
        llm_resp: LLMResponse = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )

        # Step 5: Build Grounded Citations in Application Code
        citations = []
        for res in results:
            snippet = res.text[:140] + "..." if len(res.text) > 140 else res.text
            citations.append(
                Citation(
                    citation_id=res.chunk_id,
                    doc_name=res.doc_name,
                    page_number=res.page_number,
                    score=res.score,
                    text_snippet=snippet
                )
            )

        # Determine if the answer indicates missing evidence
        insufficient_phrases = [
            "does not provide sufficient information",
            "does not contain sufficient",
            "not mentioned in the document",
            "cannot find this information in the uploaded document"
        ]
        has_insufficient = any(phrase in llm_resp.text.lower() for phrase in insufficient_phrases)

        elapsed = time.time() - start_time

        return RAGResponse(
            question=question,
            answer=llm_resp.text,
            citations=citations,
            retrieved_chunks=results,
            is_mock=llm_resp.is_mock,
            model_name=llm_resp.model_name,
            learning_level=prefs.learning_level,
            answer_length=prefs.answer_length,
            execution_time_seconds=elapsed,
            sufficient_evidence=not has_insufficient
        )

    def _build_system_instruction(self, prefs: UserPreferences) -> str:
        """Constructs persona and personalization guidelines for the LLM."""
        instructions = [
            "You are Athena AI Lite, an academic teaching assistant designed to explain concepts clearly.",
            "Always maintain strict academic integrity and ground answers in provided evidence."
        ]

        # Learning Level
        if prefs.learning_level == "Beginner":
            instructions.append(
                "LEARNING LEVEL: Beginner. Explain concepts in clear, intuitive everyday language. "
                "Define unfamiliar terminology gently and provide relatable intuition."
            )
        elif prefs.learning_level == "Intermediate":
            instructions.append(
                "LEARNING LEVEL: Intermediate. Use technical and academic terminology with explanations. "
                "Highlight practical applications and relationships between key ideas."
            )
        elif prefs.learning_level == "Advanced":
            instructions.append(
                "LEARNING LEVEL: Advanced. Use precise academic terminology. Address theoretical assumptions, "
                "methodological nuances, limitations, and edge cases with rigor."
            )

        # Answer Length
        if prefs.answer_length == "Short":
            instructions.append("ANSWER LENGTH: Short. Keep the response concise, punchy, and direct (1-2 paragraphs).")
        elif prefs.answer_length == "Balanced":
            instructions.append("ANSWER LENGTH: Balanced. Provide a well-structured, clear explanation (2-3 paragraphs).")
        elif prefs.answer_length == "Detailed":
            instructions.append("ANSWER LENGTH: Detailed. Provide an in-depth, structured explanation with thorough sections.")

        if prefs.include_examples:
            instructions.append("EXAMPLES: Include practical illustrative examples where helpful.")

        return " ".join(instructions)
