"""Supervisor Agent for Athena AI Lite.

EDUCATIONAL CONCEPT: What is a Supervisor Agent?
A Supervisor coordinates multiple specialist agents to answer complex user requests:
1. Routing / Decision-Making:
   It inspects the user question and user preferences (e.g., whether web search is enabled)
   to decide which specialist agents need to be dispatched.
2. Tool Execution:
   The supervisor invokes the actual specialist agents (DocumentAgent, WebAgent)
   and collects their typed evidence structures.
3. Activity Tracing:
   It records real execution milestones (not simulated delays) so the student can
   inspect exactly which agents executed, what they found, and how long each step took.
4. Evidence Synthesis & Citation Integrity:
   It merges document passages and external web snippets into a structured multi-source
   prompt for the LLM, preserving distinct citations for both.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional
from config import UserPreferences
from agents.document_agent import DocumentAgent, DocumentEvidence
from agents.web_agent import WebAgent, WebEvidence
from core.llm_client import GeminiLLMClient, LLMResponse
from core.rag_pipeline import Citation


@dataclass
class ActivityStep:
    """Represents a milestone in the supervisor's execution trace."""
    step_name: str
    status: str  # "completed", "skipped", "error"
    detail: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_name,
            "status": self.status,
            "detail": self.detail,
            "timestamp": round(self.timestamp, 3)
        }


@dataclass
class SupervisorDecision:
    """Routing choice determined by the supervisor."""
    use_document_agent: bool
    use_web_agent: bool
    reasoning: str


@dataclass
class SupervisorResult:
    """Comprehensive output produced by the supervisor coordination workflow."""
    question: str
    answer: str
    decision: SupervisorDecision
    document_evidence: Optional[DocumentEvidence] = None
    web_evidence: Optional[WebEvidence] = None
    document_citations: List[Citation] = field(default_factory=list)
    web_citations: List[Dict[str, str]] = field(default_factory=list)
    activity_trace: List[ActivityStep] = field(default_factory=list)
    is_mock: bool = False
    model_name: str = ""
    sufficient_evidence: bool = True
    execution_time_seconds: float = 0.0


class AgentSupervisor:
    """Coordinates DocumentAgent and WebAgent to generate grounded, multi-source answers."""

    def __init__(
        self,
        document_agent: DocumentAgent,
        web_agent: WebAgent,
        llm_client: GeminiLLMClient
    ):
        self.doc_agent = document_agent
        self.web_agent = web_agent
        self.llm_client = llm_client

    def decide_routing(
        self,
        question: str,
        preferences: UserPreferences
    ) -> SupervisorDecision:
        """Determines which specialist agents to activate based on settings and query intent."""
        # If web search is disabled by the user, route exclusively to the Document Agent
        if not preferences.include_web_search:
            return SupervisorDecision(
                use_document_agent=True,
                use_web_agent=False,
                reasoning="Web search disabled in preferences. Routing exclusively to Document Agent."
            )

        q_lower = question.lower()
        # Signals that the question specifically targets the uploaded document
        doc_signals = ["syllabus", "lecture", "course", "professor", "exam", "grading", "office hours", "document", "pdf", "page"]
        is_explicitly_doc_only = any(sig in q_lower for sig in doc_signals) and not any(k in q_lower for k in ["latest", "news", "current", "internet", "web", "compare"])

        if is_explicitly_doc_only and self.doc_agent.retriever.is_ready:
            return SupervisorDecision(
                use_document_agent=True,
                use_web_agent=False,
                reasoning="Question targets internal course materials. Document Agent prioritized."
            )

        # If web search is enabled and question can benefit from external facts or both
        return SupervisorDecision(
            use_document_agent=self.doc_agent.retriever.is_ready,
            use_web_agent=True,
            reasoning="Multi-agent routing: Combining uploaded document with live web search."
        )

    def run(
        self,
        question: str,
        preferences: Optional[UserPreferences] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> SupervisorResult:
        """Executes the supervisor workflow and coordinates specialist agents."""
        start_time = time.time()
        prefs = preferences or UserPreferences()
        activity_trace: List[ActivityStep] = []

        def record(step: str, status: str, detail: str):
            trace_item = ActivityStep(step_name=step, status=status, detail=detail)
            activity_trace.append(trace_item)
            if on_progress:
                on_progress(f"[{step}] {detail}")

        # 1. Understanding question
        record("Understanding question", "completed", f"Analyzed query intent for '{question[:60]}...'")
        decision = self.decide_routing(question, prefs)

        doc_evidence: Optional[DocumentEvidence] = None
        web_evidence: Optional[WebEvidence] = None

        # 2. Searching uploaded document
        if decision.use_document_agent:
            record("Searching uploaded document", "completed", "Invoked Document Agent on active FAISS index.")
            doc_evidence = self.doc_agent.search(question, top_k=prefs.top_k)
            status_desc = "Found matching chunks." if doc_evidence.has_evidence else "No relevant document passages found."
            record("Document search completed", "completed", status_desc)
        else:
            record("Searching uploaded document", "skipped", "Document agent not requested by routing decision.")

        # 3. Searching web, if enabled
        if decision.use_web_agent:
            record("Searching web", "completed", f"Invoked Web Agent using provider '{self.web_agent.web_engine.provider}'.")
            web_evidence = self.web_agent.search(question, max_results=4)
            status_desc = f"Retrieved {len(web_evidence.results)} web sources." if web_evidence.has_evidence else "Web search returned no results."
            record("Web search completed", "completed", status_desc)
        else:
            record("Searching web", "skipped", "Web search not enabled or not required.")

        # 4. Combining evidence
        record("Combining evidence", "completed", "Synthesizing retrieved passages into grounded context.")
        context_blocks = []
        doc_citations: List[Citation] = []
        web_citations: List[Dict[str, str]] = []

        if doc_evidence and doc_evidence.has_evidence:
            context_blocks.append("--- UNIVERSITY DOCUMENT EVIDENCE ---")
            for chunk in doc_evidence.chunks:
                context_blocks.append(
                    f"[PDF Source: {chunk.doc_name}, Page {chunk.page_number}]\n{chunk.text}"
                )
                snippet = chunk.text[:140] + "..." if len(chunk.text) > 140 else chunk.text
                doc_citations.append(
                    Citation(
                        citation_id=chunk.chunk_id,
                        doc_name=chunk.doc_name,
                        page_number=chunk.page_number,
                        score=chunk.score,
                        text_snippet=snippet
                    )
                )

        if web_evidence and web_evidence.has_evidence:
            context_blocks.append("--- LIVE WEB EVIDENCE ---")
            for item in web_evidence.results:
                context_blocks.append(
                    f"[Web Source: {item.title} | URL: {item.url}]\n{item.snippet}"
                )
                web_citations.append({
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source": item.source_provider,
                    "is_mock": str(item.is_mock)
                })

        # Check if we have any evidence at all
        has_any_evidence = bool(doc_citations or web_citations)
        if not has_any_evidence:
            record("Generating final answer", "completed", "Reported lack of evidence to user.")
            return SupervisorResult(
                question=question,
                answer="Neither the uploaded document nor external search provided sufficient information to answer this question.",
                decision=decision,
                document_evidence=doc_evidence,
                web_evidence=web_evidence,
                document_citations=[],
                web_citations=[],
                activity_trace=activity_trace,
                sufficient_evidence=False,
                execution_time_seconds=time.time() - start_time
            )

        combined_context = "\n\n".join(context_blocks)

        # 5. Generating final answer with LLM
        record("Generating final answer", "completed", "Sending evidence and personalization constraints to LLM.")

        system_instruction = self._build_system_instruction(prefs)
        prompt = (
            f"You are Athena AI Lite, an academic supervisor and teaching assistant.\n\n"
            f"EVIDENCE FROM SPECIALIST AGENTS:\n"
            f"================================\n"
            f"{combined_context}\n"
            f"================================\n\n"
            f"USER QUESTION: {question}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer strictly using the evidence provided above.\n"
            f"2. If document evidence and web evidence both exist, clearly distinguish what the uploaded document states versus what external web research found.\n"
            f"3. If evidence is conflicting, highlight the discrepancy transparently.\n"
            f"4. Do not invent facts, page numbers, or URLs.\n"
            f"5. Adhere to learning level '{prefs.learning_level}' and answer length '{prefs.answer_length}'."
        )

        llm_resp: LLMResponse = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )

        elapsed = time.time() - start_time
        record("Workflow complete", "completed", f"Answer synthesized in {round(elapsed, 2)}s.")

        return SupervisorResult(
            question=question,
            answer=llm_resp.text,
            decision=decision,
            document_evidence=doc_evidence,
            web_evidence=web_evidence,
            document_citations=doc_citations,
            web_citations=web_citations,
            activity_trace=activity_trace,
            is_mock=llm_resp.is_mock,
            model_name=llm_resp.model_name,
            sufficient_evidence=True,
            execution_time_seconds=elapsed
        )

    def _build_system_instruction(self, prefs: UserPreferences) -> str:
        """Personalization guide for multi-agent synthesis."""
        instructions = [
            "You are Athena AI Lite's synthesis engine coordinating multi-agent evidence.",
            f"Explain at the {prefs.learning_level} level."
        ]
        if prefs.learning_level == "Beginner":
            instructions.append("Use plain language, simple analogies, and define any technical words.")
        elif prefs.learning_level == "Intermediate":
            instructions.append("Use standard technical terms with practical examples.")
        elif prefs.learning_level == "Advanced":
            instructions.append("Use rigorous academic terminology, addressing assumptions and subtleties.")

        if prefs.answer_length == "Short":
            instructions.append("Keep your answer concise (1-2 paragraphs).")
        elif prefs.answer_length == "Balanced":
            instructions.append("Provide a balanced explanation (2-3 paragraphs).")
        elif prefs.answer_length == "Detailed":
            instructions.append("Provide a detailed, step-by-step explanation.")

        return " ".join(instructions)
