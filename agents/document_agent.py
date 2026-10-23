"""Document Specialist Agent for Athena AI Lite.

EDUCATIONAL CONCEPT: What is a Specialist Agent?
Instead of forcing a single prompt to do everything, Multi-Agent systems divide work
among focused components with distinct responsibilities:
- The Document Agent specializes in extracting, searching, and validating evidence
  from the student's active university PDF.
- It returns structured evidence with page provenance, or explicitly flags when
  the document lacks sufficient data to answer the query.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.retriever import Retriever
from core.vector_store import SearchResult


@dataclass
class DocumentEvidence:
    """Structured evidence returned by the Document Agent."""
    has_evidence: bool
    summary: str
    chunks: List[SearchResult] = field(default_factory=list)
    doc_name: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_evidence": self.has_evidence,
            "summary": self.summary,
            "doc_name": self.doc_name,
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
            "error_message": self.error_message
        }


class DocumentAgent:
    """Specialist agent responsible for querying the local PDF knowledge base."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def search(self, question: str, top_k: int = 4) -> DocumentEvidence:
        """Searches the active PDF and returns grounded evidence with citations.

        Args:
            question: The academic inquiry.
            top_k: Maximum number of chunks to inspect.

        Returns:
            DocumentEvidence containing matching chunks and an evidence summary.
        """
        if not self.retriever.is_ready:
            return DocumentEvidence(
                has_evidence=False,
                summary="No university PDF is currently indexed.",
                doc_name=None,
                error_message="Active document not found in vector store."
            )

        clean_q = question.strip()
        if not clean_q:
            return DocumentEvidence(
                has_evidence=False,
                summary="Empty question provided.",
                doc_name=self.retriever.active_document_name
            )

        try:
            results = self.retriever.retrieve(query=clean_q, top_k=top_k)
        except Exception as e:
            return DocumentEvidence(
                has_evidence=False,
                summary=f"Retrieval error: {e}",
                doc_name=self.retriever.active_document_name,
                error_message=str(e)
            )

        if not results:
            return DocumentEvidence(
                has_evidence=False,
                summary="The uploaded document does not contain relevant information for this query.",
                doc_name=self.retriever.active_document_name
            )

        # Build an evidence summary for the supervisor
        pages_cited = sorted(list({r.page_number for r in results}))
        summary = (
            f"Found {len(results)} relevant passage(s) across Page(s) {pages_cited} "
            f"in '{self.retriever.active_document_name}'."
        )

        return DocumentEvidence(
            has_evidence=True,
            summary=summary,
            chunks=results,
            doc_name=self.retriever.active_document_name
        )
