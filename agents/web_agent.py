"""Web Specialist Agent for Athena AI Lite.

EDUCATIONAL CONCEPT: The Web Agent's Responsibility
The Web Agent specializes in external open-web discovery:
- It receives research queries from the Supervisor when external facts or current
  context are needed.
- It returns search results with actual URLs and publication dates where available.
- It reports unavailable credentials or search failures honestly to the Supervisor
  instead of fabricating information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.web_search import WebSearchEngine, WebSearchError, WebSearchResult


@dataclass
class WebEvidence:
    """Structured evidence returned by the Web Agent."""
    has_evidence: bool
    summary: str
    results: List[WebSearchResult] = field(default_factory=list)
    is_mock: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_evidence": self.has_evidence,
            "summary": self.summary,
            "result_count": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "is_mock": self.is_mock,
            "error_message": self.error_message
        }


class WebAgent:
    """Specialist agent responsible for querying external web search engines."""

    def __init__(self, web_engine: WebSearchEngine):
        self.web_engine = web_engine

    def search(self, query: str, max_results: int = 4) -> WebEvidence:
        """Executes a web search and returns structured results for the Supervisor.

        Args:
            query: The research inquiry.
            max_results: Limit of search results.

        Returns:
            WebEvidence object with search results and error notes if any.
        """
        clean_q = query.strip()
        if not clean_q:
            return WebEvidence(
                has_evidence=False,
                summary="Empty query received by Web Agent."
            )

        try:
            results = self.web_engine.search(query=clean_q, max_results=max_results)
            if not results:
                return WebEvidence(
                    has_evidence=False,
                    summary="No relevant web search results were found.",
                    is_mock=self.web_engine.is_mock
                )

            summary = f"Retrieved {len(results)} external web source(s)."
            is_mock = any(r.is_mock for r in results)

            return WebEvidence(
                has_evidence=True,
                summary=summary,
                results=results,
                is_mock=is_mock
            )

        except WebSearchError as e:
            return WebEvidence(
                has_evidence=False,
                summary=f"Web search error: {e}",
                error_message=str(e),
                is_mock=self.web_engine.is_mock
            )
        except Exception as e:
            return WebEvidence(
                has_evidence=False,
                summary=f"Unexpected web search error: {e}",
                error_message=str(e),
                is_mock=self.web_engine.is_mock
            )
