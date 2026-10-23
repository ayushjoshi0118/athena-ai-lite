"""Web Search Adapter for Athena AI Lite.

EDUCATIONAL CONCEPT: Why Web Search in an Academic Assistant?
While RAG connects an AI to private uploaded PDFs (such as lecture notes or a course syllabus),
those documents can be outdated or incomplete.
When live web search is enabled:
1. The Supervisor Agent can recognize questions requiring current global information
   (e.g., "What are the latest developments in transformer architectures in 2026?").
2. The Web Agent queries an external search provider and returns grounded search snippets.
3. The LLM combines the university PDF context with the live web evidence, maintaining
   separate citation provenance for both!

Supported Providers:
- DuckDuckGo: Free, requires no API key.
- Tavily: Specialized search API for AI agents (requires TAVILY_API_KEY).
- Mock: Deterministic test fixtures for offline testing and verification.
"""

from dataclasses import asdict, dataclass
import os
import re
import time
from typing import Any, Dict, List, Optional
import requests
import config


class WebSearchError(Exception):
    """Custom exception raised when web search operations fail."""
    pass


@dataclass
class WebSearchResult:
    """Represents an external web source with title, URL, and excerpt."""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    source_provider: str = "web"
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebSearchEngine:
    """Configurable web search engine supporting DuckDuckGo, Tavily, and Mock."""

    def __init__(
        self,
        provider: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        force_mock: bool = False
    ):
        self.provider = (provider or config.WEB_SEARCH_PROVIDER).lower()
        self.tavily_api_key = tavily_api_key or config.TAVILY_API_KEY
        self.force_mock = force_mock

    @property
    def is_mock(self) -> bool:
        """Indicates if search is running in mock mode."""
        if self.force_mock or self.provider == "mock":
            return True
        if self.provider == "tavily" and not self.tavily_api_key:
            return True
        return False

    def search(self, query: str, max_results: int = 4) -> List[WebSearchResult]:
        """Executes a search query and returns structured results with URLs.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve (default 4).

        Returns:
            List of WebSearchResult items.

        Raises:
            WebSearchError: On critical provider failures.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        if self.is_mock:
            note = "Missing TAVILY_API_KEY" if self.provider == "tavily" and not self.tavily_api_key else None
            return self._mock_search(clean_query, max_results, note=note)

        if self.provider == "tavily":
            return self._tavily_search(clean_query, max_results)

        # Default: DuckDuckGo
        return self._duckduckgo_search(clean_query, max_results)

    def _duckduckgo_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Queries DuckDuckGo via its public HTML endpoint without requiring an API key."""
        try:
            # DuckDuckGo HTML Lite search endpoint
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            data = {"q": query}

            resp = requests.post(url, data=data, headers=headers, timeout=8.0)
            if resp.status_code == 429:
                raise WebSearchError("DuckDuckGo search rate limit reached. Please try again in a few moments.")
            if resp.status_code != 200:
                raise WebSearchError(f"DuckDuckGo search returned HTTP status {resp.status_code}")

            results: List[WebSearchResult] = []
            html = resp.text

            # Parse simple HTML results using regex to avoid extra dependencies
            # Matches <a class="result__snippet" ...>...</a> and <a class="result__url" href="...">...</a>
            raw_blocks = re.findall(r'<div class="result__body">(.*?)</div>', html, re.DOTALL)

            for block in raw_blocks[:max_results]:
                # Extract URL and title
                title_match = re.search(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)

                if title_match:
                    raw_url = title_match.group(1).strip()
                    # Clean DuckDuckGo redirect URL
                    url_clean = raw_url
                    uddg_match = re.search(r'uddg=([^&]+)', raw_url)
                    if uddg_match:
                        import urllib.parse
                        url_clean = urllib.parse.unquote(uddg_match.group(1))

                    raw_title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()

                    results.append(
                        WebSearchResult(
                            title=raw_title or "Web Resource",
                            url=url_clean,
                            snippet=snippet or "Search excerpt unavailable.",
                            source_provider="duckduckgo",
                            is_mock=False
                        )
                    )

            if not results:
                # If HTML parsing yielded 0 results, fallback to mock with clear note
                return self._mock_search(query, max_results, note="Live DDG returned 0 parsed results")

            return results

        except requests.exceptions.Timeout:
            raise WebSearchError("Web search connection timed out. Please check your network connection.")
        except requests.exceptions.RequestException as e:
            # If network error occurs, return mock with label
            return self._mock_search(query, max_results, note=f"Live search failed: {e}")

    def _tavily_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Queries the official Tavily Search API."""
        if not self.tavily_api_key:
            return self._mock_search(query, max_results, note="Missing TAVILY_API_KEY")

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results
            }
            resp = requests.post(url, json=payload, timeout=10.0)

            if resp.status_code == 401 or resp.status_code == 403:
                raise WebSearchError("Tavily API authentication failed: Invalid TAVILY_API_KEY.")
            if resp.status_code == 429:
                raise WebSearchError("Tavily API rate limit exceeded.")
            if resp.status_code != 200:
                raise WebSearchError(f"Tavily API returned status {resp.status_code}: {resp.text}")

            data = resp.json()
            results: List[WebSearchResult] = []

            for item in data.get("results", []):
                results.append(
                    WebSearchResult(
                        title=item.get("title", "Web Result"),
                        url=item.get("url", "https://example.com"),
                        snippet=item.get("content", ""),
                        published_date=item.get("published_date"),
                        source_provider="tavily",
                        is_mock=False
                    )
                )

            return results

        except requests.exceptions.Timeout:
            raise WebSearchError("Tavily search connection timed out.")
        except requests.exceptions.RequestException as e:
            raise WebSearchError(f"Tavily search network error: {e}")

    def _mock_search(
        self,
        query: str,
        max_results: int = 4,
        note: Optional[str] = None
    ) -> List[WebSearchResult]:
        """Returns deterministic, clearly labeled educational search fixtures."""
        reason = note or "Mock mode active / no external credentials required"
        return [
            WebSearchResult(
                title=f"[MOCK RESULT] Overview of {query}",
                url="https://educational.athena.ai/mock-search-resource-1",
                snippet=(
                    f"[MOCK SEARCH — {reason}] Relevant external background and academic "
                    f"information for '{query}' retrieved from verified simulated web sources."
                ),
                published_date="2026-01-15",
                source_provider="mock",
                is_mock=True
            ),
            WebSearchResult(
                title=f"[MOCK RESULT] Academic Standards and Definitions: {query}",
                url="https://educational.athena.ai/mock-search-resource-2",
                snippet=(
                    f"Simulated web article discussing standard definitions and industry applications "
                    f"pertaining to '{query}'."
                ),
                published_date="2026-03-20",
                source_provider="mock",
                is_mock=True
            )
        ][:max_results]
