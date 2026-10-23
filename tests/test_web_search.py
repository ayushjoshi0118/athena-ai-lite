"""Unit tests for the WebSearchEngine in Athena AI Lite.

Verifies:
- Mock mode operation for deterministic offline testing.
- Result data contracts (title, URL, snippet, source provider).
- Max results limits.
- Empty query handling.
- Graceful error reporting when credentials are missing or services fail.
"""

import pytest
from core.web_search import WebSearchEngine, WebSearchResult


def test_mock_web_search():
    """Mock search returns deterministic results with valid URLs and source tags."""
    engine = WebSearchEngine(provider="mock", force_mock=True)
    results = engine.search("modern transformer models", max_results=2)

    assert len(results) == 2
    for r in results:
        assert isinstance(r, WebSearchResult)
        assert r.is_mock is True
        assert r.source_provider == "mock"
        assert r.url.startswith("http")
        assert len(r.title) > 0
        assert len(r.snippet) > 0


def test_web_search_empty_query():
    """Empty or whitespace query should return an empty list without making requests."""
    engine = WebSearchEngine(provider="mock", force_mock=True)
    assert engine.search("") == []
    assert engine.search("   \n\t  ") == []


def test_tavily_missing_key_fallback():
    """When Tavily provider is selected but no key is provided, should fallback to mock."""
    engine = WebSearchEngine(provider="tavily", tavily_api_key="")
    assert engine.is_mock is True
    results = engine.search("neural architectures", max_results=1)
    assert len(results) == 1
    assert results[0].is_mock is True
    assert "Missing TAVILY_API_KEY" in results[0].snippet
