"""Configuration settings for Athena AI Lite.

This module centralizes environment variables, directory paths, and default
hyperparameters. It supports a safe, educational architecture:
- Secrets are read from environment variables or .env (never hard-coded).
- If API keys are missing, the system cleanly identifies that Mock Mode should be used.
- Paths for data storage, index caching, and user preferences are resolved dynamically.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Base directory: root of athena-ai-lite
BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# Directory layout
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEXES_DIR = DATA_DIR / "indexes"
PREFERENCES_PATH = DATA_DIR / "preferences.json"

# Ensure data directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration (Google Gemini via official google-genai SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Web Search Configuration
# Options: "duckduckgo", "tavily", "mock"
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# Embeddings & FAISS Vector Store Settings
# Local sentence-transformers model (dimension 384 for all-MiniLM-L6-v2)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()
EMBEDDING_DIMENSION = 384
SIMILARITY_METRIC = "cosine"  # normalized vectors with Inner Product (IndexFlatIP)

# Default RAG parameters
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))


@dataclass
class UserPreferences:
    """User personalization and retrieval preferences."""
    learning_level: str = "Beginner"  # "Beginner", "Intermediate", "Advanced"
    answer_length: str = "Balanced"   # "Short", "Balanced", "Detailed"
    include_examples: bool = True
    include_web_search: bool = False
    top_k: int = DEFAULT_TOP_K
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def load_preferences() -> UserPreferences:
    """Load user preferences from JSON, or return defaults if file doesn't exist."""
    if PREFERENCES_PATH.exists():
        try:
            with open(PREFERENCES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserPreferences.from_dict(data)
        except Exception:
            return UserPreferences()
    return UserPreferences()


def save_preferences(prefs: UserPreferences) -> None:
    """Persist user preferences to JSON."""
    try:
        with open(PREFERENCES_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs.to_dict(), f, indent=2)
    except Exception as e:
        # Avoid crashing app if filesystem write fails
        print(f"Warning: Failed to save preferences: {e}")


def reset_preferences() -> UserPreferences:
    """Reset preferences to factory defaults and save."""
    prefs = UserPreferences()
    save_preferences(prefs)
    return prefs


def is_llm_configured() -> bool:
    """Check if a valid LLM API key is present without exposing the key."""
    return bool(GEMINI_API_KEY)


def is_web_search_configured() -> bool:
    """Check if web search has necessary configuration."""
    if WEB_SEARCH_PROVIDER == "duckduckgo":
        return True
    if WEB_SEARCH_PROVIDER == "tavily":
        return bool(TAVILY_API_KEY)
    return False
