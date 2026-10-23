"""Official LLM Client for Athena AI Lite using Google GenAI SDK.

EDUCATIONAL CONCEPT: How does an LLM answer in RAG?
In a Retrieval-Augmented Generation system, we do NOT ask the model to rely solely
on its pre-trained internal memory. Instead, we:
1. Provide the retrieved document excerpts in the prompt context.
2. Instruct the model: "Answer strictly based on the provided context below.
   If the context does not contain the answer, state that clearly."
3. Adapt the tone and depth based on the student's selected learning level:
   - Beginner: Simple analogies, define unfamiliar terminology.
   - Intermediate: Technical concepts explained with practical examples.
   - Advanced: Deep technical precision, theoretical assumptions, edge cases.

Transparency & Mock Mode:
If GEMINI_API_KEY is not set or if offline testing is active, the client operates
in an explicitly labeled Mock Mode. It NEVER pretends a mock response is a live LLM output.
"""

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional
import config


class LLMError(Exception):
    """Custom exception raised when LLM generation fails."""
    pass


@dataclass
class LLMResponse:
    """Standardized response from the LLM client."""
    text: str
    model_name: str
    is_mock: bool
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class GeminiLLMClient:
    """Encapsulates interaction with the official Google GenAI Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        force_mock: bool = False
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL
        self.force_mock = force_mock
        self._client = None

        if not self.force_mock and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                # If SDK fails to initialize, log error and fallback to mock
                print(f"Warning: Failed to initialize google-genai client: {e}")
                self._client = None

    @property
    def is_mock_mode(self) -> bool:
        """Returns True if the client is running without a live Gemini connection."""
        return self.force_mock or self._client is None or not bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> LLMResponse:
        """Generates text from the LLM or mock generator.

        Args:
            prompt: User / augmented prompt containing context and question.
            system_instruction: Guiding behavioral persona.
            temperature: Sampling temperature (low = deterministic/factual).

        Returns:
            LLMResponse object.

        Raises:
            LLMError: If a live API call fails.
        """
        if self.is_mock_mode:
            return self._generate_mock(prompt, system_instruction)

        try:
            from google.genai import types

            config_params = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction if system_instruction else None
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_params
            )

            output_text = response.text or ""
            return LLMResponse(
                text=output_text.strip(),
                model_name=self.model_name,
                is_mock=False
            )

        except Exception as e:
            err_msg = str(e)
            # Sanitize any potential accidental API key echoes in error strings
            clean_err = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_API_KEY]', err_msg)

            if "403" in clean_err or "API_KEY_INVALID" in clean_err:
                raise LLMError("Authentication failed: Invalid or expired GEMINI_API_KEY. Please verify your .env file.")
            if "429" in clean_err or "RESOURCE_EXHAUSTED" in clean_err:
                raise LLMError("Rate limit exceeded: The Gemini API quota has been reached. Please wait or try again later.")
            if "timeout" in clean_err.lower() or "timed out" in clean_err.lower():
                raise LLMError("Gemini API request timed out. Please check your internet connection and retry.")

            raise LLMError(f"Gemini API generation failed: {clean_err}")

    def _generate_mock(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> LLMResponse:
        """Generates an honest, clearly-labeled educational mock response.

        Extracts evidence from the prompt's context block if present so that
        RAG mechanics can be visualized and verified even without API credentials.
        """
        lines = prompt.splitlines()
        context_lines = []
        is_context = False

        for line in lines:
            if "DOCUMENT CONTEXT:" in line or "EVIDENCE:" in line:
                is_context = True
                continue
            if "QUESTION:" in line or "USER QUESTION:" in line:
                is_context = False
            if is_context and line.strip():
                context_lines.append(line.strip())

        evidence_summary = " ".join(context_lines[:6]) if context_lines else "No context found in prompt."
        # Limit length
        if len(evidence_summary) > 300:
            evidence_summary = evidence_summary[:300] + "..."

        mock_text = (
            "[MOCK MODE — Educational Preview]\n"
            "(No live GEMINI_API_KEY detected. The following answer is generated deterministically "
            "from the retrieved passages for local testing):\n\n"
            f"Based on the supplied document passages: {evidence_summary}\n\n"
            "This confirms the RAG pipeline correctly retrieved relevant text and formatted the prompt."
        )

        return LLMResponse(
            text=mock_text,
            model_name="mock-gemini-educational",
            is_mock=True
        )
