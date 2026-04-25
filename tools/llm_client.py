"""
tools/llm_client.py — Centralized LLM inference client for Butler.

Supports two providers:
1. HuggingFace Inference API (primary)  — uses HF_TOKEN
2. Cursor API (fallback)                — uses CURSOR_API_KEY

All agents import from here instead of making direct API calls.
API keys are loaded from environment variables (set in .env).
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CURSOR_MODEL = "gpt-4o-mini"
MAX_NEW_TOKENS = 256


class LLMClient:
    """
    Unified LLM client for Butler.

    Priority order:
    1. HuggingFace Inference API (if HF_TOKEN is set)
    2. Cursor API (if CURSOR_API_KEY is set)
    3. Template fallback (no API needed)

    Usage:
        client = get_llm_client()
        response = client.generate(
            system_prompt="You are Butler...",
            user_prompt="Draft a reply to...",
            max_tokens=200
        )
    """

    def __init__(self):
        self._hf_token = os.environ.get("HF_TOKEN")
        self._cursor_key = os.environ.get("CURSOR_API_KEY")
        self._hf_model = os.environ.get("HF_INFERENCE_MODEL", DEFAULT_HF_MODEL)
        self._cursor_model = os.environ.get("CURSOR_MODEL", DEFAULT_CURSOR_MODEL)
        self._hf_client = None
        self._provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Detect which LLM provider is available."""
        if self._hf_token and self._hf_token.startswith("hf_"):
            logger.info("LLM provider: HuggingFace Inference API (%s)", self._hf_model)
            return "huggingface"
        if self._cursor_key:
            logger.info("LLM provider: Cursor API (%s)", self._cursor_model)
            return "cursor"
        logger.warning("No LLM API key found. Using template fallback.")
        return "fallback"

    @property
    def provider(self) -> str:
        """Return the active provider name."""
        return self._provider

    @property
    def is_available(self) -> bool:
        """Return True if any LLM provider is configured."""
        return self._provider != "fallback"

    def reload_keys(self):
        """Reload API keys from environment (useful after .env changes)."""
        self._hf_token = os.environ.get("HF_TOKEN")
        self._cursor_key = os.environ.get("CURSOR_API_KEY")
        self._hf_model = os.environ.get("HF_INFERENCE_MODEL", DEFAULT_HF_MODEL)
        self._cursor_model = os.environ.get("CURSOR_MODEL", DEFAULT_CURSOR_MODEL)
        self._hf_client = None
        self._provider = self._detect_provider()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using the best available LLM provider.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message / question.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated text string. Empty string on failure.
        """
        if self._provider == "huggingface":
            return self._generate_hf(system_prompt, user_prompt, max_tokens, temperature)
        elif self._provider == "cursor":
            return self._generate_cursor(system_prompt, user_prompt, max_tokens, temperature)
        else:
            return ""

    def _generate_hf(
        self, system_prompt: str, user_prompt: str,
        max_tokens: int, temperature: float
    ) -> str:
        """Generate using HuggingFace Inference API."""
        try:
            from huggingface_hub import InferenceClient

            if self._hf_client is None:
                self._hf_client = InferenceClient(token=self._hf_token)

            # Use chat_completion for better structured output
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = self._hf_client.chat_completion(
                model=self._hf_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content
            logger.debug("HF response: %s chars", len(content))
            return content.strip()

        except Exception as e:
            logger.error("HuggingFace API error: %s", str(e))

            # Try fallback to Cursor if available
            if self._cursor_key:
                logger.info("Falling back to Cursor API...")
                return self._generate_cursor(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            return ""

    def _generate_cursor(
        self, system_prompt: str, user_prompt: str,
        max_tokens: int, temperature: float
    ) -> str:
        """
        Generate using Cursor API (OpenAI-compatible endpoint).

        Cursor uses an OpenAI-compatible API format.
        """
        try:
            import urllib.request
            import urllib.error

            url = "https://api.cursor.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._cursor_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._cursor_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            content = result["choices"][0]["message"]["content"]
            logger.debug("Cursor response: %s chars", len(content))
            return content.strip()

        except Exception as e:
            logger.error("Cursor API error: %s", str(e))
            return ""

    def get_status(self) -> dict:
        """Return status info about the LLM client (never includes keys)."""
        return {
            "provider": self._provider,
            "hf_model": self._hf_model if self._provider == "huggingface" else None,
            "cursor_model": self._cursor_model if self._provider == "cursor" else None,
            "hf_token_set": bool(self._hf_token),
            "cursor_key_set": bool(self._cursor_key),
            "is_available": self.is_available,
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client():
    """Force re-creation of the LLM client (e.g. after env var changes)."""
    global _llm_client
    _llm_client = None
