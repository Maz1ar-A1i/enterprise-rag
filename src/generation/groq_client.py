"""
groq_client.py
==============
Thin wrapper around the Groq API with:
  - Rate limiting (stays under free-tier 30 RPM)
  - Automatic retry with exponential backoff (handles 429 errors)
  - Response caching to disk (prevents re-calling API on identical inputs)
  - Structured JSON response parsing

Models used:
  Generation : llama-3.3-70b-versatile  (128k ctx, fast on Groq LPU)
  Judge calls : llama-3.1-8b-instant    (cheaper/faster for evaluation scoring)
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GENERATE_MODEL = os.getenv("GROQ_GENERATE_MODEL", "openai/gpt-oss-20b")
DEFAULT_JUDGE_MODEL    = os.getenv("GROQ_JUDGE_MODEL",    "openai/gpt-oss-20b")

# Free tier: 30 req/min = 1 request every 2 seconds (we use 2.2s to be safe)
_MIN_REQUEST_INTERVAL_SEC = 2.2


# ---------------------------------------------------------------------------
# GroqClient
# ---------------------------------------------------------------------------

class GroqClient:
    """
    Groq API client with rate limiting, retry, and caching.

    Args:
        cache_dir:  Directory for disk-based response cache. Set to None to disable.
        api_key:    Groq API key. Defaults to GROQ_API_KEY env var.
    """

    def __init__(self, cache_dir: Optional[str] = "logs/groq_cache", api_key: str = None):
        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self._last_call_time = 0.0
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Main call ──────────────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        """
        Make a chat completion call to Groq.

        Args:
            system_prompt: Instruction context for the LLM.
            user_message:  The user's message / question.
            model:         Groq model name (defaults to generate model).
            temperature:   Sampling temperature.
            max_tokens:    Max output tokens.
            use_cache:     Whether to check/write disk cache.

        Returns:
            String response from the LLM.
        """
        model = model or DEFAULT_GENERATE_MODEL

        # Cache lookup
        if use_cache and self._cache_dir:
            cache_key = self._make_cache_key(system_prompt, user_message, model, temperature)
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        # Rate limiting — wait if we called too recently
        self._rate_limit_wait()

        # API call with retry
        response_text = self._call_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Cache store
        if use_cache and self._cache_dir:
            self._cache_set(cache_key, response_text)

        return response_text

    # ── Rate limiting ──────────────────────────────────────────────────────

    def _rate_limit_wait(self) -> None:
        """Sleep just enough to stay under the free-tier RPM limit."""
        elapsed = time.monotonic() - self._last_call_time
        wait   = _MIN_REQUEST_INTERVAL_SEC - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    # ── Retry logic ────────────────────────────────────────────────────────

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def _call_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Inner API call; tenacity retries on any exception."""
        completion = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    # ── JSON parsing helper ────────────────────────────────────────────────

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Optional[dict]:
        """
        Like call(), but parses the response as JSON.
        Falls back to None if the response isn't valid JSON.
        """
        model = model or DEFAULT_JUDGE_MODEL
        raw = self.call(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON array or object from within the response
            import re
            match_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match_arr:
                try:
                    return json.loads(match_arr.group(0))
                except Exception:
                    pass
            match_obj = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match_obj:
                try:
                    return json.loads(match_obj.group(0))
                except Exception:
                    pass
            log.warning(f"Could not parse JSON from LLM response: {raw[:200]}")
            return None

    # ── Disk cache ─────────────────────────────────────────────────────────

    def _make_cache_key(self, *args) -> str:
        combined = "|".join(str(a) for a in args)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[str]:
        path = self._cache_dir / f"{key}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if content and content.strip():
                return content
        return None

    def _cache_set(self, key: str, value: str) -> None:
        path = self._cache_dir / f"{key}.txt"
        path.write_text(value, encoding="utf-8")
