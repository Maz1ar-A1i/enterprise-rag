"""
src/generation/ollama_client.py
================================
Local LLM client using Ollama's HTTP API (http://localhost:11434).

Provides an identical interface to GroqClient:
  - call(system_prompt, user_message, model, temperature, max_tokens) -> str
  - call_json(system_prompt, user_message, model, temperature, max_tokens) -> Optional[dict]
  - is_available() -> bool
"""

import os
import json
import logging
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class OllamaClient:
    """
    Local LLM Client talking to Ollama.
    Drop-in replacement for GroqClient.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 180.0,
    ):
        self.host = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.default_model = model or DEFAULT_OLLAMA_MODEL
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        """Return list of available model names in Ollama."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            log.warning(f"[Ollama] Failed to list models: {e}")
        return []

    def call(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        use_cache: bool = False,
    ) -> str:
        """
        Send chat completion request to Ollama.
        """
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": min(max_tokens, 400),
                "num_ctx": 2048,
                "num_thread": 8,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.host}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                content = data.get("message", {}).get("content", "").strip()
                return content
        except Exception as e:
            log.error(f"[Ollama] Generation failed ({e})")
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    def call_json(
        self,
        system_prompt: str,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Optional[dict]:
        """
        Execute call and parse response as JSON.
        """
        raw = self.call(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(f"[Ollama] JSON decode failed for response: {raw[:200]}")
            return None
