from __future__ import annotations

from typing import Any

from utils.config import DEFAULT_OLLAMA_MODEL
from utils.prompts import SYSTEM_PROMPT

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


class OllamaService:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL) -> None:
        self.model = model

    def is_available(self) -> bool:
        if ollama is None:
            return False
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def chat(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        if ollama is None:
            raise ImportError("Ollama is not installed. Install it with `pip install ollama`")
        payload = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}] + messages
        response = ollama.chat(model=self.model, messages=payload)
        if hasattr(response, "message") and getattr(response.message, "content", None):
            return response.message.content
        if isinstance(response, dict) and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            return choice.get("message", {}).get("content", "") or choice.get("content", "")
        return str(response)
