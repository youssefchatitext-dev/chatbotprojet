from __future__ import annotations

from typing import Literal

from agents.local_agent import LocalAgent
from agents.ollama_agent import OllamaAgent
from models.portfolio_models import AgentResponse


class AgentRouter:
    def __init__(self) -> None:
        self.local_agent = LocalAgent()
        self.ollama_agent = OllamaAgent()

    def route(self, mode: Literal["local", "ollama"], user_input: str, **kwargs) -> AgentResponse:
        if mode == "ollama":
            return self.ollama_agent.run(user_input, **kwargs)
        return self.local_agent.run(user_input, **kwargs)
