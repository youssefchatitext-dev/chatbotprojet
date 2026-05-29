from __future__ import annotations

from abc import ABC, abstractmethod

from models.portfolio_models import AgentResponse


class BaseAgent(ABC):
    @abstractmethod
    def run(self, user_input: str, **kwargs) -> AgentResponse:
        raise NotImplementedError
