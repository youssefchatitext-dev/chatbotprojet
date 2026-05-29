from __future__ import annotations

from typing import Any

from models.portfolio_models import AgentResponse, PortfolioResult
from services.ollama_service import OllamaService
from agents.local_agent import LocalAgent
from agents.base_agent import BaseAgent
from utils.prompts import EXPLANATION_PROMPT, SYSTEM_PROMPT


class OllamaAgent(BaseAgent):
    def __init__(self, model: str | None = None) -> None:
        self.local_agent = LocalAgent()
        self.ollama = OllamaService(model=model) if model else OllamaService()

    def run(self, user_input: str, **kwargs: Any) -> AgentResponse:
        deterministic = self.local_agent.run(user_input, **kwargs)
        if deterministic.structured is not None:
            explanation = self._explain_portfolio(deterministic.structured, user_input)
            return AgentResponse(
                content=explanation,
                structured=deterministic.structured,
                explanation=explanation,
                metadata={"mode": "ollama"},
            )

        if self._is_portfolio_request(user_input):
            return deterministic

        if self.ollama.is_available():
            return AgentResponse(
                content=self._chat_with_ollama(user_input),
                structured=None,
                metadata={"mode": "ollama"},
            )

        return deterministic

    def _is_portfolio_request(self, user_input: str) -> bool:
        normalized = user_input.lower()
        keywords = [
            "portefeuille",
            "rendement",
            "risque",
            "allocation",
            "allocation",
            "actions",
            "investissement",
            "performance",
            "volatilité",
            "volatilite",
            "sharpe",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _explain_portfolio(self, portfolio: PortfolioResult, user_input: str | None = None) -> str:
        if not self.ollama.is_available():
            return self._fallback_explanation(portfolio)
        metrics = portfolio.metrics
        allocations_lines = [
            f"- {ticker}: {weight * 100:.2f}%"
            for ticker, weight in portfolio.allocations.items()
        ]
        summary = (
            "Voici les données déterministes disponibles :\n"
            f"Rendement attendu : {metrics.expected_return * 100:.2f}%\n"
            f"Volatilité : {metrics.volatility * 100:.2f}%\n"
            f"Sharpe : {metrics.sharpe_ratio:.2f}\n"
            f"Sortino : {metrics.sortino_ratio:.2f}\n"
            f"Max Drawdown : {metrics.max_drawdown * 100:.2f}%\n"
            f"VaR 95% : {metrics.var_95 * 100:.2f}%\n"
            f"CVaR 95% : {metrics.cvar_95 * 100:.2f}%\n"
            f"Diversification : {metrics.diversification_score:.2f}%\n"
            "Allocations :\n"
            + "\n".join(allocations_lines)
            + "\n"
        )
        user_context = f"\n\nQuestion utilisateur : {user_input}" if user_input else ""
        messages = [
            {
                "role": "user",
                "content": EXPLANATION_PROMPT + "\n\n" + summary + user_context,
            },
        ]
        try:
            return self.ollama.chat(messages, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            return self._fallback_explanation(portfolio) + f"\n\n[Remarque : Ollama indisponible - {exc}]"

    def _chat_with_ollama(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        try:
            return self.ollama.chat(messages, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            fallback = self.local_agent.run(user_input)
            return (
                fallback.content
                + f"\n\n[Remarque : Ollama indisponible - {exc}]"
            )

    def _fallback_explanation(self, portfolio: PortfolioResult) -> str:
        metrics = portfolio.metrics
        explanation_lines = [
            "Je suis un agent local expliquant les résultats déterministes du moteur financier.",
            f"Ce portefeuille vise un rendement attendu de {metrics.expected_return * 100:.2f}% avec une volatilité de {metrics.volatility * 100:.2f}%.",
            f"Le Sharpe ratio est de {metrics.sharpe_ratio:.2f}, ce qui indique le niveau de rendement ajusté au risque par rapport au taux sans risque.",
            f"La diversification du portefeuille est mesurée à {metrics.diversification_score:.2f}, donc les poids ne sont pas concentrés sur un seul actif.",
            "Les allocations sont calculées à partir de retours annualisés géométriques et d'une matrice de covariance stabilisée.",
        ]
        return "\n".join(explanation_lines)
