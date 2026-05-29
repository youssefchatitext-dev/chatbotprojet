from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    intents: list[tuple[str, float]] = field(default_factory=list)


class IntentEngine:
    INTENT_PATTERNS: dict[str, list[str]] = {
        "greeting": ["bonjour", "salut", "hello", "coucou", "salam", "bonsoir", "hey"],
        "smalltalk": ["ca va", "cava", "comment ca va", "tu vas bien", "quoi de neuf", "labas", "labass"],
        "help_request": ["aide", "help", "comment faire", "instructions", "exemple", "que peux tu faire"],
        "educational_question": ["explique", "definition", "c est quoi", "qu est ce que", "apprendre", "cours"],
        "portfolio_creation": ["portefeuille", "allocation", "investir", "composer", "repartition", "position"],
        "optimization_request": ["optimisation", "optimiser", "meilleur", "max sharpe", "minimum volatilite", "frontiere"],
        "diversification_request": ["diversification", "diversifier", "concentration", "hhi", "nombre effectif"],
        "risk_analysis": ["risque", "risqu", "risk", "perte", "pertes", "var", "cvar", "drawdown"],
        "allocation_explanation": ["allocation", "poids", "repartition", "composition", "exposition"],
        "performance_analysis": ["performance", "rendement", "return", "gain", "profit", "sortino"],
        "portfolio_comparison": ["compare", "comparer", "comparaison", "versus", "vs", "difference"],
        "rebalancing_request": ["reequilibrage", "rebalancing", "rebalancer", "ajuster", "frequence"],
        "market_question": ["marche", "bourse", "casablanca", "masi", "plage", "range", "tendance"],
        "sector_preference": ["secteur", "banque", "energie", "immobilier", "tech", "industrie", "sante"],
        "investment_horizon": ["horizon", "long terme", "court terme", "moyen terme", "ans", "annees"],
        "capital_allocation": ["capital", "budget", "montant", "mad", "dh", "dirham", "investir"],
        "growth_strategy": ["croissance", "growth", "valorisation", "plus value", "capitalisation"],
        "dividend_strategy": ["dividende", "dividendes", "revenu", "income", "cash flow", "rendement distribue"],
        "volatility_question": ["volatilite", "volatilitee", "volatilit", "sigma", "ecart type"],
        "sharpe_question": ["sharpe", "sharpr", "ratio sharpe"],
        "efficient_frontier_question": ["frontiere efficace", "frontiere efficiente", "efficient frontier"],
        "drawdown_question": ["drawdown", "baisse maximale", "perte maximale"],
        "beginner_question": ["debutant", "simple", "facile", "vulgarise", "je commence"],
        "advanced_finance_question": ["avance", "professionnel", "quant", "covariance", "correlation", "risk parity"],
    }

    SECTOR_ALIASES: dict[str, list[str]] = {
        "technologie": ["tech", "technologie", "informatique", "digital"],
        "banque": ["banque", "bancaire", "financier", "finance"],
        "immobilier": ["immobilier", "real estate", "fonciere"],
        "energie": ["energie", "petrole", "gaz", "utilities"],
        "sante": ["sante", "pharma", "medical"],
        "industrie": ["industrie", "industriel", "materiaux", "construction"],
        "consommation": ["consommation", "retail", "distribution", "agro"],
        "telecom": ["telecom", "telecommunications"],
    }

    TYPO_ALIASES = {
        "portfeuille": "portefeuille",
        "portefeuille": "portefeuille",
        "risqu": "risque",
        "volatilitee": "volatilite",
        "volatilit": "volatilite",
        "sharpr": "sharpe",
        "dividande": "dividende",
        "optmisation": "optimisation",
        "diversifiction": "diversification",
    }

    def analyze(self, user_input: str, context: str | None = None) -> IntentResult:
        normalized = self.normalize(user_input)
        entities = self.extract_entities(user_input)
        intents = self._classify_multi(normalized)
        primary, confidence = intents[0] if intents else ("unknown", 0.0)
        if "portfolio" in normalized or "portefeuille" in normalized:
            for candidate, candidate_confidence in intents:
                if candidate == "portfolio_creation":
                    primary, confidence = candidate, max(confidence, candidate_confidence)
                    break
        return IntentResult(intent=primary, confidence=confidence, entities=entities, intents=intents)

    def _classify_multi(self, text: str) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        tokens = text.split()
        for intent, patterns in self.INTENT_PATTERNS.items():
            raw_score = 0.0
            for pattern in patterns:
                normalized_pattern = self.normalize(pattern)
                if normalized_pattern in text:
                    raw_score += 1.0
                    continue
                pattern_tokens = normalized_pattern.split()
                if len(pattern_tokens) == 1:
                    raw_score += max((self._similarity(token, normalized_pattern) for token in tokens), default=0.0) * 0.75
                else:
                    raw_score += self._similarity(text, normalized_pattern) * 0.65
            if raw_score > 0.48:
                confidence = min(0.99, raw_score / max(2.2, len(patterns) * 0.65))
                scores.append((intent, round(confidence, 3)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:5]

    def extract_entities(self, user_input: str) -> dict[str, Any]:
        text = self.normalize(user_input)
        percentages = self._extract_percentages(user_input)
        entities: dict[str, Any] = {
            "risk_tolerance": self._extract_risk_tolerance(text),
            "investment_horizon": self._extract_horizon(text),
            "target_return": self._extract_contextual_percentage(text, percentages, ["rendement", "return", "gain", "profit", "objectif"]),
            "target_volatility": self._extract_contextual_percentage(text, percentages, ["risque", "risk", "volatilite", "sigma"]),
            "sector_preferences": self._extract_sectors(text),
            "forbidden_assets": self._extract_assets_after(text, ["sans", "exclure", "eviter", "interdire"]),
            "preferred_assets": self._extract_assets_after(text, ["avec", "inclure", "preferer", "privilegier"]),
            "investment_amount": self._extract_currency(user_input),
            "dividend_preference": self._has_any(text, ["dividende", "revenu", "income"]),
            "growth_preference": self._has_any(text, ["croissance", "growth", "plus value"]),
            "income_preference": self._has_any(text, ["revenu", "income", "cash flow"]),
            "rebalance_frequency": self._extract_rebalance_frequency(text),
            "liquidity_preference": self._extract_liquidity_preference(text),
            "response_style": self._extract_response_style(text),
            "optimization_mode": self._extract_optimization_mode(text),
        }
        if entities["target_return"] is None and len(percentages) >= 2:
            entities["target_return"] = percentages[0]
        if entities["target_volatility"] is None and len(percentages) >= 2:
            entities["target_volatility"] = percentages[1]
        return entities

    def _extract_risk_tolerance(self, text: str) -> str | None:
        if self._has_any(text, ["faible risque", "low risk", "prudent", "defensif", "conservateur", "stable"]):
            return "low"
        if self._has_any(text, ["risque eleve", "high risk", "agressif", "offensif", "dynamique"]):
            return "high"
        if self._has_any(text, ["equilibre", "modere", "balanced", "moyen risque"]):
            return "medium"
        return None

    def _extract_horizon(self, text: str) -> int | None:
        match = re.search(r"(\d{1,2})\s*(?:ans|annees|years|year)", text)
        if match:
            return int(match.group(1))
        if "court terme" in text:
            return 1
        if "moyen terme" in text:
            return 5
        if "long terme" in text:
            return 10
        return None

    def _extract_sectors(self, text: str) -> list[str]:
        sectors = []
        for sector, aliases in self.SECTOR_ALIASES.items():
            if self._has_any(text, aliases):
                sectors.append(sector)
        return sectors

    def _extract_contextual_percentage(self, text: str, percentages: list[float], keywords: list[str]) -> float | None:
        for keyword in keywords:
            pattern = rf"{keyword}\w*\s*(?:cible|target|objectif|max|maximum|<=|>=|=|:)?\s*(\d+(?:[.,]\d+)?)\s*%?"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1).replace(",", "."))
        for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%", text):
            before = text[max(0, match.start() - 28):match.start()]
            after = text[match.end():match.end() + 28]
            if any(keyword in before or keyword in after for keyword in keywords):
                return float(match.group(1).replace(",", "."))
        return percentages[0] if len(percentages) == 1 and any(k in text for k in keywords) else None

    @staticmethod
    def _extract_percentages(user_input: str) -> list[float]:
        return [float(value.replace(",", ".")) for value in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", user_input)]

    @staticmethod
    def _extract_currency(user_input: str) -> float | None:
        match = re.search(r"(\d{1,3}(?:[ .]?\d{3})*(?:[.,]\d+)?)\s*(?:mad|dh|dhs|dirhams?)", user_input, flags=re.I)
        if match:
            return float(match.group(1).replace(" ", "").replace(".", "").replace(",", "."))
        match = re.search(r"(?:capital|budget|montant|investir)\s*(?:de)?\s*(\d{3,}(?:[.,]\d+)?)", user_input, flags=re.I)
        return float(match.group(1).replace(",", ".")) if match else None

    def _extract_assets_after(self, text: str, markers: list[str]) -> list[str]:
        assets: list[str] = []
        for marker in markers:
            for match in re.finditer(rf"{marker}\s+([a-z0-9, ;_-]+)", text):
                fragment = match.group(1)
                for token in re.split(r"[,; ]+", fragment):
                    if 2 <= len(token) <= 6 and token not in {"les", "des", "actions", "secteur"}:
                        assets.append(token.upper())
        return list(dict.fromkeys(assets))

    @staticmethod
    def _extract_rebalance_frequency(text: str) -> str | None:
        if any(word in text for word in ["mensuel", "monthly", "mois"]):
            return "monthly"
        if any(word in text for word in ["trimestriel", "quarterly", "trimestre"]):
            return "quarterly"
        if any(word in text for word in ["annuel", "yearly", "annee"]):
            return "yearly"
        return None

    @staticmethod
    def _extract_liquidity_preference(text: str) -> str | None:
        if any(word in text for word in ["liquide", "liquidite", "facile a vendre"]):
            return "high"
        if any(word in text for word in ["long terme", "illiquide"]):
            return "low"
        return None

    @staticmethod
    def _extract_response_style(text: str) -> str | None:
        if any(word in text for word in ["debutant", "simple", "facile"]):
            return "beginner"
        if any(word in text for word in ["avance", "professionnel", "quant", "technique"]):
            return "advanced"
        if any(word in text for word in ["court", "bref", "resume"]):
            return "concise"
        return None

    @staticmethod
    def _extract_optimization_mode(text: str) -> str | None:
        mode_aliases = {
            "maximum_sharpe": ["max sharpe", "meilleur sharpe", "sharpe maximum"],
            "minimum_volatility": ["minimum volatilite", "min volatilite", "faible volatilite"],
            "risk_parity": ["risk parity", "parite de risque", "equal risk"],
            "diversification": ["diversification maximale", "max diversification"],
            "balanced": ["equilibre", "balanced"],
            "equal_weight": ["equal weight", "poids egaux"],
        }
        for mode, aliases in mode_aliases.items():
            if any(alias in text for alias in aliases):
                return mode
        return None

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        return difflib.SequenceMatcher(None, left, right).ratio()

    @classmethod
    def _has_any(cls, text: str, keywords: list[str]) -> bool:
        tokens = text.split()
        for keyword in keywords:
            normalized = cls.normalize(keyword)
            if normalized in text:
                return True
            if any(cls._similarity(token, normalized) >= 0.82 for token in tokens):
                return True
        return False

    @classmethod
    def normalize(cls, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower().strip())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9%.,\s_-]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        for typo, replacement in cls.TYPO_ALIASES.items():
            normalized = re.sub(rf"\b{typo}\b", replacement, normalized)
        return normalized
