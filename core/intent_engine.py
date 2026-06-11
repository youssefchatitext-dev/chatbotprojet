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
        "greeting": ["bonjour", "salut", "hello", "coucou", "salam", "bonsoir", "hey", "hy", "bjr", "slt"],
        "smalltalk": ["ca va", "cava", "comment ca va", "tu vas bien", "quoi de neuf", "labas", "labass", "tu fais quoi"],
        "goodbye": ["au revoir", "a bientot", "bye", "goodbye", "ciao", "a plus", "a plus tard", "bonne journee", "bonne nuit"],
        "thanks": ["merci", "thanks", "thank you", "parfait", "super", "excellent", "bravo"],
        "help_request": ["aide", "help", "comment faire", "instructions", "exemple", "que peux tu faire", "guide", "aider", "montre moi", "mode emploi"],
        "educational_question": ["explique", "definition", "c est quoi", "qu est ce que", "apprendre", "cours", "vulgarise", "definis", "signifie", "notion", "concept"],
        "portfolio_creation": ["portefeuille", "portfolio", "allocation", "investir", "composer", "repartition", "position", "strategie", "placer", "construire", "creer", "propose", "donne moi un portefeuille"],
        "optimization_request": ["optimisation", "optimiser", "meilleur", "max sharpe", "maximum sharpe", "minimum volatilite", "min variance", "frontiere", "optimal", "optimale", "trouve le meilleur", "slsqp"],
        "diversification_request": ["diversification", "diversifier", "concentration", "hhi", "nombre effectif", "repartir", "moins concentre", "pas concentre", "equilibrer les poids"],
        "risk_analysis": ["risque", "risqu", "risk", "perte", "pertes", "var", "cvar", "drawdown", "danger", "securite", "risque maximal", "risque maximum", "limite de risque"],
        "allocation_explanation": ["allocation", "poids", "repartition", "composition", "exposition", "pourcentage", "parts"],
        "performance_analysis": ["performance", "rendement", "return", "gain", "profit", "sortino", "rentabilite", "objectif de rendement", "rendement cible", "retour attendu"],
        "portfolio_comparison": ["compare", "comparer", "comparaison", "versus", "vs", "difference", "choisir entre"],
        "rebalancing_request": ["reequilibrage", "rebalancing", "rebalancer", "ajuster", "frequence", "rebalance"],
        "market_question": ["marche", "bourse", "casablanca", "masi", "plage", "range", "tendance", "actions marocaines"],
        "sector_preference": ["secteur", "banque", "energie", "immobilier", "tech", "industrie", "sante", "telecom", "consommation", "ciment", "mines", "assurance"],
        "investment_horizon": ["horizon", "long terme", "court terme", "moyen terme", "ans", "annees", "mois", "duree", "periode"],
        "capital_allocation": ["capital", "budget", "montant", "mad", "dh", "dirham", "investir", "combien placer", "somme", "cash"],
        "growth_strategy": ["croissance", "growth", "valorisation", "plus value", "capitalisation", "dynamique", "offensif"],
        "dividend_strategy": ["dividende", "dividendes", "revenu", "income", "cash flow", "rendement distribue", "revenus reguliers"],
        "volatility_question": ["volatilite", "volatilitee", "volatilit", "sigma", "ecart type"],
        "sharpe_question": ["sharpe", "sharpr", "ratio sharpe", "rendement ajuste au risque"],
        "efficient_frontier_question": ["frontiere efficace", "frontiere efficiente", "efficient frontier"],
        "drawdown_question": ["drawdown", "baisse maximale", "perte maximale"],
        "beginner_question": ["debutant", "simple", "facile", "vulgarise", "je commence", "explique simplement", "niveau debutant"],
        "advanced_finance_question": ["avance", "professionnel", "quant", "covariance", "correlation", "risk parity", "ledoit wolf", "markowitz", "monte carlo"],
    }

    SECTOR_ALIASES: dict[str, list[str]] = {
        "technologie": ["tech", "technologie", "informatique", "digital"],
        "banque": ["banque", "bancaire", "financier", "finance"],
        "immobilier": ["immobilier", "real estate", "fonciere", "foncier"],
        "energie": ["energie", "petrole", "gaz", "utilities", "electricite"],
        "sante": ["sante", "pharma", "medical", "clinique"],
        "industrie": ["industrie", "industriel", "materiaux", "construction", "ciment", "acier"],
        "consommation": ["consommation", "retail", "distribution", "agro", "agroalimentaire"],
        "telecom": ["telecom", "telecommunications"],
        "assurance": ["assurance", "assurances"],
        "mines": ["mine", "mines", "minier"],
        "transport": ["transport", "logistique"],
    }

    TYPO_ALIASES = {
        "aurevoir": "au revoir",
        "portfeuille": "portefeuille",
        "portefuille": "portefeuille",
        "portefeuillee": "portefeuille",
        "porte feuille": "portefeuille",
        "portefeuille": "portefeuille",
        "portfoli": "portfolio",
        "portfolo": "portfolio",
        "portflio": "portfolio",
        "rendemet": "rendement",
        "rendment": "rendement",
        "rendemnt": "rendement",
        "rendemen": "rendement",
        "rentablite": "rentabilite",
        "rentabilitee": "rentabilite",
        "risqu": "risque",
        "risq": "risque",
        "risquee": "risque",
        "riskque": "risque",
        "riske": "risque",
        "volatilitee": "volatilite",
        "volatilit": "volatilite",
        "volatilte": "volatilite",
        "volalite": "volatilite",
        "sharpr": "sharpe",
        "sharp": "sharpe",
        "sortnio": "sortino",
        "drawdon": "drawdown",
        "drowdown": "drawdown",
        "dividande": "dividende",
        "dividente": "dividende",
        "optmisation": "optimisation",
        "optimisaton": "optimisation",
        "optimzer": "optimiser",
        "optimser": "optimiser",
        "optimse": "optimise",
        "diversifiction": "diversification",
        "diversifcation": "diversification",
        "diversifier": "diversifier",
        "alocation": "allocation",
        "allocaton": "allocation",
        "alloction": "allocation",
        "repartion": "repartition",
        "repartiton": "repartition",
        "reequilibrage": "reequilibrage",
        "reequilibrer": "reequilibrer",
        "rebalencage": "rebalancing",
        "capitale": "capital",
        "montan": "montant",
        "montent": "montant",
        "horizen": "horizon",
        "horizont": "horizon",
        "longterm": "long terme",
        "courtterm": "court terme",
        "bonjoure": "bonjour",
        "salu": "salut",
        "slt": "salut",
        "bjr": "bonjour",
        "mercii": "merci",
        "mercie": "merci",
    }

    PHRASE_ALIASES = {
        "faible risq": "faible risque",
        "bas risque": "faible risque",
        "peu risque": "faible risque",
        "risque bas": "faible risque",
        "risque max": "risque maximal",
        "risque maxi": "risque maximal",
        "rendement max": "rendement maximal",
        "rendement mini": "rendement minimal",
        "je veut": "je veux",
        "j veux": "je veux",
        "donne moi": "donne moi",
        "c est koi": "c est quoi",
        "c koi": "c est quoi",
        "quest ce que": "qu est ce que",
    }

    def analyze(self, user_input: str, context: str | None = None) -> IntentResult:
        normalized = self.normalize(user_input)
        entities = self.extract_entities(user_input)
        intents = self._classify_multi(normalized)
        intents = self._apply_entity_boosts(intents, entities, normalized, context)
        primary, confidence = intents[0] if intents else ("unknown", 0.0)
        if "portfolio" in normalized or "portefeuille" in normalized:
            for candidate, candidate_confidence in intents:
                if candidate == "portfolio_creation":
                    primary, confidence = candidate, max(confidence, candidate_confidence)
                    break
            intents = self._promote_intent(intents, "portfolio_creation", confidence)
        return IntentResult(intent=primary, confidence=confidence, entities=entities, intents=intents)

    def _classify_multi(self, text: str) -> list[tuple[str, float]]:
        scores: list[tuple[str, float]] = []
        tokens = text.split()
        for intent, patterns in self.INTENT_PATTERNS.items():
            raw_score = 0.0
            for pattern in patterns:
                normalized_pattern = self.normalize(pattern)
                raw_score += self._pattern_score(text, tokens, normalized_pattern)
            if raw_score > 0.42:
                confidence = min(0.99, raw_score / max(2.0, min(len(patterns), 7) * 0.72))
                scores.append((intent, round(confidence, 3)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:5]

    def _apply_entity_boosts(
        self,
        intents: list[tuple[str, float]],
        entities: dict[str, Any],
        normalized: str,
        context: str | None = None,
    ) -> list[tuple[str, float]]:
        scores = dict(intents)
        if entities.get("target_return") is not None or entities.get("target_volatility") is not None:
            scores["portfolio_creation"] = max(scores.get("portfolio_creation", 0.0), 0.62)
        if entities.get("investment_amount") is not None:
            scores["capital_allocation"] = max(scores.get("capital_allocation", 0.0), 0.58)
        if entities.get("risk_tolerance") is not None:
            scores["risk_analysis"] = max(scores.get("risk_analysis", 0.0), 0.55)
        if entities.get("investment_horizon") is not None:
            scores["investment_horizon"] = max(scores.get("investment_horizon", 0.0), 0.55)
        if entities.get("sector_preferences"):
            scores["sector_preference"] = max(scores.get("sector_preference", 0.0), 0.58)
        if entities.get("optimization_mode") is not None:
            scores["optimization_request"] = max(scores.get("optimization_request", 0.0), 0.66)
        if any(word in normalized for word in ["option", "choix"]) and re.search(r"\d+", normalized):
            scores["portfolio_comparison"] = max(scores.get("portfolio_comparison", 0.0), 0.60)
        if context and any(word in normalized for word in ["plus prudent", "plus agressif", "variante", "autre"]):
            scores["portfolio_creation"] = max(scores.get("portfolio_creation", 0.0), 0.57)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(intent, round(min(score, 0.99), 3)) for intent, score in ranked[:5]]

    @staticmethod
    def _promote_intent(intents: list[tuple[str, float]], intent_name: str, confidence: float) -> list[tuple[str, float]]:
        remaining = [(name, score) for name, score in intents if name != intent_name]
        promoted = (intent_name, round(min(confidence, 0.99), 3))
        return [promoted, *remaining][:5]

    def extract_entities(self, user_input: str) -> dict[str, Any]:
        text = self.normalize(user_input)
        percentages = self._extract_percentages(text)
        entities: dict[str, Any] = {
            "risk_tolerance": self._extract_risk_tolerance(text),
            "investment_horizon": self._extract_horizon(text),
            "target_return": self._extract_contextual_percentage(text, percentages, ["rendement", "return", "gain", "profit", "objectif", "rentabilite", "performance"]),
            "target_volatility": self._extract_contextual_percentage(text, percentages, ["risque", "risk", "volatilite", "sigma", "perte", "drawdown"]),
            "sector_preferences": self._extract_sectors(text),
            "forbidden_assets": self._extract_assets_after(text, ["sans", "exclure", "eviter", "interdire", "sauf"]),
            "preferred_assets": self._extract_assets_after(text, ["avec", "inclure", "preferer", "privilegier", "favoriser"]),
            "investment_amount": self._extract_currency(text),
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
        normalized_text = self.normalize(text)
        if self._has_any(normalized_text, ["faible risque", "risque bas", "low risk", "prudent", "defensif", "conservateur", "stable", "securise", "pas trop risque"]):
            return "faible"
        if self._has_any(normalized_text, ["risque eleve", "high risk", "agressif", "offensif", "dynamique", "haut risque", "rendement fort"]):
            return "élevé"
        if self._has_any(normalized_text, ["equilibre", "modere", "balanced", "moyen risque", "normal", "intermediaire"]):
            return "modéré"
        return None

    def _extract_horizon(self, text: str) -> int | None:
        match = re.search(r"(\d{1,2})\s*(?:ans|annees|years|year|y)\b", text)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d{1,2})\s*(?:mois|months|month|m)\b", text)
        if match:
            months = int(match.group(1))
            return max(1, round(months / 12))
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
            pattern = rf"{keyword}\w*\s*(?:cible|target|objectif|max|maximal|maximum|min|minimal|minimum|vise|souhaite|<=|>=|=|:)?\s*(\d+(?:[.,]\d+)?)\s*%?"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1).replace(",", "."))
            reverse_pattern = rf"(\d+(?:[.,]\d+)?)\s*%?\s*(?:de|en|pour)?\s*{keyword}\w*"
            match = re.search(reverse_pattern, text)
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
        match = re.search(r"(?:capital|budget|montant|investir|placer|somme)\s*(?:de|avec|a)?\s*(\d{3,}(?:[ .]?\d{3})*(?:[.,]\d+)?)", user_input, flags=re.I)
        if match:
            return float(match.group(1).replace(" ", "").replace(".", "").replace(",", "."))
        return None

    def _extract_assets_after(self, text: str, markers: list[str]) -> list[str]:
        assets: list[str] = []
        for marker in markers:
            for match in re.finditer(rf"{marker}\s+([a-z0-9, ;_-]+)", text):
                fragment = match.group(1)
                for token in re.split(r"[,; ]+", fragment):
                    if (
                        2 <= len(token) <= 6
                        and token not in {"les", "des", "actions", "secteur", "mad", "dh", "dhs", "dirham", "dirhams"}
                        and not token.isdigit()
                    ):
                        assets.append(token.upper())
        return list(dict.fromkeys(assets))

    @staticmethod
    def _extract_rebalance_frequency(text: str) -> str | None:
        if any(word in text for word in ["mensuel", "monthly", "chaque mois"]):
            return "monthly"
        if any(word in text for word in ["trimestriel", "quarterly", "trimestre", "chaque trimestre"]):
            return "quarterly"
        if any(word in text for word in ["annuel", "yearly", "chaque an", "chaque annee"]):
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
            "maximum_sharpe": ["max sharpe", "maximum sharpe", "meilleur sharpe", "sharpe maximum", "rendement ajuste au risque"],
            "minimum_volatility": ["minimum volatilite", "min volatilite", "faible volatilite", "minimum variance", "min variance"],
            "risk_parity": ["risk parity", "parite de risque", "equal risk", "risque egal"],
            "diversification": ["diversification maximale", "max diversification", "moins concentre", "diversifier au maximum"],
            "balanced": ["equilibre", "balanced", "compromis", "rendement risque"],
            "equal_weight": ["equal weight", "poids egaux", "equipondere", "egalement reparti"],
        }
        for mode, aliases in mode_aliases.items():
            for alias in aliases:
                normalized_alias = IntentEngine.normalize(alias)
                if " " in normalized_alias:
                    if normalized_alias in text:
                        return mode
                elif IntentEngine._has_any(text, [normalized_alias]):
                    return mode
        return None

    @classmethod
    def _pattern_score(cls, text: str, tokens: list[str], pattern: str) -> float:
        if not pattern:
            return 0.0
        if pattern in text:
            return 1.25 if " " in pattern else 1.0

        pattern_tokens = pattern.split()
        if len(pattern_tokens) == 1:
            best = max((cls._similarity(token, pattern) for token in tokens), default=0.0)
            if best >= 0.92:
                return 0.90
            if best >= 0.82:
                return 0.68
            return 0.0

        coverage = cls._token_coverage(tokens, pattern_tokens)
        best_window = cls._best_window_similarity(tokens, pattern_tokens)
        token_set = cls._token_set_similarity(tokens, pattern_tokens)
        score = max(best_window * 0.86, token_set * 0.78, coverage * 0.72)
        return score if score >= 0.50 else 0.0

    @classmethod
    def _token_coverage(cls, tokens: list[str], pattern_tokens: list[str]) -> float:
        if not pattern_tokens:
            return 0.0
        hits = 0
        for pattern_token in pattern_tokens:
            if any(cls._similarity(token, pattern_token) >= 0.82 for token in tokens):
                hits += 1
        return hits / len(pattern_tokens)

    @classmethod
    def _best_window_similarity(cls, tokens: list[str], pattern_tokens: list[str]) -> float:
        size = len(pattern_tokens)
        if not tokens or size == 0:
            return 0.0
        pattern = " ".join(pattern_tokens)
        if len(tokens) < size:
            return cls._similarity(" ".join(tokens), pattern)
        best = 0.0
        for start in range(0, len(tokens) - size + 1):
            window = " ".join(tokens[start:start + size])
            best = max(best, cls._similarity(window, pattern))
        return best

    @staticmethod
    def _token_set_similarity(tokens: list[str], pattern_tokens: list[str]) -> float:
        token_set = set(tokens)
        pattern_set = set(pattern_tokens)
        if not token_set or not pattern_set:
            return 0.0
        overlap = len(token_set & pattern_set)
        return (2 * overlap) / (len(token_set) + len(pattern_set))

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
            keyword_tokens = normalized.split()
            if len(keyword_tokens) > 1:
                if cls._token_coverage(tokens, keyword_tokens) >= 0.90:
                    return True
                continue
            if any(cls._similarity(token, normalized) >= 0.82 for token in tokens):
                return True
        return False

    @classmethod
    def normalize(cls, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower().strip())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9%.,\s_-]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"([a-z])\1{2,}", r"\1\1", normalized)
        for phrase, replacement in cls.PHRASE_ALIASES.items():
            normalized = re.sub(rf"\b{phrase}\b", replacement, normalized)
        for typo, replacement in cls.TYPO_ALIASES.items():
            normalized = re.sub(rf"\b{typo}\b", replacement, normalized)
        return normalized
