from __future__ import annotations

import difflib
import numpy as np
import pandas as pd
import random
import time
import re
import unicodedata
from typing import Any
from scipy.stats import norm

from core.explanation_engine import ExplanationEngine
from core.intent_engine import IntentEngine, IntentResult
from core.optimizer import PortfolioOptimizer
from core.portfolio_engine import PortfolioEngine
from core.response_engine import ResponseEngine
from models.portfolio_models import AgentResponse, PortfolioResult
from services.data_service import DataService
from services.memory_service import MemoryService
from agents.base_agent import BaseAgent


GREETING_PREFIXES = [
    "Bonjour",
    "Salut",
    "Hello",
    "Bonsoir",
    "Coucou",
    "Hey",
    "Yo",
    "Slt",
    "Bjr",
    "Cc",
]
GREETING_ACTIONS = [
    "je peux analyser un portefeuille",
    "je peux expliquer des notions financières",
    "je peux construire une allocation optimale",
    "je peux comparer des stratégies",
    "je peux t'aider à choisir entre rendement et risque",
    "je peux transformer une idée vague en contraintes de portefeuille",
    "je peux détecter un profil prudent, équilibré ou agressif",
    "je peux répondre même si ta demande contient quelques fautes",
]
GREETING_ENDS = [
    "Dis-moi ce que tu veux faire.",
    "Tu peux me poser une question ou me demander une allocation.",
    "Je suis prêt pour ta prochaine demande.",
    "Je suis là pour t'accompagner dans ton investissement.",
    "Je peux aussi proposer des variantes prudentes ou agressives.",
    "Donne-moi juste ton objectif, ton risque, ou ton horizon.",
    "Même une phrase simple suffit pour commencer.",
]

SMALLTALK_OPENERS = [
    "Tout fonctionne bien",
    "Je vais très bien",
    "Ça marche ici",
    "Prêt à analyser tes idées",
    "Je suis opérationnel",
]
SMALLTALK_CONTINUES = [
    "Je suis prêt à analyser des portefeuilles ou répondre à des questions financières.",
    "Tu peux me demander une optimisation de portefeuille ou une analyse de risque.",
    "Dis-moi ce que tu veux analyser.",
    "Prêt pour une nouvelle optimisation financière.",
    "Je peux t'expliquer des concepts financiers ou proposer un portefeuille adapté.",
    "Je peux aussi reformuler ta demande si elle est imprécise.",
    "On peut partir d'un capital, d'un rendement cible ou d'un profil de risque.",
]

THANK_YOU_TEMPLATES = [
    "Avec plaisir.",
    "Ravi d'avoir aidé.",
    "N'hésite pas à demander d'autres simulations.",
    "Je peux aussi comparer plusieurs stratégies si tu veux.",
    "Je peux générer d'autres profils de portefeuille si tu en as besoin.",
    "Content que ce soit utile.",
    "On peut affiner encore avec un horizon, un secteur ou un capital.",
    "Je peux garder ce profil comme base pour une variante.",
]

GOODBYE_TEMPLATES = [
    "À bientôt.",
    "Reviens si tu veux tester d'autres allocations.",
    "Bonne continuation.",
    "À plus tard pour une nouvelle optimisation.",
    "Je reste disponible si tu veux analyser un autre scénario.",
    "À bientôt. Je garderai le contexte de cette session pour reprendre plus facilement.",
    "Bonne journée. Reviens quand tu veux pour comparer un nouveau portefeuille.",
    "À plus tard. Pense à préciser rendement, risque et capital pour une simulation directe.",
    "Merci pour l'échange. Je serai prêt pour la prochaine analyse.",
]

HELP_TEMPLATES = [
    "Je peux construire des portefeuilles optimisés selon ton rendement cible et ton risque maximal.",
    "Je peux analyser la diversification, le risque et les allocations.",
    "Je peux comparer plusieurs stratégies d'investissement.",
    "Je peux expliquer les notions financières importantes.",
    "Je peux générer des profils prudents, équilibrés ou agressifs.",
    "Je peux t'aider à choisir entre croissance et revenus ou à définir un horizon d'investissement.",
    "Je comprends des demandes imparfaites comme 'portfeuille prudent 5 ans 50000 mad'.",
    "Je peux détailler une option précédente si tu écris 'option 2'.",
    "Je peux comparer les dernières propositions si tu écris 'compare option 1 et 3'.",
]

EDUCATION_SUBJECTS = {
    "sharpe": "Le ratio de Sharpe mesure le rendement excédentaire par unité de risque. Plus il est élevé, mieux le portefeuille est rémunéré pour la volatilité qu'il assume.",
    "sortino": "Le ratio de Sortino mesure le rendement par rapport aux fluctuations négatives seulement, utile quand on veut limiter les pertes.",
    "risque": "Le risque mesure l'incertitude autour du rendement d'un investissement. En portefeuille, on l'approche souvent par la volatilité, le drawdown, la concentration et la corrélation entre actifs.",
    "volatilité": "La volatilité représente l'amplitude des variations de prix d'un actif ou d'un portefeuille. Une volatilité élevée signifie des gains et des pertes plus importants.",
    "covariance": "La covariance indique si deux actifs évoluent ensemble. Une covariance positive signifie qu'ils bougent souvent dans le même sens.",
    "corrélation": "La corrélation compare la relation entre deux actifs sur une échelle de -1 à +1. Elle aide à comprendre la diversification.",
    "diversification": "La diversification répartit ton investissement sur plusieurs actifs pour limiter l'impact d'une mauvaise performance sur un seul titre.",
    "drawdown": "Le drawdown est la variation négative maximale depuis un pic. Il mesure la profondeur des pertes potentielles.",
    "frontière efficace": "La frontière efficace montre les portefeuilles offrant le meilleur rendement pour chaque niveau de risque.",
    "var": "La VaR estime la perte maximale possible dans un certain intervalle de confiance, par exemple 95 % sur un horizon donné.",
    "cvar": "La CVaR mesure la moyenne des pertes extrêmes au-delà de la VaR. C'est une mesure plus prudente du risque de queue.",
    "rééquilibrage": "Le rééquilibrage ajuste périodiquement les poids pour revenir à la stratégie cible et éviter la surconcentration.",
    "horizon": "L'horizon d'investissement est la durée pendant laquelle tu gardes le capital. Un horizon long permet souvent d'absorber davantage de volatilité.",
    "dividendes": "Une stratégie dividende vise des revenus réguliers. Elle est souvent plus stable mais peut offrir moins de croissance qu'une stratégie purement croissance.",
    "croissance": "Une stratégie croissance privilégie la hausse du capital à long terme, souvent avec plus de volatilité.",
}

PORTFOLIO_STYLE_MAP = {
    "prudent": {"target_risk": 0.08, "style": "conservateur"},
    "équilibré": {"target_risk": 0.13, "style": "équilibré"},
    "agressif": {"target_risk": 0.20, "style": "agressif"},
    "croissance": {"target_risk": 0.18, "style": "croissance"},
    "revenu": {"target_risk": 0.10, "style": "revenu"},
}

FOLLOWUP_TEMPLATES = {
    "horizon": [
        "Quel horizon tu vises pour cet investissement ? 3 ans, 5 ans ou 10 ans ?",
        "Tu veux garder cet argent combien de temps ?",
        "Court terme, moyen terme ou long terme ?",
        "Pour mieux t'aider, indique-moi la durée de placement.",
    ],
    "risk": [
        "Tu préfères une approche stable ou tu veux accepter des fluctuations pour chercher plus de rendement ?",
        "Veux-tu un portefeuille plutôt défensif, équilibré ou offensif ?",
        "Quel niveau de risque te semble acceptable ?",
    ],
    "capital": [
        "Quel montant veux-tu investir ? 10 000, 50 000 ou 100 000 MAD par exemple ?",
        "Dis-moi le capital disponible pour adapter les allocations.",
        "As-tu un montant précis à investir ?",
    ],
    "sector": [
        "Tu veux privilégier un secteur comme la banque, l'énergie ou la consommation ?",
        "As-tu des secteurs préférés ou à éviter ?",
        "Tu veux une exposition large ou concentrée sur certains thèmes ?",
    ],
    "style": [
        "Plutôt une stratégie prudente, équilibrée ou agressive ?",
        "Tu veux plus de stabilité ou plus de croissance ?",
        "Cherches-tu des revenus réguliers ou de la croissance long terme ?",
    ],
}

CLARIFICATION_TEMPLATES = [
    "Quel niveau de risque souhaites-tu ?",
    "Préféres-tu la stabilité ou la croissance ?",
    "Quel est ton horizon d'investissement ?",
    "Cherches-tu des revenus réguliers ou une croissance long terme ?",
    "Veux-tu privilégier certains secteurs ?",
    "As-tu un capital précis à investir ?",
    "Tu veux que je te propose un portefeuille défensif, équilibré ou offensif ?",
    "Veux-tu plutôt maximiser le Sharpe, minimiser la volatilité ou diversifier au maximum ?",
    "Donne-moi au moins un rendement cible et un risque maximal pour calculer une allocation.",
    "Si tu n'as pas de chiffres, indique simplement prudent, équilibré ou agressif.",
]

GREETING_RESPONSES = [
    f"{random.choice(GREETING_PREFIXES)} ! {random.choice(GREETING_ACTIONS)} {random.choice(GREETING_ENDS)}"
    for _ in range(36)
]
CONVERSATION_RESPONSES = [
    f"{random.choice(SMALLTALK_OPENERS)} — {random.choice(SMALLTALK_CONTINUES)}"
    for _ in range(34)
]
THANK_YOU_RESPONSES = [
    *THANK_YOU_TEMPLATES,
    "Je suis content que cela t'ait aidé.",
    "Tu peux encore demander une autre simulation si tu veux.",
    "Avec plaisir. Je peux maintenant détailler le risque, la diversification ou une option précise.",
    "Merci à toi. On peut continuer avec une variante plus prudente ou plus dynamique.",
]
PERSONAL_RESPONSES = [
    "Je suis ton assistant local pour la Bourse de Casablanca, spécialisé dans l'analyse de portefeuille et l'éducation financière.",
    "Je fonctionne sans connexion externe et je m'appuie sur tes données locales pour t'aider.",
    "Je suis un assistant offline qui peut expliquer la démarche, les métriques et les choix d'allocation.",
    "Je suis conçu pour comprendre tes objectifs d'investissement, détecter les contraintes utiles et proposer des allocations calculées localement.",
    "Mon rôle est de transformer une demande en langage naturel en analyse rendement-risque exploitable.",
]
GENERAL_RESPONSES = [
    "Je peux t'aider avec des questions générales sur l'investissement, le risque, le rendement et la diversification.",
    "Pose-moi une question sur la finance locale ou la construction de portefeuille.",
    "Je peux expliquer des concepts ou t'aider à choisir une stratégie adaptée.",
    "Je peux répondre à des questions de finance, mais je reste centré sur les portefeuilles et l'investissement.",
    "Si ta question touche au risque, au rendement ou aux actions, je peux la traiter.",
]
INVESTMENT_GUIDANCE_RESPONSES = [
    "En investissement, diversifier permet d'éviter qu'une seule mauvaise performance n'entraîne tout le portefeuille.",
    "Le rendement attendu doit toujours être mis en balance avec le risque que tu es prêt à accepter.",
    "Plus l'horizon est long, plus tu peux te permettre une stratégie légèrement plus agressive.",
    "Il est utile de connaître tes secteurs préférés et les secteurs que tu veux éviter.",
    "Un bon portefeuille respecte tes objectifs tout en restant cohérent avec ta tolérance au risque.",
    "Un portefeuille réaliste commence par trois éléments : horizon, risque supportable et capital disponible.",
    "Une cible de rendement trop élevée sans risque correspondant conduit souvent à des allocations instables.",
    "La meilleure allocation n'est pas toujours celle qui maximise le rendement, mais celle que tu peux conserver dans les périodes difficiles.",
]
CASUAL_RESPONSES = [
    "Ça va bien, merci — prêt à analyser quelque chose ?",
    "Je suis ici pour t'aider à construire un portefeuille ou répondre à tes questions.",
    "Prêt à explorer des scénarios financiers ou à simuler des allocations.",
    "Ça va bien. Donne-moi une idée d'investissement et je la structure.",
    "Je suis prêt. Même une demande approximative peut servir de point de départ.",
]
CLARIFY_RESPONSES = [
    *CLARIFICATION_TEMPLATES,
    "Précise si tu veux un portefeuille dirigé par rendement ou par risque.",
    "Veux-tu que je développe une option plus prudente ou plus offensive ?",
]
HELP_RESPONSES = [
    *HELP_TEMPLATES,
    "Je peux te proposer un portefeuille en fonction de ton rendement et de ton risque.",
    "Je peux te guider pas à pas pour formuler ta demande.",
]
GOODBYE_RESPONSES = [
    *GOODBYE_TEMPLATES,
    "Bonne journée et à bientôt pour un autre scénario.",
    "Reviens quand tu veux pour tester d'autres portefeuilles.",
    "À bientôt. Je reste disponible pour une prochaine simulation.",
    "Bonne continuation. Reviens avec un rendement cible ou un profil de risque quand tu veux.",
]
EXAMPLE_RESPONSES = [
    "Exemple : '12% rendement, 18% risque' ou '15% rendement, 14% risque'",
    "Tu peux aussi demander : 'propose-moi un portefeuille prudent' ou 'cherche un rendement 10%'.",
    "Exemple : 'portefeuille équilibré 5 ans 100000 MAD avec risque modéré'.",
    "Exemple : 'compare option 1 et option 3' après une simulation.",
    "Exemple : 'explique le Sharpe simplement' ou 'c'est quoi la diversification ?'.",
]
EXPLAIN_RESPONSES = [
    "Je génère des portefeuilles, j'estime rendement et risque, puis je te présente le meilleur compromis.",
    "Je combine historique des prix, rendements annualisés et covariance pour proposer des allocations.",
]
DATA_RESPONSES = [
    "Les données locales viennent des fichiers CSV. Vérifie que les historiques sont bien présents.",
    "Je travaille offline avec les fichiers de prix et les informations disponibles sur ton disque.",
]
ALLOCATION_RESPONSES = [
    "Pour afficher une allocation, indique un rendement cible et un risque maximum.",
    "Je peux te proposer plusieurs allocations compatibles si tu fournis ton rendement et ton risque.",
    "Je peux construire une allocation si tu me donnes au minimum un objectif de rendement et une limite de risque.",
    "Si tu fournis un capital en MAD, je peux aussi estimer les quantités d'actions à acheter.",
]

OUT_OF_SCOPE_RESPONSES = [
    "Je suis spécialisé dans l'investissement, les portefeuilles et la Bourse de Casablanca. Reformule ta question autour du risque, du rendement, d'une action ou d'une allocation et je pourrai t'aider.",
    "Cette question semble hors de mon domaine. Je peux surtout aider sur les portefeuilles, la finance, les actions, le risque, le rendement et la diversification.",
    "Je ne suis pas le bon assistant pour ce sujet. En revanche, je peux analyser un portefeuille, expliquer une notion financière ou proposer une allocation adaptée.",
    "Je préfère rester sur mon terrain : finance, investissement et optimisation de portefeuille. Donne-moi un objectif financier et je le transforme en analyse.",
    "Ta question ne ressemble pas à une demande financière. Essaie par exemple avec un capital, un horizon, un risque maximal ou une notion comme Sharpe, volatilité ou diversification.",
]

MAX_RANDOM_PORTFOLIOS = 25000
TOP_PORTFOLIO_CHOICES = 7


class LocalAgent(BaseAgent):
    def __init__(self, data_service: DataService | None = None) -> None:
        self.data_service = data_service or DataService()
        self.portfolio_engine = PortfolioEngine()
        self.memory_service = MemoryService()
        self.memory_service.initialize()
        self.intent_engine = IntentEngine()
        self.rng = np.random.default_rng(42)

    def run(self, user_input: str, **kwargs: Any) -> AgentResponse:
        self.memory_service.initialize()
        context = self.memory_service.get_context(max_items=6)
        intent = self.intent_engine.analyze(user_input, context=context)
        self._update_profile_from_intent(intent, user_input)
        self._update_profile_from_input(user_input)

        # afficher plus d'options après une proposition
        if self._is_more_options_request(user_input):
            return self._list_recent_options()

        # selection d'une option precedente: 'option 2' ou 'choix 1'
        opt_n = self._is_option_selection(user_input)
        if opt_n is not None:
            response = self._provide_option_details(opt_n)
            return response

        # comparer des options: 'comparer' ou 'compare'
        if self._is_compare_request(user_input):
            pair = self._parse_compare_indices(user_input)
            recent = self._get_recent_option_entries()
            if pair is None:
                if len(recent) >= 3:
                    indices = [int(item.get("option_rank", -1)) for item in recent]
                    response = self._compare_multiple_options(indices)
                elif len(recent) == 2:
                    a = recent[0].get("option_rank")
                    b = recent[1].get("option_rank")
                    response = self._compare_options(int(a), int(b))
                else:
                    response = AgentResponse(content="Pas assez d'options récentes pour comparer. Génère d'abord plusieurs options.")
            else:
                response = self._compare_options(pair[0], pair[1])
            return response
        self.memory_service.initialize()

        if self._is_smalltalk_request(user_input, intent):
            response = AgentResponse(content=self._compose_smalltalk_response(user_input))
        elif self._is_goodbye_request(user_input):
            response = AgentResponse(content=self._compose_goodbye_response())
        elif self._is_thank_you_request(user_input):
            response = AgentResponse(content=self._compose_thank_you_response())
        elif self._is_pure_educational_request(user_input, intent):
            response = AgentResponse(content=self._answer_investment_question(user_input))
        elif self._is_contextual_variant_request(user_input):
            response = self._build_contextual_variant_response(user_input)
        elif self._is_range_request(user_input) or intent.intent == "market_question":
            response = AgentResponse(content=self._market_range_message())
        else:
            rendement, risque, capital, extracted = self._extract_parameters(user_input)
            rendement = rendement if rendement is not None else intent.entities.get("target_return")
            risque = risque if risque is not None else intent.entities.get("target_volatility")
            capital = capital if capital is not None else intent.entities.get("investment_amount")
            explicit_portfolio_request = self._has_explicit_portfolio_generation_request(user_input)
            complete_numeric_constraints = rendement is not None and risque is not None
            if (
                explicit_portfolio_request
                and (rendement is None or risque is None)
                and self._should_infer_portfolio_request(intent)
            ):
                rendement, risque = self._infer_constraints_from_profile(rendement, risque)
                extracted = True
            explain = self._is_method_explanation_request(user_input)
            if complete_numeric_constraints or (explicit_portfolio_request and rendement is not None and risque is not None):
                validation_error = self._validate_constraints(rendement, risque)
                if validation_error:
                    response = AgentResponse(content=validation_error)
                else:
                    response = self._build_portfolio_result(
                        rendement,
                        risque,
                        capital,
                        explain_method=explain,
                        user_input=user_input,
                    )
            elif extracted and (rendement is None or risque is None) and explicit_portfolio_request:
                missing = []
                if rendement is None:
                    missing.append("target_return")
                if risque is None:
                    missing.append("max_risk")
                response = AgentResponse(content=ResponseEngine.followup_for_missing(missing, self.memory_service.get_profile()))
            elif self._is_greeting_request(user_input):
                response = AgentResponse(content=self._compose_greeting_response())
            elif self._is_conversation_request(user_input):
                response = AgentResponse(content=self._compose_conversation_response(user_input))
            elif self._is_personal_request(user_input):
                response = AgentResponse(content=self._compose_personal_response())
            elif self._is_method_explanation_request(user_input):
                response = AgentResponse(content=self._method_explanation_message())
            elif self._is_out_of_scope_request(user_input, intent):
                response = AgentResponse(content=self._compose_out_of_scope_response(user_input))
            elif self._is_investment_question(user_input) or self._is_educational_intent(intent):
                response = AgentResponse(content=self._answer_investment_question(user_input))
            elif self._fuzzy_contains_any(user_input, ["aide", "help", "comment faire", "que faire", "instructions"]):
                response = AgentResponse(content=self._compose_help_response())
            elif self._fuzzy_contains_any(user_input, ["exemple", "example", "comment ecrire", "exemples"]):
                response = AgentResponse(content=random.choice(EXAMPLE_RESPONSES))
            elif self._fuzzy_contains_any(user_input, ["donnees", "data", "csv", "fichiers", "stock"]):
                response = AgentResponse(content=self._compose_data_response())
            elif self._fuzzy_contains_any(user_input, ["allocation", "poids", "positions", "allocations", "top"]):
                response = AgentResponse(content=self._compose_allocation_response())
            elif self._is_goodbye_request(user_input):
                response = AgentResponse(content=self._compose_goodbye_response())
            elif self._is_out_of_scope_request(user_input, intent):
                response = AgentResponse(content=self._compose_out_of_scope_response(user_input))
            elif user_input.strip().endswith("?"):
                response = AgentResponse(content=random.choice(CLARIFY_RESPONSES))
            else:
                response = AgentResponse(content=self._compose_contextual_default_response(user_input, intent))

        return response

    def _compose_greeting_response(self) -> str:
        return self._compose_smalltalk_response("salut")

    def _compose_conversation_response(self, user_input: str) -> str:
        profile_line = self._profile_summary_line()
        if self._is_general_conversation_request(user_input):
            return f"{profile_line}{ResponseEngine.choose('educational', user_input)}"
        return f"{profile_line}{ResponseEngine.choose('followup', user_input)}"

    def _compose_smalltalk_response(self, user_input: str) -> str:
        normalized = self._normalize_text(user_input)
        if any(term in normalized for term in ["ca va", "cava", "comment ca va", "tu vas bien", "labas", "labass"]):
            return "Ça va bien, merci. Et toi ?"
        if any(term in normalized for term in ["salut", "bonjour", "bonsoir", "hello", "coucou"]):
            return "Salut. Tu veux discuter, apprendre une notion financière, ou travailler sur un portefeuille ?"
        return "Je suis là. Dis-moi ce que tu veux comprendre ou analyser."

    def _compose_thank_you_response(self) -> str:
        return f"{self._profile_summary_line()}{random.choice(THANK_YOU_RESPONSES)}"

    def _compose_goodbye_response(self) -> str:
        return f"{self._profile_summary_line()}{random.choice(GOODBYE_RESPONSES)}"

    def _compose_out_of_scope_response(self, user_input: str) -> str:
        hint = ""
        if len(self._normalize_text(user_input).split()) <= 4:
            hint = " Si tu voulais parler finance, ajoute par exemple 'risque', 'rendement', 'portefeuille' ou 'action'."
        return f"{random.choice(OUT_OF_SCOPE_RESPONSES)}{hint}"

    def _compose_personal_response(self) -> str:
        profile_line = self._profile_summary_line()
        return f"{profile_line}{random.choice(PERSONAL_RESPONSES)}"

    def _compose_help_response(self) -> str:
        return f"{self._profile_summary_line()}{ResponseEngine.choose('optimization', 'help')}"

    def _compose_data_response(self) -> str:
        return f"{self._profile_summary_line()}{ResponseEngine.choose('market', 'data')}"

    def _compose_allocation_response(self) -> str:
        return f"{self._profile_summary_line()}{ResponseEngine.choose('allocation', 'allocation')}"

    def _compose_default_response(self, user_input: str) -> str:
        profile_line = self._profile_summary_line()
        if self._is_investment_question(user_input):
            return f"{profile_line}{self._answer_investment_question(user_input)}"
        return f"{profile_line}{ResponseEngine.choose('clarification', user_input)}"

    def _compose_contextual_default_response(self, user_input: str, intent: IntentResult) -> str:
        profile = self.memory_service.get_profile()
        if intent.intent in {"risk_analysis", "volatility_question", "drawdown_question"}:
            return f"{ResponseEngine.profile_tone(profile)} {ResponseEngine.choose('risk', user_input)}"
        if intent.intent in {"diversification_request", "sector_preference"}:
            return ResponseEngine.choose("diversification", user_input)
        if intent.intent in {"allocation_explanation", "portfolio_creation", "optimization_request"}:
            missing = self.memory_service.missing_profile_fields(["target_return", "max_risk"])
            return ResponseEngine.followup_for_missing(missing, profile)
        if intent.intent in {"educational_question", "advanced_finance_question", "beginner_question"}:
            return self._answer_investment_question(user_input)
        return self._compose_default_response(user_input)

    def _profile_summary_line(self) -> str:
        summary = self.memory_service.get_profile_summary()
        if summary and summary != "Aucun profil enregistré pour le moment.":
            return f"Profil actuel : {summary}. "
        return ""

    def _update_profile_from_intent(self, intent: IntentResult, user_input: str) -> None:
        updates = {key: value for key, value in intent.entities.items() if value not in (None, [], False)}
        if intent.intents:
            updates["last_intents"] = [name for name, _ in intent.intents]
        updates["last_request"] = user_input
        self.memory_service.update_profile(**updates)

    def _is_educational_intent(self, intent: IntentResult) -> bool:
        return intent.intent in {
            "educational_question",
            "risk_analysis",
            "volatility_question",
            "sharpe_question",
            "efficient_frontier_question",
            "drawdown_question",
            "advanced_finance_question",
            "beginner_question",
            "diversification_request",
            "rebalancing_request",
            "allocation_explanation",
            "performance_analysis",
        }

    def _is_smalltalk_request(self, user_input: str, intent: IntentResult) -> bool:
        normalized = self._normalize_text(user_input)
        smalltalk_terms = [
            "ca va",
            "cava",
            "comment ca va",
            "tu vas bien",
            "quoi de neuf",
            "labas",
            "labass",
        ]
        finance_terms = [
            "portefeuille",
            "allocation",
            "rendement",
            "risque",
            "volatilite",
            "sharpe",
            "investir",
            "optimiser",
        ]
        return (
            (intent.intent == "smalltalk" or any(term in normalized for term in smalltalk_terms))
            and not any(term in normalized for term in finance_terms)
        )

    def _is_pure_educational_request(self, user_input: str, intent: IntentResult) -> bool:
        normalized = self._normalize_text(user_input)
        has_numbers = bool(re.search(r"\d+(?:[.,]\d+)?\s*%?", user_input))
        explicit_portfolio_words = [
            "portefeuille",
            "allocation",
            "optimisation",
            "optimiser",
            "propose",
            "construire",
            "investir",
            "rendement cible",
            "risque max",
            "risque maximal",
        ]
        educational_words = [
            "expliquer",
            "explique",
            "definition",
            "c est quoi",
            "qu est ce que",
            "risque",
            "sharpe",
            "sortino",
            "volatilite",
            "drawdown",
            "diversification",
            "correlation",
            "covariance",
            "var",
            "cvar",
            "frontiere",
            "reequilibrage",
        ]
        return (
            self._is_educational_intent(intent)
            and any(word in normalized for word in educational_words)
            and not has_numbers
            and not any(word in normalized for word in explicit_portfolio_words)
        )

    def _should_infer_portfolio_request(self, intent: IntentResult) -> bool:
        return intent.intent in {
            "portfolio_creation",
            "optimization_request",
            "growth_strategy",
            "dividend_strategy",
            "sector_preference",
            "capital_allocation",
        } and any(score >= 0.35 for _, score in intent.intents)

    def _has_explicit_portfolio_generation_request(self, user_input: str) -> bool:
        normalized = self._normalize_text(user_input)
        portfolio_terms = [
            "portefeuille",
            "portfeuille",
            "allocation",
            "repartition",
            "positions",
        ]
        action_terms = [
            "propose",
            "proposer",
            "cree",
            "creer",
            "construis",
            "construire",
            "genere",
            "generer",
            "optimise",
            "optimiser",
            "trouve",
            "donne moi",
            "je veux",
            "cherche",
        ]
        strategy_terms = [
            "prudent",
            "conservateur",
            "equilibre",
            "agressif",
            "croissance",
            "dividende",
            "revenu",
            "risk parity",
            "minimum volatilite",
            "maximum sharpe",
        ]
        has_portfolio_word = any(term in normalized for term in portfolio_terms)
        has_action = any(term in normalized for term in action_terms)
        has_strategy = any(term in normalized for term in strategy_terms)
        return (has_portfolio_word and (has_action or has_strategy)) or (
            has_action and has_strategy and any(term in normalized for term in ["investir", "strategie", "allocation"])
        )

    def _is_contextual_variant_request(self, user_input: str) -> bool:
        normalized = self._normalize_text(user_input)
        variant_terms = [
            "plus agressif",
            "plus agressive",
            "plus aggressive",
            "plus prudent",
            "plus prudente",
            "plus conservateur",
            "plus defensif",
            "plus defensive",
            "plus equilibre",
            "variante agressive",
            "variante prudente",
            "variante defensive",
            "variante equilibree",
        ]
        if not any(term in normalized for term in variant_terms):
            return False

        profile = self.memory_service.get_profile()
        has_profile_constraints = (
            self._as_float(profile.get("target_return")) is not None
            and self._as_float(profile.get("max_risk")) is not None
        )
        return bool(self.memory_service.get_portfolio_history()) or has_profile_constraints

    def _build_contextual_variant_response(self, user_input: str) -> AgentResponse:
        target_return, max_risk = self._last_portfolio_constraints()
        if target_return is None or max_risk is None:
            return AgentResponse(
                content="Je peux te proposer une variante, mais il me faut d'abord une base chiffrée: par exemple `12% rendement 15% risque`."
            )

        normalized = self._normalize_text(user_input)
        if any(term in normalized for term in ["agressif", "agressive", "aggressive"]):
            target_return = min(target_return + 3.0, 30.0)
            max_risk = min(max_risk + 5.0, 35.0)
            prefix = "## Variante plus agressive\nJ'augmente le rendement cible et j'autorise davantage de volatilité par rapport à la dernière base.\n\n"
        elif any(term in normalized for term in ["prudent", "prudente", "conservateur", "defensif", "defensive"]):
            target_return = max(target_return - 2.0, 3.0)
            max_risk = max(max_risk - 4.0, 4.0)
            prefix = "## Variante plus prudente\nJe réduis le risque maximal et je baisse légèrement l'objectif de rendement pour chercher plus de stabilité.\n\n"
        else:
            target_return = max(min(target_return, 10.0), 7.0)
            max_risk = max(min(max_risk, 14.0), 9.0)
            prefix = "## Variante équilibrée\nJe recentre la demande vers un compromis rendement-risque plus modéré.\n\n"

        capital = self.memory_service.get_profile().get("investment_amount")
        response = self._build_portfolio_result(
            target_return,
            max_risk,
            float(capital) if isinstance(capital, (int, float)) else None,
            explain_method=False,
            user_input=user_input,
        )
        response.content = prefix + response.content
        return response

    def _last_portfolio_constraints(self) -> tuple[float | None, float | None]:
        for entry in reversed(self.memory_service.get_portfolio_history()):
            target_return = self._parse_percent_string(entry.get("target_return"))
            max_risk = self._parse_percent_string(entry.get("max_risk"))
            if target_return is not None and max_risk is not None:
                return target_return, max_risk
        profile = self.memory_service.get_profile()
        return self._as_float(profile.get("target_return")), self._as_float(profile.get("max_risk"))

    @staticmethod
    def _parse_percent_string(value: Any) -> float | None:
        if value is None or value == "N/A":
            return None
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(value))
        return float(match.group(1).replace(",", ".")) if match else None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"(\d+(?:[.,]\d+)?)", value)
            return float(match.group(1).replace(",", ".")) if match else None
        return None

    def _infer_constraints_from_profile(self, rendement: float | None, risque: float | None) -> tuple[float | None, float | None]:
        profile = self.memory_service.get_profile()
        risk_tolerance = profile.get("risk_tolerance")
        style = profile.get("style")
        if risque is None:
            if risk_tolerance in {"low", "faible"} or style == "prudent":
                risque = 8.0
            elif risk_tolerance in {"high", "élevé"} or style == "agressif":
                risque = 20.0
            else:
                risque = 13.0
        if rendement is None:
            if profile.get("growth_preference") or style in {"croissance", "agressif"}:
                rendement = 12.0
            elif profile.get("dividend_preference") or profile.get("income_preference") or style == "revenu":
                rendement = 7.0
            elif risque <= 9:
                rendement = 6.0
            elif risque >= 18:
                rendement = 12.0
            else:
                rendement = 9.0
        return rendement, risque

    def _update_profile_from_input(self, user_input: str) -> None:
        normalized = self._normalize_text(user_input)
        updates: dict[str, str | list[str] | float | int | None] = {}

        if any(keyword in normalized for keyword in ["prudent", "conservateur", "defensif", "securise", "secure"]):
            updates["style"] = "prudent"
            updates["risk_tolerance"] = "faible"
        if any(keyword in normalized for keyword in ["equilibre", "equilibrer", "balanced", "modere"]):
            updates["style"] = "équilibré"
            updates["risk_tolerance"] = "modéré"
        if any(keyword in normalized for keyword in ["agressif", "offensif", "dynamique", "haut risque", "risque eleve", "risque fort"]):
            updates["style"] = "agressif"
            updates["risk_tolerance"] = "élevé"

        if "court terme" in normalized or "court-terme" in normalized:
            updates["investment_horizon"] = 2
        if "moyen terme" in normalized or "moyen-terme" in normalized:
            updates["investment_horizon"] = 5
        if "long terme" in normalized or "long-terme" in normalized:
            updates["investment_horizon"] = 10

        sectors = self._parse_preferred_sectors(normalized)
        if sectors:
            updates["preferred_sectors"] = sectors

        montant = self._extract_currency_amount(user_input)
        if montant is not None:
            updates["investment_amount"] = montant

        target_return = self._parse_percentage(user_input, ["rendement", "return", "profit", "gain", "objectif", "target"])
        max_risk = self._parse_percentage(user_input, ["risque", "volatilité", "volatilite", "sigma", "variance", "risk"])
        if target_return is not None:
            updates["target_return"] = target_return
        if max_risk is not None:
            updates["max_risk"] = max_risk

        if updates:
            self.memory_service.update_profile(**updates)

    def _parse_preferred_sectors(self, normalized: str) -> list[str]:
        sector_keywords = [
            "banque",
            "energie",
            "immobilier",
            "sante",
            "consommation",
            "technologie",
            "industrie",
            "materiaux",
            "petrole",
            "services",
            "agro",
            "finance",
        ]
        sectors: list[str] = []
        for keyword in sector_keywords:
            if keyword in normalized and keyword not in sectors:
                sectors.append(keyword)
        return sectors

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9%\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        typo_aliases = {
            "portfeuille": "portefeuille",
            "portefuille": "portefeuille",
            "portefeuillee": "portefeuille",
            "portfoli": "portfolio",
            "rendemet": "rendement",
            "rendment": "rendement",
            "rendemnt": "rendement",
            "risqu": "risque",
            "risquee": "risque",
            "volatilitee": "volatilite",
            "volatilit": "volatilite",
            "volatilte": "volatilite",
            "diversifiction": "diversification",
            "diversifcation": "diversification",
            "optmisation": "optimisation",
            "optimisaton": "optimisation",
            "optimzer": "optimiser",
            "sharpr": "sharpe",
            "sharp": "sharpe",
            "sortnio": "sortino",
            "drawdon": "drawdown",
            "reequilibrage": "reequilibrage",
            "rebalencage": "rebalancing",
            "alocation": "allocation",
            "allocaton": "allocation",
            "repartion": "repartition",
            "capitale": "capital",
            "montan": "montant",
            "bonjoure": "bonjour",
            "salu": "salut",
            "mercii": "merci",
            "mercie": "merci",
            "aurevoir": "au revoir",
        }
        for typo, replacement in typo_aliases.items():
            normalized = re.sub(rf"\b{typo}\b", replacement, normalized)
        return normalized

    def _parse_percentage(self, text: str, keywords: list[str]) -> float | None:
        normalized = text.lower()
        for keyword in keywords:
            pattern_before = rf"(\d+(?:[.,]\d+)?)\s*%\s*(?:de\s*)?{keyword}s?"
            match = re.search(pattern_before, normalized)
            if match:
                return float(match.group(1).replace(",", "."))

            pattern = rf"{keyword}s?\s*(?:cible|target|objectif|>=|<=|=|:)?\s*(\d+(?:[.,]\d+)?)\s*%?"
            for match in re.finditer(pattern, normalized):
                number = float(match.group(1).replace(",", "."))
                matched_text = normalized[match.start(): match.end()]
                if "%" not in matched_text:
                    after = normalized[match.end():]
                    if re.match(r"^\s*(?:mad|dh|dhs|dirhams?)\b", after):
                        continue
                return number

        return None

    def _extract_currency_amount(self, text: str) -> float | None:
        normalized = text.lower()
        match = re.search(r"(\d{1,3}(?:[.,]?\d{3})*(?:[.,]\d+)?)\s*(mad|dh|dhs|dirhams?)", normalized)
        if match:
            return float(match.group(1).replace(" ", "").replace(",", "."))

        match = re.search(r"(?:capital|budget|montant)\s*(?:de)?\s*(\d{3,}(?:[.,]\d+)?)", normalized)
        if match:
            return float(match.group(1).replace(" ", "").replace(",", "."))

        return None

    def _extract_parameters(self, user_input: str) -> tuple[float | None, float | None, float | None, bool]:
        normalized = self._normalize_text(user_input)
        rendement = self._parse_percentage(user_input, ["rendement", "return", "rentabilité", "gain", "objectif", "target"])
        risque = self._parse_percentage(user_input, ["risque", "volatilité", "volatilite", "sigma", "variance", "risk"])
        capital = self._extract_currency_amount(user_input)
        extracted = False

        if rendement is None or risque is None:
            percents = [float(p.replace(",", ".")) for p in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", user_input)]
            words = self._normalize_text(user_input)
            if len(percents) >= 2 and (rendement is None or risque is None):
                if "rendement" in words and "risque" in words:
                    if words.index("rendement") < words.index("risque"):
                        rendement = rendement or percents[0]
                        risque = risque or percents[1]
                    else:
                        rendement = rendement or percents[1]
                        risque = risque or percents[0]
                else:
                    rendement = rendement or percents[0]
                    risque = risque or percents[1]
                extracted = True

        if rendement is None or risque is None:
            numbers = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", user_input)]
            if len(numbers) >= 2 and (rendement is None or risque is None):
                if rendement is None and risque is None:
                    rendement, risque = numbers[0], numbers[1]
                elif rendement is None:
                    rendement = numbers[0]
                elif risque is None:
                    risque = numbers[1]
                extracted = True

        if rendement is not None or risque is not None or capital is not None:
            extracted = True

        return rendement, risque, capital, extracted

    def _validate_constraints(self, rendement: float, risque: float) -> str | None:
        if rendement < 0 or risque < 0:
            return "Les valeurs de rendement et de risque doivent être positives."
        if rendement > 100 or risque > 100:
            return "Les pourcentages sont probablement trop élevés. Veuillez vérifier que vous utilisez des pourcentages et non des montants."
        if risque == 0:
            return "Un risque de 0% est irréaliste dans un portefeuille réel. Donne une valeur de risque positive."
        return None

    def _is_method_explanation_request(self, user_input: str) -> bool:
        normalized = self._normalize_text(user_input)
        method_context = [
            "comment tu fais",
            "demarche",
            "methode",
            "comment marche",
            "processus",
            "pourquoi ce portefeuille",
            "algorithme",
            "calcul",
            "optimisation",
        ]
        if any(term in normalized for term in method_context):
            return True
        if ("explique" in normalized or "explication" in normalized) and any(
            term in normalized for term in ["portefeuille", "allocation", "optimisation", "calcul"]
        ):
            return True
        return self._fuzzy_contains_any(
            user_input,
            [
                "comment tu fais",
                "démarche",
                "methode",
                "comment marche",
                "processus",
                "pourquoi ce portefeuille",
            ],
            threshold=0.62,
        )

    def _is_investment_question(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(
            user_input,
            [
                "investir",
                "diversification",
                "volatilité",
                "volatilite",
                "sharpe",
                "rendement",
                "risque",
                "risqué",
                "risk",
                "sortino",
                "covariance",
                "corrélation",
                "correlation",
                "drawdown",
                "var",
                "cvar",
                "rééquilibrage",
                "reequilibrage",
                "rebalancing",
                "ETF",
                "obligation",
                "action",
                "gestion passive",
                "gestion active",
                "allocation",
                "portefeuille",
            ],
            threshold=0.60,
        )

    def _is_conversation_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(
            user_input,
            [
                "ça va",
                "comment vas tu",
                "parle",
                "tu peux",
                "dis moi",
                "d'accord",
                "ok",
                "j'aimerais discuter",
                "quel est ton avis",
            ],
            threshold=0.72,
        )

    def _answer_investment_question(self, user_input: str) -> str:
        normalized = self._normalize_text(user_input)
        concise = len(normalized.split()) <= 3
        profile = self.memory_service.get_profile()
        level = profile.get("response_style") or ("concise" if concise else "beginner")
        if "sharpe" in normalized:
            self.memory_service.remember_education("sharpe")
            return ExplanationEngine.educate("sharpe", level=str(level))
        if "sortino" in normalized:
            self.memory_service.remember_education("sortino")
            return ExplanationEngine.educate("sortino", level=str(level))
        if "volatil" in normalized:
            self.memory_service.remember_education("volatilite")
            return ExplanationEngine.educate("volatilite", level=str(level))
        if "covariance" in normalized:
            return ExplanationEngine.educate("covariance", level=str(level))
        if "correlation" in normalized:
            return ExplanationEngine.educate("correlation", level=str(level))
        if "divers" in normalized:
            return f"{ExplanationEngine.educate('diversification', level=str(level))} {ResponseEngine.choose('diversification', user_input)}"
        if "frontiere" in normalized:
            return ExplanationEngine.educate("frontiere efficace", level=str(level))
        if "drawdown" in normalized:
            return ExplanationEngine.educate("drawdown", level=str(level))
        if normalized == "var" or " value at risk" in f" {normalized} ":
            return ExplanationEngine.educate("var", level=str(level))
        if "cvar" in normalized:
            return ExplanationEngine.educate("cvar", level=str(level))
        if "reequilibrage" in normalized or "rebalanc" in normalized:
            return ExplanationEngine.educate("reequilibrage", level=str(level))
        if "risk parity" in normalized or "parite" in normalized:
            return ExplanationEngine.educate("risk parity", level=str(level))
        if "monte carlo" in normalized or "simulation" in normalized:
            return ExplanationEngine.educate("monte carlo", level=str(level))
        if "gestion passive" in normalized or "passive" in normalized:
            return "La gestion passive vise à suivre un indice, avec des frais faibles et une structure simple. C'est souvent bien adapté pour un horizon long et un style patient."
        if "gestion active" in normalized or "active" in normalized:
            return "La gestion active cherche à surperformer le marché en sélectionnant des titres, mais elle peut coûter plus cher et est plus difficile à bien exécuter."
        if "obligation" in normalized:
            return "Une obligation est une dette émise par une entreprise ou un État. Elle offre des flux d'intérêts réguliers et une volatilité généralement plus faible que les actions."
        if "action" in normalized:
            return "Une action représente une part de propriété d'une entreprise. Les actions peuvent générer des rendements élevés, mais elles sont aussi plus volatiles."
        if "ETF" in normalized or "etf" in normalized:
            return "Un ETF permet d'investir dans un panier d'actifs comme un indice ou un secteur, souvent à faible coût. C'est utile pour diversifier rapidement."
        if "risque" in normalized and "rendement" in normalized:
            return (
                "Le rendement et le risque se lisent ensemble. Chercher plus de rendement implique généralement d'accepter plus de volatilité, "
                "plus de drawdown possible ou une concentration plus forte.\n\n"
                "Un bon portefeuille ne cherche donc pas seulement le rendement le plus élevé: il vérifie si ce rendement est correctement rémunéré par rapport au risque pris. "
                "C'est pour cela qu'on regarde aussi le Sharpe, la diversification, la corrélation entre actifs et la perte maximale historique."
            )
        if "risque" in normalized or "risk" in normalized:
            base = ExplanationEngine.educate("risque", level=str(level))
            if concise:
                return (
                    f"{base}\n\n"
                    "Exemple simple: si deux portefeuilles visent 10% de rendement, celui qui peut perdre 5% dans une mauvaise période est moins risqué que celui qui peut perdre 25%.\n\n"
                    "Pour réduire le risque, on diversifie, on limite les positions trop concentrées, on évite les actifs trop corrélés et on choisit une volatilité compatible avec ton horizon."
                )
            if "faible" in normalized or "bas" in normalized or "prudent" in normalized:
                return "Un risque faible signifie chercher une variation plus stable du portefeuille, souvent avec un rendement attendu plus modéré et une diversification plus large."
            if "eleve" in normalized or "haut" in normalized or "agressif" in normalized:
                return "Un risque élevé signifie accepter de fortes variations de valeur pour viser un rendement potentiel plus élevé. Il faut surtout vérifier que ton horizon et ta tolérance aux pertes le permettent."
            return (
                f"{base}\n\n"
                "Dans ton agent financier, je le mesure avec plusieurs angles:\n"
                "- Volatilité: amplitude moyenne des variations.\n"
                "- Drawdown: baisse maximale depuis un sommet.\n"
                "- VaR et CVaR: estimation des pertes dans les mauvais scénarios.\n"
                "- Concentration: dépendance à quelques titres ou secteurs.\n"
                "- Corrélation: risque caché si plusieurs actifs bougent ensemble.\n\n"
                "Donc quand tu demandes un portefeuille, le but n'est pas seulement de trouver un rendement, mais de choisir un niveau de risque que tu peux réellement supporter."
            )
        if "rendement" in normalized or "return" in normalized:
            return "Le rendement mesure le gain ou la perte d'un investissement sur une période donnée. Il doit toujours être comparé au risque pris pour l'obtenir."
        if "horizon" in normalized or "long terme" in normalized or "court terme" in normalized:
            return EDUCATION_SUBJECTS["horizon"]
        if "investir" in normalized or "conseil" in normalized or "conseils" in normalized:
            return random.choice(INVESTMENT_GUIDANCE_RESPONSES)
        return "Je peux expliquer des sujets comme le Sharpe, la volatilité, la diversification, le risque et le rééquilibrage. Dis-moi ce que tu veux approfondir."

    def _method_explanation_message(self) -> str:
        return (
            "Voici la démarche que j'utilise pour construire un portefeuille :\n"
            "1. Je charge les historiques de prix des actifs disponibles.\n"
            "2. Je transforme ces prix en rendements quotidiens puis en rendements annualisés.\n"
            "3. Je calcule la matrice de covariance pour estimer la volatilité et la corrélation entre actifs.\n"
            "4. Je génère ou j'optimise des portefeuilles pour respecter ton rendement cible et ton risque maximal.\n"
            "5. Je sélectionne le meilleur compromis entre rendement, risque et diversification.\n"
            "6. Si ta cible est trop ambitieuse, je te propose le portefeuille le plus proche de tes objectifs.\n"
            "7. Je t'explique ensuite les allocations principales, la contribution au risque et les métriques clés.\n"
            "8. Enfin, je propose un bilan claire sur le capital investi et la liquidité restante si tu as fourni un montant."
        )

    def _fuzzy_contains_any(self, text: str, keywords: list[str], threshold: float = 0.72) -> bool:
        normalized = self._normalize_text(text)
        tokens = normalized.split()
        for keyword in keywords:
            kw_norm = self._normalize_text(keyword)
            if kw_norm in normalized:
                return True
            if difflib.SequenceMatcher(None, normalized, kw_norm).ratio() >= threshold:
                return True
            if " " in kw_norm:
                continue
            for token in tokens:
                if difflib.SequenceMatcher(None, token, kw_norm).ratio() >= threshold:
                    return True
        return False

    def _is_range_request(self, user_input: str) -> bool:
        normalized = self._normalize_text(user_input)
        return any(word in normalized for word in ["plage", "range", "valeurs", "valeur"])

    def _is_greeting_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(user_input, ["bonjour", "salut", "hello", "coucou", "salam", "bonsoir", "hey", "hy"])

    def _is_thank_you_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(user_input, ["merci", "thanks", "top", "super", "parfait", "excellent", "bravo"])

    def _is_goodbye_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(
            user_input,
            ["au revoir", "a bientot", "bye", "goodbye", "ciao", "a plus", "a plus tard", "bonne journee", "bonne nuit", "fin", "quit"],
            threshold=0.68,
        )

    def _is_personal_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(user_input, ["qui es tu", "ton nom", "tu es", "que fais", "peux tu", "explique", "comment"])

    def _is_out_of_scope_request(self, user_input: str, intent: IntentResult) -> bool:
        normalized = self._normalize_text(user_input)
        if not normalized or len(normalized) <= 2:
            return False
        if self._has_finance_signal(normalized):
            return False
        if self._is_greeting_request(user_input) or self._is_thank_you_request(user_input) or self._is_goodbye_request(user_input):
            return False
        out_of_scope_terms = [
            "meteo",
            "weather",
            "recette",
            "cuisine",
            "sport",
            "football",
            "musique",
            "film",
            "serie",
            "voyage",
            "hotel",
            "restaurant",
            "programmation",
            "python",
            "java",
            "html",
            "maladie",
            "medicament",
            "medecin",
            "politique",
            "histoire",
            "traduire",
            "traduction",
            "poeme",
            "blague",
        ]
        if self._fuzzy_contains_any(normalized, out_of_scope_terms, threshold=0.78):
            return True
        if intent.intent != "unknown" and intent.confidence >= 0.28:
            return False
        question_words = ["qui", "quoi", "ou", "quand", "comment", "pourquoi", "combien", "quel", "quelle"]
        return normalized.endswith("?") or any(normalized.startswith(word + " ") for word in question_words)

    def _has_finance_signal(self, normalized: str) -> bool:
        finance_terms = [
            "portefeuille",
            "portfolio",
            "allocation",
            "investir",
            "investissement",
            "action",
            "actions",
            "bourse",
            "casablanca",
            "masi",
            "rendement",
            "return",
            "risque",
            "volatilite",
            "sharpe",
            "sortino",
            "var",
            "cvar",
            "drawdown",
            "covariance",
            "correlation",
            "diversification",
            "capital",
            "budget",
            "mad",
            "dh",
            "dirham",
            "secteur",
            "banque",
            "energie",
            "dividende",
            "croissance",
            "reequilibrage",
            "frontiere",
            "optimisation",
            "optimiser",
            "minimum volatilite",
            "risk parity",
        ]
        return self._fuzzy_contains_any(normalized, finance_terms, threshold=0.74)

    def _is_general_conversation_request(self, user_input: str) -> bool:
        if self._is_greeting_request(user_input) or self._is_thank_you_request(user_input) or self._is_personal_request(user_input):
            return True
        return self._fuzzy_contains_any(
            user_input,
            ["comment", "pourquoi", "quel", "quelle", "quels", "similaire", "difference", "conseil", "aide", "je veux", "tu peux", "dis moi"],
            threshold=0.66,
        )

    def _market_range_message(self) -> str:
        prices = self.data_service.load_price_data()
        if prices.empty:
            return "Aucune donnée de prix disponible pour calculer les plages."

        returns = self.data_service.compute_returns(prices)
        if returns.empty:
            return "Impossible de calculer les rendements à partir des données disponibles."

        annual_returns = returns.mean() * 252
        covariance = returns.cov() * 252
        results, _ = self._simulate_random_portfolios(annual_returns, covariance, num_portfolios=MAX_RANDOM_PORTFOLIOS)
        market_return_min = float(np.nanmin(annual_returns) * 100)
        market_return_max = float(np.nanmax(annual_returns) * 100)
        market_risk_min = float(np.min(results[1]))
        market_risk_max = float(np.nanmax(np.sqrt(np.diag(covariance))) * 100)
        best_idx = int(np.argmax(results[2]))
        best_sharpe_return = float(results[0, best_idx])
        best_sharpe_risk = float(results[1, best_idx])

        return (
            "## Plages possibles sur ce marché\n"
            f"{ResponseEngine.choose('market', 'range')}\n\n"
            f"- Rendement possible : **{market_return_min:.2f}%** à **{market_return_max:.2f}%**\n"
            f"- Risque possible : **{market_risk_min:.2f}%** à **{market_risk_max:.2f}%**\n"
            f"- Meilleur compromis observé : **{best_sharpe_return:.2f}%** de rendement pour **{best_sharpe_risk:.2f}%** de risque.\n"
            "Donne-moi ensuite un rendement cible (%) et un risque maximum (%) pour construire un portefeuille."
        )

    def _build_portfolio_result(
        self,
        rendement_cible_pct: float,
        risque_max_pct: float,
        capital: float | None = None,
        explain_method: bool = False,
        user_input: str | None = None,
    ) -> AgentResponse:
        prices = self.data_service.load_price_data()
        if prices.empty:
            return AgentResponse(content="Erreur technique : aucun historique de prix exploitable n'a été détecté.")

        returns = self.data_service.compute_returns(prices)
        if returns.empty:
            return AgentResponse(content="Erreur technique : impossible de calculer les rendements.")

        annual_returns = returns.mean() * 252
        covariance = returns.cov() * 252
        num_assets = len(returns.columns)
        if num_assets == 0:
            return AgentResponse(content="Erreur technique : aucun actif valide n'a été détecté.")

        results, weights_record = self._simulate_random_portfolios(annual_returns, covariance, num_portfolios=MAX_RANDOM_PORTFOLIOS)
        market_return_min = float(np.nanmin(annual_returns) * 100)
        market_return_max = float(np.nanmax(annual_returns) * 100)
        market_risk_min = float(np.min(results[1]))
        market_risk_max = float(np.nanmax(np.sqrt(np.diag(covariance))) * 100)
        best_idx = int(np.argmax(results[2]))
        best_sharpe_return = float(results[0, best_idx])
        best_sharpe_risk = float(results[1, best_idx])
        if prices.empty:
            return AgentResponse(content="Erreur technique : aucun historique de prix exploitable n'a été détecté.")
        single_asset_idx = np.where(
            (annual_returns * 100 >= rendement_cible_pct) &
            (np.sqrt(np.diag(covariance)) * 100 <= risque_max_pct)
        )[0]
        if len(single_asset_idx) > 0:
            idx = int(single_asset_idx[np.argmax((annual_returns * 100)[single_asset_idx])])
            weights = np.zeros(num_assets, dtype=float)
            weights[idx] = 1.0
            return self._build_response(
                weights,
                returns,
                covariance,
                prices,
                rendement_cible_pct,
                risque_max_pct,
                capital,
                market_return_min,
                market_return_max,
                best_sharpe_return,
                best_sharpe_risk,
                single_asset=True,
                explain_method=explain_method,
                user_input=user_input,
            )

        valid_indices = np.where(
            (results[0] >= rendement_cible_pct) &
            (results[1] <= risque_max_pct)
        )[0]
        if len(valid_indices) == 0:
            direct_response = self._try_direct_optimization(
                rendement_cible_pct,
                risque_max_pct,
                returns,
                covariance,
                prices,
                capital,
                market_return_min,
                market_return_max,
                best_sharpe_return,
                best_sharpe_risk,
                explain_method=explain_method,
                user_input=user_input,
            )
            if direct_response is not None:
                return direct_response

            nearest_idx = self._closest_portfolio_index(results, rendement_cible_pct, risque_max_pct)
            weights = weights_record[nearest_idx]
            response = self._build_nearest_match_response(
                weights,
                returns,
                covariance,
                prices,
                rendement_cible_pct,
                risque_max_pct,
                capital,
                market_return_min,
                market_return_max,
                best_sharpe_return,
                best_sharpe_risk,
                explain_method=explain_method,
                user_input=user_input,
            )
            return response

        top_indices = self._select_diverse_portfolios(
            valid_indices,
            results,
            weights_record,
            covariance,
            limit=TOP_PORTFOLIO_CHOICES,
        )
        if len(top_indices) > 1:
            return self._build_multi_choice_response(
                top_indices,
                results,
                weights_record,
                returns,
                covariance,
                prices,
                rendement_cible_pct,
                risque_max_pct,
                capital,
                market_return_min,
                market_return_max,
                best_sharpe_return,
                best_sharpe_risk,
                explain_method=explain_method,
                user_input=user_input,
            )

        best_idx = int(top_indices[0])
        optimal_weights = weights_record[best_idx]
        return self._build_response(
            optimal_weights,
            returns,
            covariance,
            prices,
            rendement_cible_pct,
            risque_max_pct,
            capital,
            market_return_min,
            market_return_max,
            best_sharpe_return,
            best_sharpe_risk,
            single_asset=False,
        )

    def _try_direct_optimization(
        self,
        rendement_cible_pct: float,
        risque_max_pct: float,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        prices: pd.DataFrame,
        capital: float | None,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
        explain_method: bool = False,
        user_input: str | None = None,
    ) -> AgentResponse | None:
        optimizer = PortfolioOptimizer(
            annual_returns=returns.mean() * 252,
            covariance=covariance,
        )
        target_return = rendement_cible_pct / 100.0
        optimization = optimizer.optimize_target_return(target_return)
        if optimization.success:
            candidate = self.portfolio_engine.build_portfolio(
                name="optimized_target_return",
                weights=optimization.weights,
                returns=returns,
                covariance=covariance,
                asset_labels=self.data_service.load_asset_metadata(),
                price_history=prices,
                target_return=target_return,
                max_risk=risque_max_pct / 100.0,
            )
            if candidate.metrics.volatility * 100 <= risque_max_pct + 0.05:
                response = self._build_response(
                    optimization.weights,
                    returns,
                    covariance,
                    prices,
                    rendement_cible_pct,
                    risque_max_pct,
                    capital,
                    market_return_min,
                    market_return_max,
                    best_sharpe_return,
                    best_sharpe_risk,
                    single_asset=False,
                    explain_method=explain_method,
                    user_input=user_input,
                )
                response.content = (
                    "## Optimisation directe réussie\n"
                    "J'ai utilisé une optimisation dédiée pour respecter au mieux ton rendement cible.\n\n"
                    + response.content
                )
                return response

        max_sharpe = optimizer.optimize_max_sharpe()
        min_variance_return = optimizer.optimize_minimum_variance_for_return(target_return)
        if min_variance_return.success:
            candidate = self.portfolio_engine.build_portfolio(
                name="optimized_minimum_variance_return",
                weights=min_variance_return.weights,
                returns=returns,
                covariance=covariance,
                asset_labels=self.data_service.load_asset_metadata(),
                price_history=prices,
                target_return=target_return,
                max_risk=risque_max_pct / 100.0,
            )
            if candidate.metrics.volatility * 100 <= risque_max_pct + 0.05:
                response = self._build_response(
                    min_variance_return.weights,
                    returns,
                    covariance,
                    prices,
                    rendement_cible_pct,
                    risque_max_pct,
                    capital,
                    market_return_min,
                    market_return_max,
                    best_sharpe_return,
                    best_sharpe_risk,
                    single_asset=False,
                    explain_method=explain_method,
                    user_input=user_input,
                )
                response.content = (
                    "## Optimisation risque minimale réussie\n"
                    "J'ai cherché le portefeuille à volatilité minimale qui respecte ton objectif de rendement.\n\n"
                    + response.content
                )
                return response

        if max_sharpe.success:
            candidate = self.portfolio_engine.build_portfolio(
                name="optimized_max_sharpe",
                weights=max_sharpe.weights,
                returns=returns,
                covariance=covariance,
                asset_labels=self.data_service.load_asset_metadata(),
                price_history=prices,
                target_return=target_return,
                max_risk=risque_max_pct / 100.0,
            )
            if candidate.metrics.expected_return * 100 >= rendement_cible_pct and candidate.metrics.volatility * 100 <= risque_max_pct + 0.05:
                response = self._build_response(
                    max_sharpe.weights,
                    returns,
                    covariance,
                    prices,
                    rendement_cible_pct,
                    risque_max_pct,
                    capital,
                    market_return_min,
                    market_return_max,
                    best_sharpe_return,
                    best_sharpe_risk,
                    single_asset=False,
                    explain_method=explain_method,
                    user_input=user_input,
                )
                response.content = (
                    "## Optimisation Sharpe réussie\n"
                    "J'ai identifié un portefeuille optimisé pour le meilleur Sharpe disponible.\n\n"
                    + response.content
                )
                return response

        return None

    def _select_diverse_portfolios(
        self,
        candidate_indices: np.ndarray,
        results: np.ndarray,
        weights_record: list[np.ndarray],
        covariance: pd.DataFrame,
        limit: int,
    ) -> np.ndarray:
        scored = sorted(
            (int(idx) for idx in candidate_indices),
            key=lambda idx: self._portfolio_quality_score(weights_record[idx], results[:, idx], covariance),
            reverse=True,
        )
        selected: list[int] = []
        for idx in scored:
            weights = weights_record[idx]
            if not selected:
                selected.append(idx)
            else:
                min_distance = min(float(np.sum(np.abs(weights - weights_record[chosen]))) for chosen in selected)
                if min_distance >= 0.35:
                    selected.append(idx)
            if len(selected) >= limit:
                break
        if len(selected) < min(limit, len(scored)):
            for idx in scored:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= limit:
                    break
        return np.array(selected, dtype=int)

    def _portfolio_quality_score(self, weights: np.ndarray, result_column: np.ndarray, covariance: pd.DataFrame) -> float:
        portfolio_return, volatility, sharpe = [float(x) for x in result_column[:3]]
        hhi = float(np.sum(np.square(weights)))
        effective_assets = 1.0 / hhi if hhi > 0 else 0.0
        max_weight = float(np.max(weights)) if len(weights) else 1.0
        corr_exposure = self._correlation_exposure(weights, covariance)
        concentration_penalty = 1.8 * hhi + 0.8 * max(0.0, max_weight - 0.22)
        diversification_bonus = min(effective_assets / 12.0, 1.0) * 0.45
        stability_bonus = max(0.0, 1.0 - volatility / 35.0) * 0.25
        return sharpe + diversification_bonus + stability_bonus + portfolio_return / 500.0 - concentration_penalty - corr_exposure * 0.35

    def _correlation_exposure(self, weights: np.ndarray, covariance: pd.DataFrame) -> float:
        corr = covariance.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        if corr.shape[0] != len(weights):
            return 0.0
        weight_outer = np.outer(weights, weights)
        off_diagonal = ~np.eye(len(weights), dtype=bool)
        return float(np.sum(np.abs(corr[off_diagonal]) * weight_outer[off_diagonal]))

    def _closest_portfolio_index(self, results: np.ndarray, rendement_cible_pct: float, risque_max_pct: float) -> int:
        target = np.array([rendement_cible_pct, risque_max_pct], dtype=float)
        return_gaps = np.maximum(target[0] - results[0], 0.0)
        risk_gaps = np.maximum(results[1] - target[1], 0.0)
        scaling = np.maximum(np.abs(target), 1.0)
        errors = np.sqrt((return_gaps / scaling[0]) ** 2 + (risk_gaps / scaling[1]) ** 2)
        return int(np.nanargmin(errors))

    def _build_nearest_match_response(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        prices: pd.DataFrame,
        rendement_cible_pct: float,
        risque_max_pct: float,
        capital: float | None,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
        explain_method: bool = False,
        user_input: str | None = None,
    ) -> AgentResponse:
        portfolio = self.portfolio_engine.build_portfolio(
            name="nearest_match",
            weights=weights,
            returns=returns,
            covariance=covariance,
            asset_labels=self.data_service.load_asset_metadata(),
            price_history=prices,
            target_return=rendement_cible_pct / 100.0,
            max_risk=risque_max_pct / 100.0,
        )
        response = self._build_response(
            weights,
            returns,
            covariance,
            prices,
            rendement_cible_pct,
            risque_max_pct,
            capital,
            market_return_min,
            market_return_max,
            best_sharpe_return,
            best_sharpe_risk,
            single_asset=False,
            explain_method=explain_method,
            user_input=user_input,
        )
        warning_lines = []
        if portfolio.metrics.volatility * 100 > risque_max_pct + 0.05:
            warning_lines.append(
                f"- Risque réalisé : **{portfolio.metrics.volatility * 100:.2f}%** (supérieur à la limite demandée de {risque_max_pct:.2f}%)."
            )
        if portfolio.metrics.expected_return * 100 < rendement_cible_pct - 0.05:
            warning_lines.append(
                f"- Rendement réalisé : **{portfolio.metrics.expected_return * 100:.2f}%** (inférieur à l'objectif demandé de {rendement_cible_pct:.2f}%)."
            )
        warning = "\n".join(warning_lines)
        if warning:
            warning = "\n" + warning + "\n"

        response.content = (
            "## Proposition la plus proche\n"
            "Je n'ai pas trouvé de portefeuille qui respecte exactement tes contraintes, mais voici le meilleur compromis le plus proche.\n"
            f"{warning}\n"
            + response.content
        )
        return response

    def _build_response(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        prices: pd.DataFrame,
        rendement_cible_pct: float,
        risque_max_pct: float,
        capital: float | None,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
        single_asset: bool,
        explain_method: bool = False,
        user_input: str | None = None,
    ) -> AgentResponse:
        portfolio = self.portfolio_engine.build_portfolio(
            name="local_random_search",
            weights=weights,
            returns=returns,
            covariance=covariance,
            asset_labels=self.data_service.load_asset_metadata(),
            price_history=prices,
            target_return=rendement_cible_pct / 100.0,
            max_risk=risque_max_pct / 100.0,
        )
        self._store_portfolio_history(portfolio)
        title = "Portefeuille proposé (actif unique)" if single_asset else "Portefeuille proposé"
        markdown = self._format_portfolio_markdown(
            title,
            portfolio,
            prices.iloc[-1],
            capital,
            rendement_cible_pct,
            risque_max_pct,
            market_return_min,
            market_return_max,
            best_sharpe_return,
            best_sharpe_risk,
        )
        if explain_method:
            explanation = ExplanationEngine.explain_portfolio(portfolio, user_input or "", style="beginner")
            markdown = (
                "## Démarche suivie\n"
                + self._method_explanation_message()
                + "\n\n"
                + markdown
            )
            return AgentResponse(content=markdown, structured=portfolio, explanation=explanation)
        return AgentResponse(content=markdown, structured=portfolio)

    def _format_portfolio_markdown(
        self,
        title: str,
        portfolio: PortfolioResult,
        latest_prices: pd.Series,
        capital: float | None,
        rendement_cible_pct: float,
        risque_max_pct: float,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
    ) -> str:
        if not portfolio.allocations:
            return "Aucune allocation significative n'a été retenue."

        profile_summary = self.memory_service.get_profile_summary()
        estimated_loss_prob = self._approximate_loss_probability(
            portfolio.metrics.expected_return, portfolio.metrics.volatility
        )
        var_95 = abs(portfolio.metrics.var_95) * 100
        cvar_95 = abs(portfolio.metrics.cvar_95) * 100
        lines = [
            f"## {title}\n",
            "_⚠️ Avertissement : ces résultats sont des estimations historiques. Ils ne garantissent pas les performances futures. La valeur du portefeuille peut fluctuer et une perte est possible._\n",
            "_Ces chiffres sont calculés à partir des données locales disponibles et doivent rester une base de réflexion plutôt qu'une promesse._\n",
            "\n",
            f"- Rendement estimé : **{portfolio.metrics.expected_return * 100:.2f}%**\n",
            f"  - Cela représente le gain moyen annuel attendu si les tendances passées se répètent. Ce n'est pas une promesse de gain.\n",
            f"- Risque estimé : **{portfolio.metrics.volatility * 100:.2f}%**\n",
            "  - C'est une mesure de la variation possible du portefeuille : plus le risque est élevé, plus les gains ou pertes peuvent être importants.\n",
            f"- Sharpe : **{portfolio.metrics.sharpe_ratio:.2f}**\n",
            "  - Il montre si le rendement attendu est bon par rapport au risque pris. Un Sharpe plus élevé est généralement meilleur.\n",
            f"- Sortino : **{portfolio.metrics.sortino_ratio:.2f}**\n",
            "  - Semblable au Sharpe, mais il prend en compte surtout les pertes. C'est utile si tu veux limiter les baisses.\n",
            f"- VaR 95% : **{var_95:.2f}%**\n",
            "  - Dans les 5 % des pires cas historiques, c'est la perte maximale que l'on peut attendre.\n",
            f"- CVaR 95% : **{cvar_95:.2f}%**\n",
            "  - En cas de très mauvais scénario, c'est la perte moyenne des cas les plus extrêmes.\n",
            f"- Drawdown historique maximum : **{portfolio.metrics.max_drawdown * 100:.2f}%**\n",
            "  - C'est la plus grande baisse observée depuis un sommet jusqu'à un creux.\n",
            f"- Probabilité approximative d'une année perdante : **{estimated_loss_prob * 100:.1f}%**\n",
            "  - C'est une estimation de la chance que ce portefeuille perde de la valeur sur une année.\n",
            f"- Concentration HHI : **{portfolio.metrics.concentration_score:.3f}**\n",
            "  - Mesure si le portefeuille est trop centré sur quelques actions. Plus c'est bas, plus la répartition est équilibrée.\n",
            f"- Nombre effectif d'actifs : **{portfolio.metrics.effective_number_of_assets:.2f}**\n",
            "  - Indique combien d'actifs sont réellement influents dans le portefeuille.\n",
            f"- Ratio de diversification : **{portfolio.metrics.diversification_ratio:.2f}**\n",
            "  - Montre si les actifs se complètent bien ou s'ils ont tendance à évoluer ensemble.\n",
            f"- Exposition aux corrélations : **{portfolio.metrics.correlation_exposure:.3f}**\n",
            "  - Si les actifs sont fortement corrélés, le portefeuille peut chuter plus fort en cas de mauvaise nouvelle.\n",
        ]
        if profile_summary and profile_summary != "Aucun profil enregistré pour le moment.":
            lines.append(f"- Profil actuel : {profile_summary}\n")
        lines.extend([
            "\n",
            f"## Contexte\n",
            f"- Objectif demandé : rendement >= **{rendement_cible_pct:.2f}%**, risque <= **{risque_max_pct:.2f}%**\n",
            f"- Plages de marché observées : rendement **{market_return_min:.2f}%** à **{market_return_max:.2f}%**, risque approximatif **{portfolio.metrics.volatility * 100:.2f}%**\n",
            f"- Meilleur compromis observé (Sharpe max) : **{best_sharpe_return:.2f}%** de rendement pour **{best_sharpe_risk:.2f}%** de risque\n\n",
            f"- Lecture diversification : {ExplanationEngine.explain_allocation(portfolio)}\n\n",
            f"## Allocations principales\n",
            ])

        weights = pd.Series(portfolio.allocations).sort_values(ascending=False)
        output_lines = []
        if capital and capital > 0:
            output_lines = [
                "| Entreprise | Ticker | Poids | Quantité | Montant |",
                "|---|---|---:|---:|---:|",
            ]
        else:
            output_lines = [
                "| Entreprise | Ticker | Poids |",
                "|---|---|---:|",
            ]

        asset_names = portfolio.asset_names or {}
        invested = 0.0
        remaining_cash = float(capital or 0.0)
        prices_dict = latest_prices.to_dict() if isinstance(latest_prices, pd.Series) else dict(latest_prices)

        if capital is not None and capital > 0:
            items = []
            for ticker, weight in weights.items():
                label = asset_names.get(ticker, ticker)
                price = float(prices_dict.get(ticker, 0.0))
                if price <= 0:
                    continue
                target_amount = capital * weight
                shares = int(target_amount // price)
                items.append({
                    "label": label,
                    "ticker": ticker,
                    "weight": weight,
                    "price": price,
                    "shares": shares,
                    "target_amount": target_amount,
                })

            invested = float(sum(item["shares"] * item["price"] for item in items))
            remaining_cash = capital - invested

            def update_remainder(item: dict[str, float]) -> float:
                return item["target_amount"] - item["shares"] * item["price"]

            while True:
                affordable = [item for item in items if remaining_cash >= item["price"]]
                if not affordable:
                    break
                candidate = max(
                    affordable,
                    key=lambda item: (
                        update_remainder(item) / item["price"],
                        item["weight"],
                        -item["price"],
                    ),
                )
                candidate["shares"] += 1
                remaining_cash -= candidate["price"]
                invested += candidate["price"]

            for item in items:
                if item["shares"] <= 0:
                    continue
                output_lines.append(
                    f"| {item['label']} | {item['ticker']} | {item['weight'] * 100:.2f}% | {item['shares']} | {item['shares'] * item['price']:,.2f} MAD |"
                )
        else:
            for ticker, weight in weights.items():
                label = asset_names.get(ticker, ticker)
                output_lines.append(
                    f"| {label} | {ticker} | {weight * 100:.2f}% |"
                )

        lines.extend(output_lines)
        if capital is not None and capital > 0:
            lines.extend([
                "\n💡 **Bilan du portefeuille :**",
                f"- Capital initial : {capital:,.2f} MAD",
                f"- Montant investi : {invested:,.2f} MAD",
                f"- Liquidité restante : {remaining_cash:,.2f} MAD",
            ])

        # Si certaines positions ont été omises car trop petites, prévenir l'utilisateur
        if capital is not None and capital > 0 and invested < capital:
            lines.append("\n_Remarque : les positions trop petites pour acheter au moins une action ont été omises du tableau._")

        lines.append(
            "\n💬 Si tu veux, je peux aussi proposer une variante plus prudente, plus agressive, ou avec un horizon différent."
        )

        return "\n".join(lines)

    def _approximate_loss_probability(self, expected_return: float, volatility: float) -> float:
        if volatility <= 0:
            return 0.0 if expected_return >= 0 else 1.0
        return float(norm.cdf(-expected_return / volatility))

    def _store_portfolio_history(self, portfolio: PortfolioResult) -> None:
        self.memory_service.append_portfolio(
            {
                "name": portfolio.name,
                "expected_return": f"{portfolio.metrics.expected_return * 100:.2f}%",
                "volatility": f"{portfolio.metrics.volatility * 100:.2f}%",
                "sharpe": f"{portfolio.metrics.sharpe_ratio:.2f}",
                "target_return": f"{portfolio.target_return * 100:.2f}%" if portfolio.target_return is not None else "N/A",
                "max_risk": f"{portfolio.max_risk * 100:.2f}%" if portfolio.max_risk is not None else "N/A",
            }
        )

    def _build_multi_choice_response(
        self,
        top_indices: np.ndarray,
        results: np.ndarray,
        weights_record: list[np.ndarray],
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        prices: pd.DataFrame,
        rendement_cible_pct: float,
        risque_max_pct: float,
        capital: float | None,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
        explain_method: bool = False,
        user_input: str | None = None,
    ) -> AgentResponse:
        best_weight = weights_record[int(top_indices[0])]
        response = self._build_response(
            best_weight,
            returns,
            covariance,
            prices,
            rendement_cible_pct,
            risque_max_pct,
            capital,
            market_return_min,
            market_return_max,
            best_sharpe_return,
            best_sharpe_risk,
            single_asset=False,
            explain_method=explain_method,
            user_input=user_input,
        )

        option_lines = [
            "\n## Autres portefeuilles possibles\n",
            "J'ai trouvé plusieurs portefeuilles compatibles avec tes contraintes. Voici les trois meilleurs en fonction du Sharpe ratio :\n",
        ]

        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            weight = weights_record[idx]
            portfolio = self.portfolio_engine.build_portfolio(
                name=f"choice_{rank}",
                weights=weight,
                returns=returns,
                covariance=covariance,
                asset_labels=self.data_service.load_asset_metadata(),
                price_history=prices,
                target_return=rendement_cible_pct / 100.0,
                max_risk=risque_max_pct / 100.0,
            )
            option_lines.append(
                f"### Option {rank} : Sharpe {portfolio.metrics.sharpe_ratio:.2f}, rendement {portfolio.metrics.expected_return * 100:.2f}%, risque {portfolio.metrics.volatility * 100:.2f}%\n"
            )
            top_allocation = pd.Series(portfolio.allocations).sort_values(ascending=False)
            if top_allocation.empty:
                option_lines.append("- (Aucune allocation significative)\n")
            else:
                top_n = len(top_allocation)
                for ticker, weight_value in top_allocation.head(top_n).items():
                    name = portfolio.asset_names.get(ticker, ticker)
                    option_lines.append(f"- {name} ({ticker}) : {weight_value * 100:.2f}%\n")
            option_lines.append("\n")
            # store option details for later selection/comparison
            try:
                opt_md = self._format_portfolio_markdown(
                    f"Option {rank}",
                    portfolio,
                    prices.iloc[-1],
                    capital,
                    rendement_cible_pct,
                    risque_max_pct,
                    market_return_min,
                    market_return_max,
                    best_sharpe_return,
                    best_sharpe_risk,
                )
            except Exception:
                opt_md = ""
            self.memory_service.append_portfolio({
                "option_rank": rank,
                "markdown": opt_md,
                "expected_return": f"{portfolio.metrics.expected_return * 100:.2f}%",
                "volatility": f"{portfolio.metrics.volatility * 100:.2f}%",
                "sharpe": f"{portfolio.metrics.sharpe_ratio:.2f}",
                "allocations": {k: f"{v*100:.2f}%" for k, v in portfolio.allocations.items()},
                "timestamp": time.time(),
            })

        deterministic_options = self._build_named_portfolio_options(
            returns,
            covariance,
            prices,
            rendement_cible_pct,
            risque_max_pct,
            capital,
            market_return_min,
            market_return_max,
            best_sharpe_return,
            best_sharpe_risk,
        )
        if deterministic_options:
            option_lines.append("\n## Profils stratégiques complémentaires\n")
            option_lines.append("Ces profils utilisent les mêmes données locales, mais changent explicitement l'objectif d'optimisation :\n")
            next_rank = len(top_indices) + 1
            for label, portfolio, markdown in deterministic_options:
                option_lines.append(
                    f"### Option {next_rank} : {label}\n"
                    f"- Rendement : **{portfolio.metrics.expected_return * 100:.2f}%**\n"
                    f"- Risque : **{portfolio.metrics.volatility * 100:.2f}%**\n"
                    f"- Sharpe : **{portfolio.metrics.sharpe_ratio:.2f}**\n"
                    f"- Actifs effectifs : **{portfolio.metrics.effective_number_of_assets:.2f}**\n"
                    f"- Concentration HHI : **{portfolio.metrics.concentration_score:.3f}**\n\n"
                )
                self.memory_service.append_portfolio({
                    "option_rank": next_rank,
                    "markdown": markdown,
                    "expected_return": f"{portfolio.metrics.expected_return * 100:.2f}%",
                    "volatility": f"{portfolio.metrics.volatility * 100:.2f}%",
                    "sharpe": f"{portfolio.metrics.sharpe_ratio:.2f}",
                    "allocations": {k: f"{v*100:.2f}%" for k, v in portfolio.allocations.items()},
                    "timestamp": time.time(),
                    "mode": label,
                })
                next_rank += 1

        response.content += (
            "\n\n⚠️ J'ai trouvé plusieurs portefeuilles compatibles avec tes critères, mais je te présente d'abord la meilleure option. "
            "Si tu veux en voir une autre variante, demande par exemple : 'option 2', 'montre-moi une autre option' ou 'voir plus d'options'."
        )
        return response

    def _build_named_portfolio_options(
        self,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
        prices: pd.DataFrame,
        rendement_cible_pct: float,
        risque_max_pct: float,
        capital: float | None,
        market_return_min: float,
        market_return_max: float,
        best_sharpe_return: float,
        best_sharpe_risk: float,
    ) -> list[tuple[str, PortfolioResult, str]]:
        optimizer = PortfolioOptimizer(annual_returns=returns.mean() * 252, covariance=covariance)
        mode_results = [
            ("Conservateur / minimum volatilité", optimizer.optimize_minimum_volatility()),
            ("Équilibré / utilité rendement-risque", optimizer.optimize_balanced()),
            ("Agressif / Sharpe maximum", optimizer.optimize_max_sharpe()),
            ("Diversification maximisée", optimizer.optimize_diversification()),
            ("Parité de risque", optimizer.optimize_risk_parity()),
            ("Poids égaux", optimizer.equal_weight()),
        ]
        options: list[tuple[str, PortfolioResult, str]] = []
        seen_signatures: list[np.ndarray] = []
        for label, opt_result in mode_results:
            if not opt_result.success:
                continue
            weights = opt_result.weights
            if any(float(np.sum(np.abs(weights - seen))) < 0.20 for seen in seen_signatures):
                continue
            seen_signatures.append(weights)
            portfolio = self.portfolio_engine.build_portfolio(
                name=label.lower().replace(" / ", "_").replace(" ", "_"),
                weights=weights,
                returns=returns,
                covariance=covariance,
                asset_labels=self.data_service.load_asset_metadata(),
                price_history=prices,
                target_return=rendement_cible_pct / 100.0,
                max_risk=risque_max_pct / 100.0,
            )
            markdown = self._format_portfolio_markdown(
                label,
                portfolio,
                prices.iloc[-1],
                capital,
                rendement_cible_pct,
                risque_max_pct,
                market_return_min,
                market_return_max,
                best_sharpe_return,
                best_sharpe_risk,
            )
            options.append((label, portfolio, markdown))
        return options

    def _is_option_selection(self, user_input: str) -> int | None:
        m = re.search(r"\b(?:option|choix)\s*(\d+)\b", user_input, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _get_recent_option_entries(self) -> list[dict]:
        entries = self.memory_service.get_portfolio_history()
        option_entries = [e for e in entries if isinstance(e, dict) and e.get("option_rank")]
        if not option_entries:
            return []
        option_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        latest_ts = option_entries[0].get("timestamp")
        recent = [e for e in option_entries if abs(e.get("timestamp", 0) - latest_ts) < 30]
        recent.sort(key=lambda x: x.get("option_rank"))
        return recent

    def _provide_option_details(self, option_number: int) -> AgentResponse:
        recent = self._get_recent_option_entries()
        if not recent:
            return AgentResponse(content="Aucune option en mémoire. Génère d'abord plusieurs options.")
        for e in recent:
            if int(e.get("option_rank", -1)) == option_number:
                md = e.get("markdown") or "Détails non disponibles pour cette option."
                return AgentResponse(content=md)
        return AgentResponse(content=f"L'option {option_number} n'a pas été trouvée parmi les dernières propositions.")

    def _is_more_options_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(
            user_input,
            [
                "plus d'options",
                "plus d options",
                "voir plus d'options",
                "voir plus d options",
                "autres options",
                "d'autres options",
                "encore des options",
                "autre option",
                "options supplémentaires",
            ],
            threshold=0.67,
        )

    def _list_recent_options(self) -> AgentResponse:
        recent = self._get_recent_option_entries()
        if not recent:
            return AgentResponse(content="Je n'ai pas d'option récente en mémoire. Génère d'abord un portefeuille pour que je puisse afficher les variantes possibles.")

        lines = [
            "## Options disponibles",
            "Voici les options récentes que je peux détailler :",
            "",
        ]
        for e in recent:
            rank = int(e.get("option_rank", -1))
            lines.append(
                f"- Option {rank} : rendement {e.get('expected_return', 'N/A')}, risque {e.get('volatility', 'N/A')}, Sharpe {e.get('sharpe', 'N/A')}"
            )
        lines.extend(
            [
                "",
                "Tape `option N` pour voir les détails complets d'une option.",
                "Tu peux aussi comparer deux options en écrivant par exemple `compare option 1 et option 2`.",
            ]
        )
        return AgentResponse(content="\n".join(lines))

    def _is_compare_request(self, user_input: str) -> bool:
        return self._fuzzy_contains_any(user_input, ["compare", "comparer", "comparez"], threshold=0.7)

    def _parse_compare_indices(self, user_input: str) -> tuple[int, int] | None:
        m = re.search(r"(\d+)\s*(?:et|,|\s)\s*(\d+)", user_input)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                return None
        return None

    def _compare_options(self, idx_a: int, idx_b: int) -> AgentResponse:
        recent = self._get_recent_option_entries()
        if not recent or len(recent) < 2:
            return AgentResponse(content="Pas assez d'options récentes pour comparer. Génère d'abord plusieurs options.")
        map_by_rank = {int(e.get("option_rank")): e for e in recent}
        a = map_by_rank.get(idx_a)
        b = map_by_rank.get(idx_b)
        if not a or not b:
            return AgentResponse(content="Impossible de trouver les options demandées pour la comparaison.")

        md_lines = [
            "## Comparaison des options",
            f"\n| Métrique | Option {idx_a} | Option {idx_b} |",
            "|---:|:---:|:---:|",
            f"| Rendement estimé | {a.get('expected_return','N/A')} | {b.get('expected_return','N/A')} |",
            f"| Risque (volatilité) | {a.get('volatility','N/A')} | {b.get('volatility','N/A')} |",
            f"| Sharpe | {a.get('sharpe','N/A')} | {b.get('sharpe','N/A')} |",
            "\n### Principales allocations pour Option {}".format(idx_a),
        ]
        for k, v in list(a.get('allocations', {}).items())[:6]:
            md_lines.append(f"- {k} : {v}")

        md_lines.append("\n### Principales allocations pour Option {}".format(idx_b))
        for k, v in list(b.get('allocations', {}).items())[:6]:
            md_lines.append(f"- {k} : {v}")

        return AgentResponse(content="\n".join(md_lines))

    def _compare_multiple_options(self, indices: list[int]) -> AgentResponse:
        recent = self._get_recent_option_entries()
        if not recent or len(recent) < 2:
            return AgentResponse(content="Pas assez d'options récentes pour comparer. Génère d'abord plusieurs options.")

        map_by_rank = {int(e.get("option_rank")): e for e in recent}
        entries = [map_by_rank.get(idx) for idx in indices if map_by_rank.get(idx)]
        if len(entries) < 2:
            return AgentResponse(content="Impossible de trouver les options demandées pour la comparaison.")

        header = ["| Métrique |"] + [f" Option {idx} |" for idx in indices]
        separator = ["|---:|"] + [":---:|" for _ in indices]
        rows = [
            "".join(header),
            "".join(separator),
        ]
        metric_keys = [
            ("Rendement estimé", "expected_return"),
            ("Risque (volatilité)", "volatility"),
            ("Sharpe", "sharpe"),
        ]
        for label, key in metric_keys:
            row = [f"| {label} |"]
            for entry in entries:
                row.append(f" {entry.get(key, 'N/A')} |")
            rows.append("".join(row))

        md_lines = ["## Comparaison des options", ""] + rows + [""]

        for entry in entries:
            rank = entry.get("option_rank")
            md_lines.append(f"### Principales allocations Option {rank}")
            allocations = entry.get("allocations", {})
            if not allocations:
                md_lines.append("- Aucune allocation disponible")
            else:
                for ticker, pct in list(allocations.items())[:6]:
                    md_lines.append(f"- {ticker} : {pct}")
            md_lines.append("")

        md_lines.append("Tape 'option N' pour voir les détails complets de l'option correspondante.")
        return AgentResponse(content="\n".join(md_lines))

    def _simulate_random_portfolios(
        self,
        annual_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        num_portfolios: int = 5000,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        num_assets = len(annual_returns)
        results = np.zeros((3, num_portfolios))
        weights_record: list[np.ndarray] = []

        for i in range(num_portfolios):
            weights = np.zeros(num_assets, dtype=float)
            max_assets_in_pf = min(20, num_assets)
            min_assets_in_pf = min(max(4, num_assets // 8), max_assets_in_pf)
            num_selected = int(self.rng.integers(min_assets_in_pf, max_assets_in_pf + 1))
            selected_indices = self.rng.choice(num_assets, num_selected, replace=False)
            alpha = float(self.rng.choice([0.65, 1.0, 1.5, 2.5]))
            random_w = self.rng.dirichlet(np.repeat(alpha, num_selected))
            max_weight = 0.24 if num_selected >= 6 else 0.32
            random_w = np.minimum(random_w, max_weight)
            random_w = random_w / np.sum(random_w)
            if self.rng.random() < 0.35:
                random_w = self._controlled_perturbation(random_w, max_weight=max_weight)
            weights[selected_indices] = random_w
            weights_record.append(weights)

            portfolio_return = float(np.dot(weights, annual_returns.values))
            portfolio_volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights))
            results[0, i] = portfolio_return * 100
            results[1, i] = portfolio_volatility * 100
            results[2, i] = (
                portfolio_return / portfolio_volatility if portfolio_volatility > 0 else -np.inf
            )

        return results, weights_record

    def _controlled_perturbation(self, weights: np.ndarray, max_weight: float = 0.25) -> np.ndarray:
        noise = self.rng.normal(0.0, 0.025, size=len(weights))
        perturbed = np.maximum(weights + noise, 0.001)
        perturbed = np.minimum(perturbed, max_weight)
        return perturbed / np.sum(perturbed)
