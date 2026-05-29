from __future__ import annotations

import hashlib
from typing import Any

from services.memory_service import MemoryService


def _expand(prefixes: list[str], middles: list[str], suffixes: list[str], limit: int = 120) -> list[str]:
    responses: list[str] = []
    for prefix in prefixes:
        for middle in middles:
            for suffix in suffixes:
                responses.append(f"{prefix} {middle} {suffix}".strip())
                if len(responses) >= limit:
                    return responses
    return responses


class ResponseEngine:
    GREETINGS = _expand(
        ["Bonjour.", "Salut.", "Bonsoir.", "Hello.", "Content de te retrouver."],
        [
            "Je peux analyser un portefeuille, expliquer une notion financière ou construire une allocation locale.",
            "On peut travailler sur le risque, le rendement, la diversification ou une stratégie complète.",
            "Je garde le calcul déterministe et je peux adapter l'explication à ton profil.",
            "Dis-moi si tu veux optimiser, comparer, apprendre ou cadrer une stratégie.",
            "Je peux raisonner comme assistant quantitatif tout en restant offline.",
        ],
        [
            "Quelle décision veux-tu éclairer aujourd'hui ?",
            "Donne-moi ton objectif, même en langage naturel.",
            "Je peux commencer avec peu d'information et te poser les bonnes questions.",
            "Si tu as déjà un profil en tête, je l'utiliserai comme contexte.",
            "On peut avancer étape par étape.",
        ],
    )
    CLARIFICATIONS = _expand(
        ["Pour cadrer proprement la demande,", "Il me manque une pièce utile:", "Je peux avancer, mais j'aimerais préciser ceci:", "Avant de calculer,", "Pour éviter une allocation mal calibrée,"],
        [
            "indique le rendement cible ou le niveau de risque maximal.",
            "choisis plutôt prudent, équilibré ou agressif.",
            "donne l'horizon de placement.",
            "précise le capital à investir si tu veux des quantités d'actions.",
            "dis si tu privilégies croissance, revenus ou stabilité.",
        ],
        [
            "Une phrase simple suffit.",
            "Je réutiliserai aussi ton profil mémorisé.",
            "Tu peux répondre sans format strict.",
            "Je transformerai ça en contraintes financières.",
            "Ensuite je proposerai des options comparables.",
        ],
    )
    EDUCATIONAL = _expand(
        ["Voici l'idée centrale:", "En termes simples,", "Financièrement,", "Le point important est que", "Pour lire cette métrique,"],
        [
            "une métrique n'a de sens que comparée au risque pris.",
            "le contexte du portefeuille compte autant que le chiffre isolé.",
            "la diversification réduit surtout le risque spécifique.",
            "la corrélation explique pourquoi deux actifs peuvent amplifier ou réduire le risque.",
            "la volatilité mesure l'amplitude des variations, pas seulement la probabilité de perte.",
        ],
        [
            "Je peux aussi te donner une version plus technique.",
            "Je peux l'appliquer à ton portefeuille si tu veux.",
            "La lecture change selon ton horizon.",
            "C'est utile pour choisir entre stabilité et rendement.",
            "On peut ensuite comparer plusieurs scénarios.",
        ],
    )
    OPTIMIZATION = _expand(
        ["Je vais raisonner en contraintes:", "La logique d'optimisation est la suivante:", "Pour cette demande,", "Le moteur local va chercher", "Je traite ça comme un problème rendement-risque:"],
        [
            "respecter le risque, viser le rendement, puis départager par qualité de portefeuille.",
            "simuler et optimiser plusieurs allocations sans inventer de métriques.",
            "comparer Sharpe, volatilité, drawdown et concentration.",
            "éviter les portefeuilles trop concentrés même si leur rendement paraît séduisant.",
            "sélectionner des alternatives réellement différentes.",
        ],
        [
            "Les chiffres restent calculés depuis les données locales.",
            "Je garde ensuite les options en mémoire pour comparaison.",
            "Le résultat inclura les compromis principaux.",
            "Je signalerai si la cible est trop ambitieuse.",
            "La diversification sera explicitement mesurée.",
        ],
    )
    DIVERSIFICATION = _expand(
        ["La diversification ne consiste pas seulement à ajouter des titres:", "Un portefeuille diversifié doit éviter", "Je surveille la diversification avec", "Le vrai sujet est l'exposition commune:", "Une bonne diversification cherche"],
        [
            "elle dépend aussi du poids de chaque position.",
            "une position dominante ou un secteur trop lourd.",
            "le HHI, le nombre effectif d'actifs et l'exposition aux corrélations.",
            "des actifs qui montent et baissent tous ensemble peuvent cacher un risque concentré.",
            "un équilibre entre rendement attendu, volatilité et indépendance des actifs.",
        ],
        [
            "C'est pourquoi je pénalise la concentration.",
            "Je peux comparer une version prudente et une version agressive.",
            "Le score de diversification aide à lire ce compromis.",
            "Le résultat doit rester réaliste, pas seulement mathématiquement joli.",
            "Je peux aussi maximiser explicitement cette dimension.",
        ],
    )
    FOLLOWUPS = _expand(
        ["Question utile:", "Pour affiner:", "Dernier point de cadrage:", "Je peux continuer; précise juste", "Pour rendre l'allocation plus personnelle,"],
        [
            "quel horizon veux-tu viser ?",
            "acceptes-tu une forte volatilité si le rendement potentiel augmente ?",
            "as-tu des secteurs à privilégier ou à éviter ?",
            "cherches-tu des dividendes, de la croissance ou un équilibre ?",
            "veux-tu des quantités d'actions avec un capital en MAD ?",
        ],
        [
            "Je garderai cette préférence pour la suite.",
            "Tu peux répondre en une phrase.",
            "Cela changera le type de portefeuille proposé.",
            "Je peux inférer le reste si tu préfères.",
            "Cette information améliore la cohérence du conseil.",
        ],
    )
    RISK = _expand(
        ["Le risque, ici,", "Dans un portefeuille,", "Le risque financier", "Pour ton agent local,", "La lecture du risque"],
        [
            "désigne l'incertitude du rendement et l'amplitude des pertes possibles.",
            "se lit avec la volatilité, le drawdown, la VaR et la concentration.",
            "n'est pas seulement un pourcentage: il dépend aussi des corrélations.",
            "doit être comparé au rendement attendu.",
            "se contrôle par diversification, limites de poids et horizon adapté.",
        ],
        [
            "Un profil prudent cherchera d'abord la stabilité.",
            "Un profil agressif acceptera plus de variation.",
            "Je peux le quantifier sur les données locales.",
            "La question clé est la perte temporaire que tu peux tolérer.",
            "C'est le coeur du compromis d'investissement.",
        ],
    )
    ALLOCATION = _expand(
        ["Une allocation robuste", "La répartition proposée", "Quand je lis une allocation,", "Un bon poids par actif", "L'allocation n'est pas juste un classement:"],
        [
            "doit équilibrer rendement, risque et concentration.",
            "doit éviter qu'un seul titre dicte tout le résultat.",
            "doit être lisible en poids, secteurs et contribution au risque.",
            "dépend du prix des actions si tu fournis un capital.",
            "doit rester cohérente avec ton horizon.",
        ],
        [
            "Je peux expliquer chaque position.",
            "Je peux aussi comparer plusieurs variantes.",
            "Le tableau montre les poids principaux.",
            "Les petites lignes peuvent être filtrées pour rester réalistes.",
            "La mémoire garde les dernières options.",
        ],
    )
    MARKET = _expand(
        ["Sur le marché local,", "Pour la Bourse de Casablanca,", "Sans données externes,", "L'analyse de marché offline", "Le cadrage de marché"],
        [
            "je peux estimer les plages de rendement et de risque observées.",
            "je m'appuie uniquement sur les CSV disponibles.",
            "je peux comparer volatilité, rendement et Sharpe des opportunités.",
            "je peux situer une demande sur la frontière rendement-risque.",
            "je peux signaler si une cible semble trop ambitieuse.",
        ],
        [
            "Cela ne remplace pas une donnée temps réel.",
            "Les calculs restent reproductibles localement.",
            "On peut ensuite transformer cela en portefeuille.",
            "La qualité dépend de l'historique disponible.",
            "Je garde les conclusions liées aux données, pas à des suppositions.",
        ],
    )

    POOLS = {
        "greeting": GREETINGS,
        "clarification": CLARIFICATIONS,
        "educational": EDUCATIONAL,
        "optimization": OPTIMIZATION,
        "diversification": DIVERSIFICATION,
        "followup": FOLLOWUPS,
        "risk": RISK,
        "allocation": ALLOCATION,
        "market": MARKET,
    }

    @classmethod
    def choose(cls, category: str, context: str = "") -> str:
        pool = cls.POOLS.get(category, cls.CLARIFICATIONS)
        recent = set(MemoryService.recent_response_keys())
        seed = int(hashlib.sha256(f"{category}|{context}".encode("utf-8")).hexdigest()[:8], 16)
        for offset in range(len(pool)):
            index = (seed + offset * 17) % len(pool)
            key = f"{category}:{index}"
            if key not in recent:
                MemoryService.remember_response(key)
                return pool[index]
        index = seed % len(pool)
        MemoryService.remember_response(f"{category}:{index}")
        return pool[index]

    @classmethod
    def followup_for_missing(cls, missing_fields: list[str], profile: dict[str, Any]) -> str:
        if not missing_fields:
            return cls.choose("followup", str(profile))
        labels = {
            "target_return": "un rendement cible",
            "max_risk": "un risque maximal",
            "investment_horizon": "un horizon de placement",
            "investment_amount": "un capital en MAD",
            "risk_tolerance": "une tolérance au risque",
        }
        readable = ", ".join(labels.get(field, field) for field in missing_fields[:3])
        return f"{cls.choose('clarification', readable)} Il me manque surtout {readable}."

    @classmethod
    def profile_tone(cls, profile: dict[str, Any]) -> str:
        if profile.get("risk_tolerance") in {"low", "faible"}:
            return "Je vais privilégier la stabilité, la diversification et les limites de concentration."
        if profile.get("risk_tolerance") in {"high", "élevé"}:
            return "Je peux explorer des profils plus dynamiques, en gardant le risque explicitement mesuré."
        if profile.get("dividend_preference") or profile.get("income_preference"):
            return "Je vais tenir compte d'une préférence pour des revenus plus réguliers."
        if profile.get("growth_preference"):
            return "Je vais orienter l'analyse vers le potentiel de croissance, sans ignorer la volatilité."
        return "Je vais chercher un compromis lisible entre rendement, risque et diversification."
