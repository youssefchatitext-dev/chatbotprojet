from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from models.portfolio_models import PortfolioResult


class ExplanationEngine:
    EDUCATION_SNIPPETS = {
        "risque": {
            "beginner": (
                "Le risque, en finance, signifie que le résultat réel peut être différent de ce que tu espères. "
                "Pour une action ou un portefeuille, il se voit surtout dans les baisses possibles, les fortes variations de prix et l'incertitude du rendement. "
                "Un investissement plus risqué peut offrir plus de rendement potentiel, mais il peut aussi perdre davantage à court terme."
            ),
            "advanced": (
                "Le risque financier combine plusieurs dimensions: volatilité des rendements, drawdown, corrélations, concentration, liquidité et pertes extrêmes. "
                "Dans un portefeuille, le risque total dépend des risques individuels mais aussi de la covariance entre actifs, via w'Σw. "
                "Deux actifs risqués peuvent réduire le risque global s'ils ne bougent pas exactement ensemble."
            ),
            "concise": "Le risque mesure l'incertitude du rendement et la possibilité de pertes ou de fortes variations de valeur.",
            "professional": (
                "Le risque est l'exposition à l'incertitude: volatilité, pertes temporaires, concentration, corrélation et événements extrêmes. "
                "Il doit toujours être lu avec le rendement attendu et l'horizon d'investissement."
            ),
        },
        "sharpe": {
            "beginner": "Le ratio de Sharpe indique combien de rendement excédentaire tu obtiens pour chaque unité de risque. Plus il est élevé, plus le couple rendement-risque est efficace.",
            "advanced": "Le Sharpe annualisé approxime (rendement attendu - taux sans risque) / volatilité. Il pénalise toute volatilité, positive ou négative, et doit être comparé sur des univers cohérents.",
            "concise": "Sharpe = rendement excédentaire par unité de volatilité.",
            "professional": "Le Sharpe mesure la rémunération du risque total. Il aide à départager deux portefeuilles quand les rendements et volatilités diffèrent.",
        },
        "sortino": {
            "beginner": "Le Sortino ressemble au Sharpe, mais il se concentre surtout sur les variations négatives. Il est utile si tu veux mesurer le risque de baisse plutôt que toute la volatilité.",
            "advanced": "Le Sortino remplace la volatilité totale par la déviation baissière autour d'un seuil. Il convient mieux aux distributions asymétriques et aux stratégies orientées protection du capital.",
            "concise": "Sortino = rendement excédentaire par unité de risque baissier.",
            "professional": "Le Sortino isole la volatilité défavorable et complète le Sharpe quand les pertes comptent plus que les hausses.",
        },
        "volatilite": {
            "beginner": "La volatilité mesure à quel point la valeur bouge. Une volatilité élevée signifie que les gains et pertes peuvent être plus marqués.",
            "advanced": "La volatilité est l'écart-type annualisé des rendements. Elle capte la dispersion, mais pas directement l'asymétrie ni les pertes extrêmes.",
            "concise": "La volatilité mesure l'amplitude des variations.",
            "professional": "La volatilité sert de proxy principal du risque de marché, à compléter avec drawdown, VaR et concentration.",
        },
        "covariance": {
            "beginner": "La covariance indique si deux actifs ont tendance à bouger ensemble. Elle aide à comprendre pourquoi certains mélanges réduisent le risque.",
            "advanced": "La covariance mesure la co-variation absolue des rendements. Dans Markowitz, elle détermine la variance du portefeuille via w'Σw.",
            "concise": "La covariance mesure le mouvement commun de deux actifs.",
            "professional": "La covariance structure le risque agrégé du portefeuille et explique l'intérêt de combiner des actifs imparfaitement liés.",
        },
        "correlation": {
            "beginner": "La corrélation va de -1 à +1. Si deux actifs sont très corrélés, ils diversifient moins le portefeuille.",
            "advanced": "La corrélation normalise la covariance par les volatilités. Les clusters de corrélation révèlent les concentrations cachées.",
            "concise": "La corrélation mesure la proximité des mouvements entre actifs.",
            "professional": "La corrélation permet d'identifier les expositions communes et les bénéfices réels de diversification.",
        },
        "diversification": {
            "beginner": "La diversification répartit le capital pour qu'un seul actif ne décide pas de tout le résultat.",
            "advanced": "La diversification combine nombre effectif d'actifs, poids, corrélations et expositions sectorielles. Un portefeuille avec beaucoup de lignes peut rester concentré.",
            "concise": "Diversifier réduit le risque spécifique.",
            "professional": "Une diversification robuste limite concentration, corrélation excessive et dépendance sectorielle.",
        },
        "frontiere efficace": {
            "beginner": "La frontière efficace regroupe les portefeuilles qui offrent le meilleur rendement possible pour un niveau de risque donné.",
            "advanced": "La frontière efficiente est l'enveloppe des portefeuilles non dominés dans l'espace rendement-volatilité.",
            "concise": "Elle montre les meilleurs compromis rendement-risque.",
            "professional": "Elle sert à situer une allocation par rapport aux alternatives non dominées.",
        },
        "drawdown": {
            "beginner": "Le drawdown mesure la baisse depuis un sommet. Il répond à la question: combien le portefeuille a pu perdre avant de se reprendre ?",
            "advanced": "Le maximum drawdown capte la pire perte pic-creux. Il complète la volatilité en montrant le risque de trajectoire.",
            "concise": "Le drawdown est la perte maximale depuis un plus haut.",
            "professional": "Le drawdown donne une lecture concrète de la profondeur des pertes temporaires.",
        },
        "var": {
            "beginner": "La VaR estime une perte qui ne devrait pas être dépassée dans la plupart des cas, selon un niveau de confiance donné.",
            "advanced": "La VaR est un quantile de distribution des rendements. Elle ne décrit pas la gravité moyenne au-delà du seuil.",
            "concise": "La VaR est un seuil de perte statistique.",
            "professional": "La VaR cadre le risque de perte sous hypothèse historique ou distributionnelle.",
        },
        "cvar": {
            "beginner": "La CVaR regarde les pertes moyennes dans les pires scénarios au-delà de la VaR.",
            "advanced": "La CVaR, ou expected shortfall, mesure l'espérance conditionnelle des pertes dans la queue de distribution.",
            "concise": "La CVaR mesure la moyenne des pertes extrêmes.",
            "professional": "La CVaR est souvent plus prudente que la VaR pour piloter le risque de queue.",
        },
        "reequilibrage": {
            "beginner": "Le rééquilibrage remet le portefeuille vers ses poids cibles après les mouvements de marché.",
            "advanced": "Le rebalancing contrôle la dérive des expositions, mais crée un compromis entre discipline du risque, coûts et fiscalité.",
            "concise": "Rééquilibrer revient aux poids cibles.",
            "professional": "Le rééquilibrage limite la surconcentration et maintient la cohérence stratégique.",
        },
        "risk parity": {
            "beginner": "Le risk parity cherche à ce que chaque actif contribue de façon plus équilibrée au risque total.",
            "advanced": "La parité de risque égalise les contributions marginales au risque, souvent au lieu d'égaliser les montants investis.",
            "concise": "Risk parity = équilibrer les contributions au risque.",
            "professional": "La parité de risque réduit la domination des actifs les plus volatils ou fortement pondérés.",
        },
        "monte carlo": {
            "beginner": "Une simulation Monte Carlo teste beaucoup de portefeuilles possibles pour repérer de bons compromis.",
            "advanced": "L'exploration Monte Carlo échantillonne l'espace des poids sous contraintes, puis classe les portefeuilles par métriques de rendement, risque et diversification.",
            "concise": "Monte Carlo explore de nombreuses allocations candidates.",
            "professional": "Monte Carlo complète l'optimisation directe en explorant des allocations proches et variées.",
        },
    }

    @classmethod
    def educate(cls, topic: str, level: str = "beginner") -> str:
        normalized = cls._normalize_topic(topic)
        for key, variants in cls.EDUCATION_SNIPPETS.items():
            if key in normalized:
                return variants.get(level, variants["professional"])
        return (
            "Je peux expliquer le Sharpe, le Sortino, la volatilité, la covariance, la corrélation, la diversification, "
            "la frontière efficace, les drawdowns, la VaR, la CVaR, le rééquilibrage, le risk parity et Monte Carlo."
        )

    @staticmethod
    def explain_portfolio(portfolio: PortfolioResult, user_input: str | None = None, style: str = "professional") -> str:
        effective_assets = 1.0 / sum(weight * weight for weight in portfolio.allocations.values()) if portfolio.allocations else 0.0
        largest = max(portfolio.allocations.values(), default=0.0)
        lines = [
            f"Analyse du portefeuille {portfolio.name}:",
            f"- Rendement attendu: {portfolio.metrics.expected_return * 100:.2f}%",
            f"- Volatilité estimée: {portfolio.metrics.volatility * 100:.2f}%",
            f"- Sharpe: {portfolio.metrics.sharpe_ratio:.2f}",
            f"- Sortino: {portfolio.metrics.sortino_ratio:.2f}",
            f"- Drawdown maximum: {portfolio.metrics.max_drawdown * 100:.2f}%",
            f"- Nombre effectif d'actifs: {effective_assets:.2f}",
            f"- Plus grande position: {largest * 100:.2f}%",
        ]
        if style == "beginner":
            lines.insert(1, "Lecture simple: je regarde d'abord si le rendement obtenu justifie le risque et si une position domine trop le portefeuille.")
        elif style == "advanced":
            lines.append("Lecture avancée: la qualité dépend du couple rendement-volatilité, de la contribution au risque, de la concentration HHI et des corrélations entre lignes.")
        if user_input:
            lines.append(f"Demande analysée: {user_input}")
        return "\n".join(lines)

    @staticmethod
    def explain_allocation(portfolio: PortfolioResult) -> str:
        if not portfolio.allocations:
            return "Aucune allocation significative n'a été retenue."
        weights = sorted(portfolio.allocations.items(), key=lambda item: item[1], reverse=True)
        top = ", ".join(f"{ticker} {weight * 100:.1f}%" for ticker, weight in weights[:5])
        largest = weights[0][1]
        if largest > 0.25:
            concentration = "La première position est élevée; il faut surveiller le risque de concentration."
        else:
            concentration = "La première position reste contenue, ce qui améliore la robustesse."
        return f"Les principales expositions sont {top}. {concentration}"

    @staticmethod
    def explain_diversification(weights: np.ndarray, covariance: pd.DataFrame | None = None) -> str:
        hhi = float(np.sum(np.square(weights)))
        effective_assets = 1.0 / hhi if hhi > 0 else 0.0
        max_weight = float(np.max(weights)) if len(weights) else 0.0
        message = f"HHI {hhi:.3f}, nombre effectif d'actifs {effective_assets:.2f}, plus grand poids {max_weight * 100:.2f}%."
        if covariance is not None and len(weights) == len(covariance.columns):
            corr = covariance.corr().values
            upper = corr[np.triu_indices_from(corr, k=1)]
            message += f" Corrélation moyenne approximative {float(np.nanmean(upper)):.2f}."
        return message

    @staticmethod
    def compare_portfolios(portfolios: Iterable[PortfolioResult], style: str = "analytical") -> str:
        lines = ["Comparaison des alternatives:"]
        for portfolio in portfolios:
            effective_assets = 1.0 / sum(weight * weight for weight in portfolio.allocations.values()) if portfolio.allocations else 0.0
            lines.append(
                f"- {portfolio.name}: rendement {portfolio.metrics.expected_return * 100:.2f}%, "
                f"risque {portfolio.metrics.volatility * 100:.2f}%, Sharpe {portfolio.metrics.sharpe_ratio:.2f}, "
                f"actifs effectifs {effective_assets:.2f}"
            )
        if style == "beginner":
            lines.append("Lis la comparaison comme un échange: plus de rendement potentiel demande souvent plus de variation ou moins de stabilité.")
        return "\n".join(lines)

    @staticmethod
    def risk_commentary(risk_tolerance: str | None) -> str:
        if risk_tolerance in {"low", "faible"}:
            return "Profil prudent: je privilégie faible volatilité, diversification et drawdown contenu."
        if risk_tolerance in {"high", "élevé"}:
            return "Profil dynamique: je peux accepter plus de volatilité, mais je garde des garde-fous de concentration."
        return "Profil équilibré: je cherche un compromis entre rendement, volatilité et diversification."

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        return topic.lower().replace("é", "e").replace("è", "e").replace("ê", "e")
