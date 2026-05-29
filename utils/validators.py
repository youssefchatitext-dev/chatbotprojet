from __future__ import annotations

import pandas as pd

from .config import MIN_OBSERVATIONS


def validate_price_matrix(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("Aucune donnée de prix disponible.")
    if prices.shape[0] < MIN_OBSERVATIONS:
        raise ValueError(
            f"Historique insuffisant : au moins {MIN_OBSERVATIONS} observations sont requises."
        )
    if prices.isna().all(axis=1).any():
        raise ValueError("Le jeu de données contient des lignes de prix entièrement manquantes.")
    if prices.columns.duplicated().any():
        raise ValueError("Des tickers dupliqués ont été trouvés dans les données des actifs.")
    if prices.isna().any().any():
        raise ValueError("Les données contiennent des valeurs manquantes. Utilisez un nettoyage préalable.")


def validate_weights(weights: list[float]) -> None:
    if not weights:
        raise ValueError("Les poids du portefeuille sont vides.")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("La somme des poids doit être égale à 1.")
