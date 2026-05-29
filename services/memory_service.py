from __future__ import annotations

from collections import Counter, deque
from typing import Any

try:
    import streamlit as st
except Exception:
    st = None


class MemoryService:
    SESSION_KEY = "portfolio_agent_history"
    PROFILE_KEY = "portfolio_agent_profile"
    PORTFOLIOS_KEY = "portfolio_agent_portfolios"
    RESPONSE_CACHE_KEY = "portfolio_agent_response_cache"
    EDUCATION_KEY = "portfolio_agent_education_topics"
    COMPARISONS_KEY = "portfolio_agent_comparisons"
    OPTIMIZATION_KEY = "portfolio_agent_optimization_modes"

    _fallback_state: dict[str, Any] = {}

    DEFAULT_PROFILE = {
        "risk_tolerance": None,
        "preferred_sectors": [],
        "forbidden_assets": [],
        "preferred_assets": [],
        "investment_horizon": None,
        "target_return": None,
        "max_risk": None,
        "investment_amount": None,
        "style": None,
        "dividend_preference": False,
        "growth_preference": False,
        "income_preference": False,
        "rebalance_frequency": None,
        "liquidity_preference": None,
        "response_style": None,
        "last_intents": [],
        "last_request": None,
    }

    @classmethod
    def _state(cls) -> dict[str, Any]:
        if st is not None:
            try:
                return st.session_state
            except Exception:
                return cls._fallback_state
        return cls._fallback_state

    @classmethod
    def initialize(cls) -> None:
        state = cls._state()
        state.setdefault(cls.SESSION_KEY, [])
        state.setdefault(cls.PROFILE_KEY, dict(cls.DEFAULT_PROFILE))
        state.setdefault(cls.PORTFOLIOS_KEY, [])
        state.setdefault(cls.RESPONSE_CACHE_KEY, deque(maxlen=32))
        state.setdefault(cls.EDUCATION_KEY, Counter())
        state.setdefault(cls.COMPARISONS_KEY, [])
        state.setdefault(cls.OPTIMIZATION_KEY, [])

    @classmethod
    def append_message(cls, role: str, content: str) -> None:
        cls.initialize()
        history = cls._state()[cls.SESSION_KEY]
        history.append({"role": role, "content": content})
        del history[:-30]

    @classmethod
    def get_history(cls) -> list[dict[str, str]]:
        cls.initialize()
        return list(cls._state()[cls.SESSION_KEY])

    @classmethod
    def clear_history(cls) -> None:
        cls.initialize()
        cls._state()[cls.SESSION_KEY] = []

    @classmethod
    def append_portfolio(cls, summary: dict[str, Any]) -> None:
        cls.initialize()
        portfolios = cls._state()[cls.PORTFOLIOS_KEY]
        portfolios.append(summary)
        del portfolios[:-20]

    @classmethod
    def get_portfolio_history(cls) -> list[dict[str, Any]]:
        cls.initialize()
        return list(cls._state()[cls.PORTFOLIOS_KEY])

    @classmethod
    def remember_education(cls, topic: str) -> None:
        cls.initialize()
        cls._state()[cls.EDUCATION_KEY][topic] += 1

    @classmethod
    def education_count(cls, topic: str) -> int:
        cls.initialize()
        return int(cls._state()[cls.EDUCATION_KEY][topic])

    @classmethod
    def remember_response(cls, response_key: str) -> None:
        cls.initialize()
        cache = cls._state()[cls.RESPONSE_CACHE_KEY]
        if not isinstance(cache, deque):
            cache = deque(cache, maxlen=32)
            cls._state()[cls.RESPONSE_CACHE_KEY] = cache
        cache.append(response_key)

    @classmethod
    def recent_response_keys(cls) -> list[str]:
        cls.initialize()
        return list(cls._state()[cls.RESPONSE_CACHE_KEY])

    @classmethod
    def remember_comparison(cls, comparison: dict[str, Any]) -> None:
        cls.initialize()
        comparisons = cls._state()[cls.COMPARISONS_KEY]
        comparisons.append(comparison)
        del comparisons[:-10]

    @classmethod
    def remember_optimization_mode(cls, mode: str) -> None:
        cls.initialize()
        modes = cls._state()[cls.OPTIMIZATION_KEY]
        modes.append(mode)
        del modes[:-12]

    @classmethod
    def update_profile(cls, **updates: Any) -> None:
        cls.initialize()
        profile = cls._state()[cls.PROFILE_KEY]
        for key, value in updates.items():
            if value is None:
                continue
            if key in {"preferred_sectors", "forbidden_assets", "preferred_assets"}:
                existing = profile.get(key) or []
                merged = list(dict.fromkeys([*existing, *value]))
                profile[key] = merged
            elif key == "target_volatility":
                profile["max_risk"] = value
            elif key == "optimization_mode":
                cls.remember_optimization_mode(str(value))
            else:
                profile[key] = value

    @classmethod
    def get_profile(cls) -> dict[str, Any]:
        cls.initialize()
        return dict(cls._state()[cls.PROFILE_KEY])

    @classmethod
    def get_context(cls, max_items: int = 8) -> str:
        cls.initialize()
        history = cls._state()[cls.SESSION_KEY][-max_items:]
        return "\n".join(f"{item['role']}: {item['content']}" for item in history)

    @classmethod
    def missing_profile_fields(cls, fields: list[str]) -> list[str]:
        profile = cls.get_profile()
        return [field for field in fields if not profile.get(field)]

    @classmethod
    def get_profile_summary(cls) -> str:
        profile = cls.get_profile()
        summary_parts = []
        if profile.get("risk_tolerance"):
            summary_parts.append(f"Tolérance au risque: {profile['risk_tolerance']}")
        if profile.get("investment_horizon"):
            summary_parts.append(f"Horizon: {profile['investment_horizon']} ans")
        if profile.get("preferred_sectors"):
            summary_parts.append(f"Secteurs préférés: {', '.join(profile['preferred_sectors'])}")
        if profile.get("target_return"):
            summary_parts.append(f"Objectif de rendement: {profile['target_return']}%")
        if profile.get("max_risk"):
            summary_parts.append(f"Limite de risque: {profile['max_risk']}%")
        if profile.get("investment_amount"):
            summary_parts.append(f"Capital: {profile['investment_amount']:,.0f} MAD")
        if profile.get("dividend_preference"):
            summary_parts.append("Préférence: revenus/dividendes")
        if profile.get("growth_preference"):
            summary_parts.append("Préférence: croissance")
        return "; ".join(summary_parts) if summary_parts else "Aucun profil enregistré pour le moment."

    @classmethod
    def reset(cls) -> None:
        state = cls._state()
        state[cls.SESSION_KEY] = []
        state[cls.PROFILE_KEY] = dict(cls.DEFAULT_PROFILE)
        state[cls.PORTFOLIOS_KEY] = []
        state[cls.RESPONSE_CACHE_KEY] = deque(maxlen=32)
        state[cls.EDUCATION_KEY] = Counter()
        state[cls.COMPARISONS_KEY] = []
        state[cls.OPTIMIZATION_KEY] = []
