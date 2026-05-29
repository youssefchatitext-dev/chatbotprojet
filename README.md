# Casablanca Portfolio Optimization Platform

## Architecture Overview

This refactor separates the project into a professional multi-agent quantitative finance platform:

- `app.py`: pure Streamlit UI layer
- `agents/`: routing and agent interfaces
  - `base_agent.py`: shared abstract agent contract
  - `local_agent.py`: deterministic financial engine interface
  - `ollama_agent.py`: conversational AI explanation layer
  - `router.py`: selects between local and Ollama modes
- `core/`: reusable financial engine modules
  - `optimizer.py`: SLSQP portfolio optimization
  - `portfolio_engine.py`: portfolio metrics and allocations
  - `risk_engine.py`: volatility, Sharpe, Sortino, VaR, CVaR, drawdown
  - `covariance_engine.py`: Ledoit-Wolf covariance shrinkage fallback
  - `allocation_engine.py`: portfolio scoring and selection
  - `frontier_engine.py`: efficient frontier generation
  - `backtesting_engine.py`: rebalance and rolling optimization support
  - `metrics_engine.py`: cumulative returns, rolling metrics, drawdown
  - `asset_analyzer.py`: liquidity and diversification filtering
- `services/`: data and AI tools
  - `data_service.py`: local CSV loader and preprocessing
  - `ollama_service.py`: centralized Ollama access
  - `memory_service.py`: Streamlit session memory
- `models/`: typed portfolio and profile schemas
- `ui/`: Streamlit presentation helpers
- `utils/`: prompts, config, validators, shared helpers

## Key Features

- Deterministic financial calculations remain local
- Sharpe ratio uses configurable risk-free rate
- Geometric annual returns are used for performance estimation
- Ledoit-Wolf covariance shrinkage with sample fallback
- SciPy SLSQP optimization for max Sharpe and minimum variance
- Portfolio constraints for min/max weights and optional sector caps
- Rebalancing and rolling window support
- Structured portfolio outputs and conversational explanation layer
- Offline-capable local mode with optional Ollama reasoning

## Installation

```bash
pip install -r requirements.txt
```

## Ollama Setup

1. Installer les dépendances Python : `pip install -r requirements.txt`
2. Installer le binaire Ollama et vérifier qu'il est accessible dans votre terminal.
3. Charger le modèle :

```bash
ollama pull mistral
```

Si Ollama n'est pas installé ou si le service n'est pas accessible, la plateforme fonctionnera toujours en mode local déterministe.

## Run Streamlit

```bash
streamlit run app.py
```

## Extending the Platform

- add new optimization models in `core/optimizer.py`
- add new UI panels in `ui/`
- add new agents by extending `agents/base_agent.py`
- add prompt templates in `utils/prompts.py`
