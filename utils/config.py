from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_STOCK_DIR = BASE_DIR / "data" / "raw" / "stock"
INFO_CSV = BASE_DIR / "data" / "raw" / "info.csv"

RISK_FREE_RATE = 0.03
RANDOM_SEED = 42
COVARIANCE_METHOD = "ledoit_wolf"
OPTIMIZATION_METHOD = "SLSQP"
REBALANCE_FREQUENCY = "monthly"
SIMULATION_COUNT = 5000
MAX_WEIGHT = 0.25
MIN_WEIGHT = 0.01
DIVERSIFICATION_PENALTY = 0.18
RISK_PARITY_PENALTY = 0.05
HHI_TARGET = 0.12
MIN_OBSERVATIONS = 252
SUPPORTED_REBALANCE_FREQUENCIES: Dict[str, int] = {
    "monthly": 21,
    "quarterly": 63,
    "yearly": 252,
}
DEFAULT_OLLAMA_MODEL = "mistral"

# Optional sector cap example if a sector map exists
SECTOR_CAPS: Dict[str, float] = {}

PORTFOLIO_MODELS = [
    "maximum_sharpe",
    "minimum_variance",
    "efficient_return",
    "risk_parity",
    "equal_weight",
]
