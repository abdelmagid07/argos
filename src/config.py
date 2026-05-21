"""Central paths and constants for Argos.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

DB_PATH = ROOT / "argos.db"

IEEE_CIS_CSV = DATA_DIR / "train_transaction.csv"

MODEL_PATH = MODELS_DIR / "fraud_detector_v1.pt"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

# Feature vector ordering
FEATURE_COLUMNS = [
    "amount",
    "user_avg_amount",
    "user_total_txns",
    "user_fraud_rate",
    "velocity_24h",
    "unique_merchants",
    "unique_countries",
    "merchant_fraud_rate",
    "merchant_avg_amount",
    "merchant_total_txns",
    "merchant_id",
]

for d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
