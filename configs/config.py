import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ROJECT PATHS
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_PATH  = DATA_DIR / "MP Fraud Takehome Task 2026 - Sheet1.csv"
PROCESSED_DIR  = DATA_DIR / "processed"

# PARSING PARAMETERS
# Minimum transactions a user must have before we use their personal history to create priors.
# Below this threshold → cold-start mode (use global priors).
COLD_START_THRESHOLD = 3

# Known cities in the dataset (used by spaCy EntityRuler and heuristic parser)
# We can also use an external geolocation database in production, but for this task we hardcode known cities.
KNOWN_CITIES = [
    "london", "glasgow", "leeds", "liverpool", "birmingham",
    "manchester", "edinburgh", "bristol", "cardiff", "belfast"
]

# Known device brands (used by heuristic parser)
KNOWN_DEVICES = [
    "iphone", "samsung", "pixel", "nokia", "huawei", "xiaomi",
    "oppo", "oneplus", "motorola", "sony"
]

#transaction type mapping
TXN_SYNONYM_MAP = {
    "withdrawal": "withdrawal", "cashout": "withdrawal", "cash_out": "withdrawal",
    "cash-out": "withdrawal", "debit": "withdrawal",
    
    "deposit": "deposit", "top-up": "deposit", "top_up": "deposit",
    "topup": "deposit", "credit": "deposit",
    
    "purchase": "purchase", "buy": "purchase",
   
    "transfer": "transfer", "send": "transfer",
    
    "refund": "refund"
}

# Currency symbol
CURRENCY_SYMBOLS = {"€", "£", "$"}

# Feature Engineering
# Velocity window in seconds
VELOCITY_WINDOW_SECONDS = 3600

# Late-night hours
LATE_NIGHT_START = 0
LATE_NIGHT_END   = 5

# Fit / Score Split
# Fraction of data (by time) used to FIT models and calculate user-specific priors.
# All data is also scored — we score everything, but models are
# parametrised only on the fit window, This is done to mirror production deployment.
FIT_WINDOW_QUANTILE = 0.70

# Rule Engine
RULE_AMOUNT_ZSCORE_THRESHOLD    = 3.0   # |z| > 3 → extreme amount
RULE_VELOCITY_THRESHOLD         = 5     # >3 txns in 1h → high velocity
RULE_NEW_CITY_ZSCORE_THRESHOLD  = 2   # new city + z>2 → suspicious
RULE_IMPOSSIBLE_TRAVEL_SECONDS  = 1800  # new city within 0.5h → impossible travel
RULE_PAIR_RARITY_THRESHOLD      = 0.01  # device-city pair seen <1% → rare pair
RULE_LATE_NIGHT_RATIO_THRESHOLD = 0.10  # <10% of user's past txns at this hour

# Isolation Forest
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.02   # assumed ~2% anomaly rate
IF_RANDOM_STATE  = 42

# Autoencoder (PyTorch)
AE_EMBEDDING_DIM_USER    = 16
AE_EMBEDDING_DIM_CITY    = 8
AE_EMBEDDING_DIM_DEVICE  = 8
AE_EMBEDDING_DIM_TXNTYPE = 8

AE_HIDDEN_DIMS    = [32, 16]   # encoder hidden layer sizes
AE_BOTTLENECK_DIM = 8

AE_DROPOUT_RATE  = 0.1
AE_LEARNING_RATE = 1e-3
AE_BATCH_SIZE    = 32
AE_EPOCHS        = 50
AE_PATIENCE      = 10          # early stopping patience
AE_VAL_FRACTION  = 0.15        # fraction of fit window used for AE validation

# Loss weights: continuous MSE + categorical cross-entropy heads
AE_LOSS_WEIGHT_CONTINUOUS = 0.60
AE_LOSS_WEIGHT_USER       = 0.10
AE_LOSS_WEIGHT_CITY       = 0.10
AE_LOSS_WEIGHT_DEVICE     = 0.10
AE_LOSS_WEIGHT_TXNTYPE    = 0.10

AE_RANDOM_STATE  = 42

# Ensemble
# Weights for final risk score.
ENSEMBLE_WEIGHT_RULES  = 0.20
ENSEMBLE_WEIGHT_IF     = 0.40
ENSEMBLE_WEIGHT_AE     = 0.40

# Risk tier thresholds (applied to normalised final_risk_score)
TIER_1_THRESHOLD = 0.75 
TIER_2_THRESHOLD = 0.50  
TIER_3_THRESHOLD = 0.30 

# Evaluation
PRECISION_AT_K_VALUES = [10, 20, 50]   # K values for Precision@K
N_SYNTHETIC_FRAUDS    = 30             # synthetic fraud cases to inject

# LLM Fallback Parser
# Set ANTHROPIC_API_KEY env variable to enable LLM fallback.
# If not set, parser silently skips this layer.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL   = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 200

# Random Seeds
RANDOM_SEED = 42
