import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_PATH  = DATA_DIR / "MP Fraud Takehome Task 2026 - Sheet1.csv"
PROCESSED_DIR  = DATA_DIR / "processed"


# We can also use an external geolocation database in production, but for this task we hardcode known cities.
KNOWN_CITIES = [
    "london", "glasgow", "leeds", "liverpool", "birmingham",
    "manchester", "edinburgh", "bristol", "cardiff", "belfast"
]

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

# LLM Fallback Parser
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL   = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 200
