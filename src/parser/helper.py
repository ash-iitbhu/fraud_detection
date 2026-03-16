import re
import logging

import pandas as pd
from dateutil import parser as dateutil_parser

from configs.config import TXN_SYNONYM_MAP

logger = logging.getLogger(__name__)

_AMT = r"(?P<cur_pre>[€£$]?)(?P<amount>\d+(?:[.,]\d+)?)(?P<cur_suf>[€£$]?)"

PATTERNS = [

    # F1: 2025-07-05 19:18:10::user1069::withdrawal::2995.12::London::iPhone 13
    (
        "F1",
        re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            r"::(?P<user>user\d+)"
            r"::(?P<txn>[a-zA-Z\-_]+)"
            r"::" + _AMT +
            r"::(?P<city>[^:]+)"
            r"::(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),

    # F2: usr:user1076|cashout|€4821.85|Glasgow|2025-07-15 12:56:05|Pixel 6
    (
        "F2",
        re.compile(
            r"usr:(?P<user>user\d+)"
            r"\|(?P<txn>[a-zA-Z\-_]+)"
            r"\|" + _AMT +
            r"\|(?P<city>[^|]+)"
            r"\|(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            r"\|(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),

    # F3: 2025-07-20 05:38:14 >> [user1034] did top-up - amt=€2191.06 - None // dev:iPhone 13
    (
        "F3",
        re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            r"\s*>>\s*\[(?P<user>user\d+)\]\s*did\s+(?P<txn>[a-zA-Z\-_]+)"
            r"\s*-\s*amt=" + _AMT +
            r"\s*-\s*(?P<city>[^/]+?)\s*//\s*dev:(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),

    # F4: 2025-06-23 14:45:58 - user=user1075 - action=debit $1215.74 - ATM: Leeds - device=Samsung Galaxy S10
    (
        "F4",
        re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            r"\s*-\s*user=(?P<user>user\d+)"
            r"\s*-\s*action=(?P<txn>[a-zA-Z\-_]+)\s+" + _AMT +
            r"\s*-\s*ATM:\s*(?P<city>[^-]+?)"
            r"\s*-\s*device=(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),

    # F5: 2025-07-29 23:47:37 | user: user1014 | txn: deposit of £3539.5 from Glasgow | device: iPhone 13
    (
        "F5",
        re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
            r"\s*\|\s*user:\s*(?P<user>user\d+)"
            r"\s*\|\s*txn:\s*(?P<txn>[a-zA-Z\-_]+)\s+of\s+" + _AMT +
            r"\s+from\s+(?P<city>[^|]+?)"
            r"\s*\|\s*device:\s*(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),

    # F6: 24/07/2025 22:47:06 ::: user1080 *** PURCHASE ::: amt:951.85$ @ Liverpool <Xiaomi Mi 11>
    (
        "F6",
        re.compile(
            r"(?P<ts>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})"
            r"\s*:::\s*(?P<user>user\d+)\s*\*\*\*\s*(?P<txn>[a-zA-Z\-_]+)"
            r"\s*:::\s*amt:" + _AMT +
            r"\s*@\s*(?P<city>[^<]+?)"
            r"\s*<(?P<device>[^>]+)>",
            re.IGNORECASE,
        ),
    ),

    # F7: user1093 2025-07-05 14:11:06 withdrawal 4926.56 None Huawei P30
    (
        "F7",
        re.compile(
            r"(?P<user>user\d+)\s+"
            r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
            r"(?P<txn>[a-zA-Z\-_]+)\s+" + _AMT +
            r"\s+(?P<city>\S+)"
            r"\s+(?P<device>.+)$",
            re.IGNORECASE,
        ),
    ),
]

# patterns for field-by-field heuristic fallback
_RE_USER    = re.compile(r"\b(user\d+)\b", re.IGNORECASE)
_RE_TS_ISO  = re.compile(r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
_RE_TS_DMY  = re.compile(r"\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\b")
_RE_AMOUNT  = re.compile(r"[€£$]?\d+(?:[.,]\d+)?[€£$]?")
_RE_TXN     = re.compile(
    r"\b(withdrawal|cashout|cash.out|deposit|top.up|purchase|debit|credit|transfer)\b",
    re.IGNORECASE,
)


# Helper Functions

def _parse_timestamp(raw):
    """Parse timestamp string using dateutil with dayfirst detection."""
    if not raw or raw.strip().lower() in ("none", "null", ""):
        return None
    raw = raw.strip()
    dayfirst = bool(re.match(r"\d{2}/\d{2}/\d{4}", raw))
    try:
        return dateutil_parser.parse(raw, dayfirst=dayfirst)
    except Exception:
        return None


def _extract_amount_currency(raw_cur_pre: str, raw_amount: str, raw_cur_suf: str):
    """Return (amount_float, currency_str) from regex named groups."""
    cur = (raw_cur_pre or raw_cur_suf or "").strip()
    if not cur:
        cur = "UNKNOWN"
    try:
        amt = float(raw_amount.replace(",", "."))
    except (ValueError, AttributeError):
        amt = None
    return amt, cur


def _normalize_txn_type(raw: str) -> str:
    """Map surface txn type to one of 4 canonical types."""
    if not raw:
        return "UNKNOWN"
    key = raw.strip().lower().replace("-", "").replace("_", "")
    return TXN_SYNONYM_MAP.get(key, TXN_SYNONYM_MAP.get(raw.strip().lower(), "UNKNOWN"))


def _normalize_city(raw: str) -> str:
    """Return UNKNOWN for None/null strings, otherwise title-case."""
    if not raw or raw.strip().lower() in ("none", "null", "n/a", ""):
        return "UNKNOWN"
    return raw.strip().title()


def _normalize_device(raw: str) -> str:
    """Return UNKNOWN for None/null strings, otherwise clean string."""
    if not raw or raw.strip().lower() in ("none", "null", "n/a", ""):
        return "UNKNOWN"
    return raw.strip()