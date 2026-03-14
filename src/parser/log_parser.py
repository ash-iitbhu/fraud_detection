"""
src/parser/log_parser.py
------------------------
Layer 1 — Regex Cascade     : 8 compiled patterns covering known formats
Layer 2 — Heuristic Fallback: Field-by-field pattern extraction
Layer 3 — spaCy EntityRuler : Rule-based NLP matcher
Layer 4 — LLM Fallback      : Optional API call (env-gated, won't break pipeline- will only run if GROK_API_KEY is set)
Layer 5 — PARSE_FAILURE     : Preserve raw string, flag for analysis

Every row produces a dict with keys:
    user_id, timestamp, txn_type, amount, currency, city, device,
    parse_method, parse_success, raw_log
"""

import re
import os
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from dateutil import parser as dateutil_parser
import spacy

from configs.config import KNOWN_CITIES, KNOWN_DEVICES, TXN_SYNONYM_MAP, CURRENCY_SYMBOLS, LLM_MODEL, LLM_MAX_TOKENS, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


# ── Compiled Regex Patterns ───────────────────────────────────────────────────
# Each tuple: (format_id, compiled_pattern, field_extractor_fn)
# Amount regex: optional currency prefix, digits, optional currency suffix

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


def _build_record(match: re.Match, format_id: str, raw_log: str) -> dict:
    """Convert a regex match into a normalized record dict."""
    g = match.groupdict()
    amount, currency = _extract_amount_currency(
        g.get("cur_pre", ""), g.get("amount", ""), g.get("cur_suf", "")
    )
    return {
        "raw_log":       raw_log,
        "user_id":       g.get("user", "UNKNOWN").strip().lower(),
        "timestamp":     _parse_timestamp(g.get("ts", "")),
        "txn_type":      _normalize_txn_type(g.get("txn", "")),
        "amount":        amount,
        "currency":      currency,
        "city":          _normalize_city(g.get("city", "")),
        "device":        _normalize_device(g.get("device", "")),
        "parse_method":  format_id,
        "parse_success": True,
    }


# Layer 1: Regex Cascade

def _try_regex_cascade(raw_log: str) -> Optional[dict]:
    """Try all compiled patterns in order. Return first match."""
    for fmt_id, pattern in PATTERNS:
        m = pattern.match(raw_log.strip())
        if m:
            return _build_record(m, fmt_id, raw_log)
    return None


# Layer 2: Heuristic Field-by-Field Fallback

def _try_heuristic_fallback(raw_log: str) -> Optional[dict]:
    """
    Extract each field independently using patterns.
    Can return a partial record (some fields may be UNKNOWN).
    """
    log = raw_log.strip()

    # User ID
    user_m = _RE_USER.search(log)
    user_id = user_m.group(1).lower() if user_m else "UNKNOWN"

    # Timestamp
    ts = None
    for pattern in (_RE_TS_ISO, _RE_TS_DMY):
        m = pattern.search(log)
        if m:
            ts = _parse_timestamp(m.group(1))
            break

    # Amount & currency
    amt_matches = _RE_AMOUNT.findall(log)
    amount, currency = None, "UNKNOWN"
    if amt_matches:
        raw_amt = amt_matches[0]
        cur = ""
        for sym in CURRENCY_SYMBOLS:
            if sym in raw_amt:
                cur = sym
                raw_amt = raw_amt.replace(sym, "")
                break
        try:
            amount = float(raw_amt.replace(",", "."))
            currency = cur if cur else "UNKNOWN"
        except ValueError:
            pass

    # Transaction type
    txn_m = _RE_TXN.search(log)
    txn_type = _normalize_txn_type(txn_m.group(1)) if txn_m else "UNKNOWN"

    # City — check against known cities list
    log_lower = log.lower()
    city = "UNKNOWN"
    for c in KNOWN_CITIES:
        if c in log_lower:
            city = c.title()
            break

    # Device — check against known device substrings
    device = "UNKNOWN"
    for d in KNOWN_DEVICES:
        if d in log_lower:
            device = d.title()
            break

    # Only return a heuristic record if we got at least user + timestamp + amount
    if user_id != "UNKNOWN" and ts is not None and amount is not None:
        return {
            "raw_log":       raw_log,
            "user_id":       user_id,
            "timestamp":     ts,
            "txn_type":      txn_type,
            "amount":        amount,
            "currency":      currency,
            "city":          city,
            "device":        device,
            "parse_method":  "HEURISTIC",
            "parse_success": True,
        }
    return None


# Layer 3: spaCy EntityRuler Fallback

def _build_spacy_ruler():
    """
    Build a spaCy pipeline with ONLY rule-based EntityRuler patterns.
    """
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
                {"label": "USER_ID",  "pattern": [{"TEXT": {"REGEX": r"user\d+"}}]},
                {"label": "CURRENCY", "pattern": [{"TEXT": {"REGEX": r"[€£$]\d+|\d+[€£$]"}}]},
                {"label": "CITY",
                "pattern": [{"LOWER": {"IN": list(KNOWN_CITIES)}}]},
            ]
    ruler.add_patterns(patterns)
    return nlp        


_SPACY_NLP = None

def _try_spacy_fallback(raw_log: str) -> Optional[dict]:
    """Use spaCy EntityRuler to extract available fields from difficult logs."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = _build_spacy_ruler()
    if _SPACY_NLP is None:
        return None

    doc = _SPACY_NLP(raw_log)
    user_id  = "UNKNOWN"
    city     = "UNKNOWN"
    currency = "UNKNOWN"
    amount   = None

    for ent in doc.ents:
        if ent.label_ == "USER_ID":
            user_id = ent.text.lower()
        elif ent.label_ == "CITY":
            city = ent.text.title()
        elif ent.label_ == "CURRENCY":
            raw = ent.text
            for sym in CURRENCY_SYMBOLS:
                if sym in raw:
                    currency = sym
                    raw = raw.replace(sym, "")
                    break
            try:
                amount = float(raw.replace(",", "."))
            except ValueError:
                pass

    if user_id == "UNKNOWN":
        return None  # Not useful without at least user_id

    # Still need timestamp and txn from heuristic micro-patterns
    ts_m = _RE_TS_ISO.search(raw_log) or _RE_TS_DMY.search(raw_log)
    ts   = _parse_timestamp(ts_m.group(1)) if ts_m else None
    txn_m = _RE_TXN.search(raw_log)
    txn = _normalize_txn_type(txn_m.group(1)) if txn_m else "UNKNOWN"

    return {
        "raw_log":       raw_log,
        "user_id":       user_id,
        "timestamp":     ts,
        "txn_type":      txn,
        "amount":        amount,
        "currency":      currency,
        "city":          city,
        "device":        "UNKNOWN",   # spaCy rules don't cover device
        "parse_method":  "SPACY_RULER",
        "parse_success": True,
    }


# Layer 4: LLM API Fallback (env-gated)

_LLM_PROMPT = """Extract the following fields from this financial transaction log and return ONLY valid JSON with no preamble or markdown.
If a field cannot be determined, use null.

Fields: user_id, timestamp (ISO format), txn_type, amount (float), currency (symbol), city, device

Log: {log}

JSON:"""


def _try_llm_fallback(raw_log: str) -> Optional[dict]:
    """
    Call a lightweight LLM API to parse difficult log entries.
    Requires ANTHROPIC_API_KEY.
    Fails silently — never breaks the pipeline.
    """
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        return None

    try:
        import urllib.request
        prompt = _LLM_PROMPT.format(log=raw_log)

        if api_key:
            payload = json.dumps({
                "model": LLM_MODEL,
                "max_tokens": LLM_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            raw_json = data["content"][0]["text"].strip()

        parsed = json.loads(raw_json)
        return {
            "raw_log":       raw_log,
            "user_id":       str(parsed.get("user_id") or "UNKNOWN").lower(),
            "timestamp":     _parse_timestamp(str(parsed.get("timestamp") or "")),
            "txn_type":      _normalize_txn_type(str(parsed.get("txn_type") or "")),
            "amount":        float(parsed["amount"]) if parsed.get("amount") is not None else None,
            "currency":      str(parsed.get("currency") or "UNKNOWN"),
            "city":          _normalize_city(str(parsed.get("city") or "None")),
            "device":        _normalize_device(str(parsed.get("device") or "None")),
            "parse_method":  "LLM_FALLBACK",
            "parse_success": True,
        }
    except Exception as e:
        logger.debug(f"LLM fallback failed for log: {raw_log[:60]}... Error: {e}")
        return None


# Layer 5: PARSE_FAILURE

def _parse_failure_record(raw_log: str) -> dict:
    """Return a dummy record for completely unparseable rows."""
    return {
        "raw_log":       raw_log,
        "user_id":       "UNKNOWN",
        "timestamp":     None,
        "txn_type":      "UNKNOWN",
        "amount":        None,
        "currency":      "UNKNOWN",
        "city":          "UNKNOWN",
        "device":        "UNKNOWN",
        "parse_method":  "PARSE_FAILURE",
        "parse_success": False,
    }


# Main Public Interface

def parse_log(raw_log: str) -> dict:
    """
    Parse a single raw log string through all layers in order.
    Returns a normalized record dict regardless of parse outcome.
    """
    if not isinstance(raw_log, str) or raw_log.strip() in ("", "MALFORMED_LOG"):
        record = _parse_failure_record(raw_log)
        record["parse_method"] = "MALFORMED" if "MALFORMED" in str(raw_log) else "EMPTY"
        return record

    # Layer 1: Regex cascade
    record = _try_regex_cascade(raw_log)
    if record:
        return record

    # Layer 2: Heuristic fallback
    record = _try_heuristic_fallback(raw_log)
    if record:
        return record

    # Layer 3: spaCy EntityRuler
    record = _try_spacy_fallback(raw_log)
    if record:
        return record

    # Layer 4: LLM fallback (only if API key present)
    record = _try_llm_fallback(raw_log)
    if record:
        return record

    # Layer 5: Failure
    return _parse_failure_record(raw_log)


def parse_dataframe(df: pd.DataFrame, log_col: str = "raw_log") -> pd.DataFrame:
    """
    Parse all rows in a DataFrame.
    """
    records = [parse_log(row) for row in df[log_col]]
    parsed_df = pd.DataFrame(records)

    # Merge back any original columns (e.g., injected anomaly labels)
    extra_cols = [c for c in df.columns if c != log_col]
    for col in extra_cols:
        parsed_df[col] = df[col].values

    # Parse quality report
    total     = len(parsed_df)
    success   = parsed_df["parse_success"].sum()
    by_method = parsed_df["parse_method"].value_counts()

    print("\n" + "="*55)
    print("  PARSE QUALITY REPORT")
    print("="*55)
    print(f"  Total rows    : {total}")
    print(f"  Parsed OK     : {success} ({100*success/total:.1f}%)")
    print(f"  Failed        : {total - success} ({100*(total-success)/total:.1f}%)")
    print("\n  Breakdown by method:")
    for method, count in by_method.items():
        print(f"    {method:<20} {count:>5}  ({100*count/total:.1f}%)")
    print("="*55 + "\n")

    return parsed_df
