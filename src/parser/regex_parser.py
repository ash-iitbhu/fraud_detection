import re
import logging
from typing import Optional


from configs.config import KNOWN_CITIES, KNOWN_DEVICES, CURRENCY_SYMBOLS
import src.parser.helper as helper
from src.parser.helper import _normalize_city, _normalize_device, _normalize_txn_type, _parse_timestamp, _extract_amount_currency

logger = logging.getLogger(__name__)

class RegexParser:
    def __init__(self):
        self.patterns = helper.PATTERNS
        self.known_cities = KNOWN_CITIES
        self.known_devices = KNOWN_DEVICES
        self.currency_symbols = CURRENCY_SYMBOLS

    def build_record(self, match: re.Match, format_id: str, raw_log: str) -> dict:
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

    def try_regex_cascade(self, raw_log: str) -> Optional[dict]:
        for fmt_id, pattern in self.patterns:
            m = pattern.match(raw_log.strip())
            if m:
                return self.build_record(m, fmt_id, raw_log)
        return None

    def try_heuristic_fallback(self, raw_log: str) -> Optional[dict]:
        log = raw_log.strip()

        user_m = helper._RE_USER.search(log)
        user_id = user_m.group(1).lower() if user_m else "UNKNOWN"

        ts = None
        for pattern in (helper._RE_TS_ISO, helper._RE_TS_DMY):
            m = pattern.search(log)
            if m:
                ts = _parse_timestamp(m.group(1))
                break

        amt_matches = helper._RE_AMOUNT.findall(log)
        amount, currency = None, "UNKNOWN"
        if amt_matches:
            raw_amt = amt_matches[0]
            cur = ""
            for sym in self.currency_symbols:
                if sym in raw_amt:
                    cur = sym
                    raw_amt = raw_amt.replace(sym, "")
                    break
            try:
                amount = float(raw_amt.replace(",", "."))
                currency = cur if cur else "UNKNOWN"
            except ValueError:
                pass

        txn_m = helper._RE_TXN.search(log)
        txn_type = _normalize_txn_type(txn_m.group(1)) if txn_m else "UNKNOWN"

        log_lower = log.lower()
        city = "UNKNOWN"
        for c in self.known_cities:
            if c in log_lower:
                city = c.title()
                break

        device = "UNKNOWN"
        for d in self.known_devices:
            if d in log_lower:
                device = d.title()
                break

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