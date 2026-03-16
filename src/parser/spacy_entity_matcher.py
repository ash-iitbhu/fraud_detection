import logging
from typing import Optional

import spacy

from configs.config import KNOWN_CITIES,CURRENCY_SYMBOLS
import src.parser.helper as helper
from src.parser.helper import _normalize_txn_type, _parse_timestamp

logger = logging.getLogger(__name__)

class SpacyEntityMatcher:
    def __init__(self):
        self._nlp = None

    def _build_spacy_ruler(self):
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "USER_ID",  "pattern": [{"TEXT": {"REGEX": r"user\d+"}}]},
            {"label": "CURRENCY", "pattern": [{"TEXT": {"REGEX": r"[€£$]\d+|\d+[€£$]"}}]},
            {"label": "CITY", "pattern": [{"LOWER": {"IN": list(KNOWN_CITIES)}}]},
        ]
        ruler.add_patterns(patterns)
        return nlp

    def parse(self, raw_log: str) -> Optional[dict]:
        if self._nlp is None:
            self._nlp = self._build_spacy_ruler()
        if self._nlp is None:
            return None

        doc = self._nlp(raw_log)
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
            return None

        ts_m = helper._RE_TS_ISO.search(raw_log) or helper._RE_TS_DMY.search(raw_log)
        ts   = _parse_timestamp(ts_m.group(1)) if ts_m else None
        txn_m = helper._RE_TXN.search(raw_log)
        txn = _normalize_txn_type(txn_m.group(1)) if txn_m else "UNKNOWN"

        return {
            "raw_log":       raw_log,
            "user_id":       user_id,
            "timestamp":     ts,
            "txn_type":      txn,
            "amount":        amount,
            "currency":      currency,
            "city":          city,
            "device":        "UNKNOWN",
            "parse_method":  "SPACY_RULER",
            "parse_success": True,
        }

