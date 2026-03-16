import json
import logging
from typing import Optional

from dateutil import parser as dateutil_parser


from configs.config import LLM_MODEL, LLM_MAX_TOKENS, ANTHROPIC_API_KEY
from src.parser.helper import _normalize_city, _normalize_device, _normalize_txn_type, _parse_timestamp

logger = logging.getLogger(__name__)


# Layer 4: LLM API Fallback (env-gated)

class LLMParser:
    def __init__(self, api_key=None, model=None, max_tokens=None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or LLM_MODEL
        self.max_tokens = max_tokens or LLM_MAX_TOKENS

    _LLM_PROMPT = """Extract the following fields from this financial transaction log and return ONLY valid JSON with no preamble or markdown.
If a field cannot be determined, use null.

Fields: user_id, timestamp (ISO format), txn_type, amount (float), currency (symbol), city, device

Log: {log}

JSON:"""

    def try_llm_fallback(self, raw_log: str) -> Optional[dict]:
        """
        Call a lightweight LLM API to parse difficult log entries.
        Requires ANTHROPIC_API_KEY.
        Fails silently — never breaks the pipeline.
        """
        if not self.api_key:
            return None

        try:
            import urllib.request
            prompt = self._LLM_PROMPT.format(log=raw_log)

            payload = json.dumps({
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key": self.api_key,
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
