import logging
import pandas as pd
from configs.config import LLM_MODEL, LLM_MAX_TOKENS, ANTHROPIC_API_KEY

from src.parser.regex_parser import RegexParser
from src.parser.spacy_entity_matcher import SpacyEntityMatcher
from src.parser.llm_parser import LLMParser
logger = logging.getLogger(__name__)


class LogParser:
    def __init__(self):
        self.regex_parser = RegexParser()
        self.spacy_matcher = SpacyEntityMatcher()
        self.llm_parser = None
        if ANTHROPIC_API_KEY:
            self.llm_parser = LLMParser(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                api_key=ANTHROPIC_API_KEY
            )

    def _parse_failure_record(self, raw_log: str) -> dict:
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

    def parse_log(self, raw_log: str) -> dict:
        if not isinstance(raw_log, str) or raw_log.strip() in ("", "MALFORMED_LOG"):
            record = self._parse_failure_record(raw_log)
            record["parse_method"] = "MALFORMED" if "MALFORMED" in str(raw_log) else "EMPTY"
            return record

        record = self.regex_parser.try_regex_cascade(raw_log)
        if record:
            return record

        record = self.regex_parser.try_heuristic_fallback(raw_log)
        if record:
            return record

        record = self.spacy_matcher.parse(raw_log)
        if record:
            return record

        record = self.llm_parser.try_llm_fallback(raw_log)
        if record:
            return record

        return self._parse_failure_record(raw_log)

    def parse_dataframe(self, df: pd.DataFrame, log_col: str = "raw_log") -> pd.DataFrame:
        records = [self.parse_log(row) for row in df[log_col]]
        parsed_df = pd.DataFrame(records)
        extra_cols = [c for c in df.columns if c != log_col]
        for col in extra_cols:
            parsed_df[col] = df[col].values

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