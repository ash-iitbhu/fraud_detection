"""
src/models/rule_engine.py
--------------------------
Deterministic rule-based anomaly detection.

Rules are user-context-aware — they use the user's own historical behaviour
rather than fixed global thresholds wherever possible.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

AMOUNT_ZSCORE_THRESHOLD    = 3.0
VELOCITY_THRESHOLD_1H      = 3
IMPOSSIBLE_TRAVEL_SECS     = 3600
LATE_NIGHT_RATIO_THRESHOLD = 0.10
NEW_CITY_AMOUNT_ZSCORE     = 1.5
PAIR_RARITY_THRESHOLD      = 0.02
AMOUNT_VELOCITY_THRESHOLD  = 5000


def _rule_amount_spike(row):
    if pd.notna(row.get("amount_zscore")) and row["amount_zscore"] > AMOUNT_ZSCORE_THRESHOLD:
        return 1, f"AMOUNT_SPIKE: z={row['amount_zscore']:.2f}"
    return 0, ""


def _rule_late_night_unusual(row):
    if (row.get("is_late_night", 0) == 1
            and row.get("user_late_night_ratio", 1.0) < LATE_NIGHT_RATIO_THRESHOLD):
        return 1, f"LATE_NIGHT_UNUSUAL: hour={int(row.get('hour_of_day',0))}"
    return 0, ""


def _rule_new_city_high_value(row):
    if (row.get("is_new_city", 0) == 1
            and row.get("amount_zscore", 0) > NEW_CITY_AMOUNT_ZSCORE):
        return 1, f"NEW_CITY_HIGH_VALUE: city={row.get('city','?')}, z={row.get('amount_zscore',0):.2f}"
    return 0, ""


def _rule_impossible_travel(row):
    if (row.get("is_new_city", 0) == 1
            and row.get("time_since_last_txn_sec", 9999) < IMPOSSIBLE_TRAVEL_SECS
            and row.get("is_cold_start", 0) == 0):
        mins = row.get("time_since_last_txn_sec", 0) / 60
        return 1, f"IMPOSSIBLE_TRAVEL: new_city={row.get('city','?')}, {mins:.0f}m since last txn"
    return 0, ""


def _rule_high_velocity(row):
    if row.get("user_txn_velocity_1h", 0) > VELOCITY_THRESHOLD_1H:
        return 1, f"HIGH_VELOCITY: {int(row.get('user_txn_velocity_1h',0))} txns/1h"
    return 0, ""


def _rule_high_amount_velocity(row):
    if row.get("user_amount_velocity_1h", 0) > AMOUNT_VELOCITY_THRESHOLD:
        return 1, f"HIGH_AMOUNT_VELOCITY: £{row.get('user_amount_velocity_1h',0):.0f}/1h"
    return 0, ""


def _rule_rare_device_city(row):
    if row.get("device_city_pair_rarity", 0) > PAIR_RARITY_THRESHOLD:
        return 1, f"RARE_DEVICE_CITY: ({row.get('device','?')}, {row.get('city','?')})"
    return 0, ""


def _rule_missing_metadata(row):
    if row.get("missing_field_count", 0) >= 2:
        return 1, "MISSING_METADATA: city and device both UNKNOWN"
    return 0, ""


def _rule_new_device_high_value(row):
    if (row.get("is_new_device", 0) == 1
            and row.get("amount_zscore", 0) > NEW_CITY_AMOUNT_ZSCORE
            and row.get("is_cold_start", 0) == 0):
        return 1, f"NEW_DEVICE_HIGH_VALUE: device={row.get('device','?')}, z={row.get('amount_zscore',0):.2f}"
    return 0, ""


ALL_RULES = [
    _rule_amount_spike,
    _rule_late_night_unusual,
    _rule_new_city_high_value,
    _rule_impossible_travel,
    _rule_high_velocity,
    _rule_high_amount_velocity,
    _rule_rare_device_city,
    _rule_missing_metadata,
    _rule_new_device_high_value,
]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all rules to every row. Adds rule_flag_count, rule_flags, rule_score."""
    out = df.copy()
    flag_counts  = np.zeros(len(out), dtype=int)
    flags_list   = [[] for _ in range(len(out))]
    explain_list = [[] for _ in range(len(out))]

    for rule_fn in ALL_RULES:
        rule_name = rule_fn.__name__.replace("_rule_", "").upper()
        for i, (_, row) in enumerate(out.iterrows()):
            flag, explanation = rule_fn(row)
            if flag:
                flag_counts[i] += 1
                flags_list[i].append(rule_name)
                explain_list[i].append(explanation)

    out["rule_flag_count"]   = flag_counts
    out["rule_flags"]        = [" | ".join(f) if f else "NONE" for f in flags_list]
    out["rule_explanations"] = [" | ".join(e) if e else "" for e in explain_list]
    out["rule_score"]        = (out["rule_flag_count"] / len(ALL_RULES)).clip(0, 1)

    n_flagged = (out["rule_flag_count"] > 0).sum()
    print(f"Rule engine: {n_flagged} transactions flagged ({100*n_flagged/len(out):.1f}%)")
    return out
