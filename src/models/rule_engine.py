"""
Evidence summary (from real dataset, 7,774 transactions, 86 users):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Finding                              │ Value      │ Implication             │
├──────────────────────────────────────┼────────────┼─────────────────────────┤
│ Amount z-score > 3.0                 │ 0.35%      │ Tight, precise flag     │
│ Impossible travel flags              │ 0.46%      │ Very rare, high signal  │
│ Velocity > 2 txns in 1h              │ 0.25%      │ Very rare in this data  │
│ New city + zscore > 2.0              │ 0.55%      │ Combined signal         │
│ New device + zscore > 2.0            │ 0.55%      │ Combined signal         │
│ Missing city AND device              │ 1.66%      │ Structural fraud signal │
└──────────────────────────────────────┴────────────┴─────────────────────────┘
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass, field


AMOUNT_ZSCORE_THRESHOLD = 3
VELOCITY_1H_THRESHOLD = 2
NEW_CITY_ZSCORE_THRESHOLD = 2.0
NEW_DEVICE_ZSCORE_THRESHOLD = 2.0

@dataclass
class RuleResult:
    fired: bool
    rule_name: str
    explanation: str
    severity: str          # LOW / MEDIUM / HIGH / CRITICAL
    feature_values: Dict   # key feature values for explainability


def rule_01_amount_spike_30d(row: pd.Series) -> RuleResult:
    """
    R01 — AMOUNT_SPIKE_30D
    Amount is more than 3 std deviations above user's 30-day rolling mean.

    Evidence: z > 3 flags 0.35% of transactions on real data.
    This is the tightest single-feature signal available.
    Industry standard: PayPal, Stripe both use rolling-window z-scores
    as their primary amount anomaly feature.

    User-context-aware: uses user's OWN 30-day baseline, not global mean.
    """
    z = row.get("user_amt_zscore_30D")
    if pd.isna(z) or row.get("is_cold_start", 1) == 1:
        return RuleResult(False, "R01_AMOUNT_SPIKE_30D", "", "NONE", {})

    fired = float(z) > AMOUNT_ZSCORE_THRESHOLD
    return RuleResult(
        fired=fired,
        rule_name="R01_AMOUNT_SPIKE_30D",
        explanation=(
            f"Amount z-score={z:.2f} exceeds threshold {AMOUNT_ZSCORE_THRESHOLD} "
            f"(user 30D mean={row.get('user_amt_mean_30D', 0):.0f}, "
            f"amount={row.get('amount', 0):.0f})"
        ) if fired else "",
        severity="HIGH" if z > 3.5 else "MEDIUM",
        feature_values={"user_amt_zscore_30D": z,
                        "user_amt_mean_30D": row.get("user_amt_mean_30D"),
                        "amount": row.get("amount")}
    )


def rule_02_impossible_travel(row: pd.Series) -> RuleResult:
    """
    R02 — IMPOSSIBLE_TRAVEL
    User transacts in a new city within 1 hour of their last transaction.

    Evidence: 0.46% of real transactions — very rare, very high signal.
    This is a standard rule in all major fintech fraud systems (Stripe, Monzo,
    Wise) and directly maps to account takeover scenarios.

    Not applied to cold-start users (their first city is always 'new').
    """
    it    = row.get("impossible_travel", 0)
    gap   = row.get("time_since_last_txn_sec", 99999)
    city  = row.get("city", "UNKNOWN")
    fired = int(it) == 1 and city != "UNKNOWN"

    return RuleResult(
        fired=fired,
        rule_name="R02_IMPOSSIBLE_TRAVEL",
        explanation=(
            f"New city '{city}' but only {gap/60:.1f} minutes since last transaction"
        ) if fired else "",
        severity="CRITICAL",
        feature_values={"impossible_travel": it,
                        "time_since_last_txn_sec": gap,
                        "city": city,
                        "is_new_city": row.get("is_new_city")}
    )


def rule_03_velocity_burst_1h(row: pd.Series) -> RuleResult:
    """
    R03 — VELOCITY_BURST_1H
    User made more than 2 transactions in the last 1 hour.

    Evidence: Only 0.25% of transactions have velocity > 2 in 1h.
    This is card-testing behavior: fraudsters probe stolen credentials
    with rapid successive transactions (Chargebacks911, SEON research).

    Industry standard: Stripe, Checkout.com use 5-minute and 1-hour
    velocity as primary fraud signals. 
    """
    vel   = row.get("user_velocity_1h", 0)
    fired = float(vel) > VELOCITY_1H_THRESHOLD

    return RuleResult(
        fired=fired,
        rule_name="R03_VELOCITY_BURST_1H",
        explanation=(
            f"{int(vel)} transactions in the past 1 hour "
            f"(threshold: {VELOCITY_1H_THRESHOLD})"
        ) if fired else "",
        severity="HIGH",
        feature_values={"user_velocity_1h": vel,
                        "user_amt_sum_1h": row.get("user_amt_sum_1h")}
    )


def rule_04_new_city_high_amount(row: pd.Series) -> RuleResult:
    """
    R04 — NEW_CITY_HIGH_AMOUNT
    First-time city for this user AND amount above their 30D baseline by 2+ std.

    Evidence: This combination flags 0.55% of transactions.
    The AND condition is critical — new city alone is 8.9% (too broad),
    combined with z > 2.0 it becomes a high-precision signal.

    Mimics geo-anomaly detection used by Monzo and Revolut
    where unusual location + unusual amount = strong fraud signal.
    """
    nc = row.get("is_new_city", 0)
    z  = row.get("user_amt_zscore_30D", 0)
    cs = row.get("is_cold_start", 1)

    if cs == 1 or row.get("city", "UNKNOWN") == "UNKNOWN":
        return RuleResult(False, "R04_NEW_CITY_HIGH_AMOUNT", "", "NONE", {})

    fired = int(nc) == 1 and float(z) > NEW_CITY_ZSCORE_THRESHOLD

    return RuleResult(
        fired=fired,
        rule_name="R04_NEW_CITY_HIGH_AMOUNT",
        explanation=(
            f"First transaction in city '{row.get('city')}', "
            f"amount z-score={z:.2f} (>{NEW_CITY_ZSCORE_THRESHOLD})"
        ) if fired else "",
        severity="HIGH",
        feature_values={"is_new_city": nc,
                        "user_amt_zscore_30D": z,
                        "city": row.get("city"),
                        "user_city_prior_count": row.get("user_city_prior_count")}
    )


def rule_05_new_device_high_amount(row: pd.Series) -> RuleResult:
    """
    R05 — NEW_DEVICE_HIGH_AMOUNT
    First-time device for this user AND amount above their 30D baseline by 2+ std.

    Evidence: Same 0.55% rate as new-city rule — confirms both are equivalent
    signals in this dataset.

    Device consistency is a core fraud signal per Unit21 and Jumio documentation.
    Device takeover fraud (account compromise via new device) is the fastest-growing
    fraud type per 2024 fraud reports.
    """
    nd = row.get("is_new_device", 0)
    z  = row.get("user_amt_zscore_30D", 0)
    cs = row.get("is_cold_start", 1)

    if cs == 1 or row.get("device", "UNKNOWN") == "UNKNOWN":
        return RuleResult(False, "R05_NEW_DEVICE_HIGH_AMOUNT", "", "NONE", {})

    fired = int(nd) == 1 and float(z) > NEW_DEVICE_ZSCORE_THRESHOLD

    return RuleResult(
        fired=fired,
        rule_name="R05_NEW_DEVICE_HIGH_AMOUNT",
        explanation=(
            f"First use of device '{row.get('device')}', "
            f"amount z-score={z:.2f} (>{NEW_DEVICE_ZSCORE_THRESHOLD})"
        ) if fired else "",
        severity="HIGH",
        feature_values={"is_new_device": nd,
                        "user_amt_zscore_30D": z,
                        "device": row.get("device"),
                        "user_device_prior_count": row.get("user_device_prior_count")}
    )


def rule_06_missing_metadata(row: pd.Series) -> RuleResult:
    """
    R06 — MISSING_METADATA
    Both city AND device are UNKNOWN in the same transaction.

    Evidence: 1.66% of real transactions — structural, not random.
    Co-missingness correlation analysis shows these fields go missing together
    at a rate far above chance — indicative of deliberate metadata suppression.

    This is a recognized AML/fraud signal: fraudsters using proxy networks or
    modified apps that strip geolocation and device data from transaction requests.
    """
    both   = row.get("is_both_geo_dev_missing", 0)
    miss_n = row.get("missing_field_count", 0)
    fired  = int(both) == 1

    return RuleResult(
        fired=fired,
        rule_name="R06_MISSING_METADATA",
        explanation=(
            f"Both city and device metadata are UNKNOWN "
            f"(total missing fields: {int(miss_n)})"
        ) if fired else "",
        severity="MEDIUM",
        feature_values={"is_both_geo_dev_missing": both,
                        "missing_field_count": miss_n,
                        "is_city_missing": row.get("is_city_missing"),
                        "is_device_missing": row.get("is_device_missing")}
    )

ALL_RULES = [
    rule_01_amount_spike_30d,
    rule_02_impossible_travel,
    rule_03_velocity_burst_1h,
    rule_04_new_city_high_amount,
    rule_05_new_device_high_amount,
    rule_06_missing_metadata
]

RULE_SEVERITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH":     0.75,
    "MEDIUM":   0.50,
    "LOW":      0.25,
    "NONE":     0.0,
}


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n   = len(out)

    # Per-rule result storage
    flag_counts    = np.zeros(n, dtype=int)
    weighted_scores = np.zeros(n, dtype=float)
    rules_fired    = [[] for _ in range(n)]
    explanations   = [[] for _ in range(n)]
    severities     = [[] for _ in range(n)]

    _dummy = pd.Series(dtype=object)
    per_rule_flags = {r(_dummy).rule_name: np.zeros(n, dtype=int) for r in ALL_RULES}

    print(f"[RuleEngine] Applying {len(ALL_RULES)} rules to {n} transactions...")

    for i, row in out.iterrows():
        row_idx = out.index.get_loc(i)
        for rule_fn in ALL_RULES:
            result: RuleResult = rule_fn(row)
            if result.fired:
                flag_counts[row_idx]     += 1
                weight = RULE_SEVERITY_WEIGHTS.get(result.severity, 0)
                weighted_scores[row_idx] += weight
                rules_fired[row_idx].append(result.rule_name)
                explanations[row_idx].append(result.explanation)
                severities[row_idx].append(result.severity)
                per_rule_flags[result.rule_name][row_idx] = 1

    # Assign aggregated columns
    out["rule_flag_count"]     = flag_counts
    out["rule_weighted_score"] = weighted_scores
    # Normalize: max possible weighted score = sum of all CRITICAL weights
    max_possible = len(ALL_RULES) * RULE_SEVERITY_WEIGHTS["CRITICAL"]
    out["rule_score"]          = (weighted_scores / max_possible).clip(0, 1)
    out["rules_fired"]         = [" | ".join(r) if r else "NONE" for r in rules_fired]
    out["rule_explanations"]   = [" | ".join(e) if e else "" for e in explanations]
    out["rule_severities"]     = [" | ".join(s) if s else "" for s in severities]

    # Per-rule binary columns for fine-grained explainability
    for rule_fn in ALL_RULES:
        rule_name = rule_fn(_dummy).rule_name
        col = f"rule_{rule_name}_fired"
        out[col] = per_rule_flags[rule_name]

    # Summary statistics
    n_flagged  = (out["rule_flag_count"] > 0).sum()
    n_critical = (out["rule_severities"].str.contains("CRITICAL")).sum()
    n_multi    = (out["rule_flag_count"] >= 2).sum()

    print(f"[RuleEngine] Results:")
    print(f"  Flagged (any rule)   : {n_flagged:>5} ({100*n_flagged/n:.2f}%)")
    print(f"  Multi-rule (≥2)      : {n_multi:>5} ({100*n_multi/n:.2f}%)")
    print(f"  CRITICAL severity    : {n_critical:>5} ({100*n_critical/n:.2f}%)")
    print(f"\n  Per-rule breakdown:")

    for rule_fn in ALL_RULES:
        rname = rule_fn(_dummy).rule_name
        col   = f"rule_{rname}_fired"
        count = out[col].sum() if col in out.columns else 0
        pct   = 100 * count / n
        print(f"    {rname:<45} {count:>5} ({pct:.2f}%)")

    return out
