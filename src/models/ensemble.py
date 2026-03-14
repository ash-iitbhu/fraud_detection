"""
src/models/ensemble.py
-----------------------
Weighted ensemble combining Rule Engine, Isolation Forest, and Autoencoder scores
into a single final_risk_score in [0, 1].

Weight justification:
    Default weights are equal (0.33 each) — this is the principled default
    when no analyst feedback is available to tune them.
    Weights should be updated via precision-at-K tuning once analyst-reviewed
    labels accumulate.

Risk tiering:
    Tier 1 (>= 0.75) : Auto-block / immediate analyst review
    Tier 2 (0.50-0.75): Step-up authentication
    Tier 3 (0.25-0.50): Silent monitoring
    Normal (< 0.25)   : No action
"""

import pandas as pd
import numpy as np

# Default equal weights — update after analyst feedback
DEFAULT_WEIGHTS = {
    "rule_score": 0.33,
    "if_score":   0.33,
    "ae_score":   0.34,
}

TIER_THRESHOLDS = {
    "TIER_1": 0.75,
    "TIER_2": 0.50,
    "TIER_3": 0.25,
}


def _minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def compute_ensemble_score(
    df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute the final composite risk score.

    Expects these columns to exist in df (added by individual model scorers):
        rule_score  : float [0,1]
        if_score    : float [0,1]
        ae_score    : float [0,1]

    Adds columns:
        final_risk_score : float [0,1]
        risk_tier        : str (TIER_1 / TIER_2 / TIER_3 / NORMAL)
        model_agreement  : int (how many models independently flagged this row)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    out = df.copy()

    # Re-normalize each score to [0,1] across the full dataset
    # (individual models may have already done this, but we re-normalize
    #  to ensure comparability before weighting)
    for col in ["rule_score", "if_score", "ae_score"]:
        if col not in out.columns:
            out[col] = 0.0
        out[f"{col}_norm"] = _minmax(out[col].fillna(0))

    # Weighted sum
    out["final_risk_score"] = (
        weights.get("rule_score", 0.33) * out["rule_score_norm"] +
        weights.get("if_score",   0.33) * out["if_score_norm"]   +
        weights.get("ae_score",   0.34) * out["ae_score_norm"]
    )

    # Re-normalize final score to [0,1]
    out["final_risk_score"] = _minmax(out["final_risk_score"])

    # Risk tier assignment
    def _assign_tier(score):
        if score >= TIER_THRESHOLDS["TIER_1"]:
            return "TIER_1"
        elif score >= TIER_THRESHOLDS["TIER_2"]:
            return "TIER_2"
        elif score >= TIER_THRESHOLDS["TIER_3"]:
            return "TIER_3"
        return "NORMAL"

    out["risk_tier"] = out["final_risk_score"].apply(_assign_tier)

    # Model agreement: count of individual model flags
    flag_cols = [c for c in ["rule_flag_count", "if_is_anomaly", "ae_is_anomaly"]
                 if c in out.columns]
    if flag_cols:
        rule_flag = (out.get("rule_flag_count", pd.Series(0, index=out.index)) > 0).astype(int)
        if_flag   = out.get("if_is_anomaly",   pd.Series(0, index=out.index))
        ae_flag   = out.get("ae_is_anomaly",   pd.Series(0, index=out.index))
        out["model_agreement"] = rule_flag + if_flag + ae_flag
    else:
        out["model_agreement"] = 0

    # Print tier distribution
    tier_counts = out["risk_tier"].value_counts()
    print("\n[Ensemble] Risk Tier Distribution:")
    for tier in ["TIER_1", "TIER_2", "TIER_3", "NORMAL"]:
        n = tier_counts.get(tier, 0)
        print(f"  {tier:<8}: {n:>5} ({100*n/len(out):.1f}%)")

    return out


def get_top_anomalies(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Return the top-N highest risk transactions sorted by final_risk_score."""
    return (
        df.sort_values("final_risk_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
