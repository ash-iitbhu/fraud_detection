import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

DEFAULT_WEIGHTS = {
    "rule_score": 0.33,
    "if_score":   0.33,
    "ae_score":   0.34,
}


@dataclass
class EnsembleScorer:
    # Per-score training ranges
    score_ranges: Dict[str, tuple] = field(default_factory=dict)
    # Tier thresholds computed from training ensemble scores
    tier_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "TIER_1": 0.75, "TIER_2": 0.50, "TIER_3": 0.25
    })
    # Ensemble weights
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out    = df.copy()
        scores = ["rule_score", "if_score", "ae_score"]

        for col in scores:
            if col not in out.columns:
                out[col] = 0.0

            if col in self.score_ranges:
                mn, mx = self.score_ranges[col]
                # Clip to [0,1]: values outside training range are extreme anomalies
                out[f"{col}_norm"] = np.clip(
                    (out[col].fillna(0) - mn) / (mx - mn + 1e-9), 0.0, 1.0
                )
            else:
                # Fallback if range not available for this score
                out[f"{col}_norm"] = out[col].fillna(0)

        # Weighted sum (already in [0,1] so no re-normalization needed)
        out["final_risk_score"] = (
            self.weights.get("rule_score", 0.33) * out["rule_score_norm"] +
            self.weights.get("if_score",   0.33) * out["if_score_norm"]   +
            self.weights.get("ae_score",   0.34) * out["ae_score_norm"]
        ).clip(0.0, 1.0)

        # Tier assignment using frozen thresholds
        out["risk_tier"] = out["final_risk_score"].apply(
            lambda s: (
                "TIER_1" if s >= self.tier_thresholds["TIER_1"] else
                "TIER_2" if s >= self.tier_thresholds["TIER_2"] else
                "TIER_3" if s >= self.tier_thresholds["TIER_3"] else
                "NORMAL"
            )
        )

        # Model agreement
        rule_flag = (out.get("rule_flag_count", pd.Series(0, index=out.index)) > 0).astype(int)
        if_flag   = out.get("if_is_anomaly",   pd.Series(0, index=out.index))
        ae_flag   = out.get("ae_is_anomaly",   pd.Series(0, index=out.index))
        out["model_agreement"] = rule_flag + if_flag + ae_flag


        return out


def fit_ensemble(
    train_scored_df: pd.DataFrame,
    weights: Optional[Dict] = None,
) -> EnsembleScorer:
    scorer  = EnsembleScorer(weights=weights or dict(DEFAULT_WEIGHTS))
    scores  = ["rule_score", "if_score", "ae_score"]

    # Freeze per-score ranges from training data
    for col in scores:
        if col in train_scored_df.columns:
            vals = train_scored_df[col].fillna(0)
            scorer.score_ranges[col] = (float(vals.min()), float(vals.max()))
            print(f"  Frozen range [{col}]: [{vals.min():.4f}, {vals.max():.4f}]")

    # Compute a temporary ensemble score on training data to set tier thresholds
    temp = scorer.score(train_scored_df)
    train_ensemble_scores = temp["final_risk_score"]

    # Tier thresholds = percentiles of the training ensemble score distribution
    # This means TIER_1 = top 1%, TIER_2 = top 5%, TIER_3 = top 25%
    scorer.tier_thresholds = {
        "TIER_1": round(float(np.percentile(train_ensemble_scores, 99)), 4),
        "TIER_2": round(float(np.percentile(train_ensemble_scores, 95)), 4),
        "TIER_3": round(float(np.percentile(train_ensemble_scores, 75)), 4),
    }

    print(f"\n  Frozen tier thresholds (from training percentiles):")
    for tier, thr in scorer.tier_thresholds.items():
        print(f"    {tier}: >= {thr:.4f}")

    return scorer
