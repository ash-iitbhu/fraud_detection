"""
  1. Batch mode  — generate_batch_reports(final_df, ...)
     For the top-N highest risk transactions in a scored DataFrame.

  2. Online mode — score_and_explain_single(raw_log, ...)
     For scoring and explaining a single raw log string in real time.
     Pass the raw log text, get back a fully scored and explained report.
     This is the production inference path.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

TIER_ACTIONS = {
    "TIER_1": "AUTO-BLOCK: Decline transaction. Notify user. Escalate to fraud team.",
    "TIER_2": "STEP-UP AUTH: Request OTP or biometric verification before proceeding.",
    "TIER_3": "SILENT MONITOR: Allow transaction. Flag for enhanced monitoring.",
    "NORMAL": "NO ACTION: Transaction appears normal.",
}

_CONTEXT_FEATURE_MAP = {
    "user_amt_mean_30D":       "User 30D avg amount",
    "user_amt_mean_7D":        "User 7D avg amount",
    "user_amt_zscore_30D":     "Amount z-score (30D)",
    "user_amt_zscore_7D":      "Amount z-score (7D)",
    "user_velocity_1h":        "Txns in past 1h",
    "user_txn_count_30D":      "Txns in past 30D",
    "is_new_city":             "New city for user",
    "is_new_device":           "New device for user",
    "is_cold_start":           "Cold-start user",
    "impossible_travel":       "Impossible travel flag",
    "missing_field_count":     "Missing metadata fields",
    "amt_ratio_from_prev":     "Amount ratio vs prev txn",
    "user_location_entropy":   "Location entropy",
    "is_burst_5min":           "Burst (<5min gap)",
    "is_amount_escalating":    "Escalating amount pattern",
}


def generate_transaction_report(
    row: pd.Series,
    feature_names: List[str],
    importance_values: Optional[np.ndarray] = None,
    per_feature_recon_error: Optional[pd.Series] = None,
    top_n_features: int = 5,
    importance_method: str = "SHAP",
) -> str:
    sep  = "─" * 65
    sep2 = "═" * 65

    lines = [
        sep2,
        "  FRAUD INVESTIGATION REPORT",
        sep2,
        f"  User     : {row.get('user_id', 'N/A')}",
        f"  Time     : {row.get('timestamp', 'N/A')}",
        f"  Amount   : {row.get('currency', '?')}{row.get('amount', 'N/A')}",
        f"  City     : {row.get('city', 'N/A')}",
        f"  Device   : {row.get('device', 'N/A')}",
        f"  Txn Type : {row.get('txn_type', 'N/A')}",
        sep,
        f"  RISK SCORE : {row.get('final_risk_score', 0):.3f}  |  "
        f"TIER: {row.get('risk_tier', 'N/A')}  |  "
        f"MODEL AGREEMENT: {int(row.get('model_agreement', 0))}/3",
        sep,
    ]

    
    rule_flag = int(row.get("rule_flag_count", 0) or 0) > 0
    if_flag   = int(row.get("if_is_anomaly",   0) or 0) == 1
    ae_flag   = int(row.get("ae_is_anomaly",   0) or 0) == 1
    lines += [
        "  Model Detections:",
        f"    {'✅' if rule_flag else '⬜'} Rule Engine       "
        f"score={row.get('rule_score', 0):.3f}",
        f"    {'✅' if if_flag   else '⬜'} Isolation Forest  "
        f"score={row.get('if_score',   0):.3f}",
        f"    {'✅' if ae_flag   else '⬜'} Autoencoder       "
        f"score={row.get('ae_score',   0):.3f}",
        sep,
    ]

    rule_expls = str(row.get("rule_explanations", "") or "")
    if rule_expls.strip():
        lines.append("  Rule Flags Triggered:")
        for expl in rule_expls.split(" | "):
            if expl.strip():
                lines.append(f"    → {expl.strip()}")
        lines.append(sep)

    # ── Feature importance (SHAP or permutation) ───────────────────────────────
    if importance_values is not None and len(importance_values) > 0 and len(feature_names) > 0:
        lines.append(f"  Top Features — Isolation Forest ({importance_method}):")
        # Align feature_names length with importance_values length
        n = min(len(feature_names), len(importance_values))
        pairs = sorted(
            zip(feature_names[:n], importance_values[:n]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for feat, val in pairs[:top_n_features]:
            direction = "↑ anomalous" if val > 0 else "↓ normal"
            lines.append(f"    {feat:<38} {val:+.5f}  {direction}")
        lines.append(sep)

    # ── AE reconstruction error ────────────────────────────────────────────────
    if per_feature_recon_error is not None and len(per_feature_recon_error) > 0:
        lines.append("  Autoencoder Reconstruction Error (top features):")
        top_recon = per_feature_recon_error.sort_values(ascending=False).head(top_n_features)
        for feat, err in top_recon.items():
            level = "HIGH" if err > 0.5 else "MED " if err > 0.2 else "LOW "
            lines.append(f"    {feat:<38} {err:.5f}  [{level}]")
        lines.append(sep)

    # ── Transaction context ────────────────────────────────────────────────────
    lines.append("  Transaction Context:")
    for col, label in _CONTEXT_FEATURE_MAP.items():
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, (int, float)):
            if col in ("is_new_city", "is_new_device", "is_cold_start",
                       "impossible_travel", "is_burst_5min", "is_amount_escalating"):
                display = "Yes" if int(val) == 1 else "No"
            else:
                display = f"{val:.2f}"
        else:
            display = str(val)
        lines.append(f"    {label:<35} : {display}")

    lines += [
        sep,
        f"  RECOMMENDED ACTION: {TIER_ACTIONS.get(row.get('risk_tier', 'NORMAL'), '?')}",
        sep2,
    ]

    return "\n".join(lines)


def generate_batch_reports(
    df: pd.DataFrame,
    top_n: int = 10,
    feature_names: Optional[List[str]] = None,
    shap_matrix: Optional[np.ndarray] = None,
    per_feature_recon_df: Optional[pd.DataFrame] = None,
    importance_method: str = "SHAP",
) -> List[str]:
    # Reset index so positional lookups are safe
    df_reset = df.reset_index(drop=True)
    top_df   = df_reset.nlargest(top_n, "final_risk_score").reset_index(drop=False)
    reports  = []

    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        # orig_idx is the position in df_reset (0-based, contiguous)
        orig_idx = int(row.get("index", rank - 1))

        row_importance = None
        if shap_matrix is not None and orig_idx < len(shap_matrix):
            row_importance = shap_matrix[orig_idx]

        row_recon = None
        if per_feature_recon_df is not None and orig_idx < len(per_feature_recon_df):
            row_recon = per_feature_recon_df.iloc[orig_idx]

        report = generate_transaction_report(
            row=row,
            feature_names=feature_names or [],
            importance_values=row_importance,
            per_feature_recon_error=row_recon,
            importance_method=importance_method,
        )
        reports.append(f"\n  ── RANK #{rank} by Risk Score ──\n{report}")

    return reports


def score_and_explain_single(
    raw_log: str,
    parser,
    feature_engine,
    if_detector,
    ae_detector,
    ensemble_scorer,
    rule_applier=None,
    top_n_features: int = 5,
) -> str:
    import pandas as pd

    single_df = pd.DataFrame({"raw_log": [raw_log]})
    parsed    = parser.parse_dataframe(single_df, log_col="raw_log")
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], errors="coerce")
    parsed["amount"]    = pd.to_numeric(parsed["amount"], errors="coerce")

    if not parsed["parse_success"].any():
        return (
            "═" * 65 + "\n"
            "  FRAUD INVESTIGATION REPORT\n"
            "═" * 65 + "\n"
            f"  ❌ PARSE FAILURE\n"
            f"  Raw log could not be parsed: {raw_log[:80]}\n"
            "═" * 65
        )
    featured = feature_engine.transform(parsed)

    if rule_applier is not None:
        featured = rule_applier(featured)
    else:
        featured["rule_score"]      = 0.0
        featured["rule_flag_count"] = 0
        featured["rule_explanations"] = ""


    featured = if_detector.score(featured)
    featured = ae_detector.score(featured)
    featured = ensemble_scorer.score(featured)

    importance_values = if_detector.get_shap_values(featured)
    feature_names     = if_detector.get_feature_names()

    row_importance = None
    importance_method = "SHAP"
    if importance_values is not None and len(importance_values) > 0:
        row_importance = importance_values[0]
        

    row_recon = None
    try:
        recon_df  = ae_detector.get_per_feature_recon_error(featured)
        if not recon_df.empty:
            row_recon = recon_df.iloc[0]
    except Exception:
        pass 

    row = featured.iloc[0]
    return generate_transaction_report(
        row=row,
        feature_names=feature_names,
        importance_values=row_importance,
        per_feature_recon_error=row_recon,
        top_n_features=top_n_features,
        importance_method=importance_method,
    )