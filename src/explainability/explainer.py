"""
src/explainability/explainer.py
--------------------------------
Per-transaction fraud explanation reports combining:
  - Rule engine flags (always human-readable)
  - SHAP values from Isolation Forest
  - Per-feature reconstruction error from Autoencoder
  - Risk tier with recommended business action
"""

import numpy as np
import pandas as pd
from typing import Optional, List

TIER_ACTIONS = {
    "TIER_1": "AUTO-BLOCK: Decline transaction. Notify user. Escalate to fraud team.",
    "TIER_2": "STEP-UP AUTH: Request OTP or biometric verification before proceeding.",
    "TIER_3": "SILENT MONITOR: Allow transaction. Flag for enhanced monitoring.",
    "NORMAL":  "NO ACTION: Transaction appears normal.",
}


def generate_transaction_report(
    row: pd.Series,
    feature_names: List[str],
    shap_values: Optional[np.ndarray] = None,
    per_feature_recon_error: Optional[pd.Series] = None,
    top_n_features: int = 5,
) -> str:
    sep  = "─" * 65
    sep2 = "═" * 65

    lines = [
        sep2,
        "  FRAUD INVESTIGATION REPORT",
        sep2,
        f"  User     : {row.get('user_id', 'N/A')}",
        f"  Time     : {row.get('timestamp', 'N/A')}",
        f"  Amount   : {row.get('currency','?')}{row.get('amount', 'N/A')}",
        f"  City     : {row.get('city', 'N/A')}",
        f"  Device   : {row.get('device', 'N/A')}",
        f"  Txn Type : {row.get('txn_type', 'N/A')}",
        sep,
        f"  RISK SCORE : {row.get('final_risk_score', 0):.3f}  |  "
        f"TIER: {row.get('risk_tier', 'N/A')}  |  "
        f"MODEL AGREEMENT: {int(row.get('model_agreement', 0))}/3",
        sep,
    ]

    # Model flags
    rule_flag = int(row.get("rule_flag_count", 0) or 0) > 0
    if_flag   = int(row.get("if_is_anomaly",   0) or 0) == 1
    ae_flag   = int(row.get("ae_is_anomaly",   0) or 0) == 1
    lines += [
        "  Model Detections:",
        f"    {'✅' if rule_flag else '⬜'} Rule Engine      "
        f"score={row.get('rule_score', 0):.3f}",
        f"    {'✅' if if_flag   else '⬜'} Isolation Forest "
        f"score={row.get('if_score',   0):.3f}",
        f"    {'✅' if ae_flag   else '⬜'} Autoencoder      "
        f"score={row.get('ae_score',   0):.3f}",
        sep,
    ]

    # Rule explanations
    rule_expls = row.get("rule_explanations", "")
    if rule_expls:
        lines.append("  Rule Flags:")
        for expl in rule_expls.split(" | "):
            if expl.strip():
                lines.append(f"    → {expl.strip()}")
        lines.append(sep)

    # SHAP
    if shap_values is not None and len(shap_values) > 0:
        lines.append("  Top Features — Isolation Forest (SHAP):")
        pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
        for feat, val in pairs[:top_n_features]:
            lines.append(f"    {feat:<35} {val:+.4f}")
        lines.append(sep)

    # AE reconstruction error
    if per_feature_recon_error is not None and len(per_feature_recon_error) > 0:
        lines.append("  Autoencoder Reconstruction Error (top features):")
        top_recon = per_feature_recon_error.sort_values(ascending=False).head(top_n_features)
        for feat, err in top_recon.items():
            level = "HIGH  " if err > 0.5 else "MED   " if err > 0.2 else "LOW   "
            lines.append(f"    {feat:<35} {err:.4f}  [{level}]")
        lines.append(sep)

    # Context
    avg_amt = row.get('user_avg_amount_hist')
    lines += [
        "  Transaction Context:",
        f"    User historical avg amount   : "
        f"{avg_amt:.2f}" if pd.notna(avg_amt) else
        f"    User historical avg amount   : N/A (cold-start)",
        f"    Amount z-score               : {row.get('amount_zscore', 0):.2f}",
        f"    Txns in past 1h              : {int(row.get('user_txn_velocity_1h', 0))}",
        f"    New city for user            : {'Yes' if row.get('is_new_city', 0) else 'No'}",
        f"    New device for user          : {'Yes' if row.get('is_new_device', 0) else 'No'}",
        f"    Cold-start user              : {'Yes' if row.get('is_cold_start', 0) else 'No'}",
        sep,
        f"  RECOMMENDED ACTION: {TIER_ACTIONS.get(row.get('risk_tier','NORMAL'),'?')}",
        sep2,
    ]

    return "\n".join(lines)


def generate_batch_reports(
    df: pd.DataFrame,
    top_n: int = 10,
    feature_names: Optional[List[str]] = None,
    shap_matrix: Optional[np.ndarray] = None,
    per_feature_recon_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Generate fraud reports for the top-N highest risk transactions."""
    top_df  = df.nlargest(top_n, "final_risk_score").reset_index()
    reports = []

    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        orig_idx = int(row.get("index", rank - 1))
        row_shap  = shap_matrix[orig_idx] if (shap_matrix is not None and orig_idx < len(shap_matrix)) else None
        row_recon = per_feature_recon_df.iloc[orig_idx] if (per_feature_recon_df is not None and orig_idx < len(per_feature_recon_df)) else None

        report = generate_transaction_report(
            row=row,
            feature_names=feature_names or [],
            shap_values=row_shap,
            per_feature_recon_error=row_recon,
        )
        reports.append(f"\n  ── RANK #{rank} ──\n{report}")

    return reports
