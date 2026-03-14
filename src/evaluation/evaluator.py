"""
src/evaluation/evaluator.py
-----------------------------
Evaluation strategy for unsupervised anomaly detection.

Since no ground-truth labels exist for real transactions, evaluation uses:

1. Synthetic Injection Test
   Known fraud patterns were injected at data generation time.
   We verify what fraction of injected anomalies appear in the top-K flagged.
   This tests recall on known fraud types without requiring real labels.

2. Precision-at-K
   Top-K transactions by final_risk_score are reviewed manually (or using
   injected labels as proxy). Reports precision @ 20, 50, 100.

3. Cross-Model Agreement Matrix
   What fraction of each model's top-5% flags overlap with other models?
   High overlap = redundancy. Low overlap = independence (good for ensemble).

4. Score Distribution Analysis
   Bimodal distribution = model found structure.
   Uniform distribution = model found noise.

5. Anomaly Rate Stability
   Conceptual check: is the flagged rate consistent with industry baselines
   (0.5-3% for fintech fraud)?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional


# ── 1. Synthetic Injection Test ───────────────────────────────────────────────

def synthetic_injection_test(
    df: pd.DataFrame,
    top_k: int = 100,
    label_col: str = "is_injected_anomaly",
    score_col: str = "final_risk_score",
) -> dict:
    """
    Evaluate recall of injected anomalies in the top-K scored transactions.

    Parameters
    ----------
    df        : Scored DataFrame with injected anomaly labels
    top_k     : Number of top-scored transactions to inspect
    label_col : Column indicating injected anomalies (1 = anomaly)
    score_col : Anomaly score column

    Returns
    -------
    dict with recall, precision, f1, and per-type breakdown
    """
    if label_col not in df.columns:
        print(f"[SyntheticTest] Column '{label_col}' not found — skipping.")
        return {}

    top_k_df    = df.nlargest(top_k, score_col)
    total_injected = df[label_col].sum()
    recovered      = top_k_df[label_col].sum()

    recall    = recovered / total_injected if total_injected > 0 else 0
    precision = recovered / top_k          if top_k > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("\n" + "="*55)
    print("  SYNTHETIC INJECTION TEST")
    print("="*55)
    print(f"  Total injected anomalies : {int(total_injected)}")
    print(f"  Recovered in top-{top_k}    : {int(recovered)}")
    print(f"  Recall                   : {recall:.3f}")
    print(f"  Precision@{top_k}          : {precision:.3f}")
    print(f"  F1                       : {f1:.3f}")

    # Per anomaly type breakdown
    if "anomaly_type" in df.columns:
        print("\n  Recall by anomaly type:")
        for atype in df[df[label_col]==1]["anomaly_type"].unique():
            total_type   = (df["anomaly_type"] == atype).sum()
            recovered_type = (
                (top_k_df["anomaly_type"] == atype) & (top_k_df[label_col] == 1)
            ).sum()
            print(f"    {atype:<25} {recovered_type}/{total_type} "
                  f"({100*recovered_type/max(total_type,1):.0f}%)")
    print("="*55)

    return {"recall": recall, "precision": precision, "f1": f1, "recovered": recovered}


# ── 2. Precision-at-K ─────────────────────────────────────────────────────────

def precision_at_k(
    df: pd.DataFrame,
    k_values: list = [20, 50, 100],
    label_col: str = "is_injected_anomaly",
    score_col: str = "final_risk_score",
) -> pd.DataFrame:
    """
    Compute precision at multiple K values.
    Uses injected labels as proxy for real analyst decisions.
    """
    if label_col not in df.columns:
        print("[PrecisionAtK] Label column not found — skipping.")
        return pd.DataFrame()

    results = []
    for k in k_values:
        top_k   = df.nlargest(k, score_col)
        prec    = top_k[label_col].sum() / k
        results.append({"K": k, "precision": prec,
                        "true_positives": int(top_k[label_col].sum())})

    result_df = pd.DataFrame(results)
    print("\n[Precision@K]")
    print(result_df.to_string(index=False))
    return result_df


# ── 3. Cross-Model Agreement Matrix ───────────────────────────────────────────

def cross_model_agreement(
    df: pd.DataFrame,
    top_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Compute pairwise overlap between models' top-X% flags.
    Low overlap = independent models (good ensemble).
    """
    n_top  = max(1, int(len(df) * top_pct))
    models = {
        "RuleEngine": df.nlargest(n_top, "rule_score").index,
        "IsolForest": df.nlargest(n_top, "if_score").index,
        "Autoencoder": df.nlargest(n_top, "ae_score").index,
    }

    model_names = list(models.keys())
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)

    for m1 in model_names:
        for m2 in model_names:
            overlap = len(set(models[m1]) & set(models[m2]))
            matrix.loc[m1, m2] = overlap / n_top

    print(f"\n[Cross-Model Agreement] Pairwise overlap in top-{top_pct*100:.0f}%:")
    print(matrix.round(3).to_string())
    return matrix


# ── 4. Score Distribution Analysis ───────────────────────────────────────────

def plot_score_distributions(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot histograms of all three model scores and the final ensemble score.
    A bimodal distribution indicates the model found meaningful structure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Anomaly Score Distributions", fontsize=14, fontweight="bold")

    score_pairs = [
        ("rule_score",       "Rule Engine Score",      axes[0, 0]),
        ("if_score",         "Isolation Forest Score", axes[0, 1]),
        ("ae_score",         "Autoencoder Score",      axes[1, 0]),
        ("final_risk_score", "Ensemble Risk Score",    axes[1, 1]),
    ]

    for col, title, ax in score_pairs:
        if col not in df.columns:
            continue
        scores = df[col].dropna()
        ax.hist(scores, bins=50, color="#2c7bb6", edgecolor="white", alpha=0.8)
        ax.axvline(scores.quantile(0.95), color="red", linestyle="--",
                   label=f"95th pct: {scores.quantile(0.95):.2f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 5. Full Evaluation Report ─────────────────────────────────────────────────

def run_full_evaluation(df: pd.DataFrame, save_dir: str = "outputs/") -> dict:
    """Run all evaluation steps and return consolidated results dict."""
    results = {}

    results["synthetic_injection"] = synthetic_injection_test(df, top_k=100)
    results["precision_at_k"]      = precision_at_k(df)
    results["model_agreement"]     = cross_model_agreement(df)

    # Anomaly rate check
    for col, label in [("if_is_anomaly", "IF"), ("ae_is_anomaly", "AE"),
                        ("rule_flag_count", "Rules")]:
        if col in df.columns:
            if col == "rule_flag_count":
                rate = (df[col] > 0).mean()
            else:
                rate = df[col].mean()
            print(f"[AnomalyRate] {label}: {rate*100:.1f}% "
                  f"(industry benchmark: 0.5–3%)")

    return results
