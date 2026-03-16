import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional, List

TIER_COLORS = {
    "TIER_1": "#e74c3c",
    "TIER_2": "#e67e22",
    "TIER_3": "#f1c40f",
    "NORMAL": "#2ecc71",
}
TIER_LABELS = {
    "TIER_1": "TIER 1 — Auto Block",
    "TIER_2": "TIER 2 — Step-up Auth",
    "TIER_3": "TIER 3 — Monitor",
    "NORMAL": "Normal",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size": 10,
})


def plot_batch_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary statistics DataFrame for the batch.
    Used as a printed table in the notebook and in Streamlit st.dataframe().
    """
    rows = []
    for tier in ["TIER_1", "TIER_2", "TIER_3", "NORMAL"]:
        sub = df[df["risk_tier"] == tier]
        n   = len(sub)
        rows.append({
            "Tier":            TIER_LABELS.get(tier, tier),
            "Count":           n,
            "% of batch":      f"{100*n/max(len(df), 1):.1f}%",
            "Avg risk score":  f"{sub['final_risk_score'].mean():.3f}" if n else "—",
            "Avg amount (£)":  f"{sub['amount'].mean():.0f}" if n else "—",
            "Rules fired":     f"{(sub['rule_flag_count'] > 0).sum()}" if n and "rule_flag_count" in sub else "—",
            "Models agree ≥2": f"{(sub['model_agreement'] >= 2).sum()}" if n and "model_agreement" in sub else "—",
        })
    return pd.DataFrame(rows)


def plot_top50_anomalies(df: pd.DataFrame) -> plt.Figure:
    """
    Visual summary of the top-50 highest-risk transactions.
    Scatter plot (risk score vs amount) coloured by tier,
    plus a ranked bar chart of risk scores.
    """
    top50 = df.nlargest(50, "final_risk_score").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for tier in ["TIER_1", "TIER_2", "TIER_3", "NORMAL"]:
        sub = top50[top50["risk_tier"] == tier]
        if len(sub):
            ax1.scatter(sub["amount"], sub["final_risk_score"],
                        c=TIER_COLORS[tier], label=TIER_LABELS[tier],
                        s=80, edgecolors="white", linewidth=0.5, zorder=3)

    ax1.set_xlabel("Transaction Amount (£)")
    ax1.set_ylabel("Risk Score")
    ax1.set_title("Top 50 — Risk Score vs Amount", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    for _, r in top50[top50["risk_tier"] == "TIER_1"].iterrows():
        ax1.annotate(
            str(r.get("user_id", ""))[-4:],
            (r["amount"], r["final_risk_score"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )

    bar_colors = [TIER_COLORS.get(t, "#95a5a6") for t in top50["risk_tier"]]
    ax2.bar(range(len(top50)), top50["final_risk_score"],
            color=bar_colors, edgecolor="white", width=0.8)
    ax2.axhline(0.75, color="darkred", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.axhline(0.50, color="orange",  linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Rank (1 = highest risk)")
    ax2.set_ylabel("Risk Score")
    ax2.set_title("Top 50 — Ranked by Risk Score", fontweight="bold")
    ax2.set_xlim(-1, 50)

    patches = [mpatches.Patch(color=v, label=TIER_LABELS[k])
               for k, v in TIER_COLORS.items()]
    ax2.legend(handles=patches, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Top 50 Highest-Risk Transactions  |  "
        f"Tier 1: {(top50['risk_tier'] == 'TIER_1').sum()}  |  "
        f"Tier 2: {(top50['risk_tier'] == 'TIER_2').sum()}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig