import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

TIER_COLORS = {
    "TIER_1": "#e74c3c",
    "TIER_2": "#e67e22",
    "TIER_3": "#f1c40f",
    "NORMAL": "#2ecc71",
}

SCORE_COLS = {
    "rule_score":       "Rule Engine",
    "if_score":         "Isolation Forest",
    "ae_score":         "Autoencoder",
    "final_risk_score": "Ensemble",
}

def eval_score_distributions(df: pd.DataFrame) -> Dict:
    """
    Bimodality Coefficient > 0.555 indicates meaningful structure

    Separation ratio = p95 / median. High ratio means the model produces
    clearly elevated scores for the transactions it considers anomalous.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes      = axes.flatten()
    results   = {}

    for i, (col, label) in enumerate(SCORE_COLS.items()):
        if col not in df.columns:
            continue
        scores = df[col].dropna()
        ax     = axes[i]

        ax.hist(scores, bins=60, color="#2c7bb6", edgecolor="white",
                alpha=0.85, density=True)

        for pct, color, lw in [(90, "orange", 1.5), (95, "red", 2.0), (99, "darkred", 2.0)]:
            v = scores.quantile(pct / 100)
            ax.axvline(v, color=color, linestyle="--", linewidth=lw,
                       label=f"p{pct}: {v:.3f}")

        try:
            kde_x = np.linspace(scores.min(), scores.max(), 300)
            ax.plot(kde_x, stats.gaussian_kde(scores)(kde_x),
                    "r-", linewidth=2, label="KDE")
        except Exception:
            pass

        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

        n    = len(scores)
        skew = scores.skew()
        kurt = scores.kurtosis()
        bc   = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))

        p95    = scores.quantile(0.95)
        median = scores.median()
        sep    = round(p95 / median, 2) if median > 0 else float("inf")

        results[col] = {
            "mean":     round(scores.mean(), 4),
            "std":      round(scores.std(),  4),
            "p95":      round(p95, 4),
            "p99":      round(scores.quantile(0.99), 4),
            "skewness": round(skew, 3),
            "kurtosis": round(kurt, 3),
            "bimodality_coefficient": round(bc, 3),
            "is_bimodal": bc > 0.555,
            "separation_ratio": sep,
        }

        ax.text(0.98, 0.97,
                f"BC={bc:.3f}\n{'✅ Bimodal' if bc > 0.5 else '⚠ Unimodal'}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.suptitle("Score Distribution Analysis\n"
                 "BC > 0.555 = model found meaningful structure",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n[Score Distribution Analysis]")
    print(f"  {'Model':<22} {'BC':>7}  {'Structure':>12}  "
          f"{'p95':>8}  {'Sep.Ratio':>11}  {'Skew':>7}")
    print("  " + "-" * 74)
    for col, s in results.items():
        label = SCORE_COLS.get(col, col)
        flag  = "✅ Bimodal" if s["is_bimodal"] else "Unimodal"
        print(f"  {label:<22} {s['bimodality_coefficient']:>7.3f}  "
              f"{flag:>12}  {s['p95']:>8.4f}  "
              f"{s['separation_ratio']:>9.2f}x  {s['skewness']:>7.2f}")
    return results


def eval_precision_at_k(
    df: pd.DataFrame,
    k_values: List[int] = None,
    agreement_threshold: int = 2,
) -> pd.DataFrame:
    """
    proxy_precision@K = (transactions with ≥ agreement_threshold models) / K
    """
    if k_values is None:
        k_values = [10, 25, 50, 100, 200]

    score_col = "final_risk_score"
    agree_col = "model_agreement"

    if score_col not in df.columns or agree_col not in df.columns:
        print("[Precision@K] Missing required columns.")
        return pd.DataFrame()

    rows = []
    for k in k_values:
        top_k     = df.nlargest(k, score_col)
        proxy_pos = int((top_k[agree_col] >= agreement_threshold).sum())
        high_risk = int(top_k["risk_tier"].isin(["TIER_1", "TIER_2"]).sum()) \
            if "risk_tier" in top_k.columns else 0
        rows.append({
            "K":                  k,
            "proxy_positives":    proxy_pos,
            "proxy_precision@K":  round(proxy_pos / k, 3),
            "high_risk_in_top_K": high_risk,
            "high_risk_rate":     round(high_risk / k, 3),
            "mean_score":         round(top_k[score_col].mean(), 4),
        })
    result_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(result_df["K"], result_df["proxy_precision@K"],
                 "b-o", linewidth=2, markersize=7,
                 label=f"Proxy Precision@K (≥{agreement_threshold} models)")
    axes[0].plot(result_df["K"], result_df["high_risk_rate"],
                 "r--s", linewidth=2, markersize=7,
                 label="High-Risk Rate (T1/T2)")
    axes[0].axhline(0.5, color="orange", linestyle=":", linewidth=1.5)
    axes[0].set_title("Operational Precision @ K\n"
                      "(proxy — multi-model agreement as pseudo-label)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Fraction")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    if "risk_tier" in df.columns:
        top100      = df.nlargest(100, score_col)
        tier_counts = top100["risk_tier"].value_counts().reindex(
            ["TIER_1", "TIER_2", "TIER_3", "NORMAL"], fill_value=0
        )
        axes[1].bar(tier_counts.index, tier_counts.values,
                    color=[TIER_COLORS[t] for t in tier_counts.index],
                    edgecolor="white")
        axes[1].set_title("Risk Tier in Top-100\nIdeal: TIER_1/2 dominate")
        axes[1].set_ylabel("Count")
        for bar, val in zip(axes[1].patches, tier_counts.values):
            if val > 0:
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.3,
                             str(int(val)), ha="center", fontsize=9)

    plt.suptitle("Precision @ K — Review Queue Efficiency",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n[Precision @ K]  (proxy_precision = lower bound — real fraud in queue unobserved)")
    print(result_df.to_string(index=False))
    return result_df


def eval_cross_model_agreement(
    df: pd.DataFrame,
    top_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise Jaccard overlap and Spearman rank correlation between detectors.
    """
    n_top   = max(1, int(len(df) * top_pct))
    models  = {
        "Rules":       df.nlargest(n_top, "rule_score").index,
        "IsolForest":  df.nlargest(n_top, "if_score").index,
        "Autoencoder": df.nlargest(n_top, "ae_score").index,
    }
    model_names = list(models.keys())

    # Jaccard
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    for m1 in model_names:
        for m2 in model_names:
            i = len(set(models[m1]) & set(models[m2]))
            u = len(set(models[m1]) | set(models[m2]))
            matrix.loc[m1, m2] = i / u if u > 0 else 0.0

    # Spearman rank correlation (correct for bounded, non-normal scores)
    available = [c for c in SCORE_COLS if c in df.columns]
    spearman  = pd.DataFrame(index=available, columns=available, dtype=float)
    for c1 in available:
        for c2 in available:
            r, _ = spearmanr(df[c1].fillna(0), df[c2].fillna(0))
            spearman.loc[c1, c2] = round(r, 3)
    spearman = spearman.rename(index=SCORE_COLS, columns=SCORE_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(matrix.astype(float), annot=True, fmt=".2%",
                cmap="YlOrRd", ax=axes[0], vmin=0, vmax=1,
                linewidths=1, linecolor="white")
    axes[0].set_title(f"Pairwise Jaccard Overlap\n(top {top_pct*100:.0f}% flags)")
    for t in axes[0].texts:
        v = float(t.get_text().strip("%")) / 100
        t.set_color("darkred" if v > 0.5 else "darkgreen" if v < 0.3 else "darkorange")

    sns.heatmap(spearman.astype(float), annot=True, fmt=".3f",
                cmap="RdYlGn_r", center=0, ax=axes[1],
                vmin=-1, vmax=1, linewidths=0.5, square=True)
    axes[1].set_title("Spearman Rank Correlation\n"
                      "(appropriate for bounded scores — not Pearson)")

    plt.suptitle("Cross-Model Agreement\n"
                 "Target: Jaccard < 0.5, Spearman < 0.5",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"\n[Cross-Model Agreement]")
    print(f"  Jaccard overlap (top {top_pct*100:.0f}%):")
    print(matrix.round(3).to_string())
    print(f"\n  Spearman rank correlations:")
    print(spearman.to_string())

    try:
        pairs = [
            ("Isolation Forest", "Autoencoder",      "IF ↔ AE"),
            ("Isolation Forest", "Rule Engine",       "IF ↔ Rules"),
            ("Autoencoder",      "Rule Engine",       "AE ↔ Rules"),
        ]
        print()
        for r, c, label in pairs:
            v    = float(spearman.loc[r, c])
            flag = "✅ Independent" if abs(v) < 0.5 else "⚠ Correlated"
            print(f"  {label:<14}: Spearman={v:+.3f}  {flag}")
    except Exception:
        pass

    return matrix


def run_full_evaluation(
    df: pd.DataFrame,
    if_detector,
    ae_detector,
):

    all_results = {}

    print("\n" + "=" * 65)
    print("  FRAUD DETECTION EVALUATION SUITE")
    print("=" * 65)

    print("\n── 1/3 Score Distribution Analysis ──────────────────────────")
    all_results["score_distributions"] = eval_score_distributions(df)

    
    print("\n── 2/3 Precision @ K ────────────────────────────────────────")
    all_results["precision_at_k"] = eval_precision_at_k(
        df, k_values=[10, 25, 50, 100, 200]
    )

    print("\n── 3/3 Cross-Model Agreement ────────────────────────────────")
    all_results["model_agreement"] = eval_cross_model_agreement(df, top_pct=0.05)

    # Summary
    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)

    for col, s in all_results.get("score_distributions", {}).items():
        label = SCORE_COLS.get(col, col)
        print(f"  {label:<22} BC={s['bimodality_coefficient']:.3f}  "
              f"{'✅ Bimodal' if s['is_bimodal'] else '⚠ Unimodal'}")

    inj = all_results.get("injection_k")
    if inj is not None and len(inj) > 0:
        r100 = inj[inj["K"] == 100]
        if len(r100) > 0:
            r = r100.iloc[0]
            efficiency = r["recall@K"] / r["max_achievable_recall@K"] if r["max_achievable_recall@K"] > 0 else 0
            print(f"\n  Recall@100 (corrected)   : {r['recall@K']:.3f} "
                  f"/ max achievable {r['max_achievable_recall@K']:.3f} "
                  f"({efficiency*100:.0f}% of maximum)")
            print(f"  NDCG@100                 : {r['NDCG@K']:.3f}")

    print("=" * 65)
    return all_results
