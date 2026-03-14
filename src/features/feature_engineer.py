"""
src/features/feature_engineer.py
---------------------------------
Point-in-time correct feature engineering for fraud detection.

All behavioral aggregates (user average, velocity, is_new_city, etc.)
are computed using ONLY data available BEFORE each transaction timestamp.
This is achieved via:
    - Sort by (user_id, timestamp)
    - .shift(1) before any .expanding() aggregate
    - Rolling windows on the time index

This makes the feature set logically equivalent to real-time inference —
the same features could be computed from a feature store in production.

Cold-start handling:
    Users with < MIN_HISTORY_TRANSACTIONS prior transactions receive
    global population priors (computed from the fit window) in place of
    user-specific aggregates. A boolean flag `is_cold_start` is added.

Output columns are documented at the bottom of this file.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
MIN_HISTORY_TRANSACTIONS = 3    # below this → use global priors
LATE_NIGHT_START = 0            # 00:00
LATE_NIGHT_END   = 5            # 05:00 (exclusive)
VELOCITY_WINDOW  = "1h"         # rolling window for velocity features

# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_entropy(series: pd.Series) -> float:
    """Shannon entropy of a categorical series. Returns 0 for single-value."""
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    return float(scipy_entropy(counts))


def _expanding_stat(series: pd.Series, stat: str, shift: bool = True) -> pd.Series:
    """
    Compute expanding statistic with optional shift(1) to exclude current row.
    stat: 'mean', 'std', 'count', 'sum'
    """
    shifted = series.shift(1) if shift else series
    if stat == "mean":
        return shifted.expanding(min_periods=1).mean()
    elif stat == "std":
        return shifted.expanding(min_periods=2).std()
    elif stat == "count":
        return shifted.expanding(min_periods=1).count()
    elif stat == "sum":
        return shifted.expanding(min_periods=1).sum()
    raise ValueError(f"Unknown stat: {stat}")


# ── Global Priors (computed on fit window, applied for cold-start) ─────────────

def compute_global_priors(df: pd.DataFrame) -> dict:
    """
    Compute population-level statistics from the fit window.
    Used as fallback for cold-start users.
    """
    parsed = df[df["parse_success"] == True].copy()
    priors = {
        "amount_mean":   parsed["amount"].median(),
        "amount_std":    parsed["amount"].std(),
        "velocity_mean": 1.0,           # typical ~1 txn per observation window
        "city_rarity":   1 / max(parsed["city"].nunique(), 1),
        "device_rarity": 1 / max(parsed["device"].nunique(), 1),
        "time_since_last_txn": parsed["amount"].median() / 100,  # rough proxy
    }
    return priors


# ── Rarity Encodings (fit on training window only) ────────────────────────────

def compute_rarity_encodings(fit_df: pd.DataFrame) -> dict:
    """
    Compute global frequency-based rarity scores from the fit window.
    These are applied to all transactions (fit + score window).

    Returns a dict of lookup Series:
        city_rarity, device_rarity, device_city_pair_rarity
    """
    parsed = fit_df[fit_df["parse_success"] == True].copy()
    n = len(parsed)

    city_counts  = parsed["city"].value_counts()
    dev_counts   = parsed["device"].value_counts()

    # Pair rarity: (device, city) combination frequency
    pair_counts  = parsed.groupby(["device", "city"]).size()

    # Convert to rarity scores: 1 / frequency (higher = rarer)
    city_rarity  = (1 / city_counts.clip(lower=1)).to_dict()
    dev_rarity   = (1 / dev_counts.clip(lower=1)).to_dict()
    pair_rarity  = (1 / pair_counts.clip(lower=1)).to_dict()

    return {
        "city_rarity":            city_rarity,
        "device_rarity":          dev_rarity,
        "device_city_pair_rarity": pair_rarity,
        "n_fit":                  n,
    }


# ── Main Feature Engineering Function ─────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    global_priors: dict,
    rarity_encodings: dict,
) -> pd.DataFrame:
    """
    Build the full feature matrix from parsed log data.

    Parameters
    ----------
    df               : Parsed DataFrame (output of log_parser.parse_dataframe)
    global_priors    : dict from compute_global_priors() on fit window
    rarity_encodings : dict from compute_rarity_encodings() on fit window

    Returns
    -------
    DataFrame with all engineered features appended.
    Rows with parse_success=False are retained but features are NaN/flagged.
    """
    out = df.copy()

    # Work only on successfully parsed rows for feature computation
    mask = out["parse_success"] == True

    # ── 0. Sort by user and time (required for all expanding operations) ──────
    out = out.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── 1. Boolean convenience flags ─────────────────────────────────────────
    out["is_city_missing"]   = (out["city"]   == "UNKNOWN").astype(int)
    out["is_device_missing"] = (out["device"] == "UNKNOWN").astype(int)
    out["missing_field_count"] = out["is_city_missing"] + out["is_device_missing"]

    # ── 2. Temporal Features ──────────────────────────────────────────────────
    out["hour_of_day"]  = out["timestamp"].dt.hour
    out["day_of_week"]  = out["timestamp"].dt.dayofweek        # 0=Mon
    out["is_weekend"]   = (out["day_of_week"] >= 5).astype(int)
    out["is_late_night"] = (
        (out["hour_of_day"] >= LATE_NIGHT_START) &
        (out["hour_of_day"] <  LATE_NIGHT_END)
    ).astype(int)

    # ── 3. User Behavioral Aggregates (point-in-time correct) ─────────────────
    # Group by user — all operations use shift(1) to exclude current row
    user_groups = out[mask].groupby("user_id")

    # 3a. Amount statistics
    out.loc[mask, "user_txn_count_hist"] = user_groups["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).count()
    )
    out.loc[mask, "user_avg_amount_hist"] = user_groups["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    out.loc[mask, "user_std_amount_hist"] = user_groups["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=2).std()
    )
    out.loc[mask, "user_total_volume_hist"] = user_groups["amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).sum()
    )

    # 3b. Cold-start flag: < MIN_HISTORY_TRANSACTIONS prior transactions
    out["is_cold_start"] = (
        out["user_txn_count_hist"].fillna(0) < MIN_HISTORY_TRANSACTIONS
    ).astype(int)

    # 3c. Apply global priors where cold-start
    cs = out["is_cold_start"] == 1
    out.loc[cs, "user_avg_amount_hist"] = global_priors["amount_mean"]
    out.loc[cs, "user_std_amount_hist"] = global_priors["amount_std"]

    # 3d. Amount Z-score: (current_amount - user_hist_mean) / user_hist_std
    std_safe = out["user_std_amount_hist"].replace(0, np.nan).fillna(global_priors["amount_std"])
    out["amount_zscore"] = (
        (out["amount"] - out["user_avg_amount_hist"]) / std_safe
    ).clip(-10, 10)   # clip extreme values

    # 3e. Amount percentile rank in global distribution (no leakage risk — rank is relative)
    out["amount_global_percentile"] = out["amount"].rank(pct=True)

    # 3f. Round amount flag
    out["is_round_amount"] = (
        (out["amount"] % 100 == 0) & (out["amount"] > 0)
    ).astype(int)

    # ── 4. Time Between Transactions ──────────────────────────────────────────
    out["prev_timestamp"] = out.groupby("user_id")["timestamp"].shift(1)
    out["time_since_last_txn_sec"] = (
        out["timestamp"] - out["prev_timestamp"]
    ).dt.total_seconds()
    # Cold-start / first transaction: fill with global median
    median_gap = out["time_since_last_txn_sec"].median()
    out["time_since_last_txn_sec"] = out["time_since_last_txn_sec"].fillna(
        median_gap if pd.notna(median_gap) else 3600.0
    )

    # ── 5. Transaction Velocity (rolling 1-hour window) ───────────────────────
    # For each user, count transactions in the 60 minutes BEFORE current row
    def _rolling_velocity(group: pd.DataFrame) -> pd.Series:
        """Count transactions in past 1h (excluding current row)."""
        g = group.set_index("timestamp").sort_index()
        # shift by 1 nanosecond so current row is excluded
        counts = (
            g["amount"]
            .shift(1)                              # exclude self
            .fillna(0)
            .rolling("1h", closed="left")
            .count()
        )
        counts.index = group.index
        return counts

    velocity_series = []
    for uid, grp in out[mask].groupby("user_id"):
        v = _rolling_velocity(grp[["timestamp", "amount"]].copy())
        velocity_series.append(v)

    if velocity_series:
        all_velocity = pd.concat(velocity_series).reindex(out[mask].index)
        out.loc[mask, "user_txn_velocity_1h"] = all_velocity.values
    out["user_txn_velocity_1h"] = out["user_txn_velocity_1h"].fillna(0)

    # Amount velocity (sum of amounts in past 1h)
    def _rolling_amount_velocity(group: pd.DataFrame) -> pd.Series:
        g = group.set_index("timestamp").sort_index()
        sums = (
            g["amount"]
            .shift(1)
            .fillna(0)
            .rolling("1h", closed="left")
            .sum()
        )
        sums.index = group.index
        return sums

    amt_velocity_series = []
    for uid, grp in out[mask].groupby("user_id"):
        v = _rolling_amount_velocity(grp[["timestamp", "amount"]].copy())
        amt_velocity_series.append(v)

    if amt_velocity_series:
        all_amt_vel = pd.concat(amt_velocity_series).reindex(out[mask].index)
        out.loc[mask, "user_amount_velocity_1h"] = all_amt_vel.values
    out["user_amount_velocity_1h"] = out["user_amount_velocity_1h"].fillna(0)

    # ── 6. Location Features (point-in-time correct) ──────────────────────────

    def _is_new_city(group: pd.DataFrame) -> pd.Series:
        """True if this is the first time user transacts from this city."""
        seen = set()
        result = []
        for city in group["city"]:
            result.append(int(city not in seen and city != "UNKNOWN"))
            seen.add(city)
        return pd.Series(result, index=group.index)

    def _city_frequency(group: pd.DataFrame) -> pd.Series:
        """How many prior transactions from this city for this user."""
        freq = {}
        result = []
        for city in group["city"]:
            result.append(freq.get(city, 0))
            freq[city] = freq.get(city, 0) + 1
        return pd.Series(result, index=group.index)

    new_city_series, city_freq_series = [], []
    for uid, grp in out[mask].groupby("user_id"):
        new_city_series.append(_is_new_city(grp[["city"]]))
        city_freq_series.append(_city_frequency(grp[["city"]]))

    if new_city_series:
        out.loc[mask, "is_new_city"]       = pd.concat(new_city_series).reindex(out[mask].index).values
        out.loc[mask, "user_city_freq"]    = pd.concat(city_freq_series).reindex(out[mask].index).values

    out["is_new_city"]    = out["is_new_city"].fillna(1)      # cold-start = new city
    out["user_city_freq"] = out["user_city_freq"].fillna(0)

    # Global city rarity (from rarity encodings fit on fit window)
    city_rar = rarity_encodings["city_rarity"]
    default_city_rarity = 1.0 / rarity_encodings["n_fit"]
    out["city_global_rarity"] = out["city"].map(city_rar).fillna(default_city_rarity)

    # ── 7. Device Features ────────────────────────────────────────────────────

    def _is_new_device(group: pd.DataFrame) -> pd.Series:
        seen = set()
        result = []
        for dev in group["device"]:
            result.append(int(dev not in seen and dev != "UNKNOWN"))
            seen.add(dev)
        return pd.Series(result, index=group.index)

    new_dev_series = []
    for uid, grp in out[mask].groupby("user_id"):
        new_dev_series.append(_is_new_device(grp[["device"]]))

    if new_dev_series:
        out.loc[mask, "is_new_device"] = pd.concat(new_dev_series).reindex(out[mask].index).values
    out["is_new_device"] = out["is_new_device"].fillna(1)

    dev_rar = rarity_encodings["device_rarity"]
    out["device_global_rarity"] = out["device"].map(dev_rar).fillna(default_city_rarity)

    # ── 8. Combination (Interaction) Features ────────────────────────────────
    pair_rar = rarity_encodings["device_city_pair_rarity"]
    out["device_city_pair_rarity"] = out.apply(
        lambda r: pair_rar.get((r["device"], r["city"]), default_city_rarity),
        axis=1,
    )

    # Location entropy per user (how spread-out are user's cities?)
    location_entropy_map = (
        out[mask]
        .groupby("user_id")["city"]
        .apply(_safe_entropy)
        .to_dict()
    )
    out["user_location_entropy"] = out["user_id"].map(location_entropy_map).fillna(0.0)

    # ── 9. RFM Features ───────────────────────────────────────────────────────
    first_txn = out[mask].groupby("user_id")["timestamp"].min().to_dict()
    out["user_recency_days"] = out.apply(
        lambda r: (r["timestamp"] - first_txn.get(r["user_id"], r["timestamp"])).days
        if pd.notna(r["timestamp"]) and r["user_id"] in first_txn else 0,
        axis=1,
    )
    out["user_frequency_hist"]  = out["user_txn_count_hist"].fillna(1)
    out["user_monetary_hist"]   = out["user_total_volume_hist"].fillna(out["amount"])

    # ── 10. Transaction-type encoding (ordinal proxy) ─────────────────────────
    TXN_ORDER = {"deposit": 0, "purchase": 1, "transfer": 2, "withdrawal": 3, "UNKNOWN": 4}
    out["txn_type_encoded"] = out["txn_type"].map(TXN_ORDER).fillna(4).astype(int)

    # ── 11. Late-night × user-ratio (user-context-aware rule feature) ─────────
    def _late_night_ratio(group: pd.DataFrame) -> pd.Series:
        """Fraction of this user's PAST transactions that were late-night."""
        result = []
        late_count, total_count = 0, 0
        for _, row in group.iterrows():
            ratio = late_count / total_count if total_count > 0 else 0.0
            result.append(ratio)
            if row.get("is_late_night", 0):
                late_count += 1
            total_count += 1
        return pd.Series(result, index=group.index)

    ln_ratio_series = []
    for uid, grp in out[mask].groupby("user_id"):
        ln_ratio_series.append(
            _late_night_ratio(grp[["is_late_night"]].copy())
        )
    if ln_ratio_series:
        out.loc[mask, "user_late_night_ratio"] = (
            pd.concat(ln_ratio_series).reindex(out[mask].index).values
        )
    out["user_late_night_ratio"] = out["user_late_night_ratio"].fillna(0.0)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    out.drop(columns=["prev_timestamp"], inplace=True, errors="ignore")

    return out


# ── Feature Column Reference ──────────────────────────────────────────────────

# Continuous features fed to Isolation Forest + Autoencoder continuous stream
CONTINUOUS_FEATURES = [
    "amount",
    "amount_zscore",
    "amount_global_percentile",
    "is_round_amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_late_night",
    "time_since_last_txn_sec",
    "user_txn_velocity_1h",
    "user_amount_velocity_1h",
    "is_new_city",
    "user_city_freq",
    "city_global_rarity",
    "is_new_device",
    "device_global_rarity",
    "device_city_pair_rarity",
    "missing_field_count",
    "user_location_entropy",
    "user_recency_days",
    "user_frequency_hist",
    "user_monetary_hist",
    "user_late_night_ratio",
    "is_cold_start",
    "txn_type_encoded",
]

# Categorical features for Autoencoder embedding layers
CATEGORICAL_FEATURES = {
    "user_id":  None,   # vocab size determined at runtime
    "city":     None,
    "device":   None,
    "txn_type": None,
}
