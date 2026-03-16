"""
Generates features across 15 groups using a config-driven blast approach.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


ROLLING_WINDOWS = ["1D", "7D", "15D", "30D", "60D"]
RECENCY_WINDOWS = ["5min", "30min", "1h", "6h", "12h"]
AMOUNT_AGGS = ["mean", "std", "max", "min", "sum", "median"]
COUNT_AGGS  = ["count"]
ENTITY_GROUPS = [
    ["user_id"],
    ["user_id", "txn_type"],
    ["user_id", "device"],
]

FINGERPRINT_GROUPS = [
    ["user_id", "device"],
    ["user_id", "city"],
    ["user_id", "txn_type"],
    ["user_id", "currency"],
    ["user_id", "device", "city"],
    ["user_id", "txn_type", "city"],
    ["user_id", "txn_type", "device"],
    ["user_id", "currency", "city"],
]

SEQUENCE_COLS = ["amount", "txn_type", "city", "device", "currency"]

RARITY_COLS = ["city", "device", "txn_type", "currency"]

COLD_START_MIN = 3


#UTILITIES

def _safe_div(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((b != 0) & (~np.isnan(b)), a / b, fill)
    return pd.Series(result, index=a.index if hasattr(a, "index") else None)


def _shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    return float(scipy_entropy(counts)) if len(counts) > 1 else 0.0


def _n_distinct_cumcount(df_group: pd.DataFrame, col: str) -> pd.Series:
    """Point-in-time count of distinct values seen BEFORE each row."""
    result, seen = [], set()
    for val in df_group[col]:
        result.append(len(seen))
        seen.add(val)
    return pd.Series(result, index=df_group.index)


def _entity_name(cols: List[str]) -> str:
    """Convert entity column list to a compact feature name prefix."""
    mapping = {
        "user_id":  "user",
        "txn_type": "txntype",
        "city":     "city",
        "device":   "device",
        "currency": "currency",
    }
    return "_".join(mapping.get(c, c) for c in cols)


class FraudFeatureEngine:
    def __init__(self, fit_df: pd.DataFrame):
        self._fit_df        = fit_df[fit_df["parse_success"] == True].copy()
        self._global_stats  = {}    # global amount priors
        self._rarity_maps   = {}    # col → {value: rarity_score}
        self._city_currency = {}    # city → expected currency
        self._feature_registry: Dict[str, List[str]] = defaultdict(list)
        self._fit()

    def _fit(self):
        df = self._fit_df
        n  = len(df)

        # Global amount statistics
        log_amt = np.log1p(df["amount"].dropna())
        self._global_stats = {
            "amount_mean":     df["amount"].median(),
            "amount_std":      df["amount"].std(),
            "amount_log_mean": log_amt.mean(),
            "amount_log_std":  log_amt.std(),
            "amount_p25":      df["amount"].quantile(0.25),
            "amount_p50":      df["amount"].quantile(0.50),
            "amount_p75":      df["amount"].quantile(0.75),
            "amount_p90":      df["amount"].quantile(0.90),
            "amount_p95":      df["amount"].quantile(0.95),
            "amount_p99":      df["amount"].quantile(0.99),
            "n_fit":           n,
            # per-txn_type stats
            "txntype_stats": {
                t: {"mean": g["amount"].mean(), "std": g["amount"].std() or df["amount"].std()}
                for t, g in df.groupby("txn_type")
            },
        }

        # Rarity maps: 1 - normalized_frequency (higher = rarer)
        for col in RARITY_COLS:
            freq = df[col].value_counts(normalize=True)
            self._rarity_maps[col] = (1 - freq).to_dict()

        # Pair rarity maps
        for c1, c2 in [("device","city"), ("currency","city"),
                       ("txn_type","city"), ("txn_type","device")]:
            pair_freq = df.groupby([c1,c2]).size() / n
            self._rarity_maps[f"{c1}_{c2}"] = (1 - pair_freq).to_dict()

        # City → expected currency (dominant)
        self._city_currency = (
            df[df["city"] != "UNKNOWN"]
            .groupby("city")["currency"]
            .apply(lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN")
            .to_dict()
        )

        # City transition rarity (from_city → to_city)
        df_s = df.sort_values(["user_id","timestamp"])
        df_s["_prev_city"] = df_s.groupby("user_id")["city"].shift(1)
        trans = df_s.dropna(subset=["_prev_city"]).groupby(["_prev_city","city"]).size()
        trans_freq = trans / trans.sum()
        self._rarity_maps["city_transition"] = (1 - trans_freq).to_dict()

        print(f"[FraudFeatureEngine] Fit on {n} rows. "
              f"Global amount median: {self._global_stats['amount_mean']:.2f}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features on the input DataFrame.
        Input df is sorted by (user_id, timestamp) internally.
        All rolling windows use closed='left' to exclude the current transaction.

        Returns full DataFrame with ~540 new feature columns appended.
        """
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["amount"]    = pd.to_numeric(out["amount"], errors="coerce")
        out = out.sort_values(["user_id","timestamp"]).reset_index(drop=True)

        mask = out["parse_success"] == True   # only scored on parsed rows

        print("[FraudFeatureEngine] Computing features...")
        out = self._grp_A_base(out, mask)          ; print("  A. Base ✓")
        out = self._grp_B_temporal(out, mask)       ; print("  B. Temporal ✓")
        out = self._grp_C_rolling_amount(out, mask) ; print("  C. Rolling Amount Blast ✓")
        out = self._grp_D_rolling_count(out, mask)  ; print("  D. Rolling Count Blast ✓")
        out = self._grp_E_deviation(out, mask)      ; print("  E. Deviation / Ratio ✓")
        out = self._grp_F_fingerprint(out, mask)    ; print("  F. Behavioral Fingerprint ✓")
        out = self._grp_G_geographic(out, mask)     ; print("  G. Geographic ✓")
        out = self._grp_H_device(out, mask)         ; print("  H. Device ✓")
        out = self._grp_I_sequence(out, mask)       ; print("  J. Sequence / Lag ✓")
        out = self._grp_J_interaction(out, mask)    ; print("  K. Interaction ✓")
        out = self._grp_K_missingness(out, mask)    ; print("  L. Missingness ✓")
        out = self._grp_L_burst(out, mask)           ; print("  N. Burst Detection ✓")

        total = sum(len(v) for v in self._feature_registry.values())
        print(f"\n[FraudFeatureEngine] Total features generated: {total}")
        return out


    def _rolling(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        window: str,
        agg_col: str,
        agg: str,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=df.index, dtype=float)

        for key, grp in df.groupby(group_cols):
            g = grp.set_index("timestamp").sort_index()
            col_data = g[agg_col]

            rolled = col_data.rolling(window, closed="left")
            if agg == "mean":
                vals = rolled.mean()
            elif agg == "std":
                vals = rolled.std()
            elif agg == "max":
                vals = rolled.max()
            elif agg == "min":
                vals = rolled.min()
            elif agg == "sum":
                vals = rolled.sum()
            elif agg == "median":
                vals = rolled.median()
            elif agg == "count":
                vals = rolled.count()
            else:
                raise ValueError(f"Unknown aggregation: {agg}")

            result.loc[grp.index] = vals.values

        return result

    def _fill_cold_start(self, series: pd.Series, fill_value: float) -> pd.Series:
        """Fill NaN (cold-start users with no history) with global prior."""
        return series.fillna(fill_value)

    def _grp_A_base(self, df, mask):
        out = df.copy()

        # Log amount
        out["log_amount"] = np.log1p(out["amount"].clip(lower=0))

        # Global percentile rank
        out["amount_global_pct_rank"] = out["amount"].rank(pct=True)

        # Amount vs global thresholds (from fit window)
        for pct_name, pct_val in [("p75","amount_p75"),("p90","amount_p90"),
                                   ("p95","amount_p95"),("p99","amount_p99")]:
            threshold = self._global_stats[pct_val]
            col = f"amount_above_{pct_name}"
            out[col] = (out["amount"] > threshold).astype(int)
            self._feature_registry["A_base"].append(col)

        # Round amount flags
        for denom, name in [(100,"round_100"), (1000,"round_1000"), (500,"round_500")]:
            col = f"is_{name}"
            out[col] = ((out["amount"] % denom == 0) & (out["amount"] > 0)).astype(int)
            self._feature_registry["A_base"].append(col)

        # Transaction type ordinal
        TXN_ORD = {"deposit":0,"purchase":1,"transfer":2,"withdrawal":3,"UNKNOWN":4}
        out["txn_type_ordinal"] = out["txn_type"].map(TXN_ORD).fillna(4)

        # Cold-start flag (users with < COLD_START_MIN prior txns)
        # Will be set properly in Group C after expanding count is computed
        out["is_cold_start"] = 0

        for col in ["log_amount","amount_global_pct_rank","txn_type_ordinal"]:
            self._feature_registry["A_base"].append(col)

        return out

    def _grp_B_temporal(self, df, mask):
        out = df.copy()
        ts  = out["timestamp"]

        # Basic components
        cols = {}
        cols["hour_of_day"]    = ts.dt.hour
        cols["day_of_week"]    = ts.dt.dayofweek
        cols["day_of_month"]   = ts.dt.day
        cols["month"]          = ts.dt.month
        cols["is_weekend"]     = (ts.dt.dayofweek >= 5).astype(int)
        cols["is_month_end"]   = (ts.dt.day >= 28).astype(int)
        cols["is_month_start"] = (ts.dt.day <= 3).astype(int)

        # Risk-window flags
        hour = ts.dt.hour
        cols["is_late_night"]     = hour.isin(range(0,5)).astype(int)
        cols["is_business_hours"] = hour.between(9,17).astype(int)
        cols["is_evening"]        = hour.between(18,23).astype(int)
        cols["is_early_morning"]  = hour.between(5,8).astype(int)

        # Cyclical encoding — avoids discontinuity at 23→0 and Mon→Sun
        cols["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        cols["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        cols["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        cols["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        cols["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        cols["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)

        for col, val in cols.items():
            out[col] = val
            self._feature_registry["B_temporal"].append(col)

        # Time since last transaction (point-in-time: shift(1))
        out["prev_ts"] = out.groupby("user_id")["timestamp"].shift(1)
        out["time_since_last_txn_sec"] = (
            out["timestamp"] - out["prev_ts"]
        ).dt.total_seconds()

        median_gap = out["time_since_last_txn_sec"].median()
        out["time_since_last_txn_sec"] = out["time_since_last_txn_sec"].fillna(
            median_gap if pd.notna(median_gap) else 3600.0
        )
        out["log_time_since_last_txn"] = np.log1p(out["time_since_last_txn_sec"])
        out["time_since_last_txn_min"] = out["time_since_last_txn_sec"] / 60

        for col in ["time_since_last_txn_sec","log_time_since_last_txn",
                    "time_since_last_txn_min"]:
            self._feature_registry["B_temporal"].append(col)

        # Is first transaction of the day for this user
        out["txn_date"]  = out["timestamp"].dt.date
        out["prev_date"] = out.groupby("user_id")["txn_date"].shift(1)
        out["is_first_txn_of_day"] = (out["txn_date"] != out["prev_date"]).astype(int)
        self._feature_registry["B_temporal"].append("is_first_txn_of_day")

        out.drop(columns=["prev_ts","txn_date","prev_date"], errors="ignore", inplace=True)
        return out


    def _grp_C_rolling_amount(self, df, mask):
        out = df.copy()

        for entity in ENTITY_GROUPS:
            ename = _entity_name(entity)

            for window in ROLLING_WINDOWS:
                for agg in AMOUNT_AGGS:
                    col = f"{ename}_amt_{agg}_{window}"
                    fill = self._global_stats["amount_mean"] if agg in ("mean","median") else \
                           self._global_stats["amount_std"]  if agg == "std"    else \
                           self._global_stats["amount_p95"]  if agg == "max"    else \
                           self._global_stats["amount_p25"]  if agg == "min"    else \
                           self._global_stats["amount_mean"] * 10

                    raw = self._rolling(out, entity, window, "amount", agg)
                    out[col] = raw.fillna(fill)
                    self._feature_registry["C_rolling_amount"].append(col)

        # Set cold-start flag using expanding user txn count
        user_expanding_count = out.groupby("user_id")["amount"].transform(
            lambda x: x.shift(1).expanding(1).count()
        )
        out["user_prior_txn_count"] = user_expanding_count.fillna(0)
        out["is_cold_start"] = (out["user_prior_txn_count"] < COLD_START_MIN).astype(int)
        self._feature_registry["C_rolling_amount"].extend(
            ["user_prior_txn_count","is_cold_start"]
        )

        return out


    def _grp_D_rolling_count(self, df, mask):
        out = df.copy()

        # Long windows — user-level and cross-entity
        for entity in ENTITY_GROUPS[:3]:   # user, user×txntype, user×city
            ename = _entity_name(entity)
            for window in ROLLING_WINDOWS:
                col = f"{ename}_txn_count_{window}"
                raw = self._rolling(out, entity, window, "amount", "count")
                out[col] = raw.fillna(0)
                self._feature_registry["D_rolling_count"].append(col)

        # Short recency/velocity windows — user-level
        for window in RECENCY_WINDOWS:
            for entity in [["user_id"], ["user_id","txn_type"]]:
                ename = _entity_name(entity)
                col   = f"{ename}_velocity_{window}"
                raw   = self._rolling(out, entity, window, "amount", "count")
                out[col] = raw.fillna(0)
                self._feature_registry["D_rolling_count"].append(col)

            # Amount sum in recency window
            col = f"user_amt_sum_{window}"
            raw = self._rolling(out, ["user_id"], window, "amount", "sum")
            out[col] = raw.fillna(0)
            self._feature_registry["D_rolling_count"].append(col)

        return out


    def _grp_E_deviation(self, df, mask):
        out = df.copy()
        amt = out["amount"]
        gs  = self._global_stats

        # E1. Amount vs rolling mean — ratio (current / window_mean)
        for window in ROLLING_WINDOWS:
            mean_col = f"user_amt_mean_{window}"
            if mean_col not in out.columns:
                continue
            col = f"user_amt_ratio_vs_mean_{window}"
            out[col] = _safe_div(amt, out[mean_col])
            self._feature_registry["E_deviation"].append(col)

        # E2. Amount z-score per rolling window
        # zscore = (amount - window_mean) / window_std
        for window in ROLLING_WINDOWS:
            mean_col = f"user_amt_mean_{window}"
            std_col  = f"user_amt_std_{window}"
            if mean_col not in out.columns or std_col not in out.columns:
                continue
            std_safe = out[std_col].replace(0, np.nan).fillna(gs["amount_std"])
            col = f"user_amt_zscore_{window}"
            out[col] = _safe_div(amt - out[mean_col], std_safe).clip(-10, 10)
            self._feature_registry["E_deviation"].append(col)

        # E3. Amount vs window max — how close to user's historical max?
        for window in ["7D","30D","60D"]:
            max_col = f"user_amt_max_{window}"
            if max_col not in out.columns:
                continue
            col = f"user_amt_ratio_vs_max_{window}"
            out[col] = _safe_div(amt, out[max_col])
            self._feature_registry["E_deviation"].append(col)

        # E4. Amount vs global mean (global z-score)
        gm, gs_std = gs["amount_mean"], gs["amount_std"]
        out["amt_global_zscore"] = _safe_div(amt - gm, gs_std).clip(-10, 10)
        out["log_amt_global_zscore"] = _safe_div(
            out["log_amount"] - gs["amount_log_mean"],
            gs["amount_log_std"]
        ).clip(-10, 10)
        for c in ["amt_global_zscore","log_amt_global_zscore"]:
            self._feature_registry["E_deviation"].append(c)

        # E5. Per txn_type z-score vs user's own txn_type history
        for window in ["7D","30D"]:
            for txntype in ["withdrawal","deposit","purchase","transfer"]:
                mean_col = f"user_txntype_{txntype}_amt_mean_{window}"
                # This col won't exist in current entity blast but is illustrative
                # We compute it from the entity = [user_id, txn_type] rolling blast
                mean_col_generic = f"user_txntype_amt_mean_{window}"
                if mean_col_generic not in out.columns:
                    continue
                col = f"user_txntype_amt_zscore_{window}"
                std_col = f"user_txntype_amt_std_{window}"
                if std_col not in out.columns:
                    continue
                std_safe = out[std_col].replace(0, np.nan).fillna(gs["amount_std"])
                out[col] = _safe_div(amt - out[mean_col_generic], std_safe).clip(-10, 10)
                if col not in self._feature_registry["E_deviation"]:
                    self._feature_registry["E_deviation"].append(col)
                break  # only compute once per window

        # E6. Sequence deviation — amount vs previous transaction
        out["prev_amount"] = out.groupby("user_id")["amount"].shift(1)
        out["amt_delta_from_prev"]     = amt - out["prev_amount"].fillna(amt)
        out["amt_ratio_from_prev"]     = _safe_div(amt, out["prev_amount"].fillna(amt))
        out["amt_pct_change_from_prev"] = (
            _safe_div(amt - out["prev_amount"], out["prev_amount"].fillna(1)) * 100
        ).clip(-500, 500)
        for c in ["amt_delta_from_prev","amt_ratio_from_prev","amt_pct_change_from_prev"]:
            self._feature_registry["E_deviation"].append(c)

        # E7. Amount vs user's median (more robust than mean for skewed data)
        for window in ["7D","30D"]:
            med_col = f"user_amt_median_{window}"
            if med_col not in out.columns:
                continue
            col = f"user_amt_ratio_vs_median_{window}"
            out[col] = _safe_div(amt, out[med_col])
            self._feature_registry["E_deviation"].append(col)

        # E8. Velocity deviation — current velocity vs average velocity
        # Is user transacting faster than their own usual rate?
        for short_w in ["1h","6h"]:
            vel_col = f"user_velocity_{short_w}"
            avg_col = f"user_txn_count_30D"
            if vel_col not in out.columns or avg_col not in out.columns:
                continue
            # daily rate from 30D count
            avg_daily = _safe_div(out[avg_col], pd.Series(30, index=out.index))
            avg_hourly = _safe_div(avg_daily, pd.Series(24, index=out.index))
            hours = float(short_w.replace("h","").replace("H",""))
            avg_per_window = avg_hourly * hours
            col = f"velocity_ratio_{short_w}_vs_30D_baseline"
            out[col] = _safe_div(out[vel_col], avg_per_window.replace(0, np.nan))
            self._feature_registry["E_deviation"].append(col)

        # E9. Amount above user's expanding p95
        out.loc[mask, "user_amt_p95_expanding"] = out[mask].groupby("user_id")["amount"].transform(
            lambda x: x.shift(1).expanding(5).quantile(0.95)
        )
        out["user_amt_p95_expanding"] = out["user_amt_p95_expanding"].fillna(gs["amount_p95"])
        out["amt_above_user_p95"] = (amt > out["user_amt_p95_expanding"]).astype(int)
        out["amt_ratio_vs_user_p95"] = _safe_div(amt, out["user_amt_p95_expanding"])
        for c in ["amt_above_user_p95","amt_ratio_vs_user_p95"]:
            self._feature_registry["E_deviation"].append(c)

        out.drop(columns=["prev_amount","city_avg_amount","user_amt_p95_expanding"],
                 errors="ignore", inplace=True)
        return out


    def _grp_F_fingerprint(self, df, mask):
        out = df.copy()

        for entity in FINGERPRINT_GROUPS:
            ename = _entity_name(entity)
            col   = f"{ename}_prior_count"

            out[col] = out.groupby(entity).cumcount()
            # Note: cumcount()[i] = number of rows in this group BEFORE row i
            # This is already correct — 0 means "first time", 1 means "seen once before", etc.

            out[f"{ename}_is_first_occurrence"] = (out[col] == 0).astype(int)
            self._feature_registry["F_fingerprint"].extend([col, f"{ename}_is_first_occurrence"])

        return out


    def _grp_G_geographic(self, df, mask):
        out = df.copy()
        default_rarity = 0.99   # treat unseen values as very rare

        # G1. City global rarity (from fit window)
        out["city_global_rarity"] = out["city"].map(
            self._rarity_maps.get("city", {})
        ).fillna(default_rarity)
        self._feature_registry["G_geographic"].append("city_global_rarity")

        # G2. Is new city for this user (point-in-time via cumcount)
        out["is_new_city"] = (out["user_city_prior_count"] == 0).astype(int)
        ndc_series = [_n_distinct_cumcount(grp, "city") for _, grp in out.groupby("user_id")]
        out["user_n_distinct_cities"] = pd.concat(ndc_series).reindex(out.index).fillna(1)
        for c in ["is_new_city","user_n_distinct_cities"]:
            self._feature_registry["G_geographic"].append(c)

        # G3. City transition rarity
        out["prev_city"] = out.groupby("user_id")["city"].shift(1)
        out["city_transition_rarity"] = out.apply(
            lambda r: self._rarity_maps.get("city_transition",{}).get(
                (r["prev_city"], r["city"]), default_rarity
            ) if pd.notna(r.get("prev_city")) else default_rarity,
            axis=1
        )
        out["city_changed_from_prev"] = (
            (out["city"] != out["prev_city"]) &
            (out["prev_city"].notna()) &
            (out["city"] != "UNKNOWN")
        ).astype(int)
        for c in ["city_transition_rarity","city_changed_from_prev"]:
            self._feature_registry["G_geographic"].append(c)

        # G4. Impossible travel flag
        out["impossible_travel"] = (
            (out["is_new_city"] == 1) &
            (out["time_since_last_txn_sec"] < 3600) &
            (out["is_cold_start"] == 0)
        ).astype(int)
        self._feature_registry["G_geographic"].append("impossible_travel")

        # G5. Rolling unique cities in recent windows
        for window in ["7D","30D"]:
            col = f"user_unique_cities_{window}"
            # Use city encoded as int for rolling apply
            city_enc = out.groupby("user_id")["city"].transform(
                lambda x: pd.factorize(x)[0].astype(float)
            )
            out_temp = out.copy()
            out_temp["_city_enc"] = city_enc
            raw = self._rolling(out_temp, ["user_id"], window, "_city_enc", "count")
            out[col] = raw.fillna(1)
            self._feature_registry["G_geographic"].append(col)

        out.drop(columns=["prev_city"], errors="ignore", inplace=True)
        return out

    def _grp_H_device(self, df, mask):
        out = df.copy()
        default_rarity = 0.99

        # H1. Device global rarity
        out["device_global_rarity"] = out["device"].map(
            self._rarity_maps.get("device", {})
        ).fillna(default_rarity)
        self._feature_registry["H_device"].append("device_global_rarity")

        # H2. Is new device, prior count, distinct device count
        out["is_new_device"]           = (out["user_device_prior_count"] == 0).astype(int)
        ndd_series = [_n_distinct_cumcount(grp, "device") for _, grp in out.groupby("user_id")]
        out["user_n_distinct_devices"] = pd.concat(ndd_series).reindex(out.index).fillna(1)
        for c in ["is_new_device","user_n_distinct_devices"]:
            self._feature_registry["H_device"].append(c)

        # H3. Device switched from previous transaction
        out["prev_device"] = out.groupby("user_id")["device"].shift(1)
        out["device_changed_from_prev"] = (
            (out["device"] != out["prev_device"]) &
            (out["prev_device"].notna()) &
            (out["device"] != "UNKNOWN")
        ).astype(int)
        self._feature_registry["H_device"].append("device_changed_from_prev")


        # H4. Simultaneous new device AND new city
        out["new_device_and_new_city"] = (
            (out["is_new_device"] == 1) & (out["is_new_city"] == 1)
        ).astype(int)
        self._feature_registry["H_device"].append("new_device_and_new_city")

        out.drop(columns=["prev_device"], errors="ignore", inplace=True)
        return out
    

    def _grp_I_sequence(self, df, mask):
        out = df.copy()

        for col in SEQUENCE_COLS:
            if col not in out.columns:
                continue
            lag_col = f"prev_{col}"
            out[lag_col] = out.groupby("user_id")[col].shift(1)
            self._feature_registry["J_sequence"].append(lag_col)

            # For numeric columns, also compute lag-2 and rolling-3 mean
            if col == "amount":
                out["prev2_amount"] = out.groupby("user_id")["amount"].shift(2)
                out["lag3_mean_amount"] = (
                    out.groupby("user_id")["amount"]
                    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
                )
                for c in ["prev2_amount","lag3_mean_amount"]:
                    self._feature_registry["J_sequence"].append(c)

        # Inter-transaction time from same city (for impossible travel)
        out.loc[mask, "time_since_last_same_city_sec"] = out[mask].groupby(
            ["user_id","city"]
        )["timestamp"].transform(
            lambda x: x.diff().dt.total_seconds()
        )
        out["time_since_last_same_city_sec"] = (
            out["time_since_last_same_city_sec"].fillna(
                out["time_since_last_txn_sec"]
            )
        )
        self._feature_registry["J_sequence"].append("time_since_last_same_city_sec")

        return out


    def _grp_J_interaction(self, df, mask):
        out = df.copy()

        def _add(col, val):
            out[col] = val
            self._feature_registry["K_interaction"].append(col)

        amt_z = out.get("amt_global_zscore", pd.Series(0, index=out.index))

        # High-signal interactions
        _add("late_night_x_amt_zscore",
             out["is_late_night"] * amt_z.clip(0, 10))

        _add("new_city_x_amt_zscore",
             out["is_new_city"] * amt_z.clip(0, 10))

        _add("new_device_x_amt_zscore",
             out.get("is_new_device", 0) * amt_z.clip(0, 10))

        _add("new_city_x_new_device",
             out.get("new_device_and_new_city", 0))

        _add("weekend_x_late_night",
             out["is_weekend"] * out["is_late_night"])

        _add("weekend_x_high_amt",
             out["is_weekend"] * out.get("amount_above_p95", 0))

        _add("late_night_x_new_city",
             out["is_late_night"] * out["is_new_city"])

        _add("impossible_travel_x_amt",
             out.get("impossible_travel", 0) * out["amount"].fillna(0))

        _add("missing_meta_x_amt_zscore",
             out.get("missing_field_count", 0) * amt_z.clip(0, 10))

        # Velocity × amount interactions
        vel_1h = out.get("user_velocity_1H", pd.Series(0, index=out.index))
        _add("high_velocity_x_high_amt",
             vel_1h.clip(0, 10) * amt_z.clip(0, 10))

        _add("currency_mismatch_x_amt",
             out.get("currency_city_mismatch", 0) * out["amount"].fillna(0))

        _add("new_device_new_city_x_amt",
             out.get("new_device_and_new_city", 0) * out["amount"].fillna(0))

        # Z-score windowed interactions
        for window in ["7D","30D"]:
            z_col = f"user_amt_zscore_{window}"
            if z_col not in out.columns:
                continue
            z = out[z_col]
            _add(f"late_night_x_zscore_{window}",    out["is_late_night"] * z.clip(0,10))
            _add(f"new_city_x_zscore_{window}",       out["is_new_city"]   * z.clip(0,10))
            _add(f"new_device_x_zscore_{window}",
                 out.get("is_new_device",
                         pd.Series(0, index=out.index)) * z.clip(0,10))

        return out


    def _grp_K_missingness(self, df, mask):
        out = df.copy()

        out["is_city_missing"]     = (out["city"]     == "UNKNOWN").astype(int)
        out["is_device_missing"]   = (out["device"]   == "UNKNOWN").astype(int)
        out["is_currency_missing"] = (out["currency"] == "UNKNOWN").astype(int)
        out["missing_field_count"] = (
            out["is_city_missing"] +
            out["is_device_missing"] +
            out["is_currency_missing"]
        )
        out["is_both_geo_dev_missing"] = (
            (out["is_city_missing"] == 1) & (out["is_device_missing"] == 1)
        ).astype(int)

        # Historical missingness rate for this user
        def _miss_rate(group):
            result, cum_miss, total = [], 0, 0
            for val in group["missing_field_count"]:
                result.append(cum_miss / total if total > 0 else 0.0)
                cum_miss += int(val > 0); total += 1
            return pd.Series(result, index=group.index)

        mr_series = []
        for uid, grp in out[mask].groupby("user_id"):
            mr_series.append(_miss_rate(grp[["missing_field_count"]]))
        if mr_series:
            out.loc[mask, "user_missing_rate_hist"] = (
                pd.concat(mr_series).reindex(out[mask].index).values
            )
        out["user_missing_rate_hist"] = out["user_missing_rate_hist"].fillna(0.0)

        for c in ["is_city_missing","is_device_missing","is_currency_missing",
                  "missing_field_count","is_both_geo_dev_missing",
                  "user_missing_rate_hist"]:
            self._feature_registry["L_missingness"].append(c)

        return out

    def _grp_L_burst(self, df, mask):
        out = df.copy()
        def _add(col, val):
            out[col] = val
            if col not in self._feature_registry["N_burst"]:
                self._feature_registry["N_burst"].append(col)

        _add("is_burst_5min",  (out["time_since_last_txn_sec"] < 300).astype(int))
        _add("is_burst_30min", (out["time_since_last_txn_sec"] < 1800).astype(int))

        def _burst_hist(group):
            result, count = [], 0
            for gap in group["time_since_last_txn_sec"]:
                result.append(count)
                if gap < 300: count += 1
            return pd.Series(result, index=group.index)

        bs = [_burst_hist(g[["time_since_last_txn_sec"]]) for _, g in out[mask].groupby("user_id")]
        if bs:
            out.loc[mask, "user_burst_count_hist"] = pd.concat(bs).reindex(out[mask].index).values
        out["user_burst_count_hist"] = out["user_burst_count_hist"].fillna(0)
        _add("user_burst_count_hist", out["user_burst_count_hist"])

        _add("burst_x_high_amount", out["is_burst_5min"] * out["amount"].fillna(0))

        vel_5min = out.get("user_velocity_5min", pd.Series(0, index=out.index))
        _add("burst_velocity_5min", vel_5min)

        out["_prev_burst"] = out.groupby("user_id")["is_burst_5min"].shift(1).fillna(0)
        _add("consecutive_bursts",
             ((out["is_burst_5min"]==1) & (out["_prev_burst"]==1)).astype(int))
        out.drop(columns=["_prev_burst"], errors="ignore", inplace=True)
        return out


    @property
    def feature_registry(self) -> Dict[str, List[str]]:
        """Dict mapping feature group → list of column names."""
        return dict(self._feature_registry)

    @property
    def all_feature_columns(self) -> List[str]:
        """Flat list of all feature columns across all groups."""
        return [col for cols in self._feature_registry.values() for col in cols]

    @property
    def feature_count(self) -> int:
        return len(self.all_feature_columns)

    def feature_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing feature counts per group."""
        rows = [
            {"group": grp, "n_features": len(cols),
             "examples": ", ".join(cols[:3])}
            for grp, cols in self._feature_registry.items()
        ]
        return pd.DataFrame(rows)