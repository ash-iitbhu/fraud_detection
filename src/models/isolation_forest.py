"""
src/models/isolation_forest.py
--------------------------------
Isolation Forest anomaly detector.

Categorical variables (user_id, city, device) are represented as
continuous rarity/frequency encodings — NOT one-hot or embeddings.
This keeps IF operating in a well-behaved continuous space where random
partitioning is meaningful.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.features.feature_engineer import CONTINUOUS_FEATURES


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


class IsolationForestDetector:

    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.feature_cols  = CONTINUOUS_FEATURES
        self.model   = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler          = StandardScaler()
        self._is_fitted      = False
        self._shap_explainer = None

    def _get_X(self, df, fit=False):
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def fit(self, fit_df: pd.DataFrame):
        """Fit on successfully parsed rows in the fit window."""
        train = fit_df[fit_df["parse_success"] == True].copy()
        X = self._get_X(train, fit=True)
        self.model.fit(X)
        self._is_fitted = True

        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                pass

        print(f"[IsolationForest] Fitted on {len(train)} rows "
              f"(contamination={self.contamination})")
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all rows. Adds if_score [0,1] and if_is_anomaly columns."""
        assert self._is_fitted
        out = df.copy()
        X   = self._get_X(out)

        raw  = self.model.decision_function(X)
        pred = self.model.predict(X)

        out["if_raw_score"]  = raw
        out["if_score"]      = _minmax_normalize(-raw)   # higher = more anomalous
        out["if_is_anomaly"] = (pred == -1).astype(int)

        n = out["if_is_anomaly"].sum()
        print(f"[IsolationForest] Flagged {n}/{len(out)} ({100*n/len(out):.1f}%)")
        return out

    def get_shap_values(self, df: pd.DataFrame):
        """Return SHAP values array (n_samples, n_features) or None."""
        if self._shap_explainer is None:
            return None
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        try:
            return self._shap_explainer.shap_values(X_scaled)
        except Exception:
            return None

    def get_feature_names(self):
        return [c for c in self.feature_cols]
