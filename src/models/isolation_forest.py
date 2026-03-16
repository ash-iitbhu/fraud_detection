import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import shap
SHAP_AVAILABLE = True



def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


class IsolationForestDetector:

    def __init__(self, contamination=0.05, n_estimators=200, features = None, random_state=42, max_samples ='auto', max_features=1.0):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.max_samples   = max_samples
        self.max_features  = max_features
        self.feature_cols  = features
        self.model   = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler          = StandardScaler()
        self._is_fitted      = False
        self._shap_explainer = None
        self._train_raw_min  = None   
        self._train_raw_max  = None  

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

        train_raw   = self.model.decision_function(X)
        self._train_raw_min = float((-train_raw).min())
        self._train_raw_max = float((-train_raw).max())

        if SHAP_AVAILABLE:
            self._shap_explainer = shap.TreeExplainer(self.model)
        else:
            warnings.warn("SHAP library not available. SHAP values will be None.")

        print(f"[IsolationForest] Fitted on {len(train)} rows "
              f"(contamination={self.contamination})")
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._is_fitted
        out = df.copy()
        X   = self._get_X(out)

        raw  = self.model.decision_function(X)
        pred = self.model.predict(X)

        out["if_raw_score"] = raw
        neg_raw = -raw
        out["if_score"] = np.clip(
            (neg_raw - self._train_raw_min) / (self._train_raw_max - self._train_raw_min + 1e-9),
            0.0, 1.0
        )
        out["if_is_anomaly"] = (pred == -1).astype(int)

        n = out["if_is_anomaly"].sum()
        print(f"[IsolationForest] Flagged {n}/{len(out)} ({100*n/len(out):.1f}%)")
        return out

    def get_shap_values(self, df: pd.DataFrame):
        """Return SHAP values array (n_samples, n_features) or None."""
        if self._shap_explainer is None:
            warnings.warn("SHAP explainer not available. Returning None.")
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
