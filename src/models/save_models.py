import os
import json
import joblib
import numpy as np

ARTIFACT_DIR = "outputs/model_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_isolation_forest(detector, path: str = ARTIFACT_DIR):
    """Save IsolationForestDetector — model + scaler + shap explainer as joblib."""
    joblib.dump(detector.model,  os.path.join(path, "if_model.joblib"))
    joblib.dump(detector.scaler, os.path.join(path, "if_scaler.joblib"))

    # Save SHAP explainer if available
    if detector._shap_explainer is not None:
        joblib.dump(detector._shap_explainer,
                    os.path.join(path, "if_shap_explainer.joblib"))
        print("  SHAP explainer saved.")

    meta = {
        "feature_cols":   detector.feature_cols,
        "train_raw_min":  detector._train_raw_min,
        "train_raw_max":  detector._train_raw_max,
        "has_shap":       detector._shap_explainer is not None,
    }
    with open(os.path.join(path, "if_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ IsolationForest saved → {path}/if_{{model,scaler,shap_explainer,meta}}")


def load_isolation_forest(path: str = ARTIFACT_DIR):
    """Reconstruct IsolationForestDetector from saved artifacts."""
    from src.models.isolation_forest import IsolationForestDetector

    detector            = IsolationForestDetector.__new__(IsolationForestDetector)
    detector.model      = joblib.load(os.path.join(path, "if_model.joblib"))
    detector.scaler     = joblib.load(os.path.join(path, "if_scaler.joblib"))
    detector._is_fitted = True

    with open(os.path.join(path, "if_meta.json")) as f:
        meta = json.load(f)
    detector.feature_cols    = meta["feature_cols"]
    detector._train_raw_min  = meta["train_raw_min"]
    detector._train_raw_max  = meta["train_raw_max"]

    # Load SHAP explainer — try saved file first, then rebuild from model
    shap_path = os.path.join(path, "if_shap_explainer.joblib")
    detector._shap_explainer = None

    if meta.get("has_shap", False) and os.path.exists(shap_path):
        try:
            detector._shap_explainer = joblib.load(shap_path)
            print("  SHAP explainer loaded from file.")
        except Exception as e:
            print(f"  SHAP explainer file load failed ({e}). Rebuilding from model...")

    # Rebuild if load failed or file was missing
    if detector._shap_explainer is None and meta.get("has_shap", False):
        try:
            import shap
            detector._shap_explainer = shap.TreeExplainer(detector.model)
            print("  SHAP explainer rebuilt from loaded model.")
        except Exception as e:
            print(f"  SHAP explainer rebuild failed ({e}). SHAP unavailable.")

    print(f"✅ IsolationForest loaded ← {path}")
    return detector


def save_autoencoder(detector, path: str = ARTIFACT_DIR):
    """
    Save AutoencoderDetector artifacts.
    PyTorch model state dict + sklearn fallback + shared metadata.
    """
    # Always save sklearn scaler and vocabularies (shared by both backends)
    joblib.dump(detector._scaler, os.path.join(path, "ae_scaler.joblib"))

    meta = {
        "threshold":            detector.threshold,
        "actual_cont_feats":    detector._actual_cont_feats,
        "vocabs":               detector._vocabs,   # {col: {token: idx}}
        "bottleneck_dim":       detector.bottleneck_dim,
        "use_torch":            detector._use_torch,
        "train_error_min":   detector._train_error_min,  
        "train_error_max":   detector._train_error_max, 
    }
    with open(os.path.join(path, "ae_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if detector._use_torch and detector._model is not None:
        # PyTorch: save state dict only (portable across devices)
        import torch
        torch.save(
            detector._model.state_dict(),
            os.path.join(path, "ae_model_state.pt")
        )
        print(f"✅ Autoencoder (PyTorch) saved → {path}/ae_{{model_state,scaler,meta}}")

    elif detector._sklearn_ae is not None:
        # sklearn fallback
        joblib.dump(detector._sklearn_ae,
                    os.path.join(path, "ae_sklearn_model.joblib"))
        print(f"✅ Autoencoder (sklearn) saved → {path}/ae_{{sklearn_model,scaler,meta}}")



def load_autoencoder(path: str = ARTIFACT_DIR):
    """Reconstruct AutoencoderDetector from saved artifacts."""
    from src.models.autoencoder import (
        AutoencoderDetector, FraudAutoencoder,
        CATEGORICAL_EMBEDDING_CONFIG, TORCH_AVAILABLE
    )

    with open(os.path.join(path, "ae_meta.json")) as f:
        meta = json.load(f)

    detector = AutoencoderDetector.__new__(AutoencoderDetector)
    detector._scaler             = joblib.load(os.path.join(path, "ae_scaler.joblib"))
    detector._vocabs             = meta["vocabs"]
    detector._threshold          = meta["threshold"]
    detector._train_error_min = meta["train_error_min"]   # ADD THIS
    detector._train_error_max = meta["train_error_max"]   # ADD THIS
    detector._actual_cont_feats  = meta["actual_cont_feats"]
    detector.bottleneck_dim      = meta["bottleneck_dim"]
    detector._use_torch          = meta["use_torch"] and TORCH_AVAILABLE
    detector._is_fitted          = True
    detector._sklearn_ae         = None
    detector._model              = None

    if detector._use_torch:
        import torch
        vocab_sizes   = {col: len(v) for col, v in detector._vocabs.items()}
        detector.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector._model = FraudAutoencoder(
            n_continuous=len(detector._actual_cont_feats),
            vocab_sizes=vocab_sizes,
            bottleneck_dim=detector.bottleneck_dim,
        ).to(detector.device)
        state = torch.load(
            os.path.join(path, "ae_model_state.pt"),
            map_location=detector.device
        )
        detector._model.load_state_dict(state)
        detector._model.eval()
        print(f"✅ Autoencoder (PyTorch) loaded ← {path}")

    else:
        sklearn_path = os.path.join(path, "ae_sklearn_model.joblib")
        if os.path.exists(sklearn_path):
            detector._sklearn_ae = joblib.load(sklearn_path)
        detector.device = None
        print(f"✅ Autoencoder (sklearn) loaded ← {path}")

    return detector

def save_ensemble_scorer(scorer, path: str = ARTIFACT_DIR):
    """Save EnsembleScorer frozen parameters as JSON."""
    meta = {
        "score_ranges":      scorer.score_ranges,       # {"rule_score": [min, max], ...}
        "tier_thresholds":   scorer.tier_thresholds,    # {"TIER_1": 0.82, ...}
        "weights":           scorer.weights,            # {"rule_score": 0.33, ...}
    }
    out_path = os.path.join(path, "ensemble_scorer.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ EnsembleScorer saved → {out_path}")


def load_ensemble_scorer(path: str = ARTIFACT_DIR):
    """Reconstruct EnsembleScorer from saved JSON."""
    from src.models.ensemble import EnsembleScorer

    with open(os.path.join(path, "ensemble_scorer.json")) as f:
        meta = json.load(f)

    scorer = EnsembleScorer()
    # JSON stores lists, convert back to tuples for score_ranges
    scorer.score_ranges    = {k: tuple(v) for k, v in meta["score_ranges"].items()}
    scorer.tier_thresholds = meta["tier_thresholds"]
    scorer.weights         = meta["weights"]

    print(f"✅ EnsembleScorer loaded ← {path}")
    print(f"   Frozen ranges     : {scorer.score_ranges}")
    print(f"   Tier thresholds   : {scorer.tier_thresholds}")
    return scorer
