"""
src/models/autoencoder.py
--------------------------
Deep Autoencoder with Entity Embeddings for transaction anomaly detection.

Architecture:
    Two input streams:
      1. Continuous features → dense encoder
      2. Categorical features (user_id, city, device, txn_type) →
         separate Embedding layers → concatenated with continuous stream

    Combined representation → bottleneck → decoder → reconstruction

    Anomaly score = reconstruction error (MSE on continuous features
    + cross-entropy on categorical reconstructions)

Why entity embeddings?
    High-cardinality categoricals like user_id (100+ users) cannot be
    one-hot encoded into Isolation Forest (too sparse, too high-dimensional).
    Embeddings learn dense 16-dim behavioral fingerprints: users with
    similar transaction patterns get similar embedding vectors.
    The autoencoder handles the categorical signal; IF handles continuous.

Fallback:
    If PyTorch is unavailable, a sklearn MLPRegressor-based autoencoder
    is used on continuous features only (no embeddings).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from src.features.feature_engineer import CONTINUOUS_FEATURES

# ── Try importing PyTorch ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Using sklearn MLPRegressor fallback "
        "(no entity embeddings — install torch for full functionality)."
    )

# ── Vocabulary builder ────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["user_id", "city", "device", "txn_type"]
UNK_TOKEN = "<UNK>"

def build_vocabularies(fit_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Build integer-index vocabulary for each categorical column.
    Index 0 is always reserved for <UNK> (unseen categories at score time).
    """
    vocabs = {}
    for col in CATEGORICAL_COLS:
        unique_vals = fit_df[col].dropna().unique().tolist()
        vocab = {UNK_TOKEN: 0}
        for i, val in enumerate(unique_vals, start=1):
            vocab[str(val)] = i
        vocabs[col] = vocab
        print(f"  Vocab '{col}': {len(vocab)} tokens (incl. <UNK>)")
    return vocabs


def encode_categoricals(df: pd.DataFrame, vocabs: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    Map categorical string values to integer indices using pre-built vocabs.
    Unknown values (including new users/cities at score time) → 0 (<UNK>).
    """
    out = df.copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            vocab = vocabs[col]
            out[f"{col}_idx"] = out[col].astype(str).map(
                lambda x, v=vocab: v.get(x, 0)
            )
    return out


# ── PyTorch Implementation ─────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class TransactionDataset(Dataset):
        """PyTorch Dataset wrapping continuous + categorical feature arrays."""

        def __init__(self, X_cont: np.ndarray, X_cat: np.ndarray):
            self.X_cont = torch.FloatTensor(X_cont)
            self.X_cat  = torch.LongTensor(X_cat)

        def __len__(self):
            return len(self.X_cont)

        def __getitem__(self, idx):
            return self.X_cont[idx], self.X_cat[idx]


    class FraudAutoencoder(nn.Module):
        """
        Autoencoder with entity embedding layers for categorical inputs.

        Embedding dimensions follow the rule of thumb:
            min(50, (cardinality // 2) + 1)
        """

        def __init__(self, n_continuous: int, vocab_sizes: Dict[str, int]):
            super().__init__()

            # ── Embedding layers (one per categorical) ────────────────────────
            self.embeddings = nn.ModuleDict()
            self.emb_dims   = {}
            total_emb_dim   = 0
            for col, vocab_size in vocab_sizes.items():
                emb_dim = min(50, (vocab_size // 2) + 1)
                self.embeddings[col] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=emb_dim,
                    padding_idx=0,    # 0 = <UNK>
                )
                self.emb_dims[col] = emb_dim
                total_emb_dim += emb_dim

            # Total input dimension = continuous features + all embeddings
            input_dim = n_continuous + total_emb_dim

            # ── Encoder ───────────────────────────────────────────────────────
            encoder_dims = [input_dim, 64, 32, 16]
            encoder_layers = []
            for i in range(len(encoder_dims) - 1):
                encoder_layers.extend([
                    nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                    nn.BatchNorm1d(encoder_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
            self.encoder = nn.Sequential(*encoder_layers)

            # Bottleneck dimension
            self.bottleneck_dim = 8
            self.bottleneck = nn.Linear(16, self.bottleneck_dim)

            # ── Decoder ───────────────────────────────────────────────────────
            decoder_dims = [self.bottleneck_dim, 16, 32, 64]
            decoder_layers = []
            for i in range(len(decoder_dims) - 1):
                decoder_layers.extend([
                    nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                    nn.BatchNorm1d(decoder_dims[i+1]),
                    nn.ReLU(),
                ])
            self.decoder = nn.Sequential(*decoder_layers)

            # Output heads
            # One head reconstructs continuous features (MSE loss)
            self.output_continuous = nn.Linear(64, n_continuous)

            # One head per categorical reconstructs the category (CrossEntropy loss)
            self.output_categorical = nn.ModuleDict({
                col: nn.Linear(64, vocab_size)
                for col, vocab_size in vocab_sizes.items()
            })

            self.n_continuous = n_continuous
            self.vocab_sizes  = vocab_sizes
            self.cat_col_order = list(vocab_sizes.keys())

        def forward(self, x_cont, x_cat):
            """
            x_cont : (batch, n_continuous)
            x_cat  : (batch, n_categoricals)  — integer indices
            """
            # Embed each categorical and concatenate
            emb_parts = []
            for i, col in enumerate(self.cat_col_order):
                emb = self.embeddings[col](x_cat[:, i])
                emb_parts.append(emb)

            x_emb = torch.cat(emb_parts, dim=1)            # (batch, total_emb_dim)
            x_in  = torch.cat([x_cont, x_emb], dim=1)     # (batch, input_dim)

            # Encode
            encoded    = self.encoder(x_in)
            bottleneck = self.bottleneck(encoded)

            # Decode
            decoded = self.decoder(bottleneck)

            # Reconstruct continuous
            recon_cont = self.output_continuous(decoded)

            # Reconstruct categoricals (logits, not softmax — CrossEntropyLoss handles that)
            recon_cat = {
                col: self.output_categorical[col](decoded)
                for col in self.cat_col_order
            }

            return recon_cont, recon_cat, bottleneck

        def get_embeddings(self, col: str, indices: torch.Tensor) -> torch.Tensor:
            """Extract embedding vectors for a specific categorical column."""
            return self.embeddings[col](indices)


# ── Sklearn fallback autoencoder (no embeddings) ─────────────────────────────

class SklearnAutoencoderFallback:
    """
    MLP-based autoencoder using sklearn. Used when PyTorch is unavailable.
    Operates on continuous features only — no entity embeddings.
    """

    def __init__(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model  = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16, 32, 64),
            activation="relu",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, X_scaled)   # reconstruction target = input
        self._is_fitted = True
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        recon    = self.model.predict(X_scaled)
        return np.mean((X_scaled - recon) ** 2, axis=1)


# ── Main Autoencoder Trainer / Scorer ────────────────────────────────────────

class AutoencoderDetector:
    """
    High-level interface for training and scoring the autoencoder.
    Handles both PyTorch (full embeddings) and sklearn (fallback) backends.
    """

    def __init__(self, epochs=50, batch_size=64, lr=1e-3,
                 cont_loss_weight=0.6, cat_loss_weight=0.1,
                 device: Optional[str] = None):
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.cont_loss_weight = cont_loss_weight
        self.cat_loss_weight  = cat_loss_weight
        self.vocabs           = None
        self.model            = None
        self._is_fitted       = False
        self._use_torch       = TORCH_AVAILABLE
        self._recon_threshold = None   # 95th percentile on fit window

        if TORCH_AVAILABLE:
            self.device = torch.device(
                device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            print(f"[Autoencoder] Using device: {self.device}")
        else:
            print("[Autoencoder] PyTorch not available — using sklearn fallback.")

    def _prepare_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (X_continuous, X_categorical) arrays from DataFrame.
        Rows with parse_success=False are kept but NaN-filled.
        """
        cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        X_cont = df[cont_cols].fillna(0).values.astype(np.float32)

        # Normalize continuous features
        if hasattr(self, '_cont_scaler') and self._cont_scaler is not None:
            X_cont = self._cont_scaler.transform(X_cont)

        cat_idx_cols = [f"{col}_idx" for col in CATEGORICAL_COLS if f"{col}_idx" in df.columns]
        X_cat = df[cat_idx_cols].fillna(0).values.astype(np.int64) if cat_idx_cols else np.zeros((len(df), 1), dtype=np.int64)

        return X_cont, X_cat

    def fit(self, fit_df: pd.DataFrame) -> "AutoencoderDetector":
        """
        Fit the autoencoder on the fit window.
        Steps:
          1. Build vocabularies from fit_df
          2. Encode categoricals to integer indices
          3. Scale continuous features
          4. Train the autoencoder
          5. Compute reconstruction error threshold (95th percentile)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        train_df = fit_df[fit_df["parse_success"] == True].copy()

        # Step 1 & 2: Vocabularies + encoding
        self.vocabs  = build_vocabularies(train_df)
        train_df     = encode_categoricals(train_df, self.vocabs)

        # Step 3: Fit continuous scaler
        self._cont_scaler = StandardScaler()
        cont_cols = [c for c in CONTINUOUS_FEATURES if c in train_df.columns]
        self._cont_scaler.fit(train_df[cont_cols].fillna(0).values)
        self.cont_cols = cont_cols

        X_cont, X_cat = self._prepare_arrays(train_df)

        if self._use_torch:
            self._fit_torch(X_cont, X_cat, train_df)
        else:
            self._fit_sklearn(X_cont)

        # Step 5: Compute threshold on fit window
        errors = self._compute_errors(train_df)
        self._recon_threshold = float(np.percentile(errors, 95))
        self._is_fitted = True

        print(f"[Autoencoder] Fitted. Reconstruction error threshold (95th pct): "
              f"{self._recon_threshold:.4f}")
        return self

    def _fit_torch(self, X_cont: np.ndarray, X_cat: np.ndarray, train_df: pd.DataFrame):
        """Full PyTorch training loop."""
        from sklearn.model_selection import train_test_split

        vocab_sizes = {col: len(v) for col, v in self.vocabs.items()}
        n_continuous = X_cont.shape[1]

        self.model = FraudAutoencoder(
            n_continuous=n_continuous,
            vocab_sizes=vocab_sizes,
        ).to(self.device)

        # Train / validation split within fit window (80/20)
        idx = np.arange(len(X_cont))
        tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

        train_ds = TransactionDataset(X_cont[tr_idx], X_cat[tr_idx])
        val_ds   = TransactionDataset(X_cont[val_idx], X_cat[val_idx])
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        cont_loss_fn = nn.MSELoss()
        cat_loss_fn  = nn.CrossEntropyLoss(ignore_index=0)  # ignore <UNK>

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        print(f"[Autoencoder] Training: {len(train_ds)} train, {len(val_ds)} val rows")

        for epoch in range(self.epochs):
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            for x_cont_batch, x_cat_batch in train_dl:
                x_cont_batch = x_cont_batch.to(self.device)
                x_cat_batch  = x_cat_batch.to(self.device)

                optimizer.zero_grad()
                recon_cont, recon_cat, _ = self.model(x_cont_batch, x_cat_batch)

                # Continuous reconstruction loss
                loss = self.cont_loss_weight * cont_loss_fn(recon_cont, x_cont_batch)

                # Categorical reconstruction losses
                for i, col in enumerate(CATEGORICAL_COLS):
                    if col in recon_cat:
                        loss += self.cat_loss_weight * cat_loss_fn(
                            recon_cat[col], x_cat_batch[:, i]
                        )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # ── Validation ──
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_cont_batch, x_cat_batch in val_dl:
                    x_cont_batch = x_cont_batch.to(self.device)
                    x_cat_batch  = x_cat_batch.to(self.device)
                    recon_cont, recon_cat, _ = self.model(x_cont_batch, x_cat_batch)

                    loss = self.cont_loss_weight * cont_loss_fn(recon_cont, x_cont_batch)
                    for i, col in enumerate(CATEGORICAL_COLS):
                        if col in recon_cat:
                            loss += self.cat_loss_weight * cat_loss_fn(
                                recon_cat[col], x_cat_batch[:, i]
                            )
                    val_loss += loss.item()

            avg_val = val_loss / len(val_dl)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | "
                      f"train_loss={train_loss/len(train_dl):.4f} | "
                      f"val_loss={avg_val:.4f}")

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                # Save best weights
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        print(f"[Autoencoder] Training complete. Best val_loss={best_val_loss:.4f}")

    def _fit_sklearn(self, X_cont: np.ndarray):
        """Sklearn MLP fallback — no embeddings."""
        self._sklearn_ae = SklearnAutoencoderFallback()
        # Note: sklearn fallback has its own internal scaler
        # We reuse X_cont that was already scaled by _cont_scaler
        self._sklearn_ae.model.fit(X_cont, X_cont)
        self._sklearn_ae._is_fitted = True
        print("[Autoencoder] sklearn fallback training complete.")

    def _compute_errors(self, df: pd.DataFrame) -> np.ndarray:
        """Compute per-row reconstruction error for a DataFrame."""
        enc_df = encode_categoricals(df, self.vocabs) if self.vocabs else df
        X_cont, X_cat = self._prepare_arrays(enc_df)

        if self._use_torch and self.model is not None:
            return self._torch_recon_error(X_cont, X_cat)
        elif hasattr(self, "_sklearn_ae"):
            return self._sklearn_ae.reconstruction_error(X_cont)
        else:
            return np.zeros(len(df))

    def _torch_recon_error(self, X_cont: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
        """Per-row reconstruction MSE from the PyTorch model."""
        self.model.eval()
        errors = []
        dataset = TransactionDataset(X_cont, X_cat)
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)

        with torch.no_grad():
            for x_cont_b, x_cat_b in loader:
                x_cont_b = x_cont_b.to(self.device)
                x_cat_b  = x_cat_b.to(self.device)
                recon_cont, _, _ = self.model(x_cont_b, x_cat_b)
                # Per-row MSE on continuous features
                mse = torch.mean((recon_cont - x_cont_b) ** 2, dim=1)
                errors.extend(mse.cpu().numpy())

        return np.array(errors)

    def get_per_feature_recon_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame of per-feature reconstruction errors.
        Used for explainability: which feature was hardest to reconstruct?
        """
        if not (self._use_torch and self.model is not None):
            return pd.DataFrame()

        enc_df = encode_categoricals(df, self.vocabs)
        X_cont, X_cat = self._prepare_arrays(enc_df)

        self.model.eval()
        all_errors = []
        dataset = TransactionDataset(X_cont, X_cat)
        loader  = DataLoader(dataset, batch_size=256, shuffle=False)

        with torch.no_grad():
            for x_cont_b, x_cat_b in loader:
                x_cont_b = x_cont_b.to(self.device)
                x_cat_b  = x_cat_b.to(self.device)
                recon_cont, _, _ = self.model(x_cont_b, x_cat_b)
                feat_err = (recon_cont - x_cont_b) ** 2   # (batch, n_features)
                all_errors.append(feat_err.cpu().numpy())

        feat_errors = np.concatenate(all_errors, axis=0)
        cont_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        return pd.DataFrame(feat_errors, columns=cont_cols[:feat_errors.shape[1]])

    def get_user_embeddings(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract learned user_id embedding vectors.
        Useful for visualizing user behavioral clusters.
        Returns DataFrame with user_id and embedding dimensions.
        """
        if not (self._use_torch and self.model is not None and self.vocabs):
            return None

        enc_df = encode_categoricals(df, self.vocabs)
        unique_users = enc_df[["user_id", "user_id_idx"]].drop_duplicates()

        indices = torch.LongTensor(unique_users["user_id_idx"].values).to(self.device)
        with torch.no_grad():
            embs = self.model.get_embeddings("user_id", indices).cpu().numpy()

        emb_df = pd.DataFrame(
            embs,
            columns=[f"user_emb_{i}" for i in range(embs.shape[1])]
        )
        emb_df.insert(0, "user_id", unique_users["user_id"].values)
        return emb_df

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score all transactions. Adds ae_recon_error and ae_score columns.
        """
        assert self._is_fitted

        # Encode categoricals using fit-window vocabulary (<UNK> for unseen)
        score_df = encode_categoricals(df, self.vocabs)
        errors   = self._compute_errors(score_df)

        out = df.copy()
        out["ae_recon_error"] = errors
        # Normalize to [0,1]
        mn, mx = errors.min(), errors.max()
        out["ae_score"] = (errors - mn) / (mx - mn + 1e-9)
        out["ae_is_anomaly"] = (errors > self._recon_threshold).astype(int)

        n = out["ae_is_anomaly"].sum()
        print(f"[Autoencoder] Flagged {n}/{len(out)} ({100*n/len(out):.1f}%) "
              f"above recon threshold {self._recon_threshold:.4f}")
        return out
