import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
TORCH_AVAILABLE = True


CATEGORICAL_EMBEDDING_CONFIG = {
    "user_id":  {"lag_col": None,            "emb_dim": 16},
    "city":     {"lag_col": "prev_city",     "emb_dim": 4},
    "device":   {"lag_col": "prev_device",   "emb_dim": 4},
    "txn_type": {"lag_col": "prev_txn_type", "emb_dim": 3},
    "currency": {"lag_col": "prev_currency", "emb_dim": 2},
}

# Total embedding dimension when all are concatenated
TOTAL_EMB_DIM = sum(
    cfg["emb_dim"] * (2 if cfg["lag_col"] else 1)
    for cfg in CATEGORICAL_EMBEDDING_CONFIG.values()
)  # = 16 + 4*2 + 4*2 + 3*2 + 2*2 = 16+8+8+6+4 = 42

# Special token indices (0=UNK, 1=FIRST_TXN)
UNK_IDX   = 0
FIRST_TXN_IDX = 1



def build_vocabularies(fit_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Build integer vocabularies for each categorical column.
    Index 0: <UNK> — unseen values at score time
    Index 1: <FIRST_TXN> — NaN lag values (first transaction for user)
    Index 2+: actual category values from fit window
    """
    vocabs = {}
    for col in CATEGORICAL_EMBEDDING_CONFIG:
        vals  = fit_df[col].dropna().astype(str).unique().tolist()
        vocab = {"<UNK>": 0, "<FIRST_TXN>": 1}
        for i, v in enumerate(sorted(vals), start=2):
            vocab[v] = i
        vocabs[col] = vocab
        lag_col = CATEGORICAL_EMBEDDING_CONFIG[col]["lag_col"]
        print(f"  Vocab '{col}' (shared with '{lag_col}'): "
              f"{len(vocab)} tokens (incl. UNK, FIRST_TXN)")
    return vocabs


def encode_categoricals(
    df: pd.DataFrame,
    vocabs: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    - Current categoricals: unknown value → UNK_IDX (0)
    - Lag categoricals: NaN (first txn) → FIRST_TXN_IDX (1),
                        unknown value    → UNK_IDX (0)
    """
    out = df.copy()
    for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
        vocab = vocabs[col]
        # Current
        out[f"{col}_idx"] = out[col].astype(str).map(
            lambda x, v=vocab: v.get(x, UNK_IDX)
        ).astype(int)
        # Lag (if it exists in the DataFrame)
        lag = cfg["lag_col"]
        if lag and lag in out.columns:
            out[f"{lag}_idx"] = out[lag].apply(
                lambda x, v=vocab: FIRST_TXN_IDX if pd.isna(x) else v.get(str(x), UNK_IDX)
            ).astype(int)
        elif lag:
            # Lag column doesn't exist yet — fill with FIRST_TXN
            out[f"{lag}_idx"] = FIRST_TXN_IDX
    return out



if TORCH_AVAILABLE:

    class TransactionDataset(Dataset):
        """
        categorical_indices shape: (n_cat_inputs,) where:
            positions 0..4 = current [user_id, city, device, txn_type, currency]
            positions 5..8 = lag     [prev_city, prev_device, prev_txn_type, prev_currency]
        """

        def __init__(
            self,
            X_cont: np.ndarray,
            X_cat:  np.ndarray,
        ):
            self.X_cont = torch.FloatTensor(X_cont)
            self.X_cat  = torch.LongTensor(X_cat)

        def __len__(self) -> int:
            return len(self.X_cont)

        def __getitem__(self, idx):
            return self.X_cont[idx], self.X_cat[idx]


    class _Block(nn.Module):
        """Fully-connected block: Linear → BatchNorm → LeakyReLU → Dropout."""
        def __init__(self, in_dim: int, out_dim: int,
                     dropout: float = 0.1, bn: bool = True):
            super().__init__()
            layers = [nn.Linear(in_dim, out_dim)]
            if bn: layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            if dropout > 0: layers.append(nn.Dropout(dropout))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)


    class FraudAutoencoder(nn.Module):
        def __init__(
            self,
            n_continuous: int,
            vocab_sizes:  Dict[str, int],
            bottleneck_dim: int = 16,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_continuous    = n_continuous
            self.bottleneck_dim  = bottleneck_dim
            self.cat_col_order   = list(CATEGORICAL_EMBEDDING_CONFIG.keys())

            # ── Shared embedding tables (one per categorical type) ─────────────
            self.embeddings = nn.ModuleDict()
            for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
                vocab_size = vocab_sizes[col]
                self.embeddings[col] = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=cfg["emb_dim"],
                    padding_idx=UNK_IDX,  # <UNK> has zero gradient
                )

            # ── Continuous normalisation (learnable scale/shift) 
            self.cont_bn = nn.BatchNorm1d(n_continuous)

            # Total input = continuous + current embeddings + lag embeddings
            input_dim = n_continuous + TOTAL_EMB_DIM

            # ── Encoder 
            self.encoder = nn.Sequential(
                _Block(input_dim, 256, dropout=dropout),
                _Block(256, 128, dropout=dropout),
                _Block(128,  64, dropout=dropout),
            )
            self.bottleneck   = nn.Linear(64, bottleneck_dim)
            self.bottleneck_act = nn.LeakyReLU(0.1)

            # ── Decoder 
            self.decoder = nn.Sequential(
                _Block(bottleneck_dim,  64, dropout=0.0),
                _Block(64,  128, dropout=0.0),
                _Block(128, 256, dropout=0.0),
            )

            # ── Output heads 
            # Continuous reconstruction
            self.out_continuous = nn.Linear(256, n_continuous)

            # One classification head per categorical
            self.out_categoricals = nn.ModuleDict({
                col: nn.Linear(256, vocab_sizes[col])
                for col in self.cat_col_order
            })

        def _embed_all(self, x_cat: torch.Tensor) -> torch.Tensor:
            parts = []
            col_idx = 0
            for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
                # Current embedding
                parts.append(self.embeddings[col](x_cat[:, col_idx]))
                col_idx += 1
                # Lag embedding (shares same table)
                if cfg["lag_col"]:
                    parts.append(self.embeddings[col](x_cat[:, col_idx]))
                    col_idx += 1
            return torch.cat(parts, dim=1)  # (batch, TOTAL_EMB_DIM)

        def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
            # Normalise continuous input
            x_cont_norm = self.cont_bn(x_cont)

            # Embed categoricals
            x_emb = self._embed_all(x_cat)

            # Concatenate and encode
            x_in      = torch.cat([x_cont_norm, x_emb], dim=1)
            encoded   = self.encoder(x_in)
            z         = self.bottleneck_act(self.bottleneck(encoded))

            # Decode
            decoded = self.decoder(z)

            # Reconstruction heads
            recon_cont = self.out_continuous(decoded)
            recon_cat  = {
                col: self.out_categoricals[col](decoded)
                for col in self.cat_col_order
            }

            return recon_cont, recon_cat, z

        def get_latent(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            """Extract bottleneck representation only (for visualisation)."""
            with torch.no_grad():
                x_norm = self.cont_bn(x_cont)
                x_emb  = self._embed_all(x_cat)
                x_in   = torch.cat([x_norm, x_emb], dim=1)
                return self.bottleneck_act(self.bottleneck(self.encoder(x_in)))



class AutoencoderDetector:
    def __init__(
        self,
        epochs: int = 60,
        batch_size: int = 128,
        lr: float = 1e-3,
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
        cont_loss_weight: float = 0.5,
        cat_loss_weight: float = 0.1,
        patience: int = 12,
        device: Optional[str] = None,
        continuous_features: Optional[List[str]] = None,
    ):
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.bottleneck_dim   = bottleneck_dim
        self.dropout          = dropout
        self.cont_loss_weight = cont_loss_weight
        self.cat_loss_weight  = cat_loss_weight
        self.patience         = patience
        self.cont_feats       = continuous_features

        self._vocabs          = None
        self._scaler          = None
        self._model           = None
        self._sklearn_ae      = None
        self._is_fitted       = False
        self._threshold       = None   # 95th pct recon error on fit window
        self._train_error_min = None
        self._train_error_max = None
        self._use_torch       = TORCH_AVAILABLE
        self._actual_cont_feats: List[str] = []  # features actually present in data

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[AutoencoderDetector] Backend: PyTorch | Device: {self.device}")

    def _resolve_continuous_features(self, df: pd.DataFrame) -> List[str]:
        """Return features from config list that actually exist in df, non-NaN > 50%."""
        available = []
        for f in self.cont_feats:
            if f in df.columns:
                nan_rate = df[f].isna().mean()
                if nan_rate < 0.50:
                    available.append(f)
        return available

    def _get_X_cont(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract and scale continuous feature matrix. NaN → 0 after scaling."""
        X = df[self._actual_cont_feats].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
        if fit:
            return self._scaler.fit_transform(X).astype(np.float32)
        return self._scaler.transform(X).astype(np.float32)

    def _get_X_cat(self, df: pd.DataFrame) -> np.ndarray:
        enc = encode_categoricals(df, self._vocabs)
        cols = []
        for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
            cols.append(enc[f"{col}_idx"].values)
            if cfg["lag_col"]:
                cols.append(enc[f"{cfg['lag_col']}_idx"].values)
        return np.stack(cols, axis=1).astype(np.int64)

    def fit(self, fit_df: pd.DataFrame) -> "AutoencoderDetector":
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        train_df = fit_df.copy()
        print(f"\n[AutoencoderDetector] Fitting on {len(train_df)} rows...")

        # Step 1 — Continuous features
        self._scaler            = StandardScaler()
        self._actual_cont_feats = self._resolve_continuous_features(train_df)
        print(f"  Continuous features : {len(self._actual_cont_feats)}")

        # Step 2 — Vocabularies
        print("  Building vocabularies...")
        self._vocabs = build_vocabularies(train_df)

        # Step 3 & 4 — Encode and scale
        X_cont = self._get_X_cont(train_df, fit=True)
        X_cat  = self._get_X_cat(train_df)

        if self._use_torch:
            self._fit_torch(X_cont, X_cat)

        # Step 6 — Threshold
        # Step 6 — Threshold and training score range
        errors = self._compute_errors_from_arrays(X_cont, X_cat)
        self._threshold       = float(np.percentile(errors, 95))
        self._train_error_min = float(errors.min())
        self._train_error_max = float(errors.max())
        self._is_fitted       = True

        print(f"  Recon error threshold (95th pct): {self._threshold:.6f}")
        print(f"  Training error range: [{self._train_error_min:.6f}, {self._train_error_max:.6f}]")
        return self

    def _fit_torch(self, X_cont: np.ndarray, X_cat: np.ndarray):
        """Full PyTorch training with early stopping."""
        n_continuous = X_cont.shape[1]
        vocab_sizes  = {col: len(v) for col, v in self._vocabs.items()}

        self._model = FraudAutoencoder(
            n_continuous=n_continuous,
            vocab_sizes=vocab_sizes,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        ).to(self.device)

        # 80/20 split within fit window for early stopping
        idx     = np.arange(len(X_cont))
        tr_idx, val_idx = self._split_train_val(idx)

        tr_ds   = TransactionDataset(X_cont[tr_idx], X_cat[tr_idx])
        val_ds  = TransactionDataset(X_cont[val_idx], X_cat[val_idx])
        tr_dl   = DataLoader(tr_ds,  batch_size=self.batch_size, shuffle=True,  drop_last=True)
        val_dl  = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer    = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        cont_loss_fn = nn.MSELoss()
        cat_loss_fn  = nn.CrossEntropyLoss(ignore_index=UNK_IDX)

        best_val     = float("inf")
        patience_cnt = 0
        best_state   = None

        print(f"  Training: {len(tr_ds)} train / {len(val_ds)} val | "
              f"epochs={self.epochs} | batch={self.batch_size}")

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────
            self._model.train()
            tr_loss = 0.0
            for xc, xk in tr_dl:
                xc, xk = xc.to(self.device), xk.to(self.device)
                optimizer.zero_grad()
                rc, rk, _ = self._model(xc, xk)
                loss = self._compute_loss(xc, xk, rc, rk, cont_loss_fn, cat_loss_fn)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                tr_loss += loss.item()

            # ── Validate ──────────────────────────────────────────────────
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xc, xk in val_dl:
                    xc, xk = xc.to(self.device), xk.to(self.device)
                    rc, rk, _ = self._model(xc, xk)
                    val_loss += self._compute_loss(
                        xc, xk, rc, rk, cont_loss_fn, cat_loss_fn
                    ).item()

            avg_val = val_loss / max(len(val_dl), 1)
            scheduler.step(avg_val)

            if (epoch % 10) == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs} | "
                      f"train={tr_loss/len(tr_dl):.5f} | val={avg_val:.5f}")

            # ── Early stopping ────────────────────────────────────────────
            if avg_val < best_val - 1e-6:
                best_val     = avg_val
                patience_cnt = 0
                best_state   = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (best val={best_val:.5f})")
                    break

        if best_state:
            self._model.load_state_dict(best_state)
        print(f"  Training complete. Best val_loss={best_val:.5f}")

    def _compute_loss(self, xc, xk, rc, rk, cont_fn, cat_fn):
        """Weighted combined loss: MSE for continuous + CrossEntropy for categoricals."""
        loss = self.cont_loss_weight * cont_fn(rc, xc)
        # xk column order: [user_id, city, prev_city, device, prev_device,
        #                   txn_type, prev_txn_type, currency, prev_currency]
        col_idx = 0
        for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
            targets = xk[:, col_idx]
            loss   += self.cat_loss_weight * cat_fn(rk[col], targets)
            col_idx += 1
            if cfg["lag_col"]:
                col_idx += 1   # skip lag index for loss (current only)
        return loss

    def _split_train_val(self, idx: np.ndarray, val_frac: float = 0.20):
        """Stratified 80/20 split (random, seeded)."""
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        split = int(len(idx) * (1 - val_frac))
        return idx[:split], idx[split:]


    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._is_fitted, "Must call fit() before score()"
        out = df.copy()

        enc_df = encode_categoricals(df, self._vocabs)
        X_cont = self._get_X_cont(enc_df, fit=False)
        X_cat  = self._get_X_cat(enc_df)

        errors = self._compute_errors_from_arrays(X_cont, X_cat)

        out["ae_recon_error"] = errors

        # Normalize using TRAINING error range, not batch range
        # clip(0, 1): errors larger than training max → 1.0 (very anomalous)
        out["ae_score"] = np.clip(
            (errors - self._train_error_min) / (self._train_error_max - self._train_error_min + 1e-9),
            0.0, 1.0
        )
        out["ae_is_anomaly"]  = (errors > self._threshold).astype(int)

        n_flag = out["ae_is_anomaly"].sum()
        print(f"[AutoencoderDetector] Scored {len(out)} rows. "
              f"Flagged {n_flag} ({100*n_flag/len(out):.1f}%) "
              f"above threshold {self._threshold:.6f}")
        return out

    def _compute_errors_from_arrays(
        self, X_cont: np.ndarray, X_cat: np.ndarray
    ) -> np.ndarray:
        """Compute per-row reconstruction MSE."""
        if self._use_torch and self._model is not None:
            return self._torch_recon_error(X_cont, X_cat)
        return np.zeros(len(X_cont))

    def _torch_recon_error(
        self, X_cont: np.ndarray, X_cat: np.ndarray
    ) -> np.ndarray:
        """Batch reconstruction MSE from PyTorch model."""
        self._model.eval()
        errors  = []
        dataset = TransactionDataset(X_cont, X_cat)
        loader  = DataLoader(dataset, batch_size=512, shuffle=False)

        with torch.no_grad():
            for xc, xk in loader:
                xc, xk = xc.to(self.device), xk.to(self.device)
                rc, _, _ = self._model(xc, xk)
                mse = torch.mean((rc - xc) ** 2, dim=1)
                errors.extend(mse.cpu().numpy())

        return np.array(errors, dtype=np.float32)

    def get_per_feature_recon_error(self, df: pd.DataFrame) -> pd.DataFrame:
        if not (self._use_torch and self._model is not None):
            return pd.DataFrame()

        enc_df = encode_categoricals(df, self._vocabs)
        X_cont = self._get_X_cont(enc_df, fit=False)
        X_cat  = self._get_X_cat(enc_df)

        self._model.eval()
        all_errors = []
        dataset = TransactionDataset(X_cont, X_cat)
        loader  = DataLoader(dataset, batch_size=512, shuffle=False)

        with torch.no_grad():
            for xc, xk in loader:
                xc, xk = xc.to(self.device), xk.to(self.device)
                rc, _, _ = self._model(xc, xk)
                feat_err = (rc - xc) ** 2   # (batch, n_features)
                all_errors.append(feat_err.cpu().numpy())

        feat_errors = np.concatenate(all_errors, axis=0)
        return pd.DataFrame(feat_errors, columns=self._actual_cont_feats,
                            index=df.index)

    def get_categorical_recon_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        if not (self._use_torch and self._model is not None):
            return pd.DataFrame()

        enc_df = encode_categoricals(df, self._vocabs)
        X_cont = self._get_X_cont(enc_df, fit=False)
        X_cat  = self._get_X_cat(enc_df)

        self._model.eval()
        all_correct = {col: [] for col in CATEGORICAL_EMBEDDING_CONFIG}

        dataset = TransactionDataset(X_cont, X_cat)
        loader  = DataLoader(dataset, batch_size=512, shuffle=False)

        col_idx = 0
        col_positions = {}
        for col, cfg in CATEGORICAL_EMBEDDING_CONFIG.items():
            col_positions[col] = col_idx
            col_idx += 1
            if cfg["lag_col"]:
                col_idx += 1

        with torch.no_grad():
            for xc, xk in loader:
                xc, xk = xc.to(self.device), xk.to(self.device)
                _, rk, _ = self._model(xc, xk)
                for col, logits in rk.items():
                    predicted = logits.argmax(dim=1)
                    actual    = xk[:, col_positions[col]]
                    correct   = (predicted == actual).cpu().numpy()
                    all_correct[col].extend(correct.tolist())

        result = pd.DataFrame({
            f"{col}_recon_correct": vals
            for col, vals in all_correct.items()
        }, index=df.index)
        return result

    def get_user_embeddings(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not (self._use_torch and self._model is not None):
            return None

        enc_df   = encode_categoricals(df, self._vocabs)
        unique   = enc_df[["user_id", "user_id_idx"]].drop_duplicates()
        indices  = torch.LongTensor(unique["user_id_idx"].values).to(self.device)

        self._model.eval()
        with torch.no_grad():
            embs = self._model.embeddings["user_id"](indices).cpu().numpy()

        emb_df = pd.DataFrame(
            embs,
            columns=[f"user_emb_{i}" for i in range(embs.shape[1])]
        )
        emb_df.insert(0, "user_id", unique["user_id"].values)
        return emb_df

    def get_latent_representations(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if not (self._use_torch and self._model is not None):
            return None

        enc_df = encode_categoricals(df, self._vocabs)
        X_cont = self._get_X_cont(enc_df, fit=False)
        X_cat  = self._get_X_cat(enc_df)

        self._model.eval()
        latents  = []
        dataset  = TransactionDataset(X_cont, X_cat)
        loader   = DataLoader(dataset, batch_size=512, shuffle=False)

        with torch.no_grad():
            for xc, xk in loader:
                xc, xk = xc.to(self.device), xk.to(self.device)
                z = self._model.get_latent(xc, xk)
                latents.append(z.cpu().numpy())

        return np.concatenate(latents, axis=0)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_count(self) -> int:
        return len(self._actual_cont_feats)

    @property
    def continuous_features(self) -> List[str]:
        return list(self._actual_cont_feats)

    @property
    def vocab_sizes(self) -> Optional[Dict[str, int]]:
        if self._vocabs is None:
            return None
        return {col: len(v) for col, v in self._vocabs.items()}

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold
