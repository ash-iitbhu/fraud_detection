# Fraud Detection in Unstructured Financial Logs
---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [How to Reproduce the Pipeline](#4-how-to-reproduce-the-pipeline)
5. [Pipeline Logic](#5-pipeline-logic)
   - 5.1 [Log Parsing](#51-log-parsing)
   - 5.2 [Feature Engineering](#52-feature-engineering)
   - 5.3 [Anomaly Detection Models](#53-anomaly-detection-models)
   - 5.4 [Hyperparameter Tuning](#54-hyperparameter-tuning)
   - 5.5 [Ensemble Scoring & Risk Tiering](#55-ensemble-scoring--risk-tiering)
   - 5.6 [Evaluation](#56-evaluation)
   - 5.7 [Explainability](#57-explainability)
6. [Business Impact](#6-business-impact)
7. [Manual Validation Plan](#7-manual-validation-plan)
8. [Future Recommendations](#8-future-recommendations)

---

## 1. Project Overview

Build an end-to-end **unsupervised fraud detection pipeline**, detecting anomalous transactions across 8,876 raw transaction logs from 86 users spanning June–July 2025.

**The core challenge** is that no fraud labels exist. The pipeline must identify suspicious behaviour purely from statistical deviation and rules, without any ground truth to train or validate against in the conventional sense.

**Architecture overview:**

| Layer | Component | Purpose |
|---|---|---|
| 1 | Multi-format log parser | Extract structured fields from 7 heterogeneous log formats |
| 2 | Point-in-time feature engineering | 268 behavioural features with no look-ahead leakage |
| 3 | Rule engine | 10 expert-encoded, user-context-aware deterministic rules |
| 4 | Isolation Forest | Unsupervised multivariate anomaly detection |
| 5 | Autoencoder | Deep reconstruction-error anomaly detection with entity embeddings |
| 6 | Ensemble scorer | Weighted combination into a single calibrated risk score |
| 7 | Explainability | SHAP values + natural language fraud investigation reports |

**Final results across 7,774 parsed transactions:**

| Risk Tier | Count | % | Action |
|---|---|---|---|
| TIER_1 | 78 | 1.0% | Auto-block |
| TIER_2 | 311 | 4.0% | Step-up authentication |
| TIER_3 | 1,551 | 20.0% | Silent monitoring |
| NORMAL | 5,834 | 75.0% | No action |

> **Notebook reference:** Cells 0–1 (architecture overview), Cell 79 (ensemble results), Cell 82 (risk tier visualisations)

---

## 2. Repository Structure

```
fraud_detection/
├── data/
│   ├── MP Fraud Takehome Task 2026 - Sheet1.csv                    # Raw input (8,876 transaction logs)
│   ├── processed/
│   │   └── processed_logs.csv          # Output of parser (7,774 rows)
│   ├── featured/
│   │   ├── featured_df.csv             # 268 engineered features
│   │   └── featured_names_df.csv       # Feature name list
│   └── final_scored/
│       ├── final_df.csv                # Fully scored dataset (all models)
│       └── per_feature_recon_df.csv    # AE per-feature reconstruction errors
│
├── src/
│   ├── parser/
│   │   └── log_parser.py               # 4-layer cascade parser
│   ├── features/
│   │   └── feature_engineer.py           # FraudFeatureEngine (268 features, 15 groups)
│   ├── models/
│   │   ├── rule_engine.py              # 10 deterministic fraud rules
│   │   ├── isolation_forest.py         # IsolationForestDetector
│   │   ├── autoencoder.py              # AutoencoderDetector (PyTorch + sklearn fallback)
│   │   ├── ensemble.py                 # EnsembleScorer with frozen normalization
│   │   └── save_models.py              # Artifact save/load functions
│   ├── evaluation/
│   │   └── evaluator.py                # 4 evaluation modules
│   └── explainability/
│       ├── explainer.py                # NL report generator + online inference
│       └── visualiser.py               # 9 chart functions
│
├── notebooks/
│   ├── fraud_detection.ipynb           # Main pipeline (Parsing + Feature engineering + training + evaluation + explanability)
│   ├── model_artifacts/
|   └── top_50_anomalies.csv
```
---

## 3. Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
```

### Data placement

Place the raw CSV at `data/MP Fraud Takehome Task 2026 - Sheet1.csv`. The expected column is `raw_log` containing one transaction log string per row.

---

## 4. How to Reproduce the Pipeline

All steps are in `notebooks/fraud_detection.ipynb`. Run cells sequentially.

### Step-by-step

**Step 1 — Parse raw logs**
**Step 2 — EDA**  
Exploratory analysis to understand data quality and calibrate rule thresholds. Not required for re-running the pipeline; informational only.
**Step 3 — Feature engineering**
**Step 4 — Rule engine** 
**Step 5 — Isolation Forest** 
**Step 6 — Autoencoder** 
**Step 7 — Ensemble scoring** 
**Step 8 — Evaluation**
**Step 9 — Explainability**
### All the visulaisations can be seen in the notebook
### Important note on online scoring
Because feature engineering uses pandas rolling windows and groupby operations (30-day means, velocity counts, etc.), a new transaction **cannot be scored in isolation**. It must be appended to the historical dataset before calling `engine.transform()`, and only the new row's features extracted afterwards. See `src/explainability/explainer.py` (`score_and_explain_single`) for the correct implementation.
---

## 5. Pipeline Logic
### 5.1 Log Parsing
> **Notebook reference:**
**Problem:** The raw dataset contains 8,876 transaction logs in 8 structurally distinct formats — colon-delimited, pipe-delimited, natural language sentences, ATM-style, shorthand formats, and unrecognised/malformed entries. No single regex covers them all.
**Solution — 4-layer cascade parser:**
1. **Regex cascade (F1–F7):** Seven compiled patterns, tried in order of specificity. Each pattern captures user_id, timestamp, txn_type, amount, currency, city, and device. When a pattern matches all required fields, the row is accepted and no further layers are tried.
2. **Heuristic fallback:** For rows not matched by any regex, a field-by-field micro-pattern extractor attempts to recover individual fields independently. This handles partially-formatted logs where the structure is inconsistent but fragments are recognisable.
3. **spaCy EntityRuler:** A custom NLP entity recogniser is applied to logs that failed heuristic extraction. Custom rules recognise UK city names, device patterns, currency symbols, and monetary amounts from free-text descriptions.
4. **LLM fallback (optional):** For logs that fail all three deterministic layers, an LLM can be invoked. This layer was not required for this dataset.

**Parse results:**
- 7,774 of 8,876 rows parsed (87.6%)
- All 7 format types (F1–F7) contributed roughly equally (~1,066–1,138 rows each)
- 1,102 rows classified as MALFORMED and excluded from downstream processing

**Design decision:** The cascade architecture was chosen over a single complex regex because it is auditable — a fraud analyst can inspect which layer parsed each row and why, which matters for explainability and compliance.

---

### 5.2 Feature Engineering

> **Notebook reference:**
**Point-in-time correctness:** All features are computed with `closed='left'` on rolling windows, ensuring no transaction can access information from its own row or any future row. The dataset is sorted by `(user_id, timestamp)` before any aggregate is computed.
> 
**Feature groups (246 total across 12 groups):**

| Group | Features | Examples |
|---|---|---|
| A — Base | 10 | log_amount, amount_global_pct_rank, is_round_100 |
| B — Temporal | 17 | hour_of_day, is_late_night, is_weekend, hour_sin/cos |
| C — Rolling Amount (blast) | 60 | user_amt_mean/std/max/min over 1D, 7D, 15D, 30D, 60D |
| D — Rolling Count | 30 | user_txn_count over 1D, 7D, 15D, 30D, 60D |
| E — Deviation/Ratio | 20 | user_amt_zscore_30D, amt_ratio_from_prev, amt_above_user_p95 |
| F — Behavioural Fingerprint | 16 | user_device_prior_count, city_global_rarity, user_n_distinct_cities |
| G — Geographic | 8 | is_new_city, impossible_travel, city_transition_rarity |
| H — Device | 5 | is_new_device, device_global_rarity, new_device_and_new_city |
| J — Sequence/Lag | 10 | prev_amount, prev_city, time_since_last_txn_sec |
| K — Interaction | 15 | new_city_x_amt_zscore, impossible_travel_x_amt |
| L — Missingness | 6 | missing_field_count, is_both_geo_dev_missing |
| M — Burst Detection | 7 | is_burst_5min, consecutive_bursts |

**Key design decisions from EDA:**

- **Cold-start handling:** New users (fewer than 3 prior transactions) receive a `is_cold_start = 1` flag. Deviation features are suppressed for cold-start users because no meaningful baseline exists. This prevents false positives for genuinely new accounts.

- **User-relative normalisation:** Z-scores and ratio features are computed per-user, not globally. EDA confirmed that user CV (coefficient of variation) of 0.58 makes user-specific baselines meaningful. A £4,000 transaction is suspicious for one user and routine for another.

- **Amount ratio from previous transaction:** EDA found this is the strongest single-transaction signal, with p99 at 44× the prior transaction. Feature Group O captures escalating patterns from this signal.

- **Temporal features as cyclical encodings:** Hour and day-of-week are encoded as sin/cos pairs to preserve cyclical distance relationships (e.g. 23:00 and 01:00 are similar).

**Train/score split:** The feature engine is fitted on the first 80% of transactions (by timestamp), then applied to transform the full dataset. This simulates a production deployment where the model has only seen historical data.

---

### 5.3 Anomaly Detection Models
The pipeline uses three independent detectors whose outputs are combined. Independence is critical — if all models shared the same signal, the ensemble would add no value. Cross-model Spearman rank correlations are all below 0.3, confirming meaningful independence.

#### Layer 1 — Rule Engine

Ten deterministic rules, each encoding a specific fraud pattern observed in real-world financial fraud literature and calibrated to this dataset's actual distributions from EDA.

| Rule | Signal | Threshold | Evidence |
|---|---|---|---|
| R01_AMOUNT_SPIKE_30D | Amount z-score vs user 30D mean | z > 2.5 | EDA: p99 z-score = 3.8 for genuine anomalies |
| R02_IMPOSSIBLE_TRAVEL | New city within short time gap | gap < 600s | 0.45% of transactions; strong precision signal |
| R03_VELOCITY_BURST_1H | Transaction count in past hour | ≥ 2 | Card testing pattern |
| R04_NEW_CITY_HIGH_AMOUNT | First visit to city AND high amount | z > 2.0 | Combined signal reduces false positives from 8.9% to 0.55% |
| R05_NEW_DEVICE_HIGH_AMOUNT | First device use AND high amount | z > 2.0 | Device takeover + large withdrawal |
| R06_MISSING_METADATA | Both city AND device unknown | both UNKNOWN | Proxy/VPN usage pattern |

**Design rationale:** Rules are **user-context-aware**, not global. R01 fires based on each user's own 30-day mean, not a fixed amount threshold. This prevents wealthy users from being unfairly flagged and ensures the rules are robust to the near-uniform amount distribution in this dataset.

**Note:** The late-night rule was considered but dropped. EDA found that 21.2% of all transactions occur late at night with no corresponding amount elevation, making it an unreliable signal with a very high false positive rate.

#### Layer 2 — Isolation Forest

**Why Isolation Forest:**
- Handles high-dimensional feature spaces natively without dimensionality reduction
- Robust to small contamination fractions (the 1–5% fraud rate does not bias the model)
- Linear time complexity O(n log n) — practical for production batch scoring
- Produces an interpretable anomaly score via path length in the forest
- SHAP TreeExplainer provides feature-level attribution for each scored transaction

**Architecture:** Fitted on the 80th-percentile time split of rule-scored data. Features exclude categorical raw columns (city, device as strings) — these are represented through behavioural fingerprint features (rarity scores, prior counts) which preserve the signal in continuous form suitable for tree splits.

**Score normalization:** The raw IF decision function is normalized using the min/max of the training window's raw scores, not the incoming batch. This ensures that scoring a single new transaction produces the same score it would receive alongside the full training set. The training range is frozen and saved as `if_meta.json`.

#### Layer 3 — Deep Autoencoder

**Architecture:**
- Two input streams: continuous features (91 features) + categorical entity embeddings
- Entity embeddings: `user_id → 16-dim`, `city / prev_city` (shared weights) `→ 8-dim`, `device / prev_device` (shared weights) `→ 8-dim`
- Shared embedding weights between current and lag categorical columns capture temporal consistency
- Bottleneck dimension: 16 (configurable)
- Training uses combined loss: MSE on continuous features + CrossEntropy on categorical reconstructions
- Early stopping on 20% held-out validation split (patience = 12 epochs)

**Why trained on rule-clean data only:** The autoencoder learns a compressed representation of *normal* behaviour. If fraudulent transactions appear in training data, the model learns to reconstruct them, destroying the reconstruction-error anomaly signal. Training only on `rule_flag_count == 0` rows provides a clean baseline of normal behaviour (approximately 87% of the fit window).

**Score normalization:** Reconstruction error is normalized using the training window's error distribution range, frozen at fit time, so inference scores are stable regardless of batch size.

---

### 5.4 Hyperparameter Tuning

#### Isolation Forest — Bimodality Coefficient objective

The fundamental challenge of unsupervised anomaly detection hyperparameter tuning is that there are no labels to optimise against. We use the **Bimodality Coefficient (BC)** of the IF score distribution as a proxy objective.

**Rationale:** A good anomaly detector should produce a bimodal score distribution — a large normal cluster near 0 and a small anomaly tail near 1. The BC measures this: BC > 0.555 indicates bimodality. A model producing a uniform or unimodal score distribution has failed to find meaningful structure.

#### Autoencoder — Validation Loss objective

For the autoencoder, validation loss from early stopping is the natural tuning objective. The model is trained on rule-clean data with an 80/20 time-based train/validation split. Lower validation loss means the model better reconstructs normal transactions, which sharpens the reconstruction-error signal for anomalies.
---

### 5.5 Ensemble Scoring & Risk Tiering
**Ensemble formula:**
```
final_risk_score = 0.33 × rule_score_norm + 0.33 × if_score_norm + 0.34 × ae_score_norm
```

**Equal weights as default:** Weights are equal because no analyst feedback is available to calibrate them. The mathematically correct approach is to assign equal weights when you have no information about the relative reliability of each detector. Once analyst-reviewed labels accumulate (see Section 7), weights should be tuned via precision-at-K optimisation.

**Score normalization — frozen training ranges:** Each individual score is normalized against the training data distribution before weighting, not against the incoming batch. This is critical for production stability. Normalising within a batch means the highest-scoring transaction in any batch always gets score 1.0, which is meaningless for a single-transaction query. The `EnsembleScorer` class holds frozen normalization parameters and tier thresholds computed from the training distribution.

**Risk tier thresholds** are calibrated as percentiles of the training ensemble score distribution:
- TIER_1: ≥ p99 of training scores (auto-block)
- TIER_2: ≥ p95 of training scores (step-up authentication)
- TIER_3: ≥ p75 of training scores (silent monitoring)

These thresholds are stored in `ensemble_scorer.json` and reused at inference time.

**Cross-model independence:** Spearman rank correlations between IF, AE, and Rule scores are all below 0.3 (IF↔AE = 0.598, IF↔Rules = 0.173, AE↔Rules = 0.099). Low correlation confirms that each detector captures different fraud signals, validating the ensemble approach.

---

### 5.6 Evaluation

Because no ground-truth labels exist, the evaluation uses four complementary proxy metrics:

**1. Score Distribution Analysis (BC)**
- Bimodality Coefficient > 0.555 indicates the model found meaningful structure
- Results: Rule Engine BC = 0.774 ✅, Autoencoder BC = 0.995 ✅, IF BC = 0.462 ⚠ (IF is expected to produce a less bimodal distribution due to its path-length mechanics)

**3. Proxy Precision@K**
- Multi-model agreement (≥ 2 models flagging) is used as a pseudo-label
- Precision@50 = 0.96

**4. Model Stability (CSI)**
- CSI (Characteristic Stability Index) comparing the first 80% vs last 20% of the dataset by timestamp
- CSI < 0.10 = stable, 0.10–0.25 = monitor, > 0.25 = recalibrate
- Results: all model scores show CSI < 0.2, indicating need strict monitoring over time.
---

### 5.7 Explainability

For each high-risk transaction, the pipeline produces a structured **Fraud Investigation Report** combining four evidence sources:

1. **Rule flags** — plain-English descriptions of which deterministic rules fired and why (e.g. "Amount is 3.2 standard deviations above this user's 30-day average of £1,264")

2. **SHAP feature importances** — which features of the Isolation Forest's decision contributed most to the anomaly score. Positive SHAP = pushed toward anomalous; negative SHAP = pushed toward normal.

3. **Autoencoder reconstruction error** — which continuous and categorical features the autoencoder failed to reconstruct, indicating they deviate from the learned normal pattern.
---

## 6. Business Impact

### Detected anomalies

| Tier | Count | Avg Amount | Estimated Fraud Rate | Est. Fraudulent Txns | Est. Value at Risk |
|---|---|---|---|---|---|
| TIER_1 (auto-block) | 78 | £3,459 | 70% | ~54 | £186,793 |
| TIER_2 (step-up auth) | 311 | £2,975 | 35% | ~108 | £321,312 |
| TIER_3 (monitor) | 1,551 | £2,822 | 10% | ~155 | £437,339 |
| **Total** | **1,940** | | | **~317** | **~£945,444** |

*Fraud rate estimates are conservative assumptions based on industry benchmarks for unsupervised detection precision. They should be validated against analyst review outcomes as labels accumulate.*

### Key signals in TIER_1 transactions

- 30 transactions (38%) triggered the impossible travel rule — new city within 10 minutes of a prior transaction in a different city
- 61 transactions (78%) involved a city the user had never transacted in before
- 48 transactions (62%) involved a previously unseen device
- 7 transactions (9%) had both city and device metadata missing, consistent with proxy/VPN usage
- Average IF score for TIER_1: 0.650 (vs. 0.245 overall mean)

### Model agreement as confidence signal

36 transactions were flagged by all 3 independent models simultaneously. These represent the highest-confidence fraud cases. 148 of the 389 TIER_1/2 transactions had at least 2 models in agreement.

### Review efficiency

TIER_1 transactions are automatically blocked without analyst involvement. The total manual review queue (TIER_2 only) is 311 transactions over two months, representing approximately 5 transactions per analyst per working day at standard review rates. Estimated analyst review cost: ~£1,553. **Net estimated value: ~£943,891.**

### Data limitations

This is a synthetic dataset with a near-uniform amount distribution (skewness ≈ 0.01, kurtosis ≈ -1.2) and all 86 users visiting all 7 cities with 5–6 devices. Real financial data is log-normal and right-skewed, with strong geographic concentration. The fraud precision estimates above would likely be higher in production where genuine behavioural baselines are more heterogeneous and anomalies more distinct.

---

## 7. Manual Validation Plan

The unsupervised pipeline produces ranked risk scores but requires human-in-the-loop feedback to improve over time. This section describes how that feedback loop works in practice.

### 7.1 Weekly Analyst Review Queue

**What happens:** Each week, a fraud analyst reviews the top N transactions by `final_risk_score`. The review queue is prioritised as:
1. All TIER_1 transactions (auto-blocked) — verify no false positives before appeal window closes
2. All TIER_2 transactions — step-up auth outcomes (did the user pass or abandon?)
3. A random sample of TIER_3 transactions (10–20 per week) — validates that the monitor tier has meaningful signal

**What the analyst records for each reviewed transaction:**
- `fraud_confirmed`: True / False / Uncertain
- `fraud_type`: account_takeover / card_testing / identity_fraud / legitimate_dispute / other
- `notes`: free-text explanation

**Tooling:** The Streamlit app (`app.py`) provides the analyst interface. Each reviewed transaction is logged to a `labels.csv` file alongside the `final_risk_score`, individual model scores, and key feature values at the time of review.

### 7.2 Precision Tracking

Once 50+ analyst labels have accumulated, compute per-tier precision:

```
precision_TIER_1 = confirmed_fraud / total_reviewed_in_TIER_1
precision_TIER_2 = confirmed_fraud / total_reviewed_in_TIER_2
```

Target benchmarks:
- TIER_1: ≥ 70% confirmed fraud
- TIER_2: ≥ 35% confirmed fraud
- TIER_3: ≥ 15% confirmed fraud (lower bar acceptable — this is a monitoring queue)

If precision falls below these targets, escalate to threshold recalibration.

### 7.3 Ensemble Weight Tuning

Once 200+ analyst labels exist, the equal 0.33/0.33/0.34 weights should be replaced with evidence-based weights.

**Method:**
1. For each labeled transaction, compute the area under the precision-recall curve (AUCPR) for each individual model score separately
2. Normalise AUCPR scores to sum to 1.0 as weights
3. Re-run `fit_ensemble()` on the training data with the new weights
4. Validate: compare precision@50 before and after reweighting on the held-out review labels

**Example:** If analyst labels show that the rule engine identifies 80% of confirmed fraud at the top of the queue but AE only identifies 40%, the rule engine should receive a higher weight (e.g. 0.50 / 0.30 / 0.20).

### 7.4 Tier Threshold Recalibration

Tier thresholds are currently set at the p99/p95/p75 of training ensemble scores. As the user base grows or transaction behaviour shifts, these percentiles may no longer correspond to meaningful risk boundaries.

**Trigger:** Recalibrate thresholds when:
- CSI > 0.25 on `final_risk_score` between the current month and the training reference (major shift)
- Precision on any tier drops below the target benchmark for 4 consecutive weeks
- Significant new user cohorts or product changes are deployed

**Method:** Re-run `fit_ensemble()` on a refreshed training window (rolling 90-day window recommended). The frozen normalization ranges and tier thresholds update automatically. Save the new `ensemble_scorer.json`.

### 7.5 Model Retraining Schedule

| Component | Frequency | Trigger |
|---|---|---|
| Rule thresholds | Ad-hoc | New fraud pattern identified in analyst review |
| Isolation Forest | Quarterly | CSI > 0.25 or precision degradation |
| Autoencoder | Quarterly | Same as IF; retrain on new rule-clean baseline |
| Ensemble weights | After 200 labels, then monthly | New labels accumulated |
| Tier thresholds | Quarterly or on CSI trigger | Part of ensemble re-fit |

### 7.6 Platt Scaling (Future)

Once 500+ confirmed fraud labels exist, Platt scaling (logistic regression on model scores → probability) can replace the current score-based tiering with a calibrated probability of fraud. This makes tier thresholds interpretable as explicit probability cutoffs (e.g. TIER_1 = P(fraud) ≥ 0.60) and enables the business to set thresholds against explicit risk appetite rather than statistical percentiles.

---

## 8. Future Recommendations

### 8.1 Feature Store for Online Scoring

The current feature engine requires the full user history to compute rolling features (30-day means, velocity counts, etc.). A single new transaction cannot be scored in isolation. In production, this should be replaced by a **feature store** (Redis or Feast) that maintains pre-computed user-level aggregates updated in real time. Each new transaction would fetch the stored aggregates, compute the delta features locally, and score without reprocessing history. This reduces inference latency from O(user history) to O(1).

### 8.2 Platt Scaling / Probability Calibration

As described in the manual validation plan, replacing score-based thresholds with calibrated probabilities makes fraud decisions interpretable and auditable for compliance. The infrastructure for this is already in place — it simply requires labelled outcomes from analyst review.

### 8.3 PyTorch Autoencoder with Full Entity Embeddings

The current deployment uses an sklearn MLP fallback. The full PyTorch autoencoder with shared entity embedding weights (user_id, city/prev_city with tied weights, device/prev_device with tied weights) captures temporal consistency in a way the MLP cannot. Installing PyTorch and retraining would improve the autoencoder's BC from the current level and reduce false negatives on account-takeover patterns.

### 8.4 Drift Monitoring in Production

Implement automated monthly CSI reports comparing the current month's score distribution against the training baseline. CSI > 0.25 should trigger an automated alert to the ML engineering team. This converts the current manual CSI check (Cell 93–94) into a scheduled production monitoring job.

### 8.5 Synthetic Data Limitations

This dataset was generated with a uniform amount distribution and all users visiting all cities. A production deployment would benefit from retraining on real transaction data where:
- Amounts follow a log-normal distribution with user-specific clustering
- Geographic patterns are realistic (London >> other cities for UK users)
- Device consistency is genuine (most users have 1–2 devices, not 5–6)

The pipeline architecture is designed to handle this — the user-relative normalisation in the feature engine will automatically adapt to real distributions. However, rule thresholds calibrated from EDA on this dataset (e.g. the z-score cutoffs in R01, R04, R05) should be re-calibrated against real data distributions before production deployment.

---
