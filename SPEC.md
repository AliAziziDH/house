# SPEC.md — Experiment-01: Baseline Calibration & Leakage Guardrails

> **Competition:** Ames Housing Prices (Kaggle)
> **Metric:** RMSLE (Root Mean Squared Logarithmic Error)
> **Single source of truth for RMSLE:** [`src/metrics.py`](./src/metrics.py)
> **Status:** 🟡 Active

---

## 1. Objective

Establish a reproducible, leakage-free cross-validation baseline and define hard OOF RMSLE gates that all future experiments must beat before any model artefact is saved or any submission is generated.

---

## 2. OOF RMSLE Targets

| Gate | Threshold | Action on Miss |
|------|-----------|----------------|
| **Hard floor (Guardrail)** | `RMSLE < 0.1180` | ❌ Block — do not save model or submission |
| **Experiment-01 target** | `RMSLE < 0.1150` | ✅ Accept as Exp-01 baseline |
| **Stretch goal** | `RMSLE < 0.1120` | 🚀 Promote to ensemble candidate |

> [!IMPORTANT]
> The `0.1180` hard floor is enforced by the evaluation harness defined in `AGENTS.md §3`.
> No PR, script, or agent action may relax this threshold without explicit written justification in this file.

---

## 3. Strict Cross-Validation Protocol

### 3.1 Fold Configuration

```python
from sklearn.model_selection import KFold

CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

- **Folds:** 5 (stratified by index; Ames is too small for stratified-target CV without leakage risk)
- **`random_state`:** `42` — fixed globally across all models and ensemble stages
- **Reproducibility check:** any run that changes `random_state` must re-report OOF before merging

### 3.2 Leakage Guardrails

All transformations that learn from data **must** be fitted exclusively on the training fold and applied to the validation fold. Violations are considered critical bugs.

| Operation | Rule | Implementation |
|-----------|------|----------------|
| Imputation (LotFrontage medians, categorical modes) | Fit on train fold only | [`AmesDataTransformer.fit()`](./src/preprocess.py#L57-L100) |
| Neighborhood target rank encoding | Fit on train fold `y` only | [`AmesDataTransformer.fit()`](./src/preprocess.py#L87-L94) |
| One-hot encoding schema | Schema locked from train fold; val/test reindexed to match | [`AmesDataTransformer.transform()`](./src/preprocess.py#L234-L238) |
| Box-Cox target transform (`PowerTransformer`) | Fit on `y_train_fold`; inverse-transform val predictions before RMSLE | [`optimize_xgboost.py`](./src/optimize_xgboost.py#L47-L48), [`train_catboost.py`](./src/train_catboost.py#L54-L55) |
| Ensemble weights (SLSQP / grid search) | Optimised on OOF predictions only — never on held-out test | [`find_ensemble_weights.py`](./src/find_ensemble_weights.py) |

> [!CAUTION]
> **Never** call `AmesDataTransformer.fit()` on the full dataset before splitting.
> The existing `preprocess_data()` helper in `preprocess.py` does this and is **not suitable** for use inside a CV loop. Always instantiate a fresh `AmesDataTransformer` per fold.

### 3.3 Canonical CV Loop Skeleton

```python
from sklearn.model_selection import KFold
from src.preprocess import AmesDataTransformer
from src.metrics import rmsle
import numpy as np

CV = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_raw))

for fold, (train_idx, val_idx) in enumerate(CV.split(X_raw)):
    X_tr_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Transformer fitted on train fold only
    transformer = AmesDataTransformer()
    transformer.fit(X_tr_raw, y_tr)
    X_tr = transformer.transform(X_tr_raw)
    X_val = transformer.transform(X_val_raw)   # schema aligned to train fold

    # ... fit model on (X_tr, y_tr) ...
    # ... predict on X_val, inverse-transform if Box-Cox was used ...

    oof_preds[val_idx] = y_val_pred

oof_rmsle = rmsle(y, oof_preds)
assert oof_rmsle < 0.1180, f"Guardrail breached: OOF RMSLE = {oof_rmsle:.4f}"
```

---

## 4. Model Inventory (Experiment-01 Scope)

| Model | Tuning Method | Folds | Target Transform | Artefact Path |
|-------|---------------|-------|-----------------|---------------|
| XGBoost | Optuna (`n_trials=50`) | 5-fold KFold | Box-Cox (`PowerTransformer`) | `models/xgboost_best_rmsle.pkl` |
| CatBoost | Optuna (`n_trials=15`) | 5-fold KFold | Box-Cox (`PowerTransformer`) | `models/catboost_best_rmsle.pkl` |
| Ensemble (XGB + CB) | Grid search over `w ∈ [0,1]` step 0.01 | OOF only | — | `submissions/submission_ensemble_rmsle_final.csv` |

---

## 5. Ensemble Blending Constraints

1. Weights must sum to **exactly 1.0** (enforced at blend time).
2. The weight grid search in [`find_ensemble_weights.py`](./src/find_ensemble_weights.py) operates on **OOF vectors** — not test predictions.
3. When promoting to SLSQP (see Ensemble Skill), the `constraints` dict must include `{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}`.
4. Test predictions are averaged across folds (`pred_test += fold_pred / n_splits`), not re-predicted on full-train models.

---

## 6. Artefact Save Policy

An artefact (`.pkl`, `.csv`) is only written when **all** of the following pass:

- [ ] OOF RMSLE < 0.1180 (hard floor)
- [ ] No transformer was fitted on data that overlapped with the validation fold
- [ ] `random_state=42` was used in both `KFold` and all model constructors
- [ ] Box-Cox inverse-transform was applied before computing RMSLE

---

## 7. Experiment Log

| Date | Change | OOF RMSLE | vs prev | Promoted? |
|------|--------|-----------|---------|-----------|
| — | Experiment-01 baseline (this spec) | TBD | — | Pending |

> Update this table after every run via `experiments/` CSV outputs.

---

## 8. Related Files

| File | Role |
|------|------|
| [`src/metrics.py`](./src/metrics.py) | Single source of truth for RMSLE |
| [`src/preprocess.py`](./src/preprocess.py) | `AmesDataTransformer` — leakage-safe stateful transformer |
| [`src/train_catboost.py`](./src/train_catboost.py) | CatBoost Optuna tuning loop |
| [`src/optimize_xgboost.py`](./src/optimize_xgboost.py) | XGBoost Optuna tuning loop |
| [`src/find_ensemble_weights.py`](./src/find_ensemble_weights.py) | OOF weight grid search |
| [`src/ensemble.py`](./src/ensemble.py) | Final inference & submission generation |
| [`AGENTS.md`](./AGENTS.md) | Agent routing rules & guardrail definitions |
