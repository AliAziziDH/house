# 🤖 Ames Housing Agent Specification (AGENTS.md)

## 📌 1. Project Overview & Architecture
This repository implements a leakage-safe, high-performance pipeline for the Ames Housing Kaggle competition.
- **Evaluation Metric:** RMSLE (Root Mean Squared Logarithmic Error) via `src/metrics.py`.
- **Target Variable:** Log-transformed `SalePrice`.

## 📂 2. Directory Layout & Routing
/src/ ├── preprocess.py       # Data cleaning & transformer (AmesDataTransformer) ├── metrics.py          # Vectorized RMSLE (Single source of truth) ├── train_catboost.py   # Baseline CatBoost model ├── optimize_xgboost.py # Hyperparameter tuning ├── find_ensemble_weights.py # SLSQP meta-learner optimizer └── ensemble.py         # Final inference & blending

## 🛡️ 3. Execution Protocols (Strict Guardrails)
1. **Zero Data Leakage:** All categorical encodings and scaling must happen strictly within cross-validation folds. Never fit transformers on the entire dataset.
2. **Deterministic Blending:** Ensemble weights optimized via SLSQP must sum strictly to 1.0.
3. **Evaluation Threshold:** The OOF baseline threshold is set to RMSLE < 0.1180. Any change degrading performance will be automatically blocked by the evaluation harness.

## 🎛️ 4. Modular Agent Skills (Decoupled State)
To prevent prompt bloat and 429 limits, detailed tasks are routed to specialized skills. Do not load these unless actively invoked:
- 📊 **Explorer Skill** -> `./.agents/skills/data_exploration/` (Audits price skewness & missing values)
- ⚙️ **Feature Skill** -> `./.agents/skills/feature_engineering/` (Interaction terms & categorical mapping)
- 🧠 **Ensemble Skill** -> `./.agents/skills/model_blending/` (SLSQP weight calibration & multi-seed bagging)
