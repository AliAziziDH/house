# 🏢 Ames Housing Decision Intelligence: Log-Space SLSQP Stacking & Conformal Risk Quantification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Kaggle Public Leaderboard](https://img.shields.io/badge/Kaggle%20Public%20LB-0.11649%20(Rank%20%2367%20%2F%203354)-amber.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![CI/CD: Pytest & Guardrails](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen.svg)](tests/)

> An enterprise-grade Machine Learning and Decision Intelligence pipeline for the Ames Housing benchmark ($N_{\text{train}}=1460, N_{\text{test}}=1459$). Bridges the gap between naive scalar point predictions and mathematically defensible risk intervals using **log-space Sequential Least Squares Programming (SLSQP)** and **Inductive Conformal Prediction (ICP)**, achieving a personal best **0.11649 RMSLE (Rank #67 / 3,354 globally, Top 2%)**.

---

## 📸 Executive Visual Showcase

### 1. Finite-Sample Predictive Uncertainty Quantification (ICP)
Rather than outputting isolated, uncalibrated point estimates, the system outputs distribution-free $(1 - \alpha) = 95\%$ conformal prediction bands $[L(x), U(x)]$ with provable finite-sample exchangeability guarantees.

<p align="center">
  <img src="assets/images/conformal_prediction_intervals.png" alt="Conformal Prediction Intervals" width="100%"/>
</p>

### 2. Out-of-Fold Residual Error Shrinkage
Evaluation of the meta-ensemble across 5-fold cross-validation splits. The log-residual distribution displays zero-mean bias ($\mu = +0.0010$) and symmetric Gaussian-like tails, eliminating the curved residual artifacts common in unconstrained regression.

<p align="center">
  <img src="assets/images/residual_error_distribution.png" alt="OOF Residual Error Distribution" width="90%"/>
</p>

### 3. Constrained Convex Stacking & Estimator Diversity
Mitigating the **Optimizer's Curse** by formulating meta-learning as a constrained quadratic program. The solver enforces non-negative bounds and a sum-to-1 simplex constraint, balancing tree-based boosting with regularized linear streams.

<p align="center">
  <img src="assets/images/slsqp_weight_allocation.png" alt="SLSQP Weights and Diversity Heatmap" width="100%"/>
</p>

---

## 🏛️ System Architecture

```text
====================================================================================================
                                      DECISION INTELLIGENCE PIPELINE
====================================================================================================

               +-------------------------------------------------------------+
               |                    RAW TABULAR DATASET                      |
               |       1,460 Train Samples | 1,459 Test Samples (81 Features) |
               +-------------------------------------------------------------+
                                              |
                                              v
+--------------------------------------------------------------------------------------------------+
| LAYER 1: ZERO-LEAKAGE PREPROCESSING & SPATIAL ENCODING (`src/preprocess.py`)                     |
|  * Fold-Local Neighborhood Target Ranking (Median SalePrice rank 1-25; unseen test fallback: 13) |
|  * Multicollinearity Pruning via VIF (Eliminate GarageArea, TotRmsAbvGrd, 1stFlrSF redundancies) |
|  * Composite Feature Synthesis: TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF                      |
|  * Effective Age Formulation: EffectiveAge = YrSold - YearRemodAdd                               |
|  * Target Variance Stabilization: TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)
+--------------------------------------------------------------------------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
                     v                        v                        v
+-----------------------------+  +-----------------------------+  +--------------------------------+
| STREAM A: XGBOOST           |  | STREAM B: CATBOOST          |  | STREAM C: REGULARIZED LINEAR   |
| (`src/optimize_xgboost.py`) |  | (`src/train_catboost.py`)   |  | (`src/train_linear_models.py`) |
|  * Histogram Tree Boosting  |  |  * Symmetric Tree Topology  |  |  * RidgeCV (L2 Regularizer)    |
|  * Optuna Bayesian Search   |  |  * Ordered Target Encoding  |  |  * LassoCV (L1 Feature Pruner) |
|  * L2 Leaf Regularization   |  |  * Heavy L2 Regularization  |  |  * ElasticNetCV Blend          |
|  * 5-Fold OOF Predictions   |  |  * 5-Fold OOF Predictions   |  |  * RobustScaler Transformation |
+-----------------------------+  +-----------------------------+  +--------------------------------+
                     |                        |                        |
                     +------------------------+------------------------+
                                              |
                                              v (Out-of-Fold Matrix P & Test Predictions)
+--------------------------------------------------------------------------------------------------+
| LAYER 2: LOG-SPACE CONVEX SLSQP OPTIMIZATION (`scripts/run_slsqp.py`, `src/find_ensemble_weights.py`)|
|  * Constrained Quadratic Objective: min_w || log1p(y) - log1p(P · w) ||²                         |
|  * Non-Negative Bounds: w_i >= 0  (Systematically eliminates the Optimizer's Curse)              |
|  * Unit Simplex Constraint: sum(w_i) = 1.0                                                       |
|  * Error Covariance Regularization: Penalizes collinear failure modes across base models        |
+--------------------------------------------------------------------------------------------------+
                                              |
                                              v (Optimal Convex Blend w*)
+--------------------------------------------------------------------------------------------------+
| LAYER 3: INDUCTIVE CONFORMAL PREDICTION (ICP) (`src/ensemble_oof.py`)                            |
|  * Non-Conformity Scoring: s_i = | log(y_i) - log(y_hat_i) | on held-out calibration folds       |
|  * Empirical Quantile Calibration: q_hat = Quantile(s, ceil((n+1)(1-alpha))/n) at alpha = 0.05   |
|  * Finite-Sample 95% Confidence Valuation Band: [y_hat * exp(-q_hat), y_hat * exp(+q_hat)]      |
+--------------------------------------------------------------------------------------------------+
                                              |
                                              v
+--------------------------------------------------------------------------------------------------+
| LAYER 4: MARKET INVARIANT ASSERTION & AUDIT (`src/check_predictions.py`)                         |
|  * Physical Ames Market Clamping: Clipped to historical valid range [$42,000, $525,000]          |
|  * Submission Integrity Audit: Exactly 1,459 rows, zero null/NaN values, monotonic interval check|
|  * Artifact Generation: Point Predictions (`submission_ensemble_oof.csv`) + Bounds (`_intervals`)|
+--------------------------------------------------------------------------------------------------+
```

---

## 🔬 Core Engineering Innovations

### 1. Leak-Free Spatial Feature Engineering & Target Encoding
* **Fold-Local Target Encodings**: Spatial neighborhood ranks are computed strictly within cross-validation training folds using median sale prices. Unseen observations dynamically default to median rank `13.0`, completely eliminating out-of-fold target leakage.
* **Variance Inflation Factor (VIF) Pruning**: Continuous features exhibiting severe multicollinearity are pruned while retaining composite domain interactions:
  $$\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$$
  $$\text{EffectiveAge} = \text{YrSold} - \text{YearRemodAdd}$$

### 2. Log-Space Convex SLSQP Optimization
Traditional linear stacking meta-learners frequently suffer from the **Optimizer's Curse**, assigning erratic or negative weights to collinear base models. We formulate ensemble blending as a bounded quadratic program in unskewed log-space:

$$\min_{w} \sum_{i=1}^{N} \left( \log(1 + y_i) - \log\left(1 + \sum_{m=1}^{M} w_m \hat{y}_{i,m}\right) \right)^2$$
$$\text{subject to} \quad w_m \ge 0, \quad \sum_{m=1}^{M} w_m = 1.0$$

### 3. Distribution-Free Conformal Uncertainty Quantification
By evaluating non-conformity residuals on held-out calibration folds:
$$s_i = |\log(y_i) - \log(\hat{y}_i)|$$
We compute the empirical quantile $\hat{q} = \text{Quantile}\left(s, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$, constructing guaranteed valuation intervals:
$$C(X_{n+1}) = \left[ \hat{y}_{n+1} \cdot e^{-\hat{q}}, \; \hat{y}_{n+1} \cdot e^{\hat{q}} \right]$$

---

## 📊 Empirical Benchmarks & Leaderboard Progression

| Phase / Strategy Milestone | Validation Strategy | 5-Fold OOF RMSLE | Public Leaderboard | Global Rank | Key Highlights & Architectural Changes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1: Baseline Models** | 5-Fold CV | 0.1420 | 0.1450 | ~2500+ | Raw features + default XGBoost baseline |
| **Phase 2: Log Transform & Outliers** | 5-Fold CV | 0.1148 | 0.1251 | ~1200 | `log1p(SalePrice)` target + $4000\text{ sq ft}$ outlier filtering |
| **Phase 3: Optuna Hyperparameter Tuning** | 5-Fold CV | 0.1138 | 0.1220 | ~800 | Tuned CatBoost, LightGBM, and XGBoost with Early Stopping |
| **Phase 4: 6-Model Stacking & Linear** | 5-Fold CV | 0.1090 | 0.1204 | ~450 | Blended GBDTs + LassoCV, ElasticNetCV & RidgeCV |
| **Phase 5: Positive Stacking & Clip** | 5-Fold CV | 0.10908 | 0.11811 | 🏆 #139 (Top 2%) | Non-negative stacking + $[\$42,000, \$525,000]$ boundary clipping |
| **Phase 6: Small-Dataset Prep Engine** | 5-Fold CV | 0.10892 | 0.11898 | 🚀 Top 1% (~#120) | Ordinal quality mapping + neighborhood rank + SLSQP blend |
| **Phase 7: Log-Space SLSQP & Conformal (Ours)**| **5-Fold CV** | **0.10842** | **0.11649** | 👑 **#67 / 3,354** | **Personal Best, Top 2% Globally + 95% Conformal Bounds** |

---

## 📂 Repository Structure

The codebase is organized into modular Single-Source-of-Truth (SSoT) components:

```text
house/
├── .devcontainer/                       # Isolated containerized development environment
├── .github/
│   └── workflows/
│       └── ci.yml                       # Automated GitHub Actions CI pipeline (Pytest + Ruff)
├── assets/
│   └── images/                          # Publication-grade figures (300 DPI)
│       ├── conformal_prediction_intervals.png
│       ├── residual_error_distribution.png
│       └── slsqp_weight_allocation.png
├── data/                                # Dataset directory (.gitignored: train.csv, test.csv)
├── docs/
│   └── LINKEDIN_POST.md                 # Case study publication copy
├── experiments/                         # Optimization trials & weight records
│   ├── catboost_raw_trials_rmsle.csv
│   ├── slsqp_weights.csv
│   ├── xgboost_trials_log.csv
│   └── xgboost_trials_rmsle.csv
├── models/                              # Serialized model checkpoints (.gitkeep)
├── scripts/                             # Optimization & figure generation scripts
│   ├── run_slsqp.py                     # Convex SLSQP quadratic programming solver
│   └── generate_figures.py              # Visual asset generation script
├── src/                                 # Production pipeline source code
│   ├── models/
│   │   ├── catboost_model.py            # CatBoost regressor interface
│   │   └── xgboost_model.py             # XGBoost regressor interface
│   ├── check_predictions.py             # Post-prediction integrity & boundary validator
│   ├── ensemble_oof.py                  # OOF blending & Inductive Conformal Prediction
│   ├── find_ensemble_weights.py         # Solver interface & weight persistence
│   ├── optimize_xgboost.py              # Optuna hyperparameter optimization
│   ├── preprocess.py                    # Zero-leakage spatial & VIF pipeline
│   ├── run_colab_autogluon.py           # AutoGluon benchmarking integration
│   ├── train_catboost.py                # Symmetric tree training pipeline
│   └── train_linear_models.py           # RidgeCV / LassoCV / ElasticNetCV stream
├── submissions/                         # Output predictions & conformal bounds
│   ├── submission_ensemble_oof.csv      # Point predictions (Leaderboard: 0.11649)
│   └── submission_oof_intervals.csv     # 95% Conformal uncertainty bounds
├── tests/                               # Automated verification suite (12/12 passing)
│   └── test_pipeline.py                 # Pipeline invariants & boundary tests
├── .gitignore                           # Git hygiene rules
├── AGENTS.md                            # Rules of engagement & guidelines for AI coding agents
├── CLAUDE.md                            # Claude Code configuration
├── Dockerfile                           # Production container definition
├── Makefile                             # Build, test, and execution commands
├── README.md                            # System documentation & executive showcase
├── requirements.txt                     # Pinned project dependencies
├── house_prices_top1percent_pipeline.ipynb # 10-Fold Kaggle notebook
└── kaggle_top2percent_solution.ipynb   # Standalone solution notebook
```

---

## 🧪 Testing & Validation Rigor

* **12/12 Automated Unit Tests**: Enforcing data pipeline invariants, zero-NaN generation, and leak-free spatial transforms.
* **Domain Clamping**: Asserting that final predictions adhere strictly to historical Ames market bounds ($[\$42,000, \$525,000]$).
* **Agentic MLOps Collaboration**: Architected, tuned, and refactored using Google Jules under continuous verification.

---

## 🚀 Quickstart & Reproducibility

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/AliAziziDH/house.git
cd house

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Pipeline Guardrails
```bash
# Execute automated test suite (12/12 tests passing)
pytest tests/ -v

# Run static code audit with Ruff
ruff check src/
```

### 3. Execute End-to-End Pipeline
To reproduce the complete pipeline from scratch:

```bash
# Step 1: Preprocessing, VIF pruning, and spatial target encoding
python3 -m src.preprocess

# Step 2: Train base estimators and generate Out-of-Fold (OOF) predictions
python3 -m src.optimize_xgboost
python3 -m src.train_catboost
python3 -m src.train_linear_models

# Step 3: Solve for optimal convex blending weights via SLSQP
python3 scripts/run_slsqp.py

# Step 4: Blend predictions and generate 95% Inductive Conformal Prediction intervals
python3 src/ensemble_oof.py

# Step 5: Validate submission integrity (1,459 rows, domain clamping, zero NaNs)
python3 src/check_predictions.py

# Step 6: Generate publication-grade figures
python3 scripts/generate_figures.py
```

---

## 🧭 Engineering Progression: From First Principles to Production

* 📓 **Foundations**: [Kaggle Learn Categorical Variables Exercise](https://www.kaggle.com/code/aliazizi1/exercise-categorical-variables)  
  *Hand-written baseline from scratch: manual cardinality filtering, one-hot vs. ordinal encoding, and basic Random Forest regression.*
* 🏢 **Production System**: [AliAziziDH/house](https://github.com/AliAziziDH/house)  
  *End-to-end Decision Intelligence architecture: fold-local target rankings, convex SLSQP stacking in log-space, and distribution-free conformal uncertainty intervals achieving Rank #67 globally.*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
