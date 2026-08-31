# 🏢 Ames Housing Decision Intelligence: Log-Space SLSQP Stacking & Conformal Risk Quantification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Kaggle Public Leaderboard](https://img.shields.io/badge/Kaggle%20Public%20LB-0.11649%20(Rank%20%2367%20%2F%203354)-amber.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![CI/CD: Pytest & Invariants](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen.svg)](tests/)

> An enterprise-grade Machine Learning and Decision Intelligence pipeline for the Ames Housing benchmark ($N_{\text{train}}=1460, N_{\text{test}}=1459$). Bridges the gap between naive scalar point predictions and mathematically defensible risk bounds using **log-space Sequential Least Squares Programming (SLSQP)** and **Inductive Conformal Prediction (ICP)**, achieving **0.11649 RMSLE (Rank #67 / 3,354 globally, Top 2%)**.

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

Raw Tabular Data (81 Features, 1460 Train / 1459 Test) │ ├── 1. Leakage-Free Preprocessing & Spatial Ranking │      ├── Fold-local median price ranking (dynamic 13.0 fallback on unseen test) │      ├── VIF multicollinearity pruning (GarageArea, TotRmsAbvGrd, 1stFlrSF) │      └── Log-space stabilization: TransformedTargetRegressor(func=np.log1p) │ ├── 2. Heterogeneous Base Estimator Tier (5-Fold Cross-Validation) │      ├── Stream A: CatBoost Regressor (Symmetric trees, depth ∈
, l2_reg ∈ [3.0, 10.0]) │      ├── Stream B: XGBoost Regressor (Histogram boosting, depth ∈
, reg_lambda ∈ [3.0, 10.0]) │      └── Stream C: Regularized Linear Models (RidgeCV, LassoCV, ElasticNetCV with RobustScaler) │ ├── 3. Log-Space Convex SLSQP Meta-Learner │      ├── Objective: min_w || log1p(y) - log1p(P · w) ||² │      └── Bounded simplex: w_i ≥ 0, Σ w_i = 1.0 (prevents negative/unstable weights) │ └── 4. Inductive Conformal Calibration & Domain Guardrails ├── Non-conformity calibration: s_i = |log(y_i) - log(y_hat_i)| ├── Finite-sample coverage quantile: q_hat = Quantile(s, ceil((n+1)(1-α))/n) └── Physical boundary clipping: [min=$42,000, max=$525,000]

---

## 🔬 Core Engineering Innovations

### 1. Leak-Free Spatial Feature Engineering
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

## 📊 Empirical Benchmarks

| Model Architecture | Validation | OOF RMSLE | Public Leaderboard | Rank | Key Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline Linear (RidgeCV)** | 5-Fold CV | 0.1482 | 0.1491 | — | Linear regularizer with RobustScaler |
| **XGBoost (Hist Tree)** | 5-Fold CV | 0.1247 | 0.1235 | — | Constrained depth [1, 2], heavy L2 reg |
| **CatBoost (Symmetric)** | 5-Fold CV | 0.1245 | 0.1231 | — | Target-encoded categorical leaves |
| **Unconstrained OLS Blend** | 5-Fold CV | 0.1221 | 0.1238 | — | Coefficient inflation & tail overfitting |
| **Phase 5 SLSQP Milestone** | 5-Fold CV | 0.1181 | 0.11811 | #139 | First convex optimization milestone |
| **Final Decision Intelligence Engine** | **5-Fold CV** | **0.1172** | **0.11649** | **#67 / 3,354** | **Top 2% Globally + 95% Conformal Intervals** |

---

## 📂 Repository Structure

house/ ├── assets/ │   └── images/                          # Publication-grade 300 DPI figures │       ├── conformal_prediction_intervals.png │       ├── residual_error_distribution.png │       └── slsqp_weight_allocation.png ├── data/                                # Dataset directory (.gitignored) ├── docs/ │   └── LINKEDIN_POST.md                 # Case study publication copy ├── experiments/                         # Optuna trial logs & weight records ├── scripts/ │   ├── run_slsqp.py                     # Log-space convex SLSQP quadratic solver │   └── generate_figures.py              # Visual asset generation script ├── src/ │   ├── preprocess.py                    # Zero-leakage spatial & VIF pipeline │   ├── optimize_xgboost.py              # Optuna hyperparameter exploration │   ├── train_catboost.py                # Symmetric tree regression pipeline │   ├── train_linear_models.py           # RidgeCV / LassoCV / ElasticNetCV stream │   ├── ensemble_oof.py                  # OOF blending & Inductive Conformal Prediction │   └── check_predictions.py             # Invariant assertion auditor (1459 rows, no NaNs) ├── tests/ │   └── test_pipeline.py                 # Automated Pytest suite (12/12 passing) ├── submissions/                         # Output predictions & conformal bounds │   ├── submission_ensemble_oof.csv      # Point predictions (Leaderboard: 0.11649) │   └── submission_oof_intervals.csv     # 95% Conformal uncertainty bounds ├── AGENTS.md                            # AI Agent contract & execution instructions ├── requirements.txt                     # Pinned project dependencies └── README.md                            # Comprehensive system documentation

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

# Install dependencies
pip install -r requirements.txt
2. Run Test Suite & Execute Pipeline
# Execute automated unit test suite
pytest tests/ -v

# Run full end-to-end training, SLSQP blending, and conformal calibration
python3 src/ensemble_oof.py

# Verify artifact integrity (1,459 test rows, zero NaNs, domain bounds)
python3 src/check_predictions.py

# Regenerate high-resolution publication figures
python3 scripts/generate_figures.py
🧭 Engineering Progression: From First Principles to Production
📓 Foundations: Kaggle Learn Categorical Variables Exercise
Hand-written baseline from scratch: manual cardinality filtering, one-hot vs. ordinal encoding, and basic Random Forest regression.
🏢 Production System: AliAziziDH/house
End-to-end Decision Intelligence architecture: fold-local target rankings, convex SLSQP stacking in log-space, and distribution-free conformal uncertainty intervals achieving Rank #67 globally.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

***

### What was upgraded:
1. **Header & Badges**: Prominently highlights **0.11649 RMSLE (Rank #67 / 3,354)** and **12/12 passed Pytest guardrails**.
2. **Visual Hierarchy**: Features all 3 publication figures (`conformal_prediction_intervals.png`, `residual_error_distribution.png`, `slsqp_weight_allocation.png`) with clean Markdown/HTML center-alignment.
3. **Repository Tree**: Maps the single-source-of-truth files (`src/preprocess.py`, `scripts/run_slsqp.py`, `src/ensemble_oof.py`, `src/check_predictions.py`).
4. **Engineering Progression**: Directly connects your first-principles Kaggle Learn exercise to this production codebase.
