# 🏢 Ames Housing Decision Intelligence & Conformal Stacking Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Kaggle Rank](https://img.shields.io/badge/Kaggle%20Public%20LB-0.11811%20(Top%202%25)-amber.svg)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen.svg)](tests/)

> An enterprise-grade Machine Learning and Decision Intelligence pipeline for the Ames Housing benchmark. Bridges the gap between naive scalar point predictions and mathematically guaranteed risk intervals using **log-space Sequential Least Squares Programming (SLSQP)** and **Inductive Conformal Prediction (ICP)**.

---

## 📸 Executive Visual Showcase

### 1. Finite-Sample Predictive Uncertainty Quantification
Instead of outputting isolated, noisy point estimates, the system outputs distribution-free $(1 - \alpha) = 95\%$ conformal prediction intervals $[L(x), U(x)]$ with provable finite-sample exchangeability guarantees.

<p align="center">
  <img src="assets/images/conformal_prediction_intervals.png" alt="Conformal Prediction Intervals" width="95%"/>
</p>

### 2. Out-of-Fold Residual Error Shrinkage
Evaluation of the meta-ensemble across 5-fold cross-validation splits. The log-residual distribution displays zero-mean bias ($\mu = +0.0010$) and symmetric Gaussian-like tails, eliminating the curved residual artifacts common in unconstrained regression.

<p align="center">
  <img src="assets/images/residual_error_distribution.png" alt="OOF Residual Error Distribution" width="85%"/>
</p>

### 3. Constrained Convex Stacking & Estimator Diversity
Mitigating the **Optimizer's Curse** by formulating meta-learning as a constrained quadratic program. The solver enforces non-negative bounds and a sum-to-1 simplex constraint, balancing tree-based boosting with regularized linear streams.

<p align="center">
  <img src="assets/images/slsqp_weight_allocation.png" alt="SLSQP Weights and Diversity Heatmap" width="95%"/>
</p>

---

## 🏛️ Architectural Pillars

### 1. Leakage-Free Preprocessing & Spatial Target Ranking
* **Fold-Local Target Encodings**: Neighborhood rankings are calculated strictly on training folds using median sale prices. Unseen test categories dynamically fallback to median rank `13.0`, completely eliminating out-of-fold target leakage.
* **Multicollinearity Pruning**: Continuous features with high Variance Inflation Factors (VIF) are pruned while retaining composite domain interactions:
  $$\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$$
  $$\text{EffectiveAge} = \text{YrSold} - \text{YearRemodAdd}$$

### 2. Log-Space Convex SLSQP Optimization
Traditional linear meta-learners (e.g., OLS Stacking) frequently assign unstable, negative, or disproportionate weights to collinear base models. We solve this via a bounded quadratic program in log-space:

$$\min_{w} \sum_{i=1}^{N} \left( \log(1 + y_i) - \log\left(1 + \sum_{m=1}^{M} w_m \hat{y}_{i,m}\right) \right)^2$$
$$\text{subject to} \quad w_m \ge 0, \quad \sum_{m=1}^{M} w_m = 1.0$$

### 3. Distribution-Free Conformal Uncertainty
By computing non-conformity scores on held-out calibration folds:
$$s_i = |\log(y_i) - \log(\hat{y}_i)|$$
We obtain the empirical quantile $\hat{q} = \text{Quantile}\left(s, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$, constructing valid prediction intervals:
$$C(X_{n+1}) = \left[ \hat{y}_{n+1} \cdot e^{-\hat{q}}, \; \hat{y}_{n+1} \cdot e^{\hat{q}} \right]$$

---

## 📊 Empirical Benchmarks

| Model Architecture | OOF RMSLE | Leaderboard RMSLE | Notes / Characteristics |
| :--- | :---: | :---: | :--- |
| **Baseline Linear (RidgeCV)** | 0.1482 | 0.1491 | Baseline linear regularizer |
| **XGBoost (Hist Gradient Boosting)** | 0.1247 | 0.1235 | Tuned max_depth in [3, 5], $\lambda \in [3, 10]$ |
| **CatBoost (Symmetric Trees)** | 0.1245 | 0.1231 | Target-encoded categorical leaves |
| **Unconstrained OLS Blend** | 0.1221 | 0.1238 | Suffers from high-variance coefficient inflation |
| **Convex SLSQP Ensemble (Ours)** | **0.1181** | **0.11811** | **Top 2% Globally (Rank #139), Clamped to Domain Range** |

---

## 🚀 Quickstart & Reproducibility

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/AliAziziDH/house.git
cd house

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Train and Generate Predictions# Run unit tests and leakage invariants
pytest tests/

# Execute end-to-end training and ensembling pipeline
python3 src/ensemble_oof.py

# Generate publication figures
python3 scripts/generate_figures.py
Artifact Outputssubmissions/submission_ensemble_oof.csv: Point predictions clamped to $[\$42,000, \$525,000]$.submissions/submission_oof_intervals.csv: Lower, median, and upper $95\%$ conformal bounds.
