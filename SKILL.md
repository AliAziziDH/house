name: model_blending
description: Optimizes ensemble blending weights using Scipy's SLSQP algorithm on Out-of-Fold (OOF) predictions to minimize RMSLE.
trigger_phrases:
  - "optimize blending weights"
  - "find ensemble weights"
  - "run SLSQP solver"
  - "calibrate model weights"
dependencies:
  python_packages:
    - scipy
    - numpy
    - pandas
    - scikit-learn
inputs:
  oof_predictions: "path/to/oof_preds.csv"
  target_values: "path/to/y_train.csv"
outputs:
  optimal_weights: "yaml file containing weights summing to 1.0"
  final_oof_rmsle: "float value"
---

# 🧠 Ensemble Model Blending via SLSQP Optimizer

## 📌 1. Objective & Mathematical Formulation
The objective of this skill is to find the optimal blending weights $W = [w_1, w_2, ..., w_n]$ for $n$ base models (e.g., CatBoost, XGBoost, MLP, Lasso) that minimize the Root Mean Squared Logarithmic Error (RMSLE) on the training set using Out-of-Fold (OOF) predictions.

$$
\min_{W} \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\log(\hat{y}_i(W) + 1) - \log(y_i + 1))^2}
$$

Subject to:
1. $\sum_{j=1}^{n} w_j = 1.0$ (Unity constraint)
2. $0.0 \le w_j \le 1.0$ for all $j$ (Non-negativity bounds)

## 🛡️ 2. Execution Guardrails & Protocols
1. **OOF Vector Evaluation Only:** Weights must be fit *strictly* using Out-of-Fold (OOF) prediction vectors from cross-validation to prevent target leakage and leaderboard shakeup.
2. **Zero In-Sample Leakage:** Never fit the blending weights on the raw, training predictions directly.
3. **Deterministic Constraints:** Ensure the sum of weight array is exactly 1.0. Do not accept manual approximations (e.g., weights like 0.35, 0.35, 0.30 if they don't solve the objective).

## ⚙️ 3. Reference Config (Nested YAML Flat Tax Reduction)
optimization_config:
  solver: "SLSQP"
  max_iterations: 1000
  tolerance: 1.0e-10
  epsilon_step: 1.0e-8
  bounds:
    lower: 0.0
    upper: 1.0
  inverse_transform_target: "BoxCox"
