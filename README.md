# House Prices - Advanced Regression Techniques

**A Complete Kaggle Competition Project**  
*Predicting house sale prices using advanced regression techniques and ensemble modeling.*

---

## Project Overview

This repository contains my complete solution for the **House Prices - Advanced Regression Techniques** competition on Kaggle. The goal of this competition is to predict the final sale price of residential homes in Ames, Iowa, based on 79 explanatory variables describing various aspects of the property.

**Key Results:**
- **Public Leaderboard Score:** 0.12123
- **Final Rank** | **513 / 4,155** (Top 12.3%) |
- **Best Model:** Weighted Ensemble (XGBoost 64% + CatBoost 36%)
- **Target Transformation:** Box-Cox

**Competition Link:** [Kaggle - House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Approach & Methodology](#approach--methodology)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Target Transformation](#target-transformation)
6. [Model Development](#model-development)
7. [Ensemble & Stacking](#ensemble--stacking)
8. [Results & Key Learnings](#results--key-learnings)
9. [Project Structure](#project-structure)
10. [How to Reproduce](#how-to-reproduce)
11. [License & Contact](#license--contact)

---

## Approach & Methodology

This project follows a structured and iterative approach to solve the regression problem:

1. **Deep Exploratory Data Analysis (EDA)**
   - Analyzed missing values, distributions, and correlations.
   - Identified key features affecting sale price (e.g., `OverallQual`, `GrLivArea`, `Neighborhood`).

2. **Data Cleaning & Preprocessing**
   - Handled missing values column‑by‑column (not a one‑size‑fits‑all approach).
   - Encoded categorical variables appropriately.

3. **Feature Engineering**
   - Created 8 new features based on domain knowledge and insights from EDA.

4. **Target Transformation**
   - Compared Log, Box-Cox, and Yeo-Johnson transformations.
   - Chose the best using 5‑fold cross‑validation.

5. **Model Development**
   - Trained and optimized multiple models (XGBoost, LightGBM, CatBoost).
   - Used Optuna for hyperparameter optimization (50 trials per model).

6. **Ensemble & Stacking**
   - Tested both weighted averaging and stacking (linear regression meta-model).
   - Selected the best-performing ensemble for final submission.

All experiments, including failed ones, are saved in the `experiments/` directory for transparency and learning.
## Data Preprocessing

Data preprocessing was the most critical step in this project. The House Prices dataset contains 79 features with varying levels of missing values, and a careless approach would have significantly degraded model performance.

### 1. Handling Missing Values (Column‑by‑Column)

Instead of using a generic imputer, I analyzed each feature individually and applied domain‑specific logic:

| Feature Group | Columns | Missing Value Treatment |
| :--- | :--- | :--- |
| **Garage** | `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | Filled with `'No Garage'` (meaning: no garage exists) |
| | `GarageYrBlt`, `GarageCars`, `GarageArea` | Filled with `0` (no garage = zero area, zero cars) |
| **Basement** | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | Filled with `'No Basement'` |
| | `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF` | Filled with `0` |
| **Masonry Veneer** | `MasVnrType` | Filled with `'None'` |
| | `MasVnrArea` | Filled with `0` |
| **Optional Features** | `Alley`, `PoolQC`, `Fence`, `FireplaceQu`, `MiscFeature` | Filled with meaningful `'No ...'` values |
| **Lot Frontage** | `LotFrontage` | Imputed with **median of the `Neighborhood`** (rather than global median) |
| **Electrical** | `Electrical` | Filled with the **mode** (most frequent value: `'SBrkr'`) |
| **Test‑Only Missing** | `MSZoning`, `Utilities`, `Exterior1st`, `Exterior2nd`, `KitchenQual`, `Functional`, `SaleType` | Filled with **mode from training set** |
| | `BsmtFullBath`, `BsmtHalfBath` | Filled with `0` (no basement) |

> **Key Insight:** The test set had missing values in columns that had no missing values in the training set. This highlights the importance of always checking the test data separately.

### 2. Categorical Encoding

Two encoding strategies were applied:

- **Ordinal Encoding** (for features with natural order):
  - Examples: `ExterQual` (Po→1, Fa→2, TA→3, Gd→4, Ex→5), `BsmtExposure` (No→1, Mn→2, Av→3, Gd→4), `Functional` (Sal→1, ... Typ→8).
  - All `'No ...'` values were mapped to `0`.
  - A total of 18 ordinal features were encoded using custom dictionaries.

- **One‑Hot Encoding** (for nominal features without order):
  - Features like `Neighborhood`, `MSZoning`, `SaleType`, etc.
  - Resulted in **87 features** after ordinal encoding and **~214 features** after one‑hot encoding.

### 3. Feature Engineering

Eight new features were created to capture hidden patterns and improve model performance:

| New Feature | Formula | Intuition |
| :--- | :--- | :--- |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total living area (above + below ground) |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | Total outdoor living space |
| `TotalBathrooms` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | Total bathrooms (full = 1, half = 0.5) |
| `HouseAge` | `YrSold - YearBuilt` | Age of the house at time of sale |
| `RemodAge` | `YrSold - YearRemodAdd` | Years since last major remodel |
| `IsNew` | `1` if `YearBuilt == YrSold` else `0` | Indicator for brand‑new homes |
| `QualityScore` | `OverallQual * OverallCond` | Interaction of quality and condition |
| `GarageAge` | `YrSold - GarageYrBlt` if garage exists, else `0` | Age of the garage (0 if no garage) |

---

### 4. Saving Processed Data

After preprocessing, the cleaned data was saved in the `./processed_data/` directory as CSV files for fast loading in subsequent modeling steps.

- `X_train.csv`, `X_test.csv`, `y_train.csv` → for XGBoost and LightGBM (with one‑hot encoding).
- `X_train_raw.csv`, `X_test_raw.csv` → for CatBoost (preserving categorical features as strings).

This separation allowed me to compare models under different encoding strategies.

## Target Transformation

The target variable `SalePrice` had a **right-skewed distribution** (skewness = 1.88), which can negatively affect linear models and sometimes tree‑based models as well. To address this, I tested three different transformations and selected the best one using 5‑fold cross‑validation.

### 1. Transformations Tested

| Method | Formula | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Log** | `log(1 + y)` | Simple, interpretable | May not fully normalize skewed data |
| **Box‑Cox** | `(y^λ - 1) / λ` | Finds optimal λ for normality | Only works for positive values |
| **Yeo‑Johnson** | Similar to Box‑Cox but for any real value | Works with zero and negative values | More complex, rarely needed for positive targets |

### 2. Experimental Setup

- A simple **XGBoost model** with default parameters (`n_estimators=100`, `learning_rate=0.1`, `max_depth=5`) was used for comparison.
- **5‑fold cross‑validation** was applied to ensure robustness.
- RMSE (Root Mean Squared Error) was measured on the **original scale** (after inverse transformation) to make results comparable.

### 3. Results

| Transformation | Average RMSE (CV) | Standard Deviation |
| :--- | :--- | :--- |
| **Log** | 194,815.54 | ± 4,381.54 |
| **Box‑Cox** | **194,702.41** | ± 4,063.72 |
| **Yeo‑Johnson** | 194,702.53 | ± 4,063.73 |

### 4. Conclusion

**Box‑Cox** performed slightly better than the other two methods, both in terms of lower average RMSE and lower standard deviation. Therefore, **Box‑Cox was chosen as the final transformation** for all subsequent modeling steps.

The `PowerTransformer` from `sklearn.preprocessing` was used to apply Box‑Cox, and the fitted transformer was saved for later use on the test set.

## Model Development

After preprocessing and transforming the target, I developed and optimized three different models. Each model was tuned using **Optuna** with 50 trials and 5‑fold cross‑validation. The goal was to find the best hyperparameters and identify which model generalizes best to unseen data.

### 1. Models Tested

| Model | Strengths | Weaknesses |
| :--- | :--- | :--- |
| **XGBoost** | Fast, robust, handles missing data well | Sensitive to hyperparameter tuning |
| **LightGBM** | Faster than XGBoost, lower memory usage | Can overfit on small datasets |
| **CatBoost** | Handles categorical features natively, great CV performance | Slower to train, sometimes overfits on test data |

### 2. Hyperparameter Optimization with Optuna

For each model, I defined a search space and ran 50 trials to minimize RMSE on cross‑validation.

**XGBoost Search Space:**
- `n_estimators`: 100–1000 (step 100)
- `max_depth`: 3–10
- `learning_rate`: 0.01–0.3 (log scale)
- `subsample`: 0.6–1.0
- `colsample_bytree`: 0.6–1.0
- `min_child_weight`: 1–10

**LightGBM Search Space:**
- `n_estimators`: 100–1000 (step 100)
- `max_depth`: 3–12
- `learning_rate`: 0.01–0.3 (log scale)
- `subsample`: 0.6–1.0
- `colsample_bytree`: 0.6–1.0
- `min_child_samples`: 5–30
- `num_leaves`: 20–100

**CatBoost Search Space:**
- `iterations`: 100–1000 (step 100)
- `depth`: 3–10
- `learning_rate`: 0.01–0.3 (log scale)
- `l2_leaf_reg`: 1–10 (log scale)
- `subsample`: 0.6–1.0
- `colsample_bylevel`: 0.6–1.0

### 3. Results

| Model | Best CV RMSE | Best Parameters | Public LB Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 27,770.51 | `n_estimators=700`, `max_depth=5`, `learning_rate=0.0186`, `subsample=0.6136`, `colsample_bytree=0.7310`, `min_child_weight=2` | **0.12335** |
| **LightGBM** | 27,857.09 | `n_estimators=800`, `max_depth=11`, `learning_rate=0.0125`, `subsample=0.8454`, `colsample_bytree=0.7215`, `min_child_samples=20`, `num_leaves=47` | 0.12772 |
| **CatBoost** | 25,787.70 | `iterations=600`, `depth=4`, `learning_rate=0.0940`, `l2_leaf_reg=3.92`, `subsample=0.9645`, `colsample_bylevel=0.6347` | 0.12341 |

### 4. Key Observations

- **XGBoost** had the best balance between CV performance and generalization to the public leaderboard. Its parameters (shallow trees, low learning rate) suggest it is a robust and well‑regularized model.
- **LightGBM** overfitted significantly. Despite a reasonable CV score, it performed poorly on the public LB, likely due to its deeper trees (`max_depth=11`) and complex structure.
- **CatBoost** achieved the best CV RMSE by a large margin (~2,000 points lower than XGBoost). However, it failed to replicate that performance on the public LB, indicating that its internal categorical handling may have over‑optimized to the training data.

> **Lesson:** CV scores are not always a reliable indicator of test performance. Always validate with multiple models and compare results on the actual competition leaderboard.

## Ensemble & Stacking

After developing and optimizing individual models, I explored two different ensemble strategies to combine their predictions and reduce overall error. The goal was to leverage the strengths of each model while mitigating their individual weaknesses.

### 1. Weighted Average Ensemble

The simplest and most effective ensemble method is to take a weighted average of the predictions from the best models. I tested different weight combinations for **XGBoost** and **CatBoost** (since LightGBM underperformed on the public LB).

**Weight Search Results:**

| XGBoost Weight | CatBoost Weight | Public LB Score |
| :--- | :--- | :--- |
| 0.65 | 0.35 | 0.12125 |
| **0.64** | **0.36** | **0.12123** |
| 0.66 | 0.34 | 0.12125 |
| 0.67 | 0.33 | 0.12127 |

**Best Weighted Ensemble:**
- XGBoost: **64%**
- CatBoost: **36%**
- Public LB Score: **0.12123**

This improved the score from **0.12335** (XGBoost alone) to **0.12123** — a reduction of ~1.7% in RMSLE.

**Ensemble Weights Clarification**

Two distinct weightings were used in the project and recorded in the repository: one that gave the best public leaderboard submission, and one that was found to be optimal on out‑of‑fold (OOF) predictions during internal validation.

- Public leaderboard (final submission used for Kaggle LB): **XGBoost 64% + CatBoost 36%** (reported public LB score: 0.12123).
- OOF-optimal weights (found via OOF grid search): **XGBoost 14% + CatBoost 86%** (OOF RMSLE: 0.123800).

Both sets of weights and their generated submission files are stored in the `submissions/` folder for comparison; see the `Reproducibility` section below for the exact commands to reproduce either submission.

**Reproducibility**

Before running the commands below, download the Kaggle competition files `train.csv` and `test.csv` and place them in the `data/` directory (create `data/` if it does not exist).

To reproduce preprocessing, model training, and final submissions locally run the following commands from the repository root (assumes Python virtualenv activated and dependencies installed):

1. Preprocess and save processed data:

```bash
python src/preprocess.py
```

2. Optimize and train XGBoost (Optuna, RMSLE):

```bash
python src/optimize_xgboost.py
```

3. Optimize and train CatBoost on raw data (Optuna, RMSLE):

```bash
python src/train_catboost.py
```

4. Find ensemble weights (OOF grid search) and generate the OOF-optimal submission:

```bash
python src/find_ensemble_weights.py
```

5. Generate final submissions from trained models (two weight variants):

```bash
# (a) Leaderboard weights: XGB 0.64, CAT 0.36
python src/make_final_submission.py

# (b) OOF-optimal weights: XGB 0.14, CAT 0.86
python -c "from pathlib import Path; import joblib, pandas as pd; X_test=pd.read_csv('processed_data/X_test.csv'); x=joblib.load('models/xgboost_best_rmsle.pkl'); c=joblib.load('models/catboost_best_rmsle.pkl'); pt=joblib.load('models/boxcox_transformer.pkl'); x_o=pt.inverse_transform(x.predict(X_test).reshape(-1,1)).flatten(); c_o=pt.inverse_transform(c.predict(X_test).reshape(-1,1)).flatten(); import os; os.makedirs('submissions',exist_ok=True); pd.DataFrame({'Id':pd.read_csv('data/test.csv')['Id'],'SalePrice':0.14*x_o+0.86*c_o}).to_csv('submissions/submission_ensemble_0.14_0.86.csv',index=False)"
```

Notes:
- The repository contains both the scripts that produced the leaderboard submission and the experiments that found the OOF-optimal weights; these were intentionally preserved for transparency and comparison.
- See `experiments/README.md` for which experiment files are canonical and which ones are archived.

> **Note:** LightGBM was excluded from the final ensemble because it consistently underperformed on the public LB despite reasonable CV scores.

### 2. Stacking (Meta‑Model)

Stacking is a more advanced ensemble technique where a meta‑model (e.g., linear regression) learns the optimal combination of base model predictions.

**Stacking Setup:**
- **Base Models:** XGBoost and CatBoost (trained with their best parameters)
- **Meta‑Model:** Linear Regression
- **Training Data:** Base model predictions on the training set
- **Cross‑Validation:** 5‑fold to avoid overfitting

**Stacking Results:**
- **CV RMSE:** 79,180.80
- **Public LB Score:** 0.12870

**Why Did Stacking Fail?**

Despite a reasonable CV score, the stacking model performed worse on the public LB than the simple weighted average. Possible reasons:

- **Overfitting to the training data:** The linear regression meta‑model may have learned patterns specific to the training set that do not generalize.
- **Limited meta‑features:** With only two base models, the meta‑model had limited information to work with.
- **Simplicity of weighted average:** Sometimes simpler is better — the weighted average introduced less variance and was more stable.

### 3. Final Model Selection

## Results

| Model | Public LB Score |
|:---|:---|
| **Weighted Ensemble (XGB 0.64 + Cat 0.36)** | **0.12123** |
| XGBoost (RMSLE) | 0.12738 (CV) |
| CatBoost (RMSLE) | 0.121296 (CV) |
| CatBoost (RMSLE) on Public LB | 0.12486 |
| Ensemble (0.14 XGB + 0.86 Cat) on Public LB | 0.12409 |

**Best Submission:** `submission_ensemble_final.csv`  
**Public LB Score:** **0.12123**  
**Final Rank:** **513 / 4,155** (Top 12.3%)

**Final Model:** Weighted Ensemble (XGBoost 64% + CatBoost 36%)

**Key Insight:** The weighted average ensemble consistently outperformed individual models and even a more complex stacking approach. This reinforces the principle that sometimes simpler solutions are more robust and generalize better to unseen data.

## Results & Key Learnings

### 1. Final Results

| Metric | Value |
| :--- | :--- |
| **Public Leaderboard Score** | **0.12123** |
| **Final Rank** | **513 / 4,155** (Top 12%) |
| **Best Model** | Weighted Ensemble (XGBoost 64% + CatBoost 36%) |
| **Target Transformation** | Box-Cox |
| **Number of Features (After Preprocessing)** | 214 |

**Kaggle Competition Link:**  
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

### 2. Key Learnings

This project was a deep and practical learning experience. Here are the most important lessons I took away from this journey:

#### 🔍 Lesson 1: CV is not the final answer
CatBoost achieved the best cross‑validation score (25,787.70), but it failed to generalize to the public leaderboard (0.12341 vs XGBoost's 0.12335). This taught me that **CV is a guide, not a guarantee**. Always validate models on real test data (if available) or use multiple validation strategies.

#### 🔍 Lesson 2: Test data can have different missing values
The test set had missing values in columns that had no missing values in the training set (e.g., `MSZoning`, `Utilities`, `KitchenQual`). This is a common but often overlooked issue. Always **separately check the test set** for new patterns or missingness.

#### 🔍 Lesson 3: Simplicity often beats complexity
A simple weighted average (XGBoost + CatBoost) outperformed a more complex stacking approach (linear regression meta‑model) on the public LB. This reminded me that **complexity should be added only when necessary**, and that simpler solutions are often more robust and easier to maintain.

#### 🔍 Lesson 4: Feature engineering with domain knowledge matters
Creating features like `TotalSF`, `TotalBathrooms`, `HouseAge`, and `GarageAge` significantly improved model performance. **Understanding the data and its context** is just as important as applying algorithms.

#### 🔍 Lesson 5: Record everything, even failures
I saved all experimental results — including failed models like LightGBM and stacking. These records are valuable for:
- Learning from mistakes
- Explaining decisions in interviews
- Improving future projects

---

### 3. What I Would Do Differently Next Time

- **More advanced feature interactions:** E.g., combining `Neighborhood` with `TotalSF` or `OverallQual` to capture local price trends.
- **Target encoding:** Replace high‑cardinality categorical features like `Neighborhood` with mean target values (with smoothing to avoid overfitting).
- **Extended hyperparameter search:** Increase Optuna trials from 50 to 100–150 for even better parameters.
- **Cross‑validation with stratified folds:** Especially useful if the target distribution is skewed.

---

### 4. Acknowledgments

This project was completed as part of my personal learning journey in Data Science. Special thanks to:

- **Kaggle** for providing a rich, real‑world dataset and a supportive competition environment.
- **The open‑source community** for tools like `pandas`, `scikit‑learn`, `XGBoost`, `CatBoost`, `Optuna`, and `Streamlit`.
- **Documentation and tutorials** from the Kaggle community, which helped guide many of my decisions.

---

### 5. Final Thoughts

I'm proud of reaching **top 12%** in my first serious Kaggle competition. But more than the rank, what matters most is the process — learning how to:

- **Think critically** about data and features.
- **Design experiments** and analyze results honestly.
- **Iterate quickly** while staying organized.

This project taught me that **Data Science is not just about algorithms — it's about asking the right questions, making thoughtful decisions, and learning from both successes and failures.**

---

> "It's not about being perfect. It's about being better than you were yesterday."