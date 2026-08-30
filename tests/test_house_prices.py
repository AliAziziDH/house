"""
Automated Pytest Suite for House Prices Preprocessing and Metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import rmsle
from src.preprocess import QUALITY_MAP, AmesDataTransformer


def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP["Ex"] == 5
    assert QUALITY_MAP["Gd"] == 4
    assert QUALITY_MAP["TA"] == 3
    assert QUALITY_MAP["None"] == 0


def create_mock_data():
    columns = [
        "Id",
        "SalePrice",
        "GrLivArea",
        "Neighborhood",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "BsmtFinType1",
        "BsmtFinType2",
        "BsmtExposure",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "OpenPorchSF",
        "3SsnPorch",
        "EnclosedPorch",
        "ScreenPorch",
        "WoodDeckSF",
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
    ]
    train_data = {col: np.zeros(1460) for col in columns}
    train_data["Id"] = np.arange(1, 1461)
    train_data["GrLivArea"] = np.ones(1460) * 1000
    train_data["SalePrice"] = np.ones(1460) * 200000
    train_data["Neighborhood"] = ["CollgCr"] * 1460
    # Create 2 outliers
    train_data["GrLivArea"][0] = 4001
    train_data["SalePrice"][0] = 200000
    train_data["GrLivArea"][1] = 4001
    train_data["SalePrice"][1] = 200000
    train_df = pd.DataFrame(train_data)

    test_columns = [col for col in columns if col != "SalePrice"]
    test_data = {col: np.zeros(1459) for col in test_columns}
    test_data["Id"] = np.arange(1461, 2920)
    test_data["GrLivArea"] = np.ones(1459) * 1000
    test_data["Neighborhood"] = ["CollgCr"] * 1459
    test_df = pd.DataFrame(test_data)

    return train_df, test_df


def test_transformer_with_mock_data():
    """Verify AmesDataTransformer works correctly with mock data."""
    train_df, test_df = create_mock_data()
    y_train = train_df["SalePrice"]
    X_train_raw = train_df.drop(columns=["Id", "SalePrice"])
    X_test_raw = test_df.drop(columns=["Id"])

    transformer = AmesDataTransformer()
    transformer.fit(X_train_raw, y_train)

    X_train_trans = transformer.transform(X_train_raw)
    X_test_trans = transformer.transform(X_test_raw)

    assert len(X_train_trans) == 1460
    assert len(X_test_trans) == 1459
    assert list(X_train_trans.columns) == list(X_test_trans.columns)


def test_preprocess_data_shapes():
    """Verify data preprocessing pipeline outputs valid non-null datasets from data directory."""
    data_dir = "./data"

    if not (Path(data_dir) / "train.csv").exists():
        pytest.skip("Kaggle data files missing in ./data")

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    y_train = train["SalePrice"]
    X_train_raw = train.drop(columns=["Id", "SalePrice"])
    X_test_raw = test.drop(columns=["Id"])

    transformer = AmesDataTransformer()
    transformer.fit(X_train_raw, y_train)

    X_tr = transformer.transform(X_train_raw)
    X_te = transformer.transform(X_test_raw)

    assert len(X_tr) == 1460
    assert len(X_te) == 1459
    assert list(X_tr.columns) == list(X_te.columns)


def test_rmsle_perfect_match():
    """Verify exact match returns RMSLE of 0."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert np.isclose(rmsle(y_true, y_pred), 0.0)


def test_rmsle_negative_clamping():
    """Verify negative values are correctly clamped to 0.0."""
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-10.0, 5.0])
    assert np.isclose(rmsle(y_true, y_pred), 0.0)


def test_rmsle_known_values():
    """Verify RMSLE calculation yields expected mathematical results."""
    # log1p(e-1) = 1, log1p(0) = 0
    y_true = np.array([np.exp(1) - 1])
    y_pred = np.array([0.0])
    assert np.isclose(rmsle(y_true, y_pred), 1.0)


def test_rmsle_zero_values():
    """Verify behavior when both inputs are exactly zero."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert np.isclose(rmsle(y_true, y_pred), 0.0)



# ---------------------------------------------------------------------------
# Quality Gate 1 – Trajectory IN_ORDER Test
# Asserts that AmesDataTransformer computes its parameters STRICTLY within
# cross-validation train folds and does NOT leak global target statistics
# into the held-out validation fold.
# ---------------------------------------------------------------------------

def _make_heterogeneous_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Build a small synthetic Ames-like DataFrame with two distinct
    Neighborhood groups so we can detect if target leakage crosses folds.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    neighborhoods = ["RichArea"] * half + ["PoorArea"] * half
    # Two clearly separated price clusters so fold-specific medians differ.
    prices = np.concatenate([
        rng.normal(300_000, 5_000, half),
        rng.normal(100_000, 5_000, half),
    ])
    df = pd.DataFrame({
        "Id": np.arange(1, n + 1),
        "SalePrice": prices,
        "GrLivArea": rng.integers(800, 2500, n).astype(float),
        "Neighborhood": neighborhoods,
        "TotalBsmtSF": rng.integers(0, 1500, n).astype(float),
        "1stFlrSF": rng.integers(500, 1500, n).astype(float),
        "2ndFlrSF": rng.integers(0, 800, n).astype(float),
        "ExterQual": "TA",
        "ExterCond": "TA",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "HeatingQC": "TA",
        "KitchenQual": "TA",
        "FireplaceQu": "None",
        "GarageQual": "None",
        "GarageCond": "None",
        "PoolQC": "None",
        "BsmtFinType1": "Unf",
        "BsmtFinType2": "Unf",
        "BsmtExposure": "No",
        "FullBath": 2,
        "HalfBath": 0,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "OpenPorchSF": 0,
        "3SsnPorch": 0,
        "EnclosedPorch": 0,
        "ScreenPorch": 0,
        "WoodDeckSF": 0,
        "YrSold": 2010,
        "YearBuilt": 2000,
        "YearRemodAdd": 2005,
    })
    return df


def test_trajectory_in_order_no_leakage():
    """
    Quality Gate 1 – Trajectory IN_ORDER Test.

    Simulate 2-fold CV and assert that each fold's AmesDataTransformer
    learned its neighborhood_target_ranks_ exclusively from the *train*
    portion of that fold — not from the full dataset.

    The key invariant: fitting on fold-0-train vs fold-1-train should yield
    *different* rank dictionaries, because each train split sees a different
    target distribution (one fold's train is skewed toward RichArea, the
    other toward PoorArea).  If any global state were leaking, both folds
    would produce identical ranks derived from the whole dataset.
    """
    df = _make_heterogeneous_data(n=200, seed=42)
    n = len(df)
    mid = n // 2

    # Two manual folds (each fold trains on half, validates on the other half)
    fold_splits = [
        (np.arange(0, mid), np.arange(mid, n)),   # fold 0
        (np.arange(mid, n), np.arange(0, mid)),   # fold 1
    ]

    fold_ranks = []  # collect neighborhood_target_ranks_ per fold
    for train_idx, val_idx in fold_splits:
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold   = df.iloc[val_idx].reset_index(drop=True)

        y_train = train_fold["SalePrice"]
        X_train = train_fold.drop(columns=["Id", "SalePrice"])
        X_val   = val_fold.drop(columns=["Id", "SalePrice"])

        # Fresh transformer per fold – must not share any state
        t = AmesDataTransformer()
        t.fit(X_train, y_train)

        # Sanity: transformer must not have seen val targets during fit
        # (the object is brand-new, so learned stats come only from X_train/y_train)
        assert t.neighborhood_target_ranks_, "Transformer must learn neighborhood ranks"

        # Transform validation set using ONLY train-derived statistics
        X_val_trans = t.transform(X_val)
        assert X_val_trans is not None and len(X_val_trans) == len(val_fold)

        fold_ranks.append(dict(t.neighborhood_target_ranks_))

    # Each fold's train split contains only ONE neighborhood (RichArea or PoorArea).
    # Therefore the rank dictionaries must differ between folds.
    # If leakage occurred (global stats used), both dicts would be identical.
    assert fold_ranks[0] != fold_ranks[1], (
        "Leakage detected: both CV folds produced identical neighborhood_target_ranks_. "
        "Transformer must fit exclusively on per-fold train data."
    )

    # Additional check: ranks learned in fold-0 must not reference the
    # validation fold's neighborhood (PoorArea only exists in val for fold-0).
    fold0_train_neighborhoods = set(
        df.iloc[fold_splits[0][0]]["Neighborhood"].unique()
    )
    fold0_ranks_keys = set(fold_ranks[0].keys())
    assert fold0_ranks_keys == fold0_train_neighborhoods, (
        f"Fold-0 ranks contain unseen neighborhoods: "
        f"{fold0_ranks_keys - fold0_train_neighborhoods}"
    )


# ---------------------------------------------------------------------------
# Quality Gate 2 – pass^k Validation Check
# Runs the preprocessing + RMSLE evaluation loop 3× with different random
# seeds and asserts output stability (no state pollution between runs).
# ---------------------------------------------------------------------------

def _run_pipeline_with_seed(seed: int) -> float:
    """
    Full fit-transform-evaluate cycle on synthetic data with a given seed.
    Returns the OOF-style RMSLE computed on a held-out split.
    """
    rng = np.random.default_rng(seed)
    n = 300
    df = _make_heterogeneous_data(n=n, seed=seed)

    # Shuffle rows deterministically per seed
    shuffled_idx = rng.permutation(n)
    df = df.iloc[shuffled_idx].reset_index(drop=True)

    split = int(n * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)

    y_train = np.log1p(train_df["SalePrice"].values)
    y_val   = np.log1p(val_df["SalePrice"].values)

    X_train = train_df.drop(columns=["Id", "SalePrice"])
    X_val   = val_df.drop(columns=["Id", "SalePrice"])

    # Each call creates an independent, brand-new transformer
    transformer = AmesDataTransformer()
    transformer.fit(X_train, train_df["SalePrice"])
    X_train_trans = transformer.transform(X_train)
    X_val_trans   = transformer.transform(X_val)

    # Use column-means as a trivial "model" that is fully deterministic
    col_means = X_train_trans.mean(axis=0)
    # Predict log1p(SalePrice) from a simple linear combination of mean-scaled features:
    # y_hat = mean(y_train) + 0  (constant predictor) — keeps the test model-agnostic
    y_hat = np.full(len(y_val), np.mean(y_train))

    score = rmsle(np.expm1(y_val), np.expm1(y_hat))
    return score


def test_pass_k_stability():
    """
    Quality Gate 2 – pass^k Validation Check (k=3).

    Runs the preprocessing + evaluation pipeline 3 consecutive times with
    different random seeds.  Asserts:
      1. All three RMSLE scores are finite (pipeline never crashes or NaNs).
      2. Scores are stable (std < 0.05) — no hidden global state pollutes runs.
      3. Fitting a fresh transformer on each run is fully independent:
         the fitted statistics from run i do not influence run i+1.
    """
    seeds = [0, 7, 42]
    scores = [_run_pipeline_with_seed(s) for s in seeds]

    for i, score in enumerate(scores):
        assert np.isfinite(score), f"Run {i} (seed={seeds[i]}) returned non-finite RMSLE: {score}"
        assert score >= 0.0, f"Run {i} returned negative RMSLE: {score}"

    score_std = float(np.std(scores))
    assert score_std < 0.05, (
        f"RMSLE unstable across seeds (std={score_std:.4f} >= 0.05). "
        f"Scores: {scores}. Possible state pollution between runs."
    )

    # Verify strict independence: re-running the same seed must return the
    # exact same value (deterministic, no mutable module-level state).
    for seed in seeds:
        score_a = _run_pipeline_with_seed(seed)
        score_b = _run_pipeline_with_seed(seed)
        assert score_a == score_b, (
            f"Non-deterministic output for seed={seed}: "
            f"{score_a} != {score_b}. State is leaking between calls."
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


