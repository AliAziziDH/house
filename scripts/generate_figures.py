import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Set seaborn style for publication quality
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

os.makedirs('assets/images', exist_ok=True)

print("Generating Conformal Prediction Intervals figure...")
# 1. Conformal Prediction Intervals
try:
    df_intervals = pd.read_csv('submissions/submission_oof_intervals.csv')

    # Sort by predicted SalePrice to get a smooth curve
    df_sorted = df_intervals.sort_values(by='SalePrice').reset_index(drop=True)

    # Take a sample of 70 houses to make the plot legible
    # We take evenly spaced indices
    indices = np.linspace(0, len(df_sorted) - 1, 70, dtype=int)
    df_sample = df_sorted.iloc[indices].reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # Plot the intervals as error bars or filled area
    plt.fill_between(
        df_sample.index,
        df_sample['Price_Lower_Bound'],
        df_sample['Price_Upper_Bound'],
        alpha=0.3,
        color='steelblue',
        label='95% Conformal Interval'
    )

    # Plot the point predictions
    plt.plot(
        df_sample.index,
        df_sample['SalePrice'],
        color='navy',
        linewidth=2,
        label='Point Prediction (Ensemble)'
    )

    plt.title('Inductive Conformal Prediction Intervals on Test Set', fontweight='bold', pad=15)
    plt.xlabel('Test Instance (Sorted by Predicted Price)', labelpad=10)
    plt.ylabel('Predicted Sale Price ($)', labelpad=10)

    # Format y-axis as currency
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])

    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('assets/images/conformal_prediction_intervals.png')
    plt.close()
    print("✅ Created assets/images/conformal_prediction_intervals.png")
except Exception as e:
    print(f"Error generating intervals plot: {e}")

print("Generating Residual Error Distribution figure...")
# 2. Residual Error Distribution
try:
    # We need to construct the ensemble predictions on the train set (OOF)
    y_train = pd.read_csv('processed_data/y_train.csv').squeeze()

    # Transform to log space since models were evaluated on log(SalePrice)
    y_train_log = np.log1p(y_train)

    # Load model OOF predictions
    oof_xgb = pd.read_csv('processed_data/oof_xgboost.csv').iloc[:, 0].squeeze()
    oof_cat = pd.read_csv('processed_data/oof_catboost.csv').iloc[:, 0].squeeze()

    # Load weights
    weights = pd.read_csv('experiments/slsqp_weights.csv')
    w_dict = dict(zip(weights['model'], weights['weight']))

    w_xgb = w_dict.get('XGBoost', 0.5)
    w_cat = w_dict.get('CatBoost', 0.5)
    w_lin = w_dict.get('Linear', 0.0)

    # Reconstruct OOF ensemble prediction (in log space)
    # The saved OOFs might be in original space depending on the script, let's check
    # In src/train.py they are inverse transformed to original space.
    # So we need to log them again to compare with y_train_log
    oof_xgb_log = np.log1p(oof_xgb)
    oof_cat_log = np.log1p(oof_cat)

    oof_ensemble_log = (w_xgb * oof_xgb_log) + (w_cat * oof_cat_log)

    # Calculate residuals
    residuals = y_train_log - oof_ensemble_log

    plt.figure(figsize=(9, 6))

    # Plot KDE and histogram
    sns.histplot(residuals, kde=True, stat="density", color="mediumpurple", alpha=0.5, bins=50)

    # Add vertical lines for mean and std deviations
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    plt.axvline(mean_res, color='k', linestyle='-', linewidth=2, label=f'Mean: {mean_res:.4f}')
    plt.axvline(mean_res + std_res, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label=f'+1 Std Dev: {std_res:.4f}')
    plt.axvline(mean_res - std_res, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label=f'-1 Std Dev: -{std_res:.4f}')

    plt.title('Out-of-Fold (OOF) Log-Residual Error Distribution', fontweight='bold', pad=15)
    plt.xlabel(r'Residual Error: $\log(y_{true}) - \log(\hat{y}_{pred})$', labelpad=10)
    plt.ylabel('Density', labelpad=10)
    plt.legend(loc='upper right', frameon=True)

    # Set symmetric x-limits based on std dev
    limit = max(abs(residuals.min()), abs(residuals.max())) * 1.05
    plt.xlim(-limit, limit)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('assets/images/residual_error_distribution.png')
    plt.close()
    print("✅ Created assets/images/residual_error_distribution.png")
except Exception as e:
    print(f"Error generating residual plot: {e}")

print("Generating SLSQP Weight Allocation figure...")
# 3. SLSQP Weight Allocation
try:
    weights = pd.read_csv('experiments/slsqp_weights.csv')

    # Sort for plotting
    weights = weights.sort_values('weight', ascending=True)

    plt.figure(figsize=(8, 5))

    # Create horizontal bar chart
    bars = plt.barh(weights['model'], weights['weight'], color='teal', alpha=0.8, edgecolor='black', height=0.6)

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        if width > 0.01:  # Only label non-zero weights
            plt.text(width - 0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}',
                     ha='right', va='center', color='white', fontweight='bold')

    plt.title('SLSQP Convex Blending Weights Allocation', fontweight='bold', pad=15)
    plt.xlabel(r'Optimal Weight ($\sum = 1.0, w_i \geq 0$)', labelpad=10)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add model labels explicitly if they are 0
    plt.yticks(range(len(weights)), weights['model'])

    plt.tight_layout()
    plt.savefig('assets/images/slsqp_weight_allocation.png')
    plt.close()
    print("✅ Created assets/images/slsqp_weight_allocation.png")
except Exception as e:
    print(f"Error generating weights plot: {e}")

print("All figures generated successfully.")
