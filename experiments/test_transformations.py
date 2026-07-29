import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load preprocessed data (assuming X_train and y_train are available)
# For this test, we'll use the data from preprocess.py
# You can save X_train and y_train to CSV files after preprocessing

print("=" * 60)
print("TARGET TRANSFORMATION COMPARISON")
print("=" * 60)

# Assuming X_train and y_train are already loaded from preprocess.py
# If not, load them from CSV files:
# X_train = pd.read_csv('X_train.csv')
# y_train = pd.read_csv('y_train.csv').squeeze()

# Define transformations to test
transformations = {
    'Log': lambda y: np.log1p(y),
    'Box-Cox': lambda y: PowerTransformer(method='box-cox').fit_transform(y.values.reshape(-1, 1)).flatten(),
    'Yeo-Johnson': lambda y: PowerTransformer(method='yeo-johnson').fit_transform(y.values.reshape(-1, 1)).flatten(),
}

# Inverse transformations for predictions
inverse_transforms = {
    'Log': lambda y_pred: np.expm1(y_pred),
    'Box-Cox': lambda y_pred: PowerTransformer(method='box-cox').inverse_transform(y_pred.reshape(-1, 1)).flatten(),
    'Yeo-Johnson': lambda y_pred: PowerTransformer(method='yeo-johnson').inverse_transform(y_pred.reshape(-1, 1)).flatten(),
}

# Store results
results = {}

# Use 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

for name, transform_func in transformations.items():
    print(f"\nTesting {name} transformation...")
    
    # Transform target
    y_transformed = transform_func(y_train)
    
    # Cross-validation scores (RMSE on original scale)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed.iloc[train_idx], y_transformed.iloc[val_idx]
        
        # Train model on transformed target
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold (in transformed scale)
        y_pred_transformed = model.predict(X_val_fold)
        
        # Inverse transform to original scale
        y_pred_original = inverse_transforms[name](y_pred_transformed)
        
        # Calculate RMSE on original scale
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_original))
        rmse_scores.append(rmse)
    
    # Store results
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    results[name] = {'avg_rmse': avg_rmse, 'std_rmse': std_rmse}
    
    print(f"   Average RMSE: {avg_rmse:.4f} (+/- {std_rmse:.4f})")

# Print comparison
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, metrics in results.items():
    print(f"   {name}: RMSE = {metrics['avg_rmse']:.4f} (+/- {metrics['std_rmse']:.4f})")

# Find best transformation
best_transform = min(results, key=lambda x: results[x]['avg_rmse'])
print(f"\n✅ Best transformation: {best_transform} with RMSE = {results[best_transform]['avg_rmse']:.4f}")