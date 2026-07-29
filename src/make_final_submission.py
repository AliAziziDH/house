import pandas as pd
import numpy as np
import joblib

# Load models and transformer
xgb_model = joblib.load('./models/xgboost_best.pkl')
catboost_model = joblib.load('./models/catboost_best.pkl')
pt = joblib.load('./models/boxcox_transformer.pkl')

# Load test data
X_test = pd.read_csv('./processed_data/X_test.csv')
test_ids = pd.read_csv('./data/test.csv')['Id']

# Best weights (from optimization)
weight_xgb = 0.64
weight_catboost = 0.36

# Predict (transformed scale)
xgb_pred = xgb_model.predict(X_test)
catboost_pred = catboost_model.predict(X_test)

# Inverse transform to original scale
xgb_pred_orig = pt.inverse_transform(xgb_pred.reshape(-1, 1)).flatten()
catboost_pred_orig = pt.inverse_transform(catboost_pred.reshape(-1, 1)).flatten()

# Weighted average
final_pred = weight_xgb * xgb_pred_orig + weight_catboost * catboost_pred_orig

# Create submission
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_pred})
submission.to_csv('./submissions/submission_ensemble_final.csv', index=False)

print("✅ Final submission saved to submissions/submission_ensemble_final.csv")
print(f"   Shape: {submission.shape}")
print(f"   First 5 rows:")
print(submission.head())