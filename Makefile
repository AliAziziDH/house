SHELL := /bin/bash

.PHONY: all preprocess optimize_xgb train_catboost optimize_lightgbm find_weights make_submissions clean

all: preprocess optimize_xgb train_catboost find_weights make_submissions

preprocess:
	python src/preprocess.py

optimize_xgb:
	python src/optimize_xgboost.py

train_catboost:
	python src/train_catboost.py

optimize_lightgbm:
	python experiments/train_lightgbm.py

find_weights:
	python src/find_ensemble_weights.py

make_submissions:
	# (a) Leaderboard weights: XGB 0.64, CAT 0.36
	python src/make_final_submission.py

	# (b) OOF-optimal weights: XGB 0.14, CAT 0.86
	python -c "from pathlib import Path; import joblib,pandas as pd; X_test=pd.read_csv('processed_data/X_test.csv'); x=joblib.load('models/xgboost_best_rmsle.pkl'); c=joblib.load('models/catboost_best_rmsle.pkl'); pt=joblib.load('models/boxcox_transformer.pkl'); x_o=pt.inverse_transform(x.predict(X_test).reshape(-1,1)).flatten(); c_o=pt.inverse_transform(c.predict(X_test).reshape(-1,1)).flatten(); import os; os.makedirs('submissions',exist_ok=True); pd.DataFrame({'Id':pd.read_csv('data/test.csv')['Id'],'SalePrice':0.14*x_o+0.86*c_o}).to_csv('submissions/submission_ensemble_0.14_0.86.csv',index=False)"

clean:
	rm -rf processed_data/* models/* submissions/* experiments/archive/*
