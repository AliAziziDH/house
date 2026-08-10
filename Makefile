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
	# (a) Leaderboard weights: XGB 0.1667, CAT 0.1665
	python src/ensemble.py

	# (b) OOF-optimal weights: XGB 0.1667, CAT 0.1665
	# removed

clean:
	rm -rf processed_data/* models/* submissions/* experiments/archive/*
