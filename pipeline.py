import subprocess

def run_pipeline():
    steps = [
        "python src/preprocess.py",
        "python src/optimize_xgboost.py",
        "python src/train_catboost.py",
        "python src/find_ensemble_weights.py",
        "python src/ensemble.py",
        "python src/recommend_portfolio.py"
    ]
    for step in steps:
        print(f"Running {step}...")
        subprocess.run(step, shell=True, check=True)

if __name__ == "__main__":
    run_pipeline()
