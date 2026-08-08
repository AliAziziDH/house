import os

import pandas as pd
import sklearn

print("⚡ Container Python Environment: HEALTHY")
print(f"📦 Pandas version: {pd.__version__}")
print(f"📦 Scikit-Learn version: {sklearn.__version__}")

# Check if Kaggle data directory is mounted correctly
data_dir = "data"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"📂 Mounted 'data' directory found with {len(files)} files: {files}")

    # Try reading the training dataset
    train_path = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print(f"✅ Success! Loaded train.csv cleanly. Shape: {df.shape}")
    else:
        print("⚠️ 'train.csv' not found inside the data folder.")
else:
    print("❌ Error: 'data' folder not found in workspace root.")
