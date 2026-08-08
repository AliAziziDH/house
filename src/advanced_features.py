import featuretools as ft
import pandas as pd

# Load data
df = pd.read_csv("./data/train.csv")

# Create entity set
es = ft.EntitySet(id="house_prices")
es = es.add_dataframe(dataframe_name="houses", dataframe=df, index="Id")

# Deep feature synthesis (DFS)
features, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="houses",
    trans_primitives=["add", "multiply", "divide"],
    agg_primitives=["mean", "sum", "std", "max", "min"],
    max_depth=2,
)

# Save new features
features.to_csv("./processed_data/advanced_features.csv", index=False)
print("✅ Advanced features saved.")
