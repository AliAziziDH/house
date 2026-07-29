import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات نمایش
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# بارگذاری داده
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print("=" * 60)
print("HOUSE PRICES - DATA OVERVIEW")
print("=" * 60)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nFirst 5 rows of train:")
print(train.head())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print(f"Missing values in train:\n{missing_train}")

print("\n" + "=" * 60)
print("TARGET VARIABLE (SalePrice)")
print("=" * 60)
print(f"Mean: {train['SalePrice'].mean():.2f}")
print(f"Median: {train['SalePrice'].median():.2f}")
print(f"Skewness: {train['SalePrice'].skew():.2f}")

# توزیع هدف
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train['SalePrice'], bins=50, kde=True)
plt.title('Distribution of SalePrice')

plt.subplot(1, 2, 2)
sns.boxplot(x=train['SalePrice'])
plt.title('Boxplot of SalePrice')

plt.tight_layout()
plt.savefig('sale_price_distribution.png', dpi=150)
plt.show()
print("Saved: sale_price_distribution.png")

print("\n" + "=" * 60)
print("INITIAL ANALYSIS COMPLETE")
print("=" * 60)