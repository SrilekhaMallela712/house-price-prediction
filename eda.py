import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# =====================================
# BASIC INFORMATION
# =====================================

print("\n===== FIRST 5 ROWS =====\n")
print(df.head())

print("\n===== DATASET SHAPE =====\n")
print(df.shape)

print("\n===== COLUMN NAMES =====\n")
print(df.columns)

print("\n===== DATA TYPES =====\n")
print(df.dtypes)

# =====================================
# MISSING VALUES
# =====================================

print("\n===== MISSING VALUES =====\n")

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_values = missing_values.sort_values(ascending=False)

print(missing_values)

# =====================================
# STATISTICAL SUMMARY
# =====================================

print("\n===== STATISTICAL SUMMARY =====\n")

print(df.describe())

# =====================================
# SALE PRICE DISTRIBUTION
# =====================================

plt.figure(figsize=(8,5))

plt.hist(df["SalePrice"], bins=30)

plt.xlabel("SalePrice")
plt.ylabel("Count")
plt.title("Sale Price Distribution")

plt.show()

# =====================================
# CORRELATION ANALYSIS
# =====================================

print("\n===== TOP CORRELATED FEATURES =====\n")

corr_matrix = df.corr(numeric_only=True)

saleprice_corr = corr_matrix["SalePrice"].sort_values(ascending=False)

print(saleprice_corr.head(15))

# =====================================
# SCATTER PLOT
# =====================================

plt.figure(figsize=(8,5))

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Ground Living Area")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")

plt.show()