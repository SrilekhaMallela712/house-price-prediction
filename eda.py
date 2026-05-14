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
import numpy as np

# =====================================
# HANDLING MISSING VALUES
# =====================================

# Fill categorical missing values

categorical_cols = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu"
]

for col in categorical_cols:
    df[col] = df[col].fillna("None")

# Fill numerical missing values using median

numerical_cols = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print("\n===== MISSING VALUES AFTER HANDLING =====\n")

print(df.isnull().sum().sort_values(ascending=False).head(10))

# =====================================
# FEATURE ENGINEERING
# =====================================

# Total square feet

df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

# Total bathrooms

df["TotalBathrooms"] = (
    df["FullBath"] +
    (0.5 * df["HalfBath"]) +
    df["BsmtFullBath"] +
    (0.5 * df["BsmtHalfBath"])
)

# House age

df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

print("\n===== NEW FEATURES =====\n")

print(df[["TotalSF", "TotalBathrooms", "HouseAge"]].head())

# =====================================
# LOG TRANSFORMATION
# =====================================

df["SalePriceLog"] = np.log1p(df["SalePrice"])

print("\n===== LOG TRANSFORMATION =====\n")

print(df[["SalePrice", "SalePriceLog"]].head())

# =====================================
# OUTLIER DETECTION
# =====================================

plt.figure(figsize=(8,5))

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.title("Outlier Detection")

plt.show()

# Remove extreme outliers

df = df[df["GrLivArea"] < 4500]

print("\n===== DATASET SHAPE AFTER OUTLIER REMOVAL =====\n")

print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =====================================
# SELECT FEATURES
# =====================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "TotalSF",
    "TotalBathrooms",
    "HouseAge"
]

X = df[features]

y = df["SalePriceLog"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== TRAIN TEST SPLIT =====\n")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =====================================
# LINEAR REGRESSION
# =====================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
lr_r2 = r2_score(y_test, lr_pred)

print("\n===== LINEAR REGRESSION =====\n")

print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

# =====================================
# RANDOM FOREST
# =====================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== RANDOM FOREST =====\n")

print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# =====================================
# XGBOOST REGRESSOR
# =====================================

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_rmse = mean_squared_error(y_test, xgb_pred) ** 0.5
xgb_r2 = r2_score(y_test, xgb_pred)

print("\n===== XGBOOST =====\n")

print("RMSE:", xgb_rmse)
print("R2 Score:", xgb_r2)
