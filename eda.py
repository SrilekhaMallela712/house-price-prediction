import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("train.csv")

# =====================================
# BASIC INFORMATION
# =====================================

print("\n===== FIRST 5 ROWS =====\n")
print(df.head())

print("\n===== DATASET SHAPE =====\n")
print(df.shape)

print("\n===== MISSING VALUES =====\n")

missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

missing_values = missing_values.sort_values(ascending=False)

print(missing_values.head(15))

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

sale_corr = corr_matrix["SalePrice"].sort_values(ascending=False)

print(sale_corr.head(15))

# =====================================
# HANDLE MISSING VALUES
# =====================================

categorical_cols = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu"
]

for col in categorical_cols:
    df[col] = df[col].fillna("None")

numerical_cols = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\n===== MISSING VALUES AFTER CLEANING =====\n")

print(df.isnull().sum().sum())

# =====================================
# FEATURE ENGINEERING
# =====================================

df["TotalSF"] = (
    df["TotalBsmtSF"] +
    df["GrLivArea"]
)

df["TotalBathrooms"] = (
    df["FullBath"] +
    (0.5 * df["HalfBath"]) +
    df["BsmtFullBath"] +
    (0.5 * df["BsmtHalfBath"])
)

df["HouseAge"] = (
    df["YrSold"] -
    df["YearBuilt"]
)

print("\n===== NEW FEATURES CREATED =====\n")

print(
    df[
        [
            "TotalSF",
            "TotalBathrooms",
            "HouseAge"
        ]
    ].head()
)

# =====================================
# LOG TRANSFORMATION
# =====================================

df["SalePriceLog"] = np.log1p(df["SalePrice"])

print("\n===== LOG TRANSFORMATION DONE =====\n")

print(
    df[
        [
            "SalePrice",
            "SalePriceLog"
        ]
    ].head()
)

# =====================================
# OUTLIER DETECTION
# =====================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["GrLivArea"],
    df["SalePrice"]
)

plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("Outlier Detection")

plt.show()

# Remove outliers

df = df[df["GrLivArea"] < 4500]

print("\n===== DATASET SHAPE AFTER OUTLIER REMOVAL =====\n")

print(df.shape)

# =====================================
# MACHINE LEARNING MODELS
# =====================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

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

print(X_train.shape)
print(X_test.shape)

# =====================================
# LINEAR REGRESSION
# =====================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_rmse = mean_squared_error(
    y_test,
    lr_pred
) ** 0.5

lr_r2 = r2_score(
    y_test,
    lr_pred
)

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

rf_rmse = mean_squared_error(
    y_test,
    rf_pred
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_pred
)

print("\n===== RANDOM FOREST =====\n")

print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# =====================================
# XGBOOST
# =====================================

xgb_model = XGBRegressor(

    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42

)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_rmse = mean_squared_error(
    y_test,
    xgb_pred
) ** 0.5

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)

print("\n===== XGBOOST =====\n")

print("RMSE:", xgb_rmse)
print("R2 Score:", xgb_r2)

# =====================================
# OPTUNA HYPERPARAMETER TUNING
# =====================================

import optuna

def objective(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),

        "random_state": 42
    }

    model = XGBRegressor(**params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    return rmse

print("\n===== RUNNING OPTUNA =====\n")

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=10
)

print("\n===== BEST PARAMETERS =====\n")

print(study.best_params)

print("\n===== BEST RMSE =====\n")

print(study.best_value)

# =====================================
# FINAL TUNED MODEL
# =====================================

best_model = XGBRegressor(

    **study.best_params,
    random_state=42

)

best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)

best_rmse = mean_squared_error(
    y_test,
    best_predictions
) ** 0.5

best_r2 = r2_score(
    y_test,
    best_predictions
)

print("\n===== TUNED XGBOOST RESULTS =====\n")

print("RMSE:", best_rmse)
print("R2 Score:", best_r2)

# =====================================
# SHAP EXPLAINABILITY
# =====================================

# =====================================
# SHAP EXPLAINABILITY
# =====================================

# =====================================
# SHAP EXPLAINABILITY
# =====================================

import shap

print("\n===== SHAP ANALYSIS STARTED =====\n")

# Use Random Forest Model

explainer = shap.TreeExplainer(rf_model)

# Generate SHAP values

shap_values = explainer.shap_values(X_test)

# =====================================
# FEATURE IMPORTANCE BAR PLOT
# =====================================

print("\n===== FEATURE IMPORTANCE =====\n")

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)

plt.show()

# =====================================
# SHAP SUMMARY PLOT
# =====================================

print("\n===== SHAP SUMMARY PLOT =====\n")

shap.summary_plot(
    shap_values,
    X_test
)

plt.show()
