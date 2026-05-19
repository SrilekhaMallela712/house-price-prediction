import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("train.csv")

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

df["SalePriceLog"] = np.log1p(df["SalePrice"])

# =====================================
# FEATURES
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
# TRAIN MODEL
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

model = XGBRegressor(

    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42

)

model.fit(X_train, y_train)

# =====================================
# STREAMLIT UI
# =====================================

st.title("🏠 House Price Prediction App")

st.write(
    "Predict house prices using Machine Learning"
)

# =====================================
# USER INPUTS
# =====================================

overall_qual = st.slider(
    "Overall Quality",
    1,
    10,
    5
)

gr_liv_area = st.number_input(
    "Ground Living Area",
    500,
    5000,
    1500
)

garage_cars = st.slider(
    "Garage Cars Capacity",
    0,
    5,
    2
)

total_bsmt_sf = st.number_input(
    "Basement Area",
    0,
    3000,
    800
)

full_bath = st.slider(
    "Full Bathrooms",
    0,
    5,
    2
)

year_built = st.number_input(
    "Year Built",
    1900,
    2025,
    2005
)

total_sf = (
    total_bsmt_sf +
    gr_liv_area
)

total_bathrooms = full_bath

house_age = 2025 - year_built

# =====================================
# PREDICTION
# =====================================

input_data = pd.DataFrame({

    "OverallQual": [overall_qual],
    "GrLivArea": [gr_liv_area],
    "GarageCars": [garage_cars],
    "TotalBsmtSF": [total_bsmt_sf],
    "FullBath": [full_bath],
    "YearBuilt": [year_built],
    "TotalSF": [total_sf],
    "TotalBathrooms": [total_bathrooms],
    "HouseAge": [house_age]

})

if st.button("Predict House Price"):

    prediction_log = model.predict(input_data)

    prediction = np.expm1(prediction_log)

    st.success(
        f"Estimated House Price: ${prediction[0]:,.2f}"
    )