import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import streamlit as st


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Ahmedabad Flat Price Predictor", layout="wide")

st.title("🏠 Ahmedabad Flat Price Prediction App")
st.write("ML-powered real estate analysis using Kaggle dataset")
st.write("---")

DATA_PATH = "data/ahmedabad_cleaned.csv"


# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Basic Cleaning
    df["location"] = df["location"].astype(str).fillna("Other")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_sqft"] = pd.to_numeric(df["price_sqft"], errors="coerce")
    
    # Drop rows where critical data is missing
    df = df.dropna(subset=["price", "total_sqft", "location"])
    return df

df = load_data()



# ----------------------------
# FEATURES & DATA LEAKAGE FIX
# ----------------------------
TARGET = "price"

# CRITICAL: We drop 'price_sqft' because it's derived from price. 
# Keeping it is "Data Leakage" and makes the model fake.
features_to_drop = [TARGET, "price_sqft", "price_segment"] 
X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
y = df[TARGET]

categorical_cols = ["location"]
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()



# ----------------------------
# MODEL TRAINING & EVALUATION
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

@st.cache_resource
def train_and_evaluate(X, y):
    # Split data to see how it performs on "unseen" flats
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100, # 100 is usually enough and less prone to heavy overfitting
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train) 

    # Calculate Metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae

model, r2_val, mae_val = train_and_evaluate(X, y)



# Show Metrics in UI
m_col1, m_col2 = st.columns(2)
m_col1.metric("Model Accuracy (R²)", f"{r2_val:.2%}")
m_col2.metric("Avg. Prediction Error", f"₹ {mae_val:.2f} Lakhs")



# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.header("📍 Enter Flat Details")

def user_input():
    data = {}
    for col in numeric_cols:
        data[col] = st.sidebar.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
    data["location"] = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
    return pd.DataFrame([data])

input_df = user_input()



# ----------------------------
# PREDICTION
# ----------------------------
prediction = model.predict(input_df)[0]
st.subheader("🏷 Predicted Flat Price")
st.success(f"₹ {prediction:,.2f} Lakhs")



# ----------------------------
# LOCATION HEATMAP (Filtered "Other")
# ----------------------------
st.write("---")
st.subheader("🗺 Ahmedabad Area-wise Price Heatmap")

area_df = df.groupby("location").agg(
    avg_price=("price", "mean"),
    count=("price", "count")
).reset_index()



# Filter out "Other" and low-sample areas
area_df = area_df[(area_df["location"] != "Other") & (area_df["count"] > 10)]
area_df = area_df.sort_values("avg_price", ascending=False).head(30)

fig = px.bar(
    area_df,
    x="location",
    y="avg_price",
    color="avg_price",
    text="count",
    labels={'count': 'Listings', 'avg_price': 'Avg Price (Lakhs)'},
    color_continuous_scale="Turbo",
    title="Top 30 Ahmedabad Locations by Price (Min 10 Listings)"
)
st.plotly_chart(fig, use_container_width=True)



# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.write("---")
st.subheader("📊 Feature Importance")

rf = model.named_steps["regressor"]
encoded_features = (
    numeric_cols +
    list(model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(["location"]))
)

importances = pd.Series(rf.feature_importances_, index=encoded_features)
importances = importances.sort_values()

fig, ax = plt.subplots()
importances.tail(10).plot(kind="barh", ax=ax, color='skyblue')
plt.title("What drives the price?")
st.pyplot(fig)



# ----------------------------
# REAL ESTATE INTELLIGENCE
# ----------------------------
st.write("---")
st.subheader("🧠 Market Insights")

# Prepare Intelligence Map Data
market_stats = df[df["location"] != "Other"].groupby("location").agg(
    avg_price=("price", "mean"),
    avg_psf=("price_sqft", "mean"),
    count=("price", "count")
).reset_index()

market_mean_psf = df["price_sqft"].mean()
market_stats["value_score"] = market_mean_psf / market_stats["avg_psf"]

def recommend(v):
    if v > 1.15: return "🟢 Strong Buy (Underpriced)"
    if v > 0.9: return "🟡 Fair Market Value"
    return "🔴 Premium/Overpriced"

market_stats["recommendation"] = market_stats["value_score"].apply(recommend)



# Filter for the final insight to ensure high confidence
reliable_stats = market_stats[market_stats["count"] > 5]
top_area = reliable_stats.sort_values("avg_price", ascending=False).iloc[0]

st.info(f"**Top Insight:** The most expensive area with reliable data is **{top_area['location']}**, averaging ₹{top_area['avg_price']:.2f} Lakhs.")

fig_scatter = px.scatter(
    market_stats[market_stats["count"] > 2],
    x="avg_psf",
    y="avg_price",
    size="count",
    color="recommendation",
    hover_name="location",
    title="Price vs. Price Per Sqft Intelligence Map"
)
st.plotly_chart(fig_scatter, use_container_width=True)