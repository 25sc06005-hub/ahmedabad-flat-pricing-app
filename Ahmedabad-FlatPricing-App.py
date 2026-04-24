import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import streamlit as st

# ----------------------------
# CONFIG (ONLY ONCE)
# ----------------------------
st.set_page_config(page_title="Ahmedabad Flat Price Predictor", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:18px !important;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Ahmedabad Flat Price Prediction App")
st.write("ML-powered real estate analysis using Kaggle dataset")
st.write("---")

DATA_PATH = "data/ahmedabad_cleaned.csv"

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

df["location"] = df["location"].replace("other", "Other").fillna("Other")
df["price_segment"] = pd.qcut(df["price"], q=3, labels=["Low", "Mid", "High"])

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# TARGET
# ----------------------------
TARGET = "price"
y = df[TARGET]
X = df.drop(columns=[TARGET])

categorical_cols = ["location"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ----------------------------
# PREPROCESSING + MODEL
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

@st.cache_resource
def train_model(X, y):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model

model = train_model(X, y)

# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.header("📍 Enter Flat Details")

def user_input():
    data = {}

    for col in numeric_cols:
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    data["location"] = st.sidebar.selectbox(
        "location",
        sorted(df["location"].unique())
    )

    return pd.DataFrame([data])

input_df = user_input()

# ----------------------------
# PREDICTION
# ----------------------------
prediction = model.predict(input_df)[0]

st.subheader("🏷 Predicted Flat Price")
st.success(f"₹ {prediction:,.2f} Lakhs")

# ----------------------------
# LOCATION HEATMAP
# ----------------------------
st.write("---")
st.subheader("🗺 Ahmedabad Area-wise Price Heatmap")

area_df = df.groupby("location").agg(
    avg_price=("price", "mean"),
    avg_price_sqft=("price_sqft", "mean"),
    count=("price", "count")
).reset_index()

area_df = area_df[area_df["count"] > 20]
area_df = area_df.sort_values("avg_price", ascending=False).head(30)

fig = px.bar(
    area_df,
    x="location",
    y="avg_price",
    color="avg_price",
    text="count",
    color_continuous_scale="Turbo",
    title="Top 30 Ahmedabad Locations by Average Price"
)

fig.update_layout(
    xaxis_title="Location",
    yaxis_title="Average Price (Lakhs)",
    xaxis_tickangle=-45
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FEATURE IMPORTANCE (FIXED)
# ----------------------------
st.write("---")
st.subheader("📊 Feature Importance")

rf_model = model.named_steps["regressor"]
encoded_features = (
    numeric_cols +
    list(model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(["location"]))
)

importances = pd.Series(rf_model.feature_importances_, index=encoded_features)
importances = importances.sort_values()

fig, ax = plt.subplots()
importances.tail(15).plot(kind="barh", ax=ax)
st.pyplot(fig)

# ----------------------------
# INSIGHT PANEL
# ----------------------------
st.write("---")
st.subheader("🧠 Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Top Features")
    st.dataframe(importances.tail(10)[::-1])

with col2:
    st.info("Price is strongly influenced by location and size-related features.")

# ----------------------------
# REAL ESTATE INTELLIGENCE
# ----------------------------
area_stats = df.groupby("location").agg(
    avg_price=("price", "mean"),
    avg_psf=("price_sqft", "mean"),
    count=("price", "count")
).reset_index()

market_mean = df["price_sqft"].mean()
area_stats["value_score"] = market_mean / area_stats["avg_psf"]

def recommend(v):
    if v > 1.2:
        return "🟢 Strong Buy"
    elif v > 1.0:
        return "🟡 Moderate Buy"
    return "🔴 Overpriced"

area_stats["recommendation"] = area_stats["value_score"].apply(recommend)

fig = px.scatter(
    area_stats,
    x="avg_psf",
    y="avg_price",
    size="count",
    color="recommendation",
    hover_name="location",
    title="Ahmedabad Real Estate Intelligence Map"
)

st.plotly_chart(fig, use_container_width=True)

st.success(
    f"Top Insight: Most expensive area is {area_stats.sort_values('avg_price', ascending=False).iloc[0]['location']}"
)

st.subheader("🧠 AI Insights")

st.write("### 🟢 Best Value Locations")
st.dataframe(area_stats.sort_values("value_score", ascending=False).head(5)[["location","value_score","avg_price"]])

st.write("### 🔴 Overpriced Locations")
st.dataframe(area_stats.sort_values("value_score").head(5)[["location","value_score","avg_price"]])