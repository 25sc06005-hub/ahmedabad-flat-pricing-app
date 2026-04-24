import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt

from model.predict import load_trained_model, make_prediction
from preprocessing import clean_data


# ----------------------------
# CONFIG 
# ----------------------------
st.set_page_config(page_title="Amdavad Estate Pro", layout="wide")


# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def get_model():
    return load_trained_model()


# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    path = os.path.join("data", "ahmedabad_cleaned.csv")
    df = pd.read_csv(path)
    df = clean_data(df)
    return df


model = get_model()
df = load_data()


# ----------------------------
# HEADER
# ----------------------------
st.title("🏙️ Amdavad Estate Pro")
st.caption("AI-powered Real Estate Price Prediction System")


# ----------------------------
# SIDEBAR INPUT
# ----------------------------
st.sidebar.header("📍 Property Details")

loc = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
sqft = st.sidebar.number_input("Area (Sqft)", 300, 10000, 1200)
bhk = st.sidebar.slider("BHK", 1, 5, 2)


# ----------------------------
# PREDICTION
# ----------------------------
if st.sidebar.button("Predict Price", type="primary"):

    payload = {
        "location": loc,
        "total_sqft": sqft,
        "bhk": bhk
    }

    try:
        price = make_prediction(model, payload)

        st.success("Prediction Completed 🎯")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Estimated Price", f"₹ {price:,.2f} Lakhs")

        with col2:
            avg_loc_price = df[df["location"] == loc]["price"].mean()
            st.metric("Avg Market Price", f"₹ {avg_loc_price:,.2f} Lakhs")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ----------------------------
# MARKET ANALYSIS
# ----------------------------
st.write("---")
st.subheader("📊 Market Overview")

top_locations = (
    df.groupby("location")["price"]
    .mean()
    .reset_index()
    .sort_values("price", ascending=False)
    .head(20)
)

fig = px.bar(
    top_locations,
    x="location",
    y="price",
    title="Top 20 Expensive Locations in Ahmedabad"
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Insight: Location is the strongest predictor of price, followed by area (sqft) and BHK."
)


# ----------------------------
# FEATURE IMPORTANCE (SAFE VERSION)
# ----------------------------
st.write("---")
st.subheader("🧠 Model Explainability (Feature Importance)")

try:
    # safely access pipeline steps
    steps = model.named_steps
    preprocessor = steps[list(steps.keys())[0]]
    rf_model = steps[list(steps.keys())[-1]]

    # feature names
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["location"])
    num_features = ["total_sqft", "bhk"]

    feature_names = list(num_features) + list(cat_features)

    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig = plt.figure()
    importances.tail(15).plot(kind="barh")
    st.pyplot(fig)
    plt.close()

    st.write("### 📊 Top Features")
    st.dataframe(
        importances.sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )

except Exception as e:
    st.warning(f"Feature importance unavailable: {e}")