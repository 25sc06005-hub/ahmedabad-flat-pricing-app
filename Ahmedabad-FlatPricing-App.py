import streamlit as st
import pandas as pd
import plotly.express as px
import os
import shap
import matplotlib.pyplot as plt

from model.predict import load_trained_model, make_prediction
from preprocessing import clean_data


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Amdavad Estate Pro", layout="wide")



# ----------------------------
# LOAD RESOURCES
# ----------------------------
@st.cache_resource
def get_model():
    return load_trained_model()

@st.cache_data
def load_data():
    path = os.path.join("data", "ahmedabad_cleaned.csv")
    df = pd.read_csv(path)
    return clean_data(df)

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




# ----------------------------
# SHAP EXPLAINABILITY
# ----------------------------
st.write("---")
st.subheader("🧠 Model Explainability (SHAP)")


@st.cache_data
def compute_shap(_model, X_sample):
    X_transformed = _model.named_steps["prep"].transform(X_sample)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.TreeExplainer(_model.named_steps["reg"])
    shap_values = explainer.shap_values(X_transformed)

    return X_transformed, shap_values


X_sample = df[["total_sqft", "bhk", "location"]].sample(200, random_state=42)

X_transformed, shap_values = compute_shap(model, X_sample)

fig = plt.figure()
shap.summary_plot(shap_values, X_transformed, show=False)

st.pyplot(fig)
plt.close()