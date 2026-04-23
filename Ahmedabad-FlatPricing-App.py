import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="Ahmedabad Flat Price Predictor")

st.title("🏠 Ahmedabad Flat Price Prediction App")
st.write("Predict flat prices using Machine Learning on Kaggle dataset.")
st.write("---")

# ----------------------------
# Load dataset
# ----------------------------
DATA_PATH = "data"  # folder where you unzipped Kaggle dataset

csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

if len(csv_files) == 0:
    st.error("No CSV file found in data folder. Please unzip dataset properly.")
    st.stop()

file_path = os.path.join(DATA_PATH, csv_files[0])
df_raw = pd.read_csv("data/ahmedabad_cleaned.csv")


st.subheader("Dataset Preview")
st.write(df_raw.head())

# ----------------------------
# Target selection (AUTO)
# ----------------------------
# assume last column is target (price)
TARGET = df_raw.columns[-1]

X = df_raw.drop(columns=[TARGET])
y = df_raw[TARGET]

# handle non-numeric columns
X = X.select_dtypes(include=["int64", "float64"])
y = y

st.write("Features used for training:")
st.write(X.columns.tolist())

# ----------------------------
# Sidebar input
# ----------------------------
st.sidebar.header("Input Features")

def user_input():
    data = {}
    for col in X.columns:
        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ----------------------------
# Train model
# ----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# prediction
prediction = model.predict(input_df)

st.subheader("🏷 Predicted Price")
st.success(prediction[0])

# ----------------------------
# SHAP explainability
# ----------------------------
st.write("---")
st.subheader("📊 Feature Importance (SHAP)")

explainer = shap.TreeExplainer(model)

# use sample for speed
X_sample = X.sample(min(200, len(X)), random_state=1)
shap_values = explainer.shap_values(X_sample)

fig1 = plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
st.pyplot(fig1)

fig2 = plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(fig2)