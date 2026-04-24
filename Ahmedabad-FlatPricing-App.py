import streamlit as st
import pandas as pd
import plotly.express as px
from model.predict import load_trained_model, make_prediction
from preprocessing import clean_data




st.set_page_config(page_title="Amdavad Estate Pro", layout="wide")



# Load Resources
@st.cache_resource
def get_resources():
    model = load_trained_model()
    # Load data for UI elements like dropdowns
    df = pd.read_csv("data/ahmedabad_cleaned.csv")
    df = clean_data(df)
    return model, df

model, df = get_resources()

st.title("🏙️ Amdavad Estate Pro")




# UI Sidebar for inputs
st.sidebar.header("Property Details")
loc = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
sqft = st.sidebar.number_input("Area (Sqft)", 300, 10000, 1200)
bhk = st.sidebar.slider("BHK", 1, 5, 2)




if st.sidebar.button("Predict Price", type="primary"):
    payload = {"location": loc, "total_sqft": sqft, "bhk": bhk}
    price = make_prediction(model, payload)
    
    st.metric("Estimated Market Value", f"₹ {price:,.2f} Lakhs")
    
    # Simple Visual Context
    avg_loc_price = df[df["location"] == loc]["price"].mean()
    st.write(f"The average price in **{loc}** is ₹{avg_loc_price:.2f} Lakhs.")

# Add your tabs for Maps and Analytics here...