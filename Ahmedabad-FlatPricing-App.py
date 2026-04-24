import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

from model.predict import load_trained_model, make_prediction
from preprocessing import clean_data

# ----------------------------
# 1. CONFIG & STYLING
# ----------------------------
st.set_page_config(page_title="Amdavad Estate Pro", layout="wide", page_icon="🏢")

# Injecting some professional styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: white; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 2. RESOURCE LOADING
# ----------------------------
@st.cache_resource
def get_model():
    return load_trained_model()

@st.cache_data
def load_data():
    path = os.path.join("data", "ahmedabad_cleaned.csv")
    df = pd.read_csv(path)
    df = clean_data(df)
    return df

model = get_model()
df = load_data()

# ----------------------------
# 3. HEADER & TABS
# ----------------------------
st.title("🏙️ Amdavad Estate Pro")
st.markdown("#### The Intelligent Way to Value Property in Ahmedabad")

# Use tabs to keep the interface clean
tab1, tab2, tab3 = st.tabs(["🎯 Valuation Calculator", "📈 Market Trends", "🧬 Model Insights"])

# ----------------------------
# TAB 1: VALUATION CALCULATOR
# ----------------------------
with tab1:
    col_sidebar, col_main = st.columns([1, 2], gap="large")

    with col_sidebar:
        st.subheader("Property Specs")
        with st.container(border=True):
            loc = st.selectbox("Select Location", sorted(df["location"].unique()))
            sqft = st.number_input("Area (Sqft)", 300, 10000, 1200, step=50)
            bhk = st.select_slider("Number of BHK", options=[1, 2, 3, 4, 5], value=2)
            
            predict_btn = st.button("Generate Estimate", type="primary")

    with col_main:
        if predict_btn:
            payload = {"location": loc, "total_sqft": sqft, "bhk": bhk}
            
            with st.spinner("Analyzing market data..."):
                try:
                    price = make_prediction(model, payload)
                    avg_loc_price = df[df["location"] == loc]["price"].mean()
                    psf = (price * 100000) / sqft # Assuming price is in Lakhs
                    
                    st.success(f"### Estimated Value: ₹ {price:,.2f} Lakhs")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Estimate", f"₹{price:.1f}L")
                    
                    # Calculate delta vs area average
                    diff = price - avg_loc_price
                    m2.metric("Area Average", f"₹{avg_loc_price:.1f}L", f"{diff:.1f}L", delta_color="inverse")
                    
                    m3.metric("Price per Sqft", f"₹{psf:,.0f}")
                    
                    # Visual comparison
                    comp_df = pd.DataFrame({
                        "Type": ["Your Estimate", f"Average in {loc}"],
                        "Price": [price, avg_loc_price]
                    })
                    fig_comp = px.bar(comp_df, x="Type", y="Price", color="Type", 
                                     color_discrete_sequence=["#007BFF", "#6c757d"])
                    st.plotly_chart(fig_comp, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.info("👈 Enter property details and click 'Generate Estimate' to see the AI valuation.")

# ----------------------------
# TAB 2: MARKET TRENDS
# ----------------------------
with tab2:
    st.subheader("Ahmedabad Market Heatmap")
    
    # Filter for top locations with enough data points
    min_listings = st.slider("Minimum listings per area", 5, 50, 15)
    
    market_df = df.groupby("location").agg(
        avg_price=("price", "mean"),
        count=("price", "count"),
        avg_psf=("price_sqft", "mean")
    ).reset_index()
    
    market_df = market_df[market_df["count"] >= min_listings].sort_values("avg_price", ascending=False)

    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        fig_market = px.scatter(market_df, x="avg_psf", y="avg_price", size="count", 
                               color="avg_price", hover_name="location",
                               labels={"avg_psf": "Price per Sqft", "avg_price": "Avg Flat Price (Lakhs)"},
                               title="Market Maturity: Price vs. Value per Sqft")
        st.plotly_chart(fig_market, use_container_width=True)

    with col_list:
        st.write("**Top 10 Premium Areas**")
        st.dataframe(market_df[["location", "avg_price"]].head(10), hide_index=True)

# ----------------------------
# TAB 3: MODEL INSIGHTS
# ----------------------------
with tab3:
    st.subheader("How the AI thinks")
    
    try:
        steps = model.named_steps
        preprocessor = steps[list(steps.keys())[0]]
        rf_model = steps[list(steps.keys())[-1]]

        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["location"])
        num_features = ["total_sqft", "bhk"]
        feature_names = list(num_features) + list(cat_features)

        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)

        # Plotly version for better interactivity
        fig_imp = px.bar(importances, orientation='h', 
                         labels={"value": "Influence Score", "index": "Feature"},
                         title="Top 10 Price Drivers")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.caption("The model assigns weight based on historical transaction patterns in your dataset.")
        
    except Exception as e:
        st.warning("Feature analysis is loading...")