import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

# Importing your modular logic
from model.predict import load_trained_model, make_prediction
from preprocessing import clean_data

# ----------------------------
# 1. GLOBAL CONFIG & THEME-AWARE CSS
# ----------------------------
st.set_page_config(
    page_title="Amdavad Estate Pro",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional UI Styling
st.markdown("""
    <style>
    /* Metric Card Styling - Works for both Light & Dark Mode */
    [data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 15px 20px;
        border-radius: 12px;
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #d43f3f;
        border: none;
    }
    /* Fix Sidebar Title Padding */
    .css-1d391kg { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# 2. RESOURCE LOADING
# ----------------------------
@st.cache_resource
def get_model():
    """Loads the pre-trained model artifact."""
    try:
        return load_trained_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_and_clean_data():
    """Loads and pre-processes the dataset for the UI."""
    path = os.path.join("data", "ahmedabad_cleaned.csv")
    if not os.path.exists(path):
        st.error("Data file not found in 'data/' folder.")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df = clean_data(df)
    return df

# Initialize Data and Model
model = get_model()
df = load_and_clean_data()

# ----------------------------
# 3. HEADER
# ----------------------------
st.title("🏙️ Amdavad Estate Pro")
st.markdown("##### AI-Driven Real Estate Analytics & Valuation for Ahmedabad")
st.write("---")

# ----------------------------
# 4. SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("📍 Property Configuration")
st.sidebar.info("Enter the flat details below to generate a market valuation.")

with st.sidebar:
    loc = st.selectbox("Select Neighborhood", sorted(df["location"].unique()))
    sqft = st.number_input("Total Area (Sqft)", min_value=300, max_value=10000, value=1200, step=50)
    bhk = st.select_slider("Select BHK", options=[1, 2, 3, 4, 5, 6], value=2)
    
    predict_btn = st.button("Generate Valuation 🎯")

# ----------------------------
# 5. MAIN INTERFACE (TABS)
# ----------------------------
tab_calc, tab_trends, tab_model = st.tabs([
    "🎯 Valuation Estimate", 
    "📊 Market Trends", 
    "🧬 AI Intelligence"
])

# --- TAB 1: CALCULATION ---
with tab_calc:
    if predict_btn:
        if model is not None:
            payload = {"location": loc, "total_sqft": sqft, "bhk": bhk}
            
            with st.spinner("Analyzing current market patterns..."):
                try:
                    # Prediction Logic
                    price = make_prediction(model, payload)
                    avg_loc_price = df[df["location"] == loc]["price"].mean()
                    price_per_sqft = (price * 100000) / sqft # Convert Lakhs to absolute for PSF
                    
                    st.success(f"### Results for {loc}")
                    
                    # Metric Row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Estimated Market Price", f"₹ {price:.2f} L")
                    
                    delta = price - avg_loc_price
                    m2.metric("Vs. Neighborhood Avg", f"₹ {avg_loc_price:.1f} L", f"{delta:+.1f} L", delta_color="inverse")
                    
                    m3.metric("Unit Price", f"₹ {price_per_sqft:,.0f} /sqft")
                    
                    # Comparison Plot
                    st.write("---")
                    col_chart, col_text = st.columns([2, 1])
                    
                    with col_chart:
                        comp_data = pd.DataFrame({
                            "Metric": ["AI Estimate", "Area Average"],
                            "Price (Lakhs)": [price, avg_loc_price]
                        })
                        fig = px.bar(comp_data, x="Metric", y="Price (Lakhs)", color="Metric",
                                     color_discrete_sequence=["#ff4b4b", "#6c757d"],
                                     title=f"Valuation Comparison in {loc}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_text:
                        st.markdown("#### Market Context")
                        st.write(f"""
                        The estimated price of **₹{price:.2f} Lakhs** reflects current trends in **{loc}**. 
                        Based on your input of **{sqft} sqft**, this property is valued at approximately **₹{price_per_sqft:,.0f} per square foot**.
                        """)
                        if delta > 0:
                            st.warning("This property is valued higher than the area average, likely due to the BHK/Size configuration.")
                        else:
                            st.info("This property is valued below or at the area average.")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.error("Model artifact not found. Please ensure 'model/model.pkl' is uploaded.")
    else:
        st.info("👈 Please enter property details in the sidebar and click 'Generate Valuation'.")

# --- TAB 2: MARKET TRENDS ---
with tab_trends:
    st.subheader("Ahmedabad Real Estate Hotspots")
    
    # Aggregated Data
    market_df = df.groupby("location").agg(
        avg_price=("price", "mean"),
        count=("price", "count"),
        avg_psf=("price_sqft", "mean")
    ).reset_index()
    
    # Filter for significant locations
    market_df = market_df[market_df["count"] > 10].sort_values("avg_price", ascending=False)
    
    col_plot, col_stats = st.columns([2, 1])
    
    with col_plot:
        fig_market = px.scatter(
            market_df, x="avg_psf", y="avg_price", size="count", color="avg_price",
            hover_name="location", color_continuous_scale="Viridis",
            labels={"avg_psf": "Price Per Sqft", "avg_price": "Average Flat Price (Lakhs)"},
            title="Price vs. Unit Value (Min. 10 Listings)"
        )
        st.plotly_chart(fig_market, use_container_width=True)
        
    with col_stats:
        st.write("**Top Premium Locations**")
        st.dataframe(market_df[["location", "avg_price"]].head(10), hide_index=True)

# --- TAB 3: AI INTELLIGENCE ---
with tab_model:
    st.subheader("Model Explainability")
    st.write("This section shows which factors have the most weight in determining a property's price.")
    
    try:
        # Extracting feature importances from the pipeline
        rf_step = model.named_steps['reg']
        prep_step = model.named_steps['prep']
        
        # Get feature names from OneHotEncoder
        cat_features = prep_step.named_transformers_['cat'].get_feature_names_out(['location'])
        num_features = ['total_sqft', 'bhk']
        all_features = num_features + list(cat_features)
        
        importances = pd.Series(rf_step.feature_importances_, index=all_features)
        top_10 = importances.sort_values(ascending=True).tail(10)
        
        fig_imp = px.bar(
            top_10, orientation='h', 
            title="Top 10 Drivers of Property Price",
            labels={'value': 'Relative Importance', 'index': 'Feature'},
            color_discrete_sequence=['#1c83e1']
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("💡 **Observation:** Area (Total Sqft) and specific high-end locations like 'Satellite' or 'Bopal' typically dominate the model's decision-making.")
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

# ----------------------------
# 6. FOOTER
# ----------------------------
st.write("---")
st.caption("Amdavad Estate Pro v2.1 | Data source: Kaggle Ahmedabad Housing Dataset | Modular ML Architecture")