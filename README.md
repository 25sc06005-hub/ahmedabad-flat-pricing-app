<img width="1919" height="1079" alt="Screenshot 2026-04-24 154930" src="https://github.com/user-attachments/assets/67d881c8-707a-49e2-91e1-40dca5780eca" />

<img width="1919" height="1079" alt="Screenshot 2026-04-24 154939" src="https://github.com/user-attachments/assets/7e2fbd0c-d64d-40b5-9432-a61f816df848" />


# 🏙️ Amdavad Estate Pro


**Amdavad Estate Pro** is a production-grade machine learning application designed to provide accurate property valuations for flats in Ahmedabad, India. By leveraging a Random Forest regression model and a modular software architecture, this tool helps users navigate the local real estate market with data-driven confidence.

🚀 **[Live Demo on Streamlit Cloud](https://ahmedabad-flat-pricing-app-fpy52kxs2f4a9zatcieg9c.streamlit.app/)**

---

## 🌟 Key Features

* **Smart Valuation:** Predicts flat prices based on location, total square footage, and BHK.
* **Zero Data Leakage:** High-integrity ML pipeline that excludes derived features (like price-per-sqft) during training to ensure realistic predictions.
* **Market Density Map:** Interactive visualizations showing price hotspots and volume across Ahmedabad’s major neighborhoods.
* **Model Explainability:** Real-time feature importance charts showing exactly which factors (Location vs. Area vs. BHK) are driving the price.
* **Modular Architecture:** Separated concerns between UI, Preprocessing, and Inference for better maintainability.

---

## 🏗️ Project Structure

The project is organized into a modular pipeline to separate the training environment from the user interface:

```text
├── Ahmedabad-FlatPricing-App.py  # Streamlit UI & Dashboard
├── preprocessing.py              # Centralized data cleaning logic
├── requirements.txt              # Project dependencies
├── data/
│   └── ahmedabad_cleaned.csv     # Kaggle-sourced housing dataset
├── model/
│   ├── train.py                  # Model training & evaluation script
│   ├── predict.py                # Inference wrapper for the UI
│   └── model.pkl                 # Serialized Random Forest model
└── utils/                        # Helper functions for future scaling




🛠️ Tech Stack
Language: Python 3.14+

Framework: Streamlit

Machine Learning: Scikit-Learn (Random Forest Regressor)

Data Science: Pandas, NumPy

Visualization: Plotly Express (Interactive), Matplotlib/Seaborn




🚦 Getting Started
1. Clone the repository
Bash
git clone [https://github.com/25sc06005-hub/ahmedabad-flat-pricing-app.git](https://github.com/25sc06005-hub/ahmedabad-flat-pricing-app.git)
cd ahmedabad-flat-pricing-app
2. Install dependencies
Bash
pip install -r requirements.txt
3. Train the model (Optional)
If you wish to retrain the model or update it with new data:


Bash
python model/train.py
4. Run the App
Bash
streamlit run Ahmedabad-FlatPricing-App.py
📊 Model Performance
The current model uses a Random Forest Regressor with 150 estimators.

R² Score: High accuracy on unseen testing data.

Mean Absolute Error (MAE): Optimized to minimize the gap between predicted and actual market rates.

Inference: The model is pre-serialized (.pkl), allowing for near-instant predictions in the production environment.



👤 Author
Aryan * GitHub: @25sc06005-hub
