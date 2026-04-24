import joblib
import pandas as pd
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")



# ----------------------------
# LOAD MODEL
# ----------------------------
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "❌ model.pkl not found! Run train.py first to generate the model."
        )
    return joblib.load(MODEL_PATH)



# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def make_prediction(model, input_dict):
    """
    Converts input dict → DataFrame → aligns features → returns prediction
    """

    input_df = pd.DataFrame([input_dict])

    # Get expected features from trained pipeline
    try:
        expected_features = model.named_steps["prep"].feature_names_in_
    except Exception:
        # fallback safety
        expected_features = input_df.columns

    # Align columns (VERY IMPORTANT for production stability)
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    return float(prediction)