import joblib
import pandas as pd
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found! Run train.py first.")
    return joblib.load(MODEL_PATH)

def make_prediction(model, input_dict):
    """Takes a dictionary of inputs and returns a float prediction."""
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return prediction