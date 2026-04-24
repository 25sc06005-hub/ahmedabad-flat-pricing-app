import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
import os



# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import clean_data, get_features_target



def train():
    print("🚀 Loading data...")
    df = pd.read_csv("data/ahmedabad_cleaned.csv")
    df = clean_data(df)
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["total_sqft", "bhk"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    ])

    print("🧠 Training model...")
    pipeline.fit(X_train, y_train)



    # Evaluation
    predictions = pipeline.predict(X_test)
    print(f"✅ Training Complete.")
    print(f"📊 R2 Score: {r2_score(y_test, predictions):.4f}")
    print(f"📊 MAE: {mean_absolute_error(y_test, predictions):.2f}")



    # Save
    joblib.dump(pipeline, "model/model.pkl")
    print("💾 Model saved to model/model.pkl")

if __name__ == "__main__":
    train()