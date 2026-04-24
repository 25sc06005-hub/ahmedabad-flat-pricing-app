import os
import sys
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# Path setup (IMPORTANT for deployment)
# ----------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import clean_data, get_features_target




# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = os.path.join("data", "ahmedabad_cleaned.csv")
MODEL_PATH = os.path.join("model", "model.pkl")




def train():
    print("🚀 Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Clean data
    df = clean_data(df)

    # Split features/target
    X, y = get_features_target(df)

    # Safety check
    if y is None:
        raise ValueError("Target column 'price' not found in dataset!")

    # Ensure no NaN leaks
    X = X.dropna()
    y = y.loc[X.index]

    print(f"📊 Training data shape: {X.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # Preprocessing
    # ----------------------------
    categorical_features = ["location"]
    numeric_features = ["total_sqft", "bhk"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    # ----------------------------
    # Model
    # ----------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # ----------------------------
    # Training
    # ----------------------------
    print("🧠 Training model...")
    pipeline.fit(X_train, y_train)

    # ----------------------------
    # Evaluation
    # ----------------------------
    preds = pipeline.predict(X_test)

    print("\n✅ Training Complete")
    print(f"📊 R2 Score: {r2_score(y_test, preds):.4f}")
    print(f"📊 MAE: {mean_absolute_error(y_test, preds):.2f}")

    # ----------------------------
    # Save model
    # ----------------------------
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"💾 Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()