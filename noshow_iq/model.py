import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from noshow_iq.preprocess import preprocess, get_features_and_target

# Path to save the trained model
MODEL_PATH = Path(__file__).parent.parent / "model.joblib"


def train(filepath: str):
    """Train the model and save it to disk."""

    print("Loading and preprocessing data...")
    df = preprocess(filepath)
    X, y = get_features_and_target(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance using SMOTE
    # SMOTE creates synthetic samples of the minority class
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train, y_train
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_balanced, y_train_balanced)

    # Evaluate the model
    metrics = evaluate(model, X_test, y_test)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, metrics


def evaluate(model, X_test, y_test):
    """Evaluate model and return precision, recall, F1."""
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True
    )

    print("\n--- Model Evaluation ---")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"]
    ))

    return report


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train() first."
        )
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Make a prediction for one appointment.
    Input: dict of features
    Output: dict with risk_level, probability, recommendation
    """
    model = load_model()

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Get probability of no-show (class 1)
    proba = model.predict_proba(df)[0][1]

    # Determine risk level
    if proba >= 0.5:
        risk_level = "HIGH"
        recommendation = (
            "Call the patient to confirm. "
            "Consider overbooking this slot."
        )
    else:
        risk_level = "LOW"
        recommendation = (
            "Patient likely to show up. "
            "No action needed."
        )

    return {
        "risk_level": risk_level,
        "probability": round(float(proba), 4),
        "recommendation": recommendation
    }


if __name__ == "__main__":
    data_path = (
        Path(__file__).parent.parent / "data" / "KaggleV2-May-2016.csv"
    )
    model, metrics = train(str(data_path))
    print("Training complete!")
    print(
        "No-show F1:",
        round(metrics["no_show"]["f1-score"], 4)
    )
