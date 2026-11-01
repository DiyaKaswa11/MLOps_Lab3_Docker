"""Trains a Gradient Boosting classifier on the Wine dataset and saves the model and metadata."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings come from environment variables, so they can be changed without rebuilding the image
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.1"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "3"))
RANDOM_STATE = 42
CLASS_NAMES = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]


def main():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=RANDOM_STATE
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    now = datetime.now(timezone.utc)
    metadata = {
        "dataset": "Wine (scikit-learn)",
        "model": "StandardScaler + GradientBoostingClassifier",
        "model_version": now.strftime("%Y%m%d%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "hyperparameters": {
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "feature_names": list(wine.feature_names),
        "class_names": CLASS_NAMES,
        # One real test sample, used to pre-fill the web form
        "example_input": {
            name: round(float(value), 3) for name, value in zip(wine.feature_names, X_test[0])
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print(f"Test accuracy: {metadata['test_accuracy']}")
    print(f"Model version: {metadata['model_version']}")
    print(f"Saved model and metadata to {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
