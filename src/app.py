"""Flask app that serves the trained Wine model through a web form and a JSON API."""
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

app = Flask(__name__)

# Load the model and its metadata once, when the server starts
model = joblib.load(ARTIFACT_DIR / "model.joblib")
metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
FEATURES = metadata["feature_names"]
CLASSES = metadata["class_names"]


def parse_features(payload):
    """Checks that all 13 features are present, numeric and non-negative."""
    missing = [name for name in FEATURES if name not in payload]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    values = []
    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"'{name}' must be a non-negative number")
        values.append(value)
    return np.array([values])


@app.get("/")
def index():
    return render_template(
        "index.html",
        features=FEATURES,
        example=metadata["example_input"],
        accuracy=metadata["test_accuracy"],
    )


@app.get("/health")
def health():
    return jsonify(status="healthy", model_version=metadata["model_version"])


@app.get("/model-info")
def model_info():
    return jsonify(metadata)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify(error="Send the 13 wine features as JSON or form data"), 400
    try:
        X = parse_features(payload)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    probabilities = model.predict_proba(X)[0]
    class_id = int(np.argmax(probabilities))
    return jsonify(
        predicted_class=CLASSES[class_id],
        class_id=class_id,
        confidence=round(float(probabilities[class_id]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(CLASSES, probabilities)},
    )
