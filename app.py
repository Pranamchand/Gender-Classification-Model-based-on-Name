import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ---------- Load artifacts relative to application directory ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "gender_classification_model.h5")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Load trained model and preprocessors
model = load_model(MODEL_PATH)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)


# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a name and return predicted gender + confidence."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "").strip()

        if not name:
            return jsonify({"error": "Name is required"}), 400

        # Pre-process exactly like the model training pipeline
        clean = name.lower()

        # TF-IDF transform
        name_tfidf = vectorizer.transform([clean])

        # Model inference
        probability = float(model.predict(name_tfidf, verbose=0)[0][0])
        prediction = int(probability >= 0.5)
        gender = encoder.inverse_transform([prediction])[0]
        confidence = probability if prediction == 1 else 1 - probability

        return jsonify({
            "name": name.title(),
            "gender": "Male" if gender == "m" else "Female",
            "gender_code": gender,
            "confidence": round(confidence * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running at http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
