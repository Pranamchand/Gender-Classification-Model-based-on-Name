import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ---------- Load artifacts at startup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "gender_classification_model.h5"))

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)


# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a name and return predicted gender + confidence."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()

    if not name:
        return jsonify({"error": "Name is required"}), 400

    # Pre-process exactly like the notebook
    clean = name.lower()

    # TF-IDF transform
    name_tfidf = vectorizer.transform([clean])

    # Predict
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


if __name__ == "__main__":
    print("Gender Classification server running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
