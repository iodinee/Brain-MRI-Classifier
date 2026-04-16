import os
import random
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

# ---------------- SETUP ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")

# IMPORTANT: must match your actual folder name EXACTLY
STUDY_DIR = os.path.join(BASE_DIR, "testing_images")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_model = None

# ---------------- MODEL ----------------

def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

# ---------------- HOME ROUTES ----------------

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/scan")
def scan():
    return render_template("index.html")

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        raw = file.read()

        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        img128 = pil.resize((128, 128))
        arr = np.asarray(img128).astype(np.float32) / 255.0
        batch = np.expand_dims(arr, axis=0)

        model = load_model()
        preds = model.predict(batch, verbose=0)[0]

        pred_index = int(np.argmax(preds))

        return jsonify({
            "prediction": CLASS_NAMES[pred_index],
            "confidence": float(preds[pred_index]),
            "probabilities": {
                "glioma": float(preds[0]),
                "meningioma": float(preds[1]),
                "pituitary": float(preds[2]),
                "notumor": float(preds[3]),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- STUDY MODE (FIXED) ----------------

@app.route("/api/study/image")
def study_image():
    try:
        # ONLY choose folders that actually exist
        available_labels = [
            d for d in CLASS_NAMES
            if os.path.exists(os.path.join(STUDY_DIR, d))
        ]

        if not available_labels:
            return jsonify({"error": "No dataset folders found"}), 400

        label = random.choice(available_labels)
        folder_path = os.path.join(STUDY_DIR, label)

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not files:
            return jsonify({"error": f"No images in {label}"}), 400

        filename = random.choice(files)

        return jsonify({
            "image_url": f"/study_images/{label}/{filename}",
            "label": label,
            "explanation": f"This MRI shows a {label} case. Learn the visual patterns carefully."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- SERVE IMAGES ----------------

@app.route("/study_images/<label>/<filename>")
def serve_study_image(label, filename):
    return send_from_directory(os.path.join(STUDY_DIR, label), filename)

# ---------------- ERRORS ----------------

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({"error": "File too large"}), 413

# ---------------- RUN ----------------

if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5001)