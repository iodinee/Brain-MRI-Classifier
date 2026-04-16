import base64
import io
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from matplotlib import colormaps
    def _jet_cmap():
        return colormaps["jet"]
except Exception:
    from matplotlib import cm
    def _jet_cmap():
        return cm.get_cmap("jet")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_model = None


def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def forward_to_conv_and_logits(x, keras_model):
    vgg = keras_model.layers[0]
    h = x
    conv_out = None

    for layer in vgg.layers[1:]:
        h = layer(h, training=False)
        if layer.name == "block5_conv3":
            conv_out = h

    for layer in keras_model.layers[1:]:
        h = layer(h, training=False)

    return conv_out, h


def grad_cam_heatmap(keras_model, img_batch, pred_index):
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = forward_to_conv_and_logits(x, keras_model)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(tf.multiply(conv_out[0], pooled_grads), axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max() + 1e-8

    return heatmap


def overlay_jet_on_image(original_rgb: Image.Image, heatmap_2d: np.ndarray, alpha=0.4):
    w, h = original_rgb.size

    hm = Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    hm = np.asarray(hm).astype(np.float32) / 255.0

    jet = _jet_cmap()
    heatmap_rgb = jet(hm)[:, :, :3].astype(np.float32)

    orig = np.asarray(original_rgb).astype(np.float32) / 255.0

    blended = alpha * heatmap_rgb + (1 - alpha) * orig
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def image_to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/scan")
def scan():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        raw = file.read()

        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        original = pil.copy()

        img128 = pil.resize((128, 128), Image.BILINEAR)
        arr = np.asarray(img128).astype(np.float32) / 255.0
        batch = np.expand_dims(arr, axis=0)

        model = load_model()
        preds = model.predict(batch, verbose=0)[0]

        pred_index = int(np.argmax(preds))
        prediction = CLASS_NAMES[pred_index]
        confidence = float(preds[pred_index])

        heatmap = grad_cam_heatmap(model, batch, pred_index)
        overlay = overlay_jet_on_image(original, heatmap, alpha=0.4)
        heatmap_b64 = image_to_png_base64(overlay)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "heatmap_image": heatmap_b64,
            "probabilities": {
                "glioma": float(preds[0]),
                "meningioma": float(preds[1]),
                "pituitary": float(preds[2]),
                "notumor": float(preds[3]),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({"error": "File too large"}), 413


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5001, debug=False)

    