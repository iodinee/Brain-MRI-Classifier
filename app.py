import os
import io
import random
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
import cv2
import gdown


if not os.path.exists("brain_tumor_model.h5"):
    gdown.download("https://drive.google.com/uc?id=1bAAlMoY_Bjq3h_P_T-8HHLZOogkugua3", "brain_tumor_model.h5", quiet=False)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "brain_tumor_model.h5")
STUDY_DIR = os.path.join(BASE_DIR, "testing_images")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = 224

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_model = None
_cam_models = {}


def preprocess_image(pil_img):
    arr = np.asarray(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    return arr


def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded. Layers:")
        for i, l in enumerate(_model.layers):
            print(f" [{i}] {l.name} — {type(l).__name__}")
    return _model


def get_grad_model(depth="Bottom"):
    if depth in _cam_models:
        return _cam_models[depth]

    layer_map = {
        "Bottom": "block5_conv3",
        "Middle": "block4_conv3",
        "Top": "block3_conv3",
    }
    target_name = layer_map.get(depth, "block5_conv3")
    model = load_model()
    vgg = model.get_layer("vgg16")
    target_layer = vgg.get_layer(target_name)

    vgg_grad_model = tf.keras.Model(
        inputs=vgg.input,
        outputs=[target_layer.output, vgg.output]
    )

    head_layers = []
    found_vgg = False
    for layer in model.layers:
        if layer.name == "vgg16":
            found_vgg = True
            continue
        if found_vgg:
            head_layers.append(layer)

    _cam_models[depth] = (vgg_grad_model, head_layers)
    print(f"Grad-CAM model built: {depth} → {target_name}")
    return _cam_models[depth]


def get_gradcam_heatmap(pil_img, depth="Bottom"):
    orig_w, orig_h = pil_img.size
    img_arr = preprocess_image(pil_img)
    img_batch = tf.constant(img_arr[np.newaxis, ...])

    vgg_grad_model, head_layers = get_grad_model(depth)

    with tf.GradientTape() as tape:
        outputs = vgg_grad_model(img_batch, training=False)
        conv_out = outputs[0]
        x = outputs[1]
        tape.watch(conv_out)
        for layer in head_layers:
            x = layer(x, training=False)
        preds = x
        preds = tf.reshape(preds, [-1])
        pred_idx = tf.argmax(preds)
        pred_idx = tf.cast(pred_idx, tf.int32)
        class_score = preds[pred_idx]

    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_np = conv_out.numpy()[0]

    cam = np.sum(conv_np * pooled, axis=-1)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    cam_u8 = np.uint8(255 * cam)
    cam_resized = cv2.resize(cam_u8, (orig_w, orig_h))
    heat_rgb = cv2.cvtColor(
        cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(heat_rgb)


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def process_prediction(pil_img, depth="Bottom"):
    model = load_model()
    arr = preprocess_image(pil_img)

    preds = model.predict(arr[np.newaxis, ...], verbose=0)
    preds = np.asarray(preds).squeeze()
    preds = preds.reshape(-1)

    idx = int(np.argmax(preds))
    probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

    return {
        "prediction": CLASSES[idx],
        "confidence": float(preds[idx]),
        "probabilities": probs,
        "heatmap_image": pil_to_base64(get_gradcam_heatmap(pil_img, depth)),
    }


@app.route("/")
def home():
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
        depth = request.form.get("depth", "Bottom")
        pil = Image.open(io.BytesIO(file.read())).convert("RGB")
        return jsonify(process_prediction(pil, depth))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/study/image")
def study_image():
    try:
        labels = [d for d in CLASSES if os.path.exists(os.path.join(STUDY_DIR, d))]
        if not labels:
            return jsonify({"error": "No study images found"}), 404
        label = random.choice(labels)
        folder = os.path.join(STUDY_DIR, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            return jsonify({"error": f"No images in {label}"}), 404
        filename = random.choice(files)
        return jsonify({"image_url": f"/study_images/{label}/{filename}", "label": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/study/analyze", methods=["POST"])
def study_analyze():
    try:
        data = request.get_json(force=True)
        image_url = data.get("image_url", "")
        depth = data.get("depth", "Bottom")

        parts = image_url.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "study_images":
            return jsonify({"error": "Invalid image_url format"}), 400

        _, label, filename = parts
        img_path = os.path.join(STUDY_DIR, label, filename)
        if not os.path.isfile(img_path):
            return jsonify({"error": f"Image not found: {img_path}"}), 404

        pil = Image.open(img_path).convert("RGB")
        arr = preprocess_image(pil)
        preds = load_model().predict(arr[np.newaxis, ...], verbose=0)
        preds = np.asarray(preds).reshape(-1)
        probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}

        return jsonify({
            "heatmap_image": pil_to_base64(get_gradcam_heatmap(pil, depth)),
            "probabilities": probs,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/study_images/<label>/<filename>")
def serve_image(label, filename):
    return send_from_directory(os.path.join(STUDY_DIR, label), filename)


@app.route("/explanation_images/<filename>")
def explanation_image(filename):
    return send_from_directory(os.path.join(BASE_DIR, "explanation_images"), filename)


if __name__ == "__main__":
    app.run(debug=True, port=3000)