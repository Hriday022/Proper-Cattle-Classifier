"""
Cattle Breed Classifier — Flask web app around a pretrained MobileNetV2
transfer-learning model (.keras format, 224x224x3 input, 15-class softmax
output).

Flow:
  1. User uploads an image via the web form.
  2. Image is preprocessed to match training-time input (224x224, MobileNetV2
     preprocessing).
  3. Model predicts a probability distribution over the 15 breed classes.
  4. If the top confidence is below OOD_THRESHOLD, the image is flagged as
     "not confidently a cattle image" instead of forcing a wrong label.
"""
import io
import os

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload cap

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cattle_breed_model.keras")
IMG_SIZE = (224, 224)

# --- OOD (out-of-distribution) handling ------------------------------------
# If the model's top softmax probability is below this, we don't trust the
# prediction enough to show it as a confident breed match. Tune this value
# against your own validation set — 0.5 is a reasonable starting point for
# a 15-class softmax, but you may want it higher/lower depending on how
# "confident but wrong" your model tends to be on non-cattle images.
OOD_THRESHOLD = 0.50

# --- Class labels ------------------------------------------------------------
# Taken from your dataset's folder names, in alphabetical order — this matches
# how Keras/TensorFlow data loaders (ImageDataGenerator, image_dataset_from_directory)
# assign class indices by default (sorted folder name order). If your training
# pipeline used a different order, update this list to match.
CLASS_NAMES = [
    "Dangi", "Deoni", "Gir", "Hallikar", "Hariana",
    "Kangayam", "Kankrej", "Khillari", "Ladakhi", "Malnad_gidda",
    "Ongole", "Pulikulam", "Red_Sindhi", "Sahiwal", "siri",
]

print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("Model loaded. Ready for predictions.")


def preprocess_image(file_stream) -> np.ndarray:
    """Turns an uploaded image file stream into a model-ready batch of 1."""
    img = Image.open(file_stream).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)          # MobileNetV2's own preprocessing (scales to [-1, 1])
    arr = np.expand_dims(arr, axis=0)    # shape becomes (1, 224, 224, 3)
    return arr


def predict_breed(file_stream) -> dict:
    """Runs inference and returns a structured, template-friendly result."""
    batch = preprocess_image(file_stream)
    probs = model.predict(batch, verbose=0)[0]   # shape (15,)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    # Top-3 predictions, sorted by confidence, for a richer result view
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"label": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]

    is_confident = top_prob >= OOD_THRESHOLD

    return {
        "predicted_label": CLASS_NAMES[top_idx] if is_confident else None,
        "confidence": round(top_prob * 100, 2),
        "is_confident": is_confident,
        "top3": top3,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", result=None, error="Please choose an image file.")

    file = request.files["image"]
    try:
        result = predict_breed(file.stream)
    except Exception as exc:
        return render_template("index.html", result=None, error=f"Couldn't process that image: {exc}")

    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    # debug=True is fine locally; never in production
    app.run(debug=True, port=5000)
