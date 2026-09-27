import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "cattle_breed_model.keras"
)

IMG_SIZE = (224, 224)

OOD_THRESHOLD = 0.50


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Dangi",
    "Deoni",
    "Gir",
    "Hallikar",
    "Hariana",
    "Kangayam",
    "Kankrej",
    "Khillari",
    "Ladakhi",
    "Malnad_gidda",
    "Ongole",
    "Pulikulam",
    "Red_Sindhi",
    "Sahiwal",
    "siri",
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_cattle_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Converts uploaded image into the format expected
    by the MobileNetV2-based model.
    """

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    arr = np.array(image, dtype=np.float32)

    # MobileNetV2 preprocessing
    # Converts pixel values from [0, 255] to [-1, 1]
    arr = preprocess_input(arr)

    # Add batch dimension
    # (224, 224, 3) -> (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr


# ============================================================
# PREDICTION
# ============================================================

def predict_breed(model, image):

    batch = preprocess_image(image)

    # Model prediction
    probs = model.predict(batch, verbose=0)[0]

    # Highest probability
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    # Top 3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = []

    for i in top3_idx:
        top3.append({
            "label": CLASS_NAMES[i],
            "confidence": float(probs[i]) * 100
        })

    # OOD / confidence check
    is_confident = top_prob >= OOD_THRESHOLD

    return {
        "predicted_label": (
            CLASS_NAMES[top_idx]
            if is_confident
            else None
        ),
        "confidence": top_prob * 100,
        "is_confident": is_confident,
        "top3": top3
    }


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Cattle Breed Classifier",
    page_icon="🐄",
    layout="centered"
)


st.title("🐄 Cattle Breed Classifier")

st.write(
    "Upload an image of a cattle breed and the "
    "MobileNetV2-based model will predict the breed."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_cattle_model()

except Exception as e:

    st.error("❌ Failed to load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a cattle image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# RUN PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Classify Image",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            result = predict_breed(
                model,
                image
            )

        st.divider()

        # ====================================================
        # MAIN RESULT
        # ====================================================

        if result["is_confident"]:

            st.success(
                f"🐄 Predicted Breed: "
                f"**{result['predicted_label']}**"
            )

            st.metric(
                "Confidence",
                f"{result['confidence']:.2f}%"
            )

        else:

            st.warning(
                "⚠️ Not confident enough"
            )

            st.write(
                "The model's highest confidence is below "
                "the 50% threshold. This image may be "
                "outside the trained cattle breeds, not "
                "cattle, or unclear in quality/angle."
            )

            st.metric(
                "Highest Confidence",
                f"{result['confidence']:.2f}%"
            )


        # ====================================================
        # TOP 3 PREDICTIONS
        # ====================================================

        st.subheader("Top 3 Predictions")

        for rank, prediction in enumerate(
            result["top3"],
            start=1
        ):

            col1, col2 = st.columns([3, 1])

            with col1:

                st.write(
                    f"**{rank}. {prediction['label']}**"
                )

            with col2:

                st.write(
                    f"{prediction['confidence']:.2f}%"
                )

            st.progress(
                min(
                    prediction["confidence"] / 100,
                    1.0
                )
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    st.write("**Architecture:** MobileNetV2 Transfer Learning")

    st.write("**Input size:** 224 × 224 × 3")

    st.write("**Number of classes:** 15")

    st.write("**OOD confidence threshold:** 50%")

    st.write("**Model format:** Keras `.keras`")