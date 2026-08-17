import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/plant_disease_model.keras")
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

def prettify(name: str) -> str:
    return name.replace("Tomato___", "").replace("_", " ").strip()

model, class_names = load_model()

st.title("🌿 Plant Disease Detector")
st.caption("Upload a photo of a tomato leaf to identify possible disease.")

with st.sidebar:
    st.header("About")
    st.write(f"Classes: **{len(class_names)}**")
    st.write("MobileNetV2 (ImageNet) + fine-tuning")
    st.warning(
        "Trained on PlantVillage laboratory images (uniform background). "
        "Accuracy on real field photos will be substantially lower. "
        "This is a demonstration, not agricultural advice."
    )

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_column_width=True)

    img = image.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))

    st.success(f"### {prettify(class_names[idx])}")
    st.metric("Confidence", f"{preds[idx]:.1%}")

    if preds[idx] < 0.60:
        st.warning("Low confidence — this may be a leaf type or condition the model has not seen.")

    st.subheader("Top 3 predictions")
    top3 = np.argsort(preds)[::-1][:3]
    for i in top3:
        st.write(f"**{prettify(class_names[i])}** — {preds[i]:.1%}")
        st.progress(float(preds[i]))