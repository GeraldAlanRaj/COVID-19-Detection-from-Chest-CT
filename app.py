import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from gradcam import preprocess_img, get_gradcam_heatmap, apply_heatmap
from PIL import Image

st.set_page_config(page_title="COVID CT Detector", layout="centered")
st.title("🦠 COVID Detection from Lung CT using CNN + Grad-CAM")

uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)

    # Load model
    model = load_model("models/covid_cnn_model.keras")

    st.info("Making prediction...")
    img_array = preprocess_img(img_path)

    # Ensure model is built (important in Keras 3)
    _ = model(img_array)

    prediction = model.predict(img_array)[0][0]

    # Dataset: 0 = COVID, 1 = NON-COVID
    label = "Non-COVID" if prediction > 0.5 else "COVID"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    # Grad-CAM visualization
    st.subheader("Grad-CAM Heatmap")
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="last_conv")
    cam_image = apply_heatmap(img_path, heatmap)
    st.image(cam_image, caption="Grad-CAM Overlay", use_column_width=True)
