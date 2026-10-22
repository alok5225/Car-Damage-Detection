import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_helper import predict, MODEL_PATH

st.set_page_config(page_title="Vehicle Damage Detection", page_icon="🚗")
st.title("Vehicle Damage Detection")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. "
             "Put saved_model.pth inside a 'model' folder next to app.py.")
    st.stop()

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("Could not read this file as an image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded File", width="stretch")

    if st.button("Predict"):
        try:
            with st.spinner("Analyzing image..."):
                label, confidence = predict(image)
            st.success(f"Predicted Class: {label}")
            st.caption(f"Confidence: {confidence:.1%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")