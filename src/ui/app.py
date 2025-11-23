import io
import requests
import streamlit as st
from PIL import Image

# REST endpoint exposed by the FastAPI service
API_URL = "http://127.0.0.1:8000/predict"

# Basic page heading
st.title("Cats vs Dogs Classifier")

# Image uploader accepts common formats
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure consistent 3-channel RGB input for the model
    image = Image.open(uploaded_file).convert("RGB")
    # Display the uploaded image (use_container_width is deprecated)
    st.image(image, caption="Uploaded image", width="stretch")


    if st.button("Predict"):
        # Convert PIL image to bytes for multipart/form-data POST
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        # Call the API with the uploaded image
        files = {"file": ("image.jpg", buf, "image/jpeg")}
        resp = requests.post(API_URL, files=files)

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Prediction: {data['label']} (confidence: {data['confidence']:.2%})")
        else:
            st.error(f"Error: {resp.status_code} {resp.text}")
