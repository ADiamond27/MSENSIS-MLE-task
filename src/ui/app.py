import io
import requests
import streamlit as st
from PIL import Image

# Default REST endpoint for the FastAPI service
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="MSENSIS Cats vs Dogs", page_icon=":cat:", layout="wide")

st.title("MSENSIS Cats vs Dogs")
st.caption("Upload an image, send it to the FastAPI model, and get a prediction.")

# Sidebar controls
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
st.sidebar.info("Start FastAPI locally with: `uvicorn src.api.main:app --reload`")

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview", use_container_width=True)

with col_right:
    st.write("### Prediction")
    if uploaded_file:
        if st.button("Send to model", type="primary", use_container_width=True):
            with st.spinner("Sending to API..."):
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                buf.seek(0)

                files = {"file": ("image.jpg", buf, "image/jpeg")}
                try:
                    resp = requests.post(api_url, files=files, timeout=30)
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Prediction: {data['label']}")
                        st.metric("Confidence", f"{data['confidence']:.2%}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
    else:
        st.info("Upload a JPG or PNG to enable prediction.")
