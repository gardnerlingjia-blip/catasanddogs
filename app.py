import streamlit as st
import pandas as pd
from PIL import Image

# Title
st.title("🐶🐱 Image Classification Viewer")


# Upload image
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if df is not None and not df.empty:
        uploaded_filename = uploaded_file.name.strip().lower()
        df['filename'] = df['filename'].astype(str).str.strip().str.lower()
        match = df[df["filename"] == uploaded_filename]

        if not match.empty:
            label = match.iloc[0]["prediction"]
            st.success(f"Prediction: {label}")
        else:
            st.warning("No prediction found for this image in batch_predictions.csv.")
    else:
        st.warning("Prediction data is not available.")
















