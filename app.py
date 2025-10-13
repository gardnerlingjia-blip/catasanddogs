
import streamlit as st
import pandas as pd
from PIL import Image

# Load predictions from CSV
predictions_df = pd.read_csv("batch_predictions.csv")

st.title("Cat vs Dog Classifier (Batch Predictions)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    filename = uploaded_file.name
    match = predictions_df[predictions_df["filename"] == filename]

    if not match.empty:
        label = match.iloc[0]["predicted_label"]
        confidence = match.iloc[0].get("confidence_score", None)

        if confidence is not None:
            st.write(f"### Prediction: {label} ({confidence:.2f} confidence)")
        else:
            st.write(f"### Prediction: {label}")
    else:
        st.write("### No prediction found for this image in batch_predictions.csv.")






