
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Title
st.title("🐶🐱 Image Classification Viewer")

# Load predictions
csv_path = "cats_and_dogs_dataset/batch_predictions.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("Predictions loaded successfully.")

    # Show prediction distribution
    if not df.empty:
        st.subheader("Prediction Distribution")
        st.bar_chart(df['prediction'].value_counts())

        # Show full prediction table
        st.subheader("All Predictions")
        st.dataframe(df)
    else:
        st.warning("The CSV file is empty. Please check your inference script.")
else:
    st.warning("Prediction file not found.")

# Upload image
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Match uploaded image to prediction
    uploaded_filename = uploaded_file.name.strip().lower()
    df['filename'] = df['filename'].str.strip().str.lower()
    match = df[df["filename"] == uploaded_filename]

    if not match.empty:
        label = match.iloc[0]["prediction"]
        st.success(f"Prediction: {label}")
    else:
        st.warning("No prediction found for this image in batch_predictions.csv.")











