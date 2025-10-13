import streamlit as st
import pandas as pd
from PIL import Image
import torch
import requests
from io import BytesIO
import os

response = requests.get("https://huggingface.co/spaces/Lingjia2025/cat-dog-classifier/blob/main/cat_dog_model.pth")
model = torch.load(BytesIO(response.content), map_location=torch.device("cpu"))


# Title
st.title("🐶🐱 Image Classification Viewer")


csv_path = "batch_predictions.csv"
df = None  # Initialize df

# Try loading from local file
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("Predictions loaded successfully from local file.")
else:
    st.warning("Local prediction file not found. You can upload one below.")

# Fallback: Upload CSV manually
uploaded_csv = st.file_uploader("Or upload a prediction CSV", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.success("Predictions loaded from uploaded file.")



# Load predictions
csv_path = "batch_predictions.csv"
df = None  # Initialize df

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



st.write("Files in current directory:", os.listdir())
st.write("Files in 'catsanddogs' folder:", os.listdir("catsanddogs"))










