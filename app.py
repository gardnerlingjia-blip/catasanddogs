

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import os

# Title
st.title("🐶🐱 Image Classification Viewer (PyTorch)")

# --- PyTorch Model Loading ---
@st.cache_resource
def load_model():
    # Recreate the model architecture
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cat and dog

    # Load the saved weights
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Define Preprocessing (adjust as needed for your model) ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Match model input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Match ResNet normalization
    ])
    return transform(image).unsqueeze(0)

# --- Map Model Output to Label ---
def decode_prediction(output):
    class_names = ['cat', 'dog']
    _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# --- Optional: CSV batch predictions (legacy support) ---
csv_path = "batch_predictions.csv"
df = None

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("Predictions loaded successfully.")
else:
    st.warning("Prediction file not found. You can upload one below.")

uploaded_csv = st.file_uploader("Or upload a prediction CSV", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.success("Predictions loaded from uploaded file.")

# --- Image Upload and Prediction ---
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- PyTorch Prediction ---
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_label = decode_prediction(output)
    st.success(f"PyTorch Prediction: {predicted_label}")

    # --- (Optional) Legacy CSV Lookup ---
    if df is not None and not df.empty:
        uploaded_filename = uploaded_file.name.strip().lower()
        df['filename'] = df['filename'].astype(str).str.strip().str.lower()
        match = df[df["filename"] == uploaded_filename]
        if not match.empty:
            label = match.iloc[0]["prediction"]
            st.info(f"Legacy CSV Prediction: {label}")
        else:
            st.info("No prediction found for this image in batch_predictions.csv.")
    else:
        st.info("No batch prediction data available.")

