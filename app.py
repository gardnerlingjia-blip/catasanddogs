<<<<<<< HEAD
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Title
st.title("🐶🐱 Image Classification Viewer")


# Load predictions
csv_path = "batch_predictions.csv"
df = None  # Initialize df

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("Predictions loaded successfully.")
else:
    st.warning("Prediction file not found. You can upload one below.")

# Optional: CSV upload fallback
uploaded_csv = st.file_uploader("Or upload a prediction CSV", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.success("Predictions loaded from uploaded file.")

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


=======
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Title
st.title("🐶🐱 Image Classification Viewer")


# Load predictions
csv_path = "batch_predictions.csv"
df = None  # Initialize df

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success("Predictions loaded successfully.")
else:
    st.warning("Prediction file not found. You can upload one below.")

# Optional: CSV upload fallback
uploaded_csv = st.file_uploader("Or upload a prediction CSV", type=["csv"])
if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.success("Predictions loaded from uploaded file.")

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

















>>>>>>> b25e77100e7142cd9a7d90c96fa626c84b22bb74
