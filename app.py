


import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Title of the app
st.title("🐶🐱 Real-Time Image Classifier")

# Load the trained model with error handling
@st.cache_resource
def load_model():
    model_path = "model.h5"
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    else:
        st.error("model.h5 is missing or empty.")
        return None

model = load_model()

# Define class names (adjust based on your model's training labels)
class_names = ["cat", "dog"]

# Preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((224, 224))  # Resize to model's expected input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Upload image
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Handle binary or multi-class output
    if prediction.shape[1] == 1:
        label = "dog" if prediction[0][0] > 0.5 else "cat"
        confidence = prediction[0][0] if label == "dog" else 1 - prediction[0][0]
    else:
        predicted_index = np.argmax(prediction[0])
        label = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

    st.success(f"Prediction: {label} ({confidence:.2%} confidence)")
elif uploaded_file is not None and model is None:
    st.warning("Model could not be loaded. Please check the model.h5 file.")

