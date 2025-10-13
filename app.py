
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model

import pandas as pd
predictions_df = pd.read_csv("batch_predictions.csv")


# Define class names based on training order
class_names = ["cat", "dog"]

# Preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((224, 224))  # Match model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.title("Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

    st.write(f"### Prediction: {label} ({confidence:.2f} confidence)")





