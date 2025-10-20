
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

st.title("🐶🐱 Image Classification Viewer (PyTorch)")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)  # Avoid downloading pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def decode_prediction(output):
    class_names = ['cat', 'dog']
    _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_label = decode_prediction(output)
    st.success(f"PyTorch Prediction: {predicted_label}")



