
# 🐶🐱 CatasAndDogs: Real-Time Image Classification with PyTorch and Streamlit

This project is a Streamlit web app that performs real-time image classification to distinguish between cats and dogs using a fine-tuned ResNet18 model in PyTorch.

## 🚀 Features
- Upload an image and get an instant prediction: `cat` or `dog`
- Uses transfer learning with ResNet18
- Trained on a custom dataset of cat and dog images
- Compatible with Streamlit Cloud deployment

## 🧠 Model Training
The model is trained using PyTorch with the following setup:
- Architecture: ResNet18 with a modified final layer for 2 classes
- Training script: `train.py`
- Saved using `state_dict` to `best_model.pt`

## 🖥️ App Deployment
The app is built with Streamlit and defined in `app.py`. It loads the model weights and performs inference on uploaded images.

### To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
