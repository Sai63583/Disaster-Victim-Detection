import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, request, render_template, jsonify
import pickle
from PIL import Image

# ✅ Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")

# ✅ Load Trained Random Forest Model
import joblib

# Correct loading
rf_classifier = joblib.load("random_forest.pkl")


# ✅ Load Pretrained ResNet50 (Feature Extractor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])  # Remove classifier
resnet50_model = resnet50_model.to(device).eval()

# ✅ Image Preprocessing (Same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Feature Extraction from Image
def extract_features_from_image(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet50_model(image)
        features = features.view(1, -1).cpu().numpy()

    return features

# ✅ Homepage Route
@app.route("/")
def home():
    return render_template("index1.html")  # Ensure "index.html" exists in "templates"

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    image = Image.open(file)
    features = extract_features_from_image(image)

    # ✅ Ensure Correct Shape
    if features.shape[1] != 2048:
        return jsonify({"error": f"Incorrect feature shape: {features.shape}, expected (1, 2048)"})

    prediction = rf_classifier.predict(features)[0]

    result = {
        "prediction": "Victim Detected" if prediction == 1 else "No Victim Detected"
    }
    return jsonify(result)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
