import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def extract_features(loader, model, device):
    features, labels = [], []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(device)
            output = model(images)
            output = output.view(images.size(0), -1).cpu().numpy()
            features.extend(output)
            labels.extend(label.numpy())
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    # ✅ Paths
    DATASET_PATH = r"C:\Users\sidda\OneDrive\Desktop\web page\Dataset"
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    TEST_PATH = os.path.join(DATASET_PATH, "test")

    # ✅ Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ✅ Load datasets
    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"✅ Class Labels Mapping: {train_dataset.class_to_idx}")

    # ✅ Load ResNet50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model = resnet_model.to(device).eval()

    # ✅ Feature extraction
    print("Extracting features...")
    start_time = time.time()
    X_train, y_train = extract_features(train_loader, resnet_model, device)
    X_test, y_test = extract_features(test_loader, resnet_model, device)
    print(f"✅ Feature extraction done in {time.time() - start_time:.2f} seconds.")

    print(f"✅ X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # ✅ Train Random Forest
    print(" Training Random Forest...")
    start_train = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"✅ Training completed in {time.time() - start_train:.2f} seconds.")

    # ✅ Save model
    joblib.dump(rf, "random_forest.pkl")
    print(" Model saved as 'random_forest.pkl'")

    # ✅ Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=train_dataset.classes)

    print(f"\n Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(report)
