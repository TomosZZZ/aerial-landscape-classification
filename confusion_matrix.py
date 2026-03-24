import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import create_dataloaders
from model import AerialLandscapeCNN
import os

def plot_confusion_matrix(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = create_dataloaders(data_dir)
    model = AerialLandscapeCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix - Aerial Landscapes')
    plt.xlabel('Model prediction')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "/content/dataset/Aerial_Landscapes"
    MODEL_PATH = "aerial_model_v2.pth"
    plot_confusion_matrix(MODEL_PATH, DATA_DIR)