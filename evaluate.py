import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import create_dataloaders
from model import AerialLandscapeCNN
import os

def evaluate_model(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = create_dataloaders(data_dir)
    
    model = AerialLandscapeCNN(num_classes=len(classes))
    
 
    dummy_input = torch.randn(1, 3, 224, 224)
    model(dummy_input)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0
    loss_fn = nn.CrossEntropyLoss()


    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    
    print("\n" + "="*30)
    print(f"Test results")
    print(f"Average Loss: {avg_loss:.4f}")
    print("="*30)
    
    print("\n Classification report")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Final confusion matrix - test')
    plt.xlabel('Model prediction')
    plt.ylabel('Ground truth')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig("test_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    DATA_DIR = "/content/dataset/Aerial_Landscapes"
    MODEL_PATH = "aerial_model_v2.pth"
    
    if os.path.exists(MODEL_PATH):
        evaluate_model(MODEL_PATH, DATA_DIR)
    else:
        print(f"Błąd: Nie znaleziono pliku modelu {MODEL_PATH}!")