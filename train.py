import torch
import torch.nn as nn
import torch.optim as optim
import json

from tqdm import tqdm
from dataset import create_dataloaders
from model import AerialLandscapeCNN

DATA_DIR = "/content/dataset/Aerial_Landscapes"
LR = 0.001
NUM_CLASSES = 15
EPOCHS = 50

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_loader, val_loader, test_loader, classes = create_dataloaders(DATA_DIR)

  print(f"Found classes : {len(classes)}")

  model = AerialLandscapeCNN(NUM_CLASSES)

  loss_fn = nn.CrossEntropyLoss()

  optimizer = optim.Adam(model.parameters(),lr=LR)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',  
    factor=0.1,    
    patience=3  
)

  history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
  model = model.to(device)

  

  for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0
    
   
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training", leave=False)
    
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        
        train_loop.set_postfix(loss=loss.item())

    model.eval()
    val_loss, val_correct = 0, 0
    
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Validation", leave=False)
    
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            
            val_loop.set_postfix(val_loss=loss.item())

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss / len(val_loader)
    t_acc = train_correct / len(train_loader.dataset)
    v_acc = val_correct / len(val_loader.dataset)

    scheduler.step(v_loss)

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Summary Epoch {epoch+1}: LR: {current_lr:.6f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
  
  torch.save(model.state_dict(), "aerial_model_v2.pth")
  print("Model saved!")

  with open("history.json", "w") as f:
    json.dump(history, f)


if __name__ == "__main__":
    main()


  
  