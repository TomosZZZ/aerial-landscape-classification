import torch.nn as nn
from torch.nn.modules import MaxPool2d

class AerialLandscapeCNN(nn.Module):
  def __init__(self, num_classes):
    super(AerialLandscapeCNN,self).__init__()

    self.features1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    self.features2 = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.flatten = nn.Flatten()

    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.LazyLinear(out_features=512),
      nn.ReLU(),
      nn.Dropout(p=0.3),             
      nn.Linear(512,num_classes)
    )

  def forward(self,x):
    return self.classifier(self.flatten(self.features2(self.features1(x))))
    
