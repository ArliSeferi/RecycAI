import torch
import torch.nn as nn
from torchvision import models

class RecycleModel(nn.Module):
    def __init__(self):
        super(RecycleModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2 output classes

    def forward(self, x):
        return self.resnet(x)

# Function to save the model
def save_model(model, filepath='recycle_model.pth'):
    torch.save(model.state_dict(), filepath)

# Function to load the model
def load_model(filepath='recycle_model.pth', device='cpu'):
    model = RecycleModel()
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model
