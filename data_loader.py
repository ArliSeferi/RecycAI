import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations to apply to the images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the images to a fixed size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (ImageNet default)
])

def load_data():
    dataset_path = "assets"  # Replace with the correct path to your dataset folder
    
    # Load training data
    train_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Load testing data
    test_data = datasets.ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader
