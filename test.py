import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import RecycleModel  # Ensure the model file is correct

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecycleModel().to(device)
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_images_in_folder(root_folder):
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder '{root_folder}' not found!")
        return
    
    # Go through subfolders
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):  
            continue  # Skip if not a folder

        print(f"\n📂 Checking folder: {subfolder}...\n")

        image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"⚠️ No images found in '{subfolder_path}'")
            continue

        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

            class_labels = ["Non-Recyclable", "Recyclable"]
            print(f"🔍 Image: {image_file} | Prediction: {class_labels[predicted_class.item()]}, Confidence: {confidence.item() * 100:.2f}%")

# Specify the root test folder that contains subfolders
test_folder_path = "assets/test/"  # Update this if needed
classify_images_in_folder(test_folder_path)
