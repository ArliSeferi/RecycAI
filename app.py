import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
from model import RecycleModel  # Import trained model

app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecycleModel().to(device)
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    """Classify an image as recyclable or non-recyclable."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_labels = ["Non-Recyclable", "Recyclable"]
    return class_labels[predicted_class.item()], confidence.item() * 100

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Classify the uploaded image
            label, confidence = classify_image(filepath)

            return render_template("index.html", filename=file.filename, label=label, confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
