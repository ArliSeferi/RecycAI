import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import RecycleModel  # Make sure this is the correct import

# Load data
train_loader, test_loader = load_data()

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecycleModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if a checkpoint exists
checkpoint_path = "checkpoint.pth"
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"🔄 Found checkpoint! Loading from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
    print(f"✅ Resuming training from epoch {start_epoch}")
else:
    print("❌ No checkpoint found, starting from scratch.")

# Number of epochs
num_epochs = 10  # Adjust based on your needs

# Training loop
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print progress
    print(f"✅ Epoch {epoch+1}/{num_epochs} completed - Loss: {running_loss/len(train_loader)}")

    # Save checkpoint
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Checkpoint saved at: {os.path.abspath(checkpoint_path)}")

print("🎉 Training finished!")
