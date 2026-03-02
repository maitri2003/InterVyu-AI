import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.video_model import VideoEmotionModel

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "video", "train")
MODEL_PATH = os.path.join(BASE_DIR, "models", "video_emotion_model.pt")

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# -----------------------
# Dataset
# -----------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# -----------------------
# Model
# -----------------------
num_classes = len(train_dataset.classes)
model = VideoEmotionModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Training Loop
# -----------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# -----------------------
# Save Model
# -----------------------
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Video emotion model saved successfully")
