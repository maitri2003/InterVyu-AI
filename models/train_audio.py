import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from models.audio_model import AudioEmotionModel, AudioDataset, load_audio_data

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "audio")
MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_emotion_model.pt")

# Load data
X, y = load_audio_data(DATA_PATH)

dataset = AudioDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
num_classes = len(set(y))
model = AudioEmotionModel(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 15
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Audio emotion model saved successfully")
