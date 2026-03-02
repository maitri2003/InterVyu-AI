import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils.feature_extraction import extract_mfcc

# -----------------------
# Emotion Mapping
# -----------------------
EMOTION_MAP = {
    "01": "Neutral",
    "03": "Confident",
    "04": "Nervous",
    "05": "Stressed",
    "06": "Nervous"
}

# -----------------------
# Data Loader
# -----------------------
def load_audio_data(data_path):
    X, y = [], []

    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)

        # skip if not a directory
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            # process only wav files
            if not file.endswith(".wav"):
                continue

            parts = file.split("-")
            if len(parts) < 3:
                continue

            emotion_code = parts[2]

            if emotion_code in EMOTION_MAP:
                file_path = os.path.join(actor_path, file)
                mfcc = extract_mfcc(file_path)
                X.append(mfcc)
                y.append(EMOTION_MAP[emotion_code])

    return np.array(X), np.array(y)



# -----------------------
# Dataset
# -----------------------
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.le = LabelEncoder()
        self.y = torch.tensor(self.le.fit_transform(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# CNN-LSTM Model
# -----------------------
class AudioEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32 * 19,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 40, 174)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
