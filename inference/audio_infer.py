import os
import torch
import numpy as np
from models.audio_model import AudioEmotionModel
from utils.feature_extraction import extract_mfcc

# Emotion labels (must match training)
LABELS = ["Confident", "Neutral", "Nervous", "Stressed"]

CONFIDENCE_MAP = {
    "Confident": 0.9,
    "Neutral": 0.7,
    "Nervous": 0.5,
    "Stressed": 0.3
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_emotion_model.pt")


def predict_audio_emotion(audio_path):
    # Load model
    model = AudioEmotionModel(num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Extract MFCC
    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(mfcc)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    emotion = LABELS[predicted_idx]
    confidence = CONFIDENCE_MAP[emotion]

    return {
        "dominant_emotion": emotion,
        "confidence_score": confidence
    }
