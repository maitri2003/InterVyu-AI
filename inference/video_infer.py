import os
import cv2
import torch
from torchvision import transforms
from models.video_model import VideoEmotionModel


# Emotion labels (must match training order)
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "video_emotion_model.pt")

# Use OpenCV built-in Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_frames(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return []

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames


def predict_video_emotion(video_path):
    print("📹 Processing video:", video_path)

    model = VideoEmotionModel(num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    frames = extract_frames(video_path)
   

    emotion_counts = {label: 0 for label in LABELS}

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        

        if len(faces) == 0:
            continue

        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]

        img = transform(face).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()

        emotion_counts[LABELS[pred]] += 1

    total = sum(emotion_counts.values())

    if total == 0:
        return {
        "emotion_distribution": emotion_counts,
        "dominant_emotion": "No Face Detected",
        "confidence_score": 0.0
    }

    neutral = emotion_counts["neutral"]
    happy = emotion_counts["happy"]
    fear = emotion_counts["fear"]
    sad = emotion_counts["sad"]
    angry = emotion_counts["angry"]
    disgust = emotion_counts["disgust"]

    raw_score = (neutral + happy) - (fear + sad + angry + disgust)

# Normalize between 0 and 1
    confidence_score = (raw_score / total + 1) / 2

    dominant_emotion = max(emotion_counts, key=emotion_counts.get)

    return {
    "emotion_distribution": emotion_counts,
    "dominant_emotion": dominant_emotion,
    "confidence_score": round(confidence_score, 2)
}
