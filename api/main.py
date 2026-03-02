from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import shutil
import os
import uuid

from moviepy import VideoFileClip

from inference.audio_infer import predict_audio_emotion
from inference.video_infer import predict_video_emotion
from inference.text_infer import predict_text_confidence
from inference.fusion import compute_final_score, generate_feedback


# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="AI Interview Performance Analyzer")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "AI Interview Analyzer API is running"}


# -----------------------------
# API Endpoint (JSON Response)
# -----------------------------
@app.post("/analyze")
async def analyze_interview(
    video: UploadFile = File(...),
    answer_text: str = Form(...)
):
    # Save uploaded video
    video_path = os.path.join(UPLOAD_DIR, video.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # --- VIDEO ANALYSIS ---
    video_result = predict_video_emotion(video_path)
    video_conf = video_result["confidence_score"]

    # --- AUDIO EXTRACTION ---
    temp_audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")

    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(temp_audio_path)
    video_clip.close()

    audio_result = predict_audio_emotion(temp_audio_path)
    audio_conf = audio_result["confidence_score"]

    os.remove(temp_audio_path)

    # --- TEXT ANALYSIS ---
    text_result = predict_text_confidence(answer_text)
    text_conf = text_result["text_confidence_score"]

    # --- FUSION ---
    final_score = compute_final_score(audio_conf, video_conf, text_conf)
    feedback = generate_feedback(audio_conf, video_conf, text_conf)

    return {
        "audio_confidence": audio_conf,
        "video_confidence": video_conf,
        "text_confidence": text_conf,
        "final_score": final_score,
        "feedback": feedback
    }


# -----------------------------
# UI Page
# -----------------------------
@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# UI Analyze Endpoint
# -----------------------------
@app.post("/analyze-ui", response_class=HTMLResponse)
async def analyze_ui(
    request: Request,
    video: UploadFile = File(...),
    answer_text: str = Form(...)
):
    video_path = os.path.join(UPLOAD_DIR, video.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # --- VIDEO ---
    video_result = predict_video_emotion(video_path)
    video_conf = video_result["confidence_score"]

    # --- AUDIO ---
    temp_audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")

    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(temp_audio_path)
    video_clip.close()

    audio_result = predict_audio_emotion(temp_audio_path)
    audio_conf = audio_result["confidence_score"]

    os.remove(temp_audio_path)

    # --- TEXT ---
    text_result = predict_text_confidence(answer_text)
    text_conf = text_result["text_confidence_score"]

    # --- FUSION ---
    final_score = compute_final_score(audio_conf, video_conf, text_conf)
    feedback = generate_feedback(audio_conf, video_conf, text_conf)

    return HTMLResponse(f"""
        <html>
        <head>
            <title>Analysis Result</title>
            <style>
                body {{ font-family: Arial; text-align: center; padding: 40px; background: #f4f6f9; }}
                .card {{
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    width: 500px;
                    margin: auto;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Interview Analysis Result</h2>
                <p>Audio Confidence: {audio_conf}</p>
                <p>Video Confidence: {video_conf}</p>
                <p>Text Confidence: {text_conf}</p>
                <h3>Final Score: {final_score}</h3>
                <p>{feedback}</p>
                <br>
                <a href="/ui">Analyze Another</a>
            </div>
        </body>
        </html>
    """)
