from inference.audio_infer import predict_audio_emotion
from inference.video_infer import predict_video_emotion
from inference.text_infer import predict_text_confidence
from inference.fusion import compute_final_score, generate_feedback

# ---- AUDIO ----
audio_result = predict_audio_emotion(
    "data/audio/Actor_01/03-01-01-01-01-01-01.wav"
)
audio_conf = audio_result["confidence_score"]

# ---- VIDEO ----
video_result = predict_video_emotion("data/video/fast.mkv")
video_conf = video_result["confidence_score"]

# ---- TEXT ----
answer_text = """
I believe my strengths include consistency, structured thinking,
and the ability to solve complex problems step by step.
"""
text_result = predict_text_confidence(answer_text)
text_conf = text_result["text_confidence_score"]

# ---- FUSION ----
final_score = compute_final_score(audio_conf, video_conf, text_conf)
feedback = generate_feedback(audio_conf, video_conf, text_conf)

print("\n🎯 FINAL TRI-MODAL INTERVIEW ANALYSIS")
print("Audio Confidence:", audio_conf)
print("Video Confidence:", video_conf)
print("Text Confidence:", text_conf)
print("Final Confidence Score:", final_score)
print("Feedback:", feedback)
