from inference.video_infer import predict_video_emotion

video_path = "data/video/fast.mkv"  # change to your filename

result = predict_video_emotion(video_path)

print("\n🎯 FINAL RESULT:")
print(result)
