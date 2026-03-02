def compute_final_score(audio_confidence, video_confidence, text_confidence):
    """
    Combine audio, video, and text confidence scores
    """

    final_score = (
        0.4 * audio_confidence +
        0.3 * video_confidence +
        0.3 * text_confidence
    )

    return round(final_score, 2)


def generate_feedback(audio_confidence, video_confidence, text_confidence):
    """
    Generate human-readable interview feedback
    """

    feedback = []

    # Audio feedback
    if audio_confidence < 0.6:
        feedback.append("Your voice suggests nervousness. Try speaking more steadily and at a controlled pace.")

    # Video feedback
    if video_confidence < 0.6:
        feedback.append("Facial analysis indicates tension. Maintain relaxed expressions and natural eye contact.")

    # Text feedback
    if text_confidence < 0.6:
        feedback.append("Your answer lacks strong confidence indicators. Use clearer structure and decisive language.")

    # Strong performance case
    if audio_confidence >= 0.75 and video_confidence >= 0.7 and text_confidence >= 0.75:
        feedback.append("Excellent overall performance. You appeared confident, composed, and articulate.")

    if not feedback:
        feedback.append("Overall performance is balanced. Minor improvements can further enhance confidence.")

    return " ".join(feedback)
