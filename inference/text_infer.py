from transformers import pipeline


# Load pretrained sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def predict_text_confidence(answer_text):
    """
    Uses DistilBERT sentiment model to estimate answer confidence.
    """

    result = sentiment_model(answer_text)[0]

    label = result["label"]
    score = result["score"]

    # Map sentiment to interview confidence
    if label == "POSITIVE":
        confidence_score = score
    else:
        confidence_score = 1 - score

    # Categorize answer quality
    if confidence_score > 0.75:
        quality = "Strong"
    elif confidence_score > 0.55:
        quality = "Average"
    else:
        quality = "Weak"

    return {
        "text_confidence_score": round(confidence_score, 2),
        "answer_quality": quality
    }
