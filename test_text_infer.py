from inference.text_infer import predict_text_confidence

answer = """
I believe my strengths include consistency, structured thinking,
and the ability to solve complex problems step by step.
"""

result = predict_text_confidence(answer)

print("\n🎯 TEXT ANALYSIS RESULT")
print(result)
