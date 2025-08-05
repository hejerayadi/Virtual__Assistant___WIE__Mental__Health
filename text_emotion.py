from transformers import pipeline

# Initialize once
emotion_classifier = pipeline("text-classification",
                              model="j-hartmann/emotion-english-distilroberta-base",
                              return_all_scores=True)

def detect_emotions(text):
    print(f"🧪 Detecting emotion from text: {text}")

    if not text.strip():
        print("⚠️ Empty input text.")
        return {"error": "No text provided"}

    try:
        results = emotion_classifier(text)[0]
        print(f"✅ Raw results: {results}")
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_emotions = {emotion['label']: float(f"{emotion['score']:.4f}") for emotion in sorted_results[:3]}
        return top_emotions
    except Exception as e:
        print(f"❌ Error in detect_emotions: {e}")
        return {"error": str(e)}
