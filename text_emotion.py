from transformers import pipeline

# Initialize once
emotion_classifier = pipeline("text-classification",
                              model="j-hartmann/emotion-english-distilroberta-base",
                              return_all_scores=True)

def detect_emotions(text):
    results = emotion_classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    # Return top 3 emotions and their scores as a dict
    top_emotions = {emotion['label']: emotion['score'] for emotion in sorted_results[:3]}
    return top_emotions
