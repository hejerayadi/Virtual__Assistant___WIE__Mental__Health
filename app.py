from flask import Flask, render_template, request, jsonify
from text_emotion import detect_emotions
from llm import get_llm_response
import os
import cv2
import base64
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from voice_emotion import detect_voice_emotion
from transcript import transcribe_with_language
from text_to_speech import synthesize_speech
import whisper

app = Flask(__name__)


# Load model and face detection
face_classifier = cv2.CascadeClassifier('Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
model = load_model('Emotion_Detection_CNN/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    synthesize_speech(text, "static/llm_response.wav")
    return jsonify({"audio_path": "/static/llm_response.wav"})



@app.route("/analyze", methods=["POST"])
def analyze_text():
    try:
        data = request.get_json(force=True)
        print(f"📩 Raw Data Received: {data}")
        user_input = data.get("text", "")
        if not user_input:
            return jsonify({"error": "No text provided"}), 400

        print(f"🎙️ Voice received: {user_input}")
        emotions = detect_emotions(user_input)
        llm_response = get_llm_response(user_input)

        return jsonify({
            "text_emotion": {
                "label": f"Text Emotion: {emotions.get('emotion')}",
                "confidence": emotions.get("confidence")
            },
            "response": llm_response
        })

    except Exception as e:
        print(f"❌ Error in analyze_text: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/voice_input", methods=["POST"])
def voice_input():
    data = request.get_json()
    user_input = data.get("text", "")
    print(f"🗣️ Voice transcription: {user_input}")
    return jsonify({
        "status": "ok",
        "message": "Voice received successfully",
        "text": user_input
    })


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.get_json()
    image_data = data.get("image", "")
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image_data = image_data.split(",")[1]  # Remove base64 header
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("⚠️ No face detected in frame.")
            return jsonify({"results": []})

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            confidence = float(np.max(prediction)) * 100

            print(f"🎯 Emotion Detected: {emotion_label} ({confidence:.2f}%)")

            return jsonify({
                "results": [
                    {
                        "label": emotion_label,
                        "confidence": f"{confidence:.2f}"
                    }
                ]
            })

    except Exception as e:
        print(f"❌ Error analyzing frame: {e}")
        return jsonify({"error": "Failed to process image"}), 500
    
    
@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Assure que le dossier existe

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Check file size
    if os.stat(filepath).st_size < 10_000:
        return jsonify({"error": "Audio file too short or empty"}), 400

    try:
        print("🔍 Loading Whisper base model...")
        model = whisper.load_model("base")  # or "tiny" for speed

        print("🗣️ Transcribing audio...")
        result = model.transcribe(filepath, task="transcribe", language=None)

        transcription = result.get("text", "").strip()
        language = result.get("language", "unknown")

        if not transcription:
            return jsonify({"error": "Whisper failed to transcribe audio"}), 500

        print(f"📜 Transcription: {transcription}")
        print(f"🌐 Detected Language: {language}")

        # 🎭 Emotion Detection
        emotion_result = detect_voice_emotion(filepath)

        # 🤖 LLM Response
        from llm import get_llm_response
        llm_response = get_llm_response(transcription)
        print(f"💬 LLM Response: {llm_response}")

        # 🔊 Text-to-Speech
        audio_output_path = os.path.join("static", "llm_response.wav")
        print(f"$$$$$$$$$$$$$$$$$$$$$$[Whisper] Detected language: {language}")
        synthesize_speech(llm_response, lang_code=language, output_path=audio_output_path)
        return jsonify({
            "transcription": transcription,
            "language": language,
            "voice_emotion": {
                "label": f"Voice Emotion: {emotion_result.get('emotion')}",
                "confidence": emotion_result.get("confidence")
            },
            "response": llm_response,
            "audio_url": "/static/llm_response.wav"
        })


    except Exception as e:
        print(f"❌ Error in voice emotion detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
