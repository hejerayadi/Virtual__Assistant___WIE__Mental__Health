from flask import Flask, render_template, request, jsonify, Response
from text_emotion import detect_emotions
from llm import get_llm_response
import os
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and cascade
face_classifier = cv2.CascadeClassifier(r'Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
classifier = load_model(r'Emotion_Detection_CNN/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Keep track of selected camera index
camera_index = 0

def list_available_cameras(max_devices=5):
    available = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.read()[0]:
            available.append(index)
        cap.release()
    return available

def generate_frames(cam_id=0):
    camera = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    confidence = prediction.max() * 100

                    label_text = f'{label} ({int(confidence)}%)'
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/", methods=["GET"])
def index():
    cameras = list_available_cameras()
    return render_template("index.html", cameras=cameras)

@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    user_input = data.get("text", "")
    emotions = detect_emotions(user_input)
    llm_response = get_llm_response(user_input)
    return jsonify({
        "emotions": emotions,
        "response": llm_response
    })

@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    return Response(generate_frames(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
