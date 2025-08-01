import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load model and cascade
face_classifier = cv2.CascadeClassifier(r'Emotion_Detection_CNN/haarcascade_frontalface_default.xml')
classifier = load_model(r'Emotion_Detection_CNN/model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def generate_frames(cam_id=0):
    camera = cv2.VideoCapture(cam_id)
    if not camera.isOpened():
        raise RuntimeError(f"Camera {cam_id} not available")

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
