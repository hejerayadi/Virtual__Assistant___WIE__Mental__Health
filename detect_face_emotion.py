import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load face detector
face_classifier = cv2.CascadeClassifier('./Emotion_Detection_CNN/haarcascade_frontalface_default.xml')

# Load emotion classifier
classifier = load_model('./Emotion_Detection_CNN/model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def predict_emotion_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Erreur : Impossible de lire l'image à partir de {image_path}.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = prediction.max() * 100

            cv2.putText(frame, f"{label} ({confidence:.0f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Face Error', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
predict_emotion_on_image("images test/angry woman.jpg")  # Replace with your image path
