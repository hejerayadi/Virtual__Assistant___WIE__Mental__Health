import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.read()[0]:
        print(f"Camera index {i} is available.")
    cap.release()
