import cv2
import numpy as np
import dlib

# CONSTANTS
ESCAPE_KEY_CODE = 27

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        # Show bounding box on the detected face
        img_original = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray_frame, face)

        for landmark in range(68):
            x = landmarks.part(landmark).x
            y = landmarks.part(landmark).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ESCAPE_KEY_CODE:
        break