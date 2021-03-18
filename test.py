from typing import final
import cv2
import numpy as np
import dlib
from math import hypot, pow, sqrt

# CONSTANTS
ESCAPE_KEY_CODE = 27

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cooler_img = cv2.imread('./assets/cooler.png')

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        # Show bounding box on the detected face
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        img_original = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray_frame, face)

        head_center = (landmarks.part(27).x, landmarks.part(27).y)
        head_left = (landmarks.part(0).x, landmarks.part(0).y)
        head_right = (landmarks.part(16).x, landmarks.part(16).y)

        # Calculate distance between left head point and right head point
        cooler_width = int(sqrt(pow(head_left[0] - head_right[0], 2) - pow(head_left[1] - head_right[1], 2)) * 1.1)
        cooler_height = int(cooler_width * 0.33)

        cooler = cv2.resize(cooler_img, (cooler_width, cooler_height))

        # Adding a factor of 10px to y in-order to make the effect look more natural
        top_left = (int(head_center[0] - cooler_width / 2),
                    int(head_center[1] - cooler_height / 2) + 10)
        bottom_right = (int(head_center[0] + cooler_width / 2),
                    int(head_center[1] + cooler_height / 2) + 10)

        area = frame[top_left[1]: top_left[1] + cooler_height,
                     top_left[0]: top_left[0] + cooler_width]
        cv2.imshow("Cooler area", area)

        # Area where the cooler has to be placed (using a mask)
        cooler_area = frame[top_left[1]: top_left[1] + cooler_height,
                            top_left[0]: top_left[0] + cooler_width]
        cooler_area_gray = cv2.cvtColor(cooler, cv2.COLOR_BGR2GRAY)
        _, cooler_mask = cv2.threshold(cooler_area_gray, 25, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('cooler pic', cooler)
        cv2.imshow('cooler area', cooler_area)
        cv2.imshow('cooler mask', cooler_mask)

        # Taking bitwise AND to remove all empty space in the cooler image mask
        eye_area_cooler = cv2.bitwise_and(cooler_area, cooler_area, mask=cooler_mask)

        final_cooler_area = cv2.add(eye_area_cooler, cooler)

        # Replacing the part of frame where the cooler is supposed to be
        frame[top_left[1]: top_left[1] + cooler_height,
            top_left[0]: top_left[0] + cooler_width] = final_cooler_area
        
        for landmark in range(68):
            x = landmarks.part(landmark).x
            y = landmarks.part(landmark).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ESCAPE_KEY_CODE:
        break

cap.release()
cv2.destroyAllWindows()
