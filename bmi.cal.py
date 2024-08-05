import cv2
import os
import numpy as np
import math

# Load the Haar cascade file
opencv_dir = os.path.dirname(cv2.__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(opencv_dir, 'data', 'haarcascade_frontalface_default.xml'))
#open the default camera (index0)
cap=cv2.VideoCapture(0)
def extract_features(face_gray):
    # Convert the face image to a numpy array
    face_array = np.array(face_gray)

    # Calculate the facial features (e.g., distance between eyes, width of nose, shape of jawline)
    # For this example, let's assume we're calculating the distance between the eyes
    eye_distance = calculate_eye_distance(face_array)

    # Return the extracted features as a list or dictionary
    return {'eye_distance': eye_distance}

def calculate_eye_distance(face_array):
    # Implement your algorithm to calculate the distance between the eyes
    # For this example, let's assume we're using a simple thresholding approach
    _, thresh = cv2.threshold(face_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    eye_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if area > 100 and aspect_ratio > 2:
            eye_contours.append(contour)
    if len(eye_contours) == 2:
        x1, y1, w1, h1 = cv2.boundingRect(eye_contours[0])
        x2, y2, w2, h2 = cv2.boundingRect(eye_contours[1])
        eye_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return eye_distance
    else:
        return 0

def estimate_bmi(face_features):
    # Estimate the BMI based on the extracted facial features
    # For this example, let's assume we're using a simple linear regression model
    bmi = 20 + face_features['eye_distance'] * 0.1
    return bmi

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract facial features
        face_gray = gray[y:y+h, x:x+w]
        face_features = extract_features(face_gray)

        # Estimate BMI
        bmi = estimate_bmi(face_features)

        # Display the BMI
        cv2.putText(frame, f'BMI: {bmi:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the output
    cv2.imshow('Face Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()