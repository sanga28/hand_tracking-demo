
import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Color mapping for fingers
finger_colors = {
    'thumb': (255, 0, 0),
    'index': (0, 255, 0),
    'middle': (0, 0, 255),
    'ring': (255, 255, 0),
    'pinky': (255, 0, 255)
}

# Helper function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Connect to ESP32
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Allow ESP32 to reset

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark points
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

            # Finger joints for angles
            fingers = {
                'thumb': (4, 3, 2),  
                'index': (8, 7, 6),
                'middle': (12, 11, 10),
                'ring': (16, 15, 14),
                'pinky': (20, 19, 18)
            }

            # Calculate angles for all fingers
            angles = {}
            for finger, (a, b, c) in fingers.items():
                angle = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
                angles[finger] = round(angle, 2)

            # Calculate the Thumb Side Angle (between wrist, thumb MCP, and thumb tip)
            thumb_side_angle = calculate_angle(landmarks[0], landmarks[2], landmarks[4])

            # Send data to ESP32 in the required format
            angle_string = f"ind_{angles['index']}:mid_{angles['middle']}:rin_{angles['ring']}:lit_{angles['pinky']}:thu_{angles['thumb']}:ths_{thumb_side_angle}\n"
            ser.write(angle_string.encode())

            # Display angles on the frame
            y_offset = 30
            cv2.putText(frame, f"Thumb Side: {round(thumb_side_angle, 2)}°", (frame.shape[1] - 200, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for finger, angle in angles.items():
                y_offset += 30
                cv2.putText(frame, f"{finger.capitalize()}: {angle}°", 
                            (frame.shape[1] - 200, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            finger_colors[finger], 2)

    # Display the frame
    cv2.imshow('Hand Angle Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
