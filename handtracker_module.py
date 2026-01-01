import cv2
import mediapipe as mp
import zmq
import numpy as np
import json

# ZeroMQ setup
context = zmq.Context()
socket_pub = context.socket(zmq.PUB)  
socket_pub.bind("tcp://*:5556")  # Binds to a TCP port for communication

def main():
    # Initialize Mediapipe Hand Tracking
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Open the webcam
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Mirror the image horizontally (flip around the y-axis)
            image = cv2.flip(image, 1)

            # Convert image to RGB (MediaPipe requires RGB format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Performance optimization

            # Process the image and detect hands
            results = hands.process(image_rgb)
            image.flags.writeable = True  # Allow modifications again

            # Prepare landmark data (normalized; x is already mirrored because the image was flipped)
            hand_landmarks_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = [{"x": lm.x, "y": lm.y} for lm in hand_landmarks.landmark]
                    hand_landmarks_data.append(landmarks)

            # Encode image to JPEG format for efficient transmission
            _, encoded_img = cv2.imencode(".jpg", image)

            # Send data over ZeroMQ
            message = {
                "landmarks": hand_landmarks_data
            }
            socket_pub.send_multipart([json.dumps(message).encode(), encoded_img.tobytes()])

    cap.release()

if __name__ == '__main__':
    main()
