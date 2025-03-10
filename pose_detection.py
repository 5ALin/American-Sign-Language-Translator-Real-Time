import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import joblib

class PoseDetector:
    def __init__(self):
        """Initialize the PoseDetector with MediaPipe Hands and Load ASL Model."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        # Load trained CNN + LSTM model & scaler
        self.model = tf.keras.models.load_model("asl_cnn_lstm_model.h5")
        self.scaler = joblib.load("scaler.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")

    def detect_hands(self, frame):
        """Detect hand landmarks and return translated ASL letter."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)

        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # Extract x, y, z coordinates
                
                hand_landmarks_list.append(landmarks)

                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        # If hand landmarks detected, classify the sign
        asl_letter = "..."
        if hand_landmarks_list:
            features = np.array(hand_landmarks_list[0]).reshape(1, -1)  # Reshape for model input
            features_scaled = self.scaler.transform(features)

            # 🔹 **Fix the Reshaping Issue Here**
            features_reshaped = features_scaled.reshape(1, 3, 21)  # (batch_size=1, time_steps=3, features=21)

            # Predict ASL letter
            prediction = self.model.predict(features_reshaped)
            predicted_index = np.argmax(prediction)
            asl_letter = self.label_encoder.inverse_transform([predicted_index])[0]

        return frame, asl_letter

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
