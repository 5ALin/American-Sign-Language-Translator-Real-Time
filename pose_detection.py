import mediapipe as mp
import cv2
import numpy as np
import tensorflow.lite as tflite
import joblib

class PoseDetector:
    def __init__(self):
        """Initialize PoseDetector with MediaPipe Hands and Selfie Segmentation."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        # Load TensorFlow Lite model
        self.interpreter = tflite.Interpreter(model_path="asl_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load label encoder & scaler
        self.scaler = joblib.load("scaler.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")

        # Initialize Selfie Segmentation for background removal
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def detect_hands(self, frame):
        """Detect hand landmarks and classify ASL sign using TFLite model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply Selfie Segmentation
        segmentation_result = self.selfie_segmentation.process(frame_rgb)
        mask = segmentation_result.segmentation_mask

        # Create a black background
        black_background = np.zeros_like(frame)

        # Apply mask: Keep only the person, replace the rest with black
        condition = mask > 0.5  # Pixels where segmentation is strong
        frame_no_bg = np.where(condition[:, :, None], frame, black_background)

        # Detect hands
        hand_results = self.hands.process(frame_rgb)

        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                hand_landmarks_list.append(landmarks)

                # Draw landmarks only on the processed frame
                self.mp_drawing.draw_landmarks(
                    frame_no_bg, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        asl_letter = "..."
        if hand_landmarks_list:
            features = np.array(hand_landmarks_list[0]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            features_reshaped = features_scaled.reshape(1, 3, 21)

            # Run inference using TFLite
            self.interpreter.set_tensor(self.input_details[0]['index'], features_reshaped.astype(np.float32))
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

            predicted_index = np.argmax(prediction)
            asl_letter = self.label_encoder.inverse_transform([predicted_index])[0]

        return frame_no_bg, asl_letter

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.selfie_segmentation.close()
