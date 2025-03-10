import cv2
from pose_detection import PoseDetector
import time

def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Starting ASL Translator. Press 'ESC' to exit.")

    last_detected_letter = "..."  # Track last detected letter
    last_time_detected = time.time()  # Track time of last detection

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Detect ASL letter
        frame, current_letter = detector.detect_hands(frame)

        # Only update if a new letter is detected (different from last one)
        if current_letter != last_detected_letter and current_letter != "...":
            last_detected_letter = current_letter  # Update letter
            last_time_detected = time.time()  # Reset detection timer

        # Display only the stable letter
        cv2.putText(frame, f"ASL: {last_detected_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("ASL Translator", frame)

        # Exit when 'ESC' is pressed
        if cv2.waitKey(10) & 0xFF == 27:
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
