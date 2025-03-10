import cv2
from pose_detection import PoseDetector
from collections import deque

def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Starting ASL Translator. Press 'ESC' to exit.")

    letter_buffer = deque(maxlen=10)  # Store last 10 detected letters
    translated_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Detect ASL letter
        frame, asl_letter = detector.detect_hands(frame)
        
        if asl_letter != "...":  # Ignore "no detection"
            letter_buffer.append(asl_letter)

        # Convert letters to words
        translated_text = "".join(letter_buffer)

        # Display ASL translation
        cv2.putText(frame, f"ASL: {translated_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
