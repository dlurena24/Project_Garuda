
import cv2
import numpy as np

def webcam_test():
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame was successfully captured
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Test', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_test()
