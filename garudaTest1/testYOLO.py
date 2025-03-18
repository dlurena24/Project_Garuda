import cv2
import time
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(1)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Perform object detection on the frame
        results = model.predict(frame, show=True)

        # Access the bounding box coordinates
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]

                # Calculate the bottom-center coordinate
                bottom_center_x = (x1 + x2) // 2
                bottom_center_y = y2
                bottom_center = (bottom_center_x, bottom_center_y)

                print(f"Bottom-Center: {bottom_center}")

                # Draw the bottom center point on the frame
                cv2.circle(frame, bottom_center, 5, (0, 255, 0), -1)

            # Display the frame with detections
            cv2.imshow("YOLO Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
