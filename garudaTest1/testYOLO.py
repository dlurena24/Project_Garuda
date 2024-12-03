import cv2
import time
from ultralytics import YOLO


# Load a model
model = YOLO("best.pt")

# Perform object detection on an image
results = model.predict("Imgs/t1.jpg", save=True, show=True)

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
        time.sleep(15)