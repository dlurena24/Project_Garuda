import cv2
import numpy as np
import math
import shapeDetection as sd
from ArUco_detection_test import MarkerDetectionSystem
from ultralytics import YOLO
import time

# Camera parameters
CAMERA_ANGLE = 30  # degrees
DISTANCE_TO_SHAPE = 170  # cm
B = 20  # Distance between cameras (cm)
f = 26  # Focal length (mm)
FOV = 80  # Field of view (degrees)

# Program phases
PHASE_CAMERA_VIEW = 0  # Just show camera feeds
PHASE_SHAPE_DETECTION = 1  # Detect shapes and show bird's eye view
PHASE_OBJECT_DETECTION = 2  # Detect objects and show on map
current_phase = PHASE_CAMERA_VIEW

# Flags to track if detection has been performed
shape_detection_done = False
birds_eye_view = None
corner_3d_positions = []
object_position = None


# Calculates the coordinates of the corners for the received image
def calculate_obj_coordinates(frame):
    try:
        results = model.predict(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:  # Check if any objects were detected
                box = boxes[0]  # Take first detected object
                x1, y1, x2, y2 = [int(coord.item()) for coord in box.xyxy[0]]
                x = (x1 + x2) // 2
                y = y2
                return x, y
        return None, None  # Return None if no objects detected
    except Exception as e:
        print(f"Error in object detection: {e}")
        return None, None


# Calculate 3D positions of the received corners
def calculate_3d_point(x1, x2, width):
    fPixel = (width * 0.5) / np.tan(FOV * 0.5 * np.pi / 180)

    # Calculate disparity
    disparity = abs(x2 - x1)

    if disparity == 0:
        return None  # Avoid division by zero

    # Calculate Z (depth)
    Z = (B * fPixel) / disparity

    # Calculate X (horizontal position)
    X = (x2) / 30

    return X, Z


# Window constructor for the top view of the scanned map and detected objects
def create_birds_eye_view(corner_3d_positions, object_position=None, canvas_size=(600, 600), margin=50):
    # Create a white canvas
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255

    if not corner_3d_positions:
        cv2.putText(canvas, "No valid 3D points found", (50, canvas_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return canvas

    # Extract X and Z coordinates
    x_coords = [x for x, z in corner_3d_positions]
    z_coords = [z for x, z in corner_3d_positions]

    # Find the range of coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    z_min, z_max = min(z_coords), max(z_coords)

    # Calculate scaling factors to fit the shape in the canvas
    available_width = canvas_size[0] - 2 * margin
    available_height = canvas_size[1] - 2 * margin

    x_scale = available_width / (x_max - x_min) if x_max != x_min else 1
    z_scale = available_height / (z_max - z_min) if z_max != z_min else 1

    scale = min(x_scale, z_scale)

    # Function to transform 3D coordinates to canvas coordinates
    def transform_point(x, z):
        canvas_x = int(margin + (x - x_min) * scale)
        canvas_y = int(canvas_size[1] - (margin + (z - z_min) * scale))
        return (canvas_x, canvas_y)

    # Transform all points
    canvas_points = [transform_point(x, z) for x, z in corner_3d_positions]

    # Draw the shape
    for i in range(len(canvas_points)):
        # Draw lines between points
        cv2.line(canvas, canvas_points[i], canvas_points[(i + 1) % len(canvas_points)], (0, 0, 255), 2)

        # Draw and label corners
        cv2.circle(canvas, canvas_points[i], 5, (255, 0, 0), -1)
        cv2.putText(canvas, f"{i}", (canvas_points[i][0] + 10, canvas_points[i][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add distance labels
        mid_x = (canvas_points[i][0] + canvas_points[(i + 1) % len(canvas_points)][0]) // 2
        mid_y = (canvas_points[i][1] + canvas_points[(i + 1) % len(canvas_points)][1]) // 2

        # Calculate actual distance between points
        dx = corner_3d_positions[i][0] - corner_3d_positions[(i + 1) % len(corner_3d_positions)][0]
        dz = corner_3d_positions[i][1] - corner_3d_positions[(i + 1) % len(corner_3d_positions)][1]
        distance = math.sqrt(dx ** 2 + dz ** 2)

        cv2.putText(canvas, f"{distance:.1f}cm", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add object marker if detected
    if object_position:
        obj_canvas_point = transform_point(object_position[0], object_position[1])
        cv2.circle(canvas, obj_canvas_point, 8, (0, 255, 0), -1)  # Green circle for object
        cv2.putText(canvas, "Object", (obj_canvas_point[0] + 10, obj_canvas_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return canvas


# Function to display phase information on frames
def add_phase_info(frame, phase):
    phase_text = ""
    key_info = ""

    if phase == PHASE_CAMERA_VIEW:
        phase_text = "Phase: Camera View"
        key_info = "Press 's' for Shape Detection"
    elif phase == PHASE_SHAPE_DETECTION:
        phase_text = "Phase: Shape Detection"
        key_info = "Press 'o' for Object Detection, 'r' to rescan"
    elif phase == PHASE_OBJECT_DETECTION:
        phase_text = "Phase: Object Detection"
        key_info = "Press 'c' to return to Camera View, 'r' to rescan"

    # Add phase information at the bottom of the frame
    cv2.putText(frame, phase_text, (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, key_info, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


# Function to perform shape detection once
def perform_shape_detection(frame1, frame2):
    global corner_3d_positions, birds_eye_view

    mds1 = MarkerDetectionSystem()
    mds2 = MarkerDetectionSystem()

    # Process frames for shape detection
    #contours1, mask1, corners1 = sd.detectar_contornos_y_esquinas(frame1)
    #contours2, mask2, corners2 = sd.detectar_contornos_y_esquinas(frame2)

    corners1 = mds1.detect_markers(frame1)
    print(corners1)
    corners2 = mds2.detect_markers(frame2)
    print(corners2)

    height, width, depth = frame1.shape

    # Calculate 3D positions of the corners
    corner_3d_positions = []
    for i in range(min(len(corners1), len(corners2))):
        x1, y1 = corners1[i]
        x2, y2 = corners2[i]

        result = calculate_3d_point(x1, x2, width)
        if result:
            X, Z = result
            corner_3d_positions.append((X, Z))

    # Create bird's eye view
    birds_eye_view = create_birds_eye_view(corner_3d_positions)

    # Return contours and corners for display
    return contours1, corners1, contours2, corners2


# Function to perform object detection once
def perform_object_detection(frame1, frame2):
    global object_position, birds_eye_view

    height, width, depth = frame1.shape

    # Object detection
    xL, yL = calculate_obj_coordinates(frame1)
    xR, yR = calculate_obj_coordinates(frame2)

    if xL is not None and xR is not None:
        resultOBJ = calculate_3d_point(xL, xR, width)
        if resultOBJ:
            X, Z = resultOBJ
            object_position = (X, Z)

            # Update bird's eye view with object position
            birds_eye_view = create_birds_eye_view(corner_3d_positions, object_position)

            return xL, yL, xR, yR

    return None, None, None, None


# .::Main code:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::.

print(".:::::::::::::::Welcome to the Garuda Project!:::::::::::::::.")
print("Loading base parameters...")

# Initialize YOLO model
model = YOLO("best.pt")
print("YOLO loaded")
print("Loading webcams...")

# Initialize webcams
cap1 = cv2.VideoCapture(2)  # Left camera
cap2 = cv2.VideoCapture(1)  # Right camera

# Check if the webcams are opened correctly
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open webcams.")
    exit()

# Set resolution for both cameras (optional)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create windows
cv2.namedWindow('Combined View', cv2.WINDOW_NORMAL)
print("Webcams loaded")

# Print instructions
print("\nProject Garuda - Multi-Phase Detection System")
print("--------------------------------------------")
print("Phase 1: Camera View - Just showing camera feeds")
print("Press 's' to enter Shape Detection phase")
print("Phase 2: Shape Detection - Detecting shapes and showing bird's eye view")
print("Press 'o' to enter Object Detection phase")
print("Phase 3: Object Detection - Detecting objects and showing on map")
print("Press 'r' to rescan in current phase")
print("Press 'c' to return to Camera View phase")
print("Press 'q' to quit the program")

print("\n Loading Main loop...")
# Variables to store detection results
contours1, corners1, contours2, corners2 = [], [], [], []
xL, yL, xR, yR = None, None, None, None
last_object_scan_time = 0

# Main loop
while True:
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Failed to grab frames from one or both cameras.")
        break

    # Mirror the images horizontally
    frame1 = cv2.flip(frame1, 1)
    frame2 = cv2.flip(frame2, 1)

    # Create copies for display
    display_frame1 = frame1.copy()
    display_frame2 = frame2.copy()

    # Add phase information to frames
    display_frame1 = add_phase_info(display_frame1, current_phase)
    display_frame2 = add_phase_info(display_frame2, current_phase)

    # Process based on current phase
    if current_phase == PHASE_SHAPE_DETECTION and not shape_detection_done:
        # Perform shape detection once when entering this phase
        contours1, corners1, contours2, corners2 = perform_shape_detection(frame1, frame2)
        shape_detection_done = True
        print("Shape detection completed")

    elif current_phase == PHASE_OBJECT_DETECTION:
        current_time = time.time()
        # Perform object detection once when entering this phase or every 5 seconds
        if object_position is None or (current_time - last_object_scan_time) >= 5:
            xL, yL, xR, yR = perform_object_detection(frame1, frame2)
            last_object_scan_time = current_time
            #if object_position is not None:
            #    print("Object detection updated")
    # Draw detection results on display frames if available
    if current_phase >= PHASE_SHAPE_DETECTION and shape_detection_done:
        # Draw contours and corners on display frames
        cv2.drawContours(display_frame1, contours1, -1, (0, 255, 0), 2)
        cv2.drawContours(display_frame2, contours2, -1, (0, 255, 0), 2)

        # Draw and number corners
        for i, corner in enumerate(corners1):
            cv2.circle(display_frame1, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(display_frame1, f"{i}", tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for i, corner in enumerate(corners2):
            cv2.circle(display_frame2, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(display_frame2, f"{i}", tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw object markers if detected
    if current_phase == PHASE_OBJECT_DETECTION and object_position is not None:
        if xL is not None:
            cv2.circle(display_frame1, (xL, yL), 10, (0, 255, 255), -1)
            cv2.putText(display_frame1, "Object", (xL + 10, yL),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if xR is not None:
            cv2.circle(display_frame2, (xR, yR), 10, (0, 255, 255), -1)
            cv2.putText(display_frame2, "Object", (xR + 10, yR),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display bird's eye view if available
    if current_phase >= PHASE_SHAPE_DETECTION and birds_eye_view is not None:
        cv2.imshow('Bird\'s Eye View', birds_eye_view)

    # Create combined view (side by side)
    # Resize if needed to make them the same height
    height_1 = display_frame1.shape[0]
    height_2 = display_frame2.shape[0]

    # Use the smaller height for both
    target_height = min(height_1, height_2)

    # Resize maintaining aspect ratio
    aspect_1 = display_frame1.shape[1] / display_frame1.shape[0]
    aspect_2 = display_frame2.shape[1] / display_frame2.shape[0]

    width_1 = int(target_height * aspect_1)
    width_2 = int(target_height * aspect_2)

    resized_1 = cv2.resize(display_frame1, (width_1, target_height))
    resized_2 = cv2.resize(display_frame2, (width_2, target_height))

    # Concatenate horizontally
    combined_frame = np.hstack((display_frame2, display_frame1))
    cv2.imshow('Combined View', combined_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Switch to Shape Detection phase
        current_phase = PHASE_SHAPE_DETECTION
        shape_detection_done = False  # Reset flag to trigger new detection
        print("Switched to Shape Detection phase")
    elif key == ord('o'):  # Switch to Object Detection phase
        if current_phase == PHASE_SHAPE_DETECTION and shape_detection_done:
            current_phase = PHASE_OBJECT_DETECTION
            object_position = None  # Reset to trigger new object detection
            print("Switched to Object Detection phase")
        else:
            print("Please complete shape detection first")
    elif key == ord('c'):  # Switch back to Camera View phase
        current_phase = PHASE_CAMERA_VIEW
        shape_detection_done = False
        object_position = None
        # Close Bird's Eye View window if open
        try:
            cv2.destroyWindow('Bird\'s Eye View')
        except:
            pass
        print("Switched to Camera View phase")
    elif key == ord('r'):  # Rescan in current phase
        if current_phase == PHASE_SHAPE_DETECTION:
            shape_detection_done = False
            print("Rescanning shape...")
        elif current_phase == PHASE_OBJECT_DETECTION:
            object_position = None
            print("Rescanning object...")

    # Add a small delay to reduce CPU usage
    time.sleep(0.01)

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()

