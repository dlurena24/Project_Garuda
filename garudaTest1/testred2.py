import cv2
import numpy as np
import math
import shapeDetection as sd

# Camera parameters

CAMERA_ANGLE = 30           # degrees
DISTANCE_TO_SHAPE = 70      # cm

B = 20                      # Distance between cameras (cm)
f = 26                      # Focal length (mm)
FOV = 80                    # FIeld of view (degrees)

fPixel = ()/

def calculate_3d_point(x1, x2, focal_length=1000):
    # Convert camera angle to radians
    angle_rad = math.radians(CAMERA_ANGLE)

    # Calculate disparity
    disparity = abs(x1 - x2)

    if disparity == 0:
        return None  # Avoid division by zero

    # Calculate Z (depth)
    Z = (CAMERA_DISTANCE * focal_length) / disparity

    # Calculate X (horizontal position)
    X = (Z * (x1 + x2)) / (2 * focal_length)

    return X, Z


def match_corners(corners1, corners2):
    # Simple matching based on vertical position
    # You might need a more sophisticated matching algorithm depending on your setup
    corners1_sorted = sorted(corners1, key=lambda x: x[1])
    corners2_sorted = sorted(corners2, key=lambda x: x[1])

    return list(zip(corners1_sorted, corners2_sorted))


def create_birds_eye_view(corner_3d_positions, canvas_size=(600, 600), margin=50):
    # Create a white canvas
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255

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

    return canvas

# Main code
img1 = cv2.imread('Imgs/izq.jpg')
img2 = cv2.imread('Imgs/der.jpg')

contours1, mask1, corners1 = sd.detectar_contornos_y_esquinas(img1)
contours2, mask2, corners2 = sd.detectar_contornos_y_esquinas(img2)

# Match corners between images
matched_corners = match_corners(corners1, corners2)

# Calculate 3D positions
corner_3d_positions = []
for (corner1, corner2) in matched_corners:
    x1, y1 = corner1
    x2, y2 = corner2

    result = calculate_3d_point(x1, x2)
    if result:
        X, Z = result
        corner_3d_positions.append((X, Z))

# After calculating corner_3d_positions, add:
birds_eye_view = create_birds_eye_view(corner_3d_positions)
cv2.imshow('Bird\'s Eye View', birds_eye_view)

# Visualize the results
for idx, (img, contours, corners) in enumerate([(img1, contours1, corners1), (img2, contours2, corners2)]):
    img_result = img.copy()

    # Draw contours
    cv2.drawContours(img_result, contours, -1, (0, 255, 0), 2)

    # Draw and number corners
    for i, corner in enumerate(corners):
        cv2.circle(img_result, tuple(corner), 5, (0, 0, 255), -1)
        cv2.putText(img_result, f"{i}", tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    imgResize = cv2.resize(img_result, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow(f'Image {idx + 1} with Contours and Corners', imgResize)

# Print 3D positions
print("Estimated 3D positions of corners (X, Z):")
for i, pos in enumerate(corner_3d_positions):
    print(f"Corner {i}: X={pos[0]:.2f}cm, Z={pos[1]:.2f}cm")

cv2.waitKey(0)
cv2.destroyAllWindows()