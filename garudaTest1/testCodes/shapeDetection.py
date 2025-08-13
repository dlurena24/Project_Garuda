import cv2
import numpy as np
import math

def detectar_contornos_y_esquinas(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Changed from red to yellow detection
    lower_yellow = np.array([95, 100, 150])
    upper_yellow = np.array([115, 255, 255])
    # Create mask for yellow color
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #corners = []
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        corners = [point[0] for point in approx]
        # Sort corners from left to right using the x axis
        corners_sorted = sorted(corners, key=lambda point: point[0])

    return contours, yellow_mask, corners_sorted
