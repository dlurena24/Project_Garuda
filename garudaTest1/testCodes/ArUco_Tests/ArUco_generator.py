import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate a marker
for marker_id in range(5):  # Generar 5 marcadores del 0 al 4
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 700)
    filename = f"aruco_marker_id_{marker_id}.png"
    cv2.imwrite(filename, marker_image)
