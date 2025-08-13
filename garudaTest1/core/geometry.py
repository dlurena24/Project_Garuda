"""
Geometry utilities for stereo vision processing.

This module provides functions to:
- Calculate 3D coordinates from stereo image points using the pinhole camera model.
- Determine if a given point lies inside a polygonal region.

Parameters from Configuration
-----------------------------
B : float
    Baseline distance between the two cameras (in meters).
FOV : float
    Field of view of the cameras (in degrees).

Requirements
------------
- OpenCV (cv2)
- NumPy
"""

import cv2
import numpy as np
from config import B, FOV

def calculate_3d_point(x1, x2, width):
    """
    Calculate the 3D coordinates (X, Z) of a point from stereo camera image coordinates.

    Parameters
    ----------
    x1 : float
        X-coordinate of the point in the left image (in pixels).
    x2 : float
        X-coordinate of the point in the right image (in pixels).
    width : int
        Width of the image in pixels.

    Returns
    -------
    tuple of float or None
        A tuple containing:
            - X (float): Horizontal position in meters (approximation).
            - Z (float): Depth/distance from the camera in meters.
        Returns None if disparity is zero (cannot calculate depth).

    Notes
    -----
    - Uses the baseline distance (B) between the cameras and the field of view (FOV)
      from the configuration file.
    - X is scaled based on an arbitrary factor (division by 30.0) for positioning.
    - Z is computed from disparity using the pinhole camera model.
    """

    fPixel = (width * 0.5) / np.tan(FOV * 0.5 * np.pi / 180)
    disparity = abs(x2 - x1)
    if disparity == 0:
        return None
    Z = (B * fPixel) / disparity
    X = ((x1 + x2) / 2.0) / 30.0
    return float(X), float(Z)

def punto_en_zona(punto, zona):
    """
    Check if a given 2D point lies inside a defined polygonal zone.

    Parameters
    ----------
    punto : tuple of float
        The (x, y) coordinates of the point to check.
    zona : list of tuple of float
        List of vertices defining the polygonal zone.

    Returns
    -------
    bool
        True if the point is inside or on the edge of the polygon, False otherwise.

    Notes
    -----
    - Uses OpenCV's `pointPolygonTest` for point-in-polygon detection.
    - The polygon must have at least three vertices to form a valid zone.
    """

    if not zona or len(zona) < 3:
        return False
    pts = np.array(zona, np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, punto, False)
    return result >= 0
