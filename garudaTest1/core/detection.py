"""
Object detection and ArUco marker scanning utilities.

This module provides:
- A class for detecting objects in images using a YOLO model and drawing bounding boxes.
- A function for scanning and matching ArUco markers from stereo images to estimate their 3D positions.

Requirements
------------
- OpenCV (cv2)
- ultralytics.YOLO
- core.geometry.calculate_3d_point
"""

import cv2
from ultralytics import YOLO
from core.geometry import calculate_3d_point

class ObjectDetector:
    """
    Class for detecting objects in image frames using a YOLO model.

    Parameters
    ----------
    model_path : str
        Path to the YOLO model file to be loaded.

    Attributes
    ----------
    model : ultralytics.YOLO
        The loaded YOLO model used for object detection.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_draw(self, frame):
        """
        Detect objects in a frame, draw bounding boxes, and return object data.

        Parameters
        ----------
        frame : numpy.ndarray
            The image frame in which to detect objects.

        Returns
        -------
        list of tuple
            A list of tuples containing:
                - label (str): Detected object class name.
                - (cx, cy) (tuple of float): Coordinates of the object's centroid in pixels.

        Notes
        -----
        - Bounding boxes are drawn in green with labels positioned above them.
        - Uses the YOLO model loaded in the `ObjectDetector` instance.
        """
        detections = []
        results = self.model(frame)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.model.names.get(cls, str(cls))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), max(10, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append((label, (float(cx), float(cy))))
        return detections

def scan_aruco(frameL, frameR, mdsL, mdsR):
    """
    Scan stereo frames for ArUco markers, find common markers, and calculate their 3D positions.

    Parameters
    ----------
    frameL : numpy.ndarray
        The left camera frame.
    frameR : numpy.ndarray
        The right camera frame.
    mdsL : object
        Marker detection system for the left camera, with methods:
        - detect_markers(frame)
        - draw_markers(frame)
        - detected_markers (dict with 'Centroid' entries).
    mdsR : object
        Marker detection system for the right camera, with the same API as `mdsL`.

    Returns
    -------
    list of tuple
        A list of tuples containing:
            - mid (int): Marker ID.
            - (X, Z) (tuple of float): 3D coordinates in meters.

    Notes
    -----
    - Only markers detected in both frames are processed.
    - 3D positions are calculated using `calculate_3d_point` from `core.geometry`.
    """
    mdsL.detect_markers(frameL)
    mdsR.detect_markers(frameR)
    mdsL.draw_markers(frameL)
    mdsR.draw_markers(frameR)
    idsL = set(mdsL.detected_markers.keys())
    idsR = set(mdsR.detected_markers.keys())
    common_ids = sorted(list(idsL & idsR))
    points_arucos = []
    if common_ids:
        height, width, _ = frameL.shape
        for mid in common_ids:
            x1, y1 = mdsL.detected_markers[mid]['Centroid']
            x2, y2 = mdsR.detected_markers[mid]['Centroid']
            result = calculate_3d_point(x1, x2, width)
            if result:
                points_arucos.append((mid, result))
    return points_arucos
