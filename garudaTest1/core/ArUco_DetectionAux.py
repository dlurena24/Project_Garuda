"""
ArUco marker detection, calibration, and visualization utilities.

This module provides a `MarkerDetectionSystem` class for:
- Detecting ArUco markers in image frames.
- Calibrating a camera using chessboard images.
- Drawing detected markers, their IDs, and connections.
- Estimating distances between markers.
- Visualizing 3D poses of detected markers.

Requirements
------------
- OpenCV (cv2, cv2.aruco)
- NumPy

Notes
-----
- The module uses both the old and new OpenCV ArUco APIs for compatibility.
- Camera calibration improves the accuracy of pose estimation.
- The marker detection system stores both raw detections and sorted centroid positions.
"""
import cv2
from cv2 import aruco
import numpy as np


class MarkerDetectionSystem:
    def __init__(self, marker_size_cm=18.7):
        self.marker_size_cm = marker_size_cm
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.detected_markers = {}
        self.centroids_sorted = []  # New variable - list of [x, y] arrays
        self.camera_matrix = None
        self.distortion_coefficients = None

    def calibrate_camera(self, calibration_images, chessboard_size=(7, 6)):
        """
        Calibrate the camera.

        Parameters
        ----------
        calibration_images : list of numpy.ndarray
            List of images containing a visible chessboard pattern.
        chessboard_size : tuple of int, optional
            Number of inner corners per chessboard row and column (rows, cols),
            by default (7, 6).

        Notes
        -----
        - The calibration computes the camera matrix and distortion coefficients.
        - Uses `cv2.findChessboardCorners` and `cv2.calibrateCamera`.
        """

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners_subpix)

        ret, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                          gray.shape[::-1], None, None)

    def detect_markers(self, frame):
        """
        Detect ArUco markers in a given frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The input image frame to analyze.

        Returns
        -------
        dict
            A dictionary mapping marker IDs to their information:
                - 'ID': Marker ID.
                - 'Centroid': (x, y) coordinates of the marker center in pixels.
                - 'Corners': 4x2 array of marker corner coordinates.

        Notes
        -----
        - Clears previous detections before processing.
        - Supports both the new (`cv2.aruco.ArucoDetector`) and old
          (`cv2.aruco.detectMarkers`) APIs for compatibility.
        - Populates `self.detected_markers` and `self.centroids_sorted`.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Clear previous detections
        self.detected_markers = {}
        self.centroids_sorted = []  # Clear previous centroids

        try:
            # New API (OpenCV 4.7+)
            detector = cv2.aruco.ArucoDetector(self.dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Old API (OpenCV < 4.7)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

        if ids is not None:
            print(f"Detected {len(ids)} markers with IDs: {ids.flatten()}")
            for i in range(len(ids)):
                marker_id = ids[i][0]
                corner_points = corners[i][0]

                # Calculate centroid using corner points
                cX = int(np.mean(corner_points[:, 0]))
                cY = int(np.mean(corner_points[:, 1]))

                marker_info = {
                    'ID': marker_id,
                    'Centroid': (cX, cY),
                    'Corners': corner_points
                }
                self.detected_markers[marker_id] = marker_info

            # Create centroids_sorted - similar structure to corners_sorted
            # Sort by marker ID (from lower to higher)
            sorted_marker_ids = sorted(self.detected_markers.keys())
            self.centroids_sorted = []

            for marker_id in sorted_marker_ids:
                centroid = self.detected_markers[marker_id]['Centroid']
                # Convert to numpy array format like corners_sorted
                centroid_array = np.array([centroid[0], centroid[1]])
                self.centroids_sorted.append(centroid_array)

            print(f"Centroids sorted by ID: {[f'ID{sorted_marker_ids[i]}:({self.centroids_sorted[i][0]},{self.centroids_sorted[i][1]})' for i in range(len(self.centroids_sorted))]}")
        else:
            print("No markers detected")
        print(self.centroids_sorted)
        return self.detected_markers

    def calculate_distance(self, marker_size_pixels, cap_width):
        """
        Calculate the distance from the camera to a marker.

        Parameters
        ----------
        marker_size_pixels : float
            The size of the detected marker in pixels.
        cap_width : float
            The width of the camera's capture frame in pixels.

        Returns
        -------
        float
            The estimated distance to the marker in centimeters.
        """

        return self.marker_size_cm * cap_width / marker_size_pixels

    def draw_markers(self, frame):
        """
        Draw detected markers on the frame.

        Parameters
        ----------
        frame : numpy.ndarray
            The image frame where detected markers will be drawn.

        Notes
        -----
        - Draws the marker center as a green dot.
        - Draws the marker's polygon outline in blue.
        - Displays the marker ID above the centroid.
        """

        if self.detected_markers:
            for marker_id, marker_info in self.detected_markers.items():
                cX, cY = marker_info['Centroid']

                # Draw marker center
                cv2.circle(frame, (cX, cY), 8, (0, 255, 0), -1)

                # Draw marker corners
                corners = marker_info['Corners'].astype(int)
                cv2.polylines(frame, [corners], True, (255, 0, 0), 2)

                # Draw marker ID
                cv2.putText(frame, f"ID: {marker_id}", (cX - 20, cY - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_connections(self, frame):
        """
        Draw connections and distances between all detected markers.

        Parameters
        ----------
        frame : numpy.ndarray
            The image frame where connections will be drawn.

        Notes
        -----
        - Draws lines between every pair of detected markers.
        - Displays the pixel distance between markers at the midpoint of the line.
        """

        marker_ids = list(self.detected_markers.keys())
        if len(marker_ids) >= 2:
            for i in range(len(marker_ids) - 1):
                for j in range(i + 1, len(marker_ids)):
                    id1 = marker_ids[i]
                    id2 = marker_ids[j]

                    # Draw line between markers
                    pt1 = self.detected_markers[id1]['Centroid']
                    pt2 = self.detected_markers[id2]['Centroid']
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                    # Calculate pixel distance
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))

                    # Draw distance text
                    mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(frame, f"{dist:.1f}px", mid_point,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def visualize_3d_coordinates(self, frame):
        """
        Visualize 3D pose axes of detected markers.

        Parameters
        ----------
        frame : numpy.ndarray
            The image frame where pose axes will be overlaid.

        Returns
        -------
        numpy.ndarray
            The frame with 3D pose visualization.

        Notes
        -----
        - Requires `self.camera_matrix` and `self.distortion_coefficients` from calibration.
        - Uses `cv2.aruco.estimatePoseSingleMarkers` and `cv2.aruco.drawAxis`.
        """
        if self.camera_matrix is not None and self.distortion_coefficients is not None:
            for marker_id, marker_info in self.detected_markers.items():
                corners_array = np.array([marker_info['Corners']], dtype=np.float32)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners_array, self.marker_size_cm,
                    self.camera_matrix, self.distortion_coefficients
                )
                frame = cv2.aruco.drawAxis(frame, self.camera_matrix,
                                           self.distortion_coefficients, rvec, tvec, 0.1)
        return frame

# TESTING ONLY::::::::::::::::::::::::::::::::::
def main():
    cap = cv2.VideoCapture(1)
    mds = MarkerDetectionSystem()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_markers = mds.detect_markers(frame)
        mds.draw_markers(frame)
        mds.draw_connections(frame)
        frame = mds.visualize_3d_coordinates(frame)

        cv2.imshow('ArUco Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
