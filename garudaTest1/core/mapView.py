"""
mapView.py

Functions for generating a Bird's Eye View representation of an environment
using detected ArUco markers, detected objects, and restricted zones.

These utilities are mainly used by the main detection pipeline to visualize
the layout of the scene from a top-down perspective.

Requirements
------------
- OpenCV
- NumPy
"""
import cv2
import numpy as np

def draw_zonas_prohibidas(canvas, zonas, transform):
    """
        Draw restricted zones on the Bird's Eye View canvas.

        Parameters
        ----------
        canvas : numpy.ndarray
            The canvas image where the zones will be drawn.
        zonas : list of list of tuple
            List of zones, where each zone is a list of (x, z) coordinate tuples
            in world coordinates.
        transform : callable
            A function that maps world coordinates `(x, z)` to canvas pixel coordinates `(cx, cy)`.

        Notes
        -----
        - Zones with fewer than 3 points are ignored.
        - Each zone is filled with a semi-transparent red overlay.
        """
    overlay = canvas.copy()
    for zona in zonas:
        if not zona or len(zona) < 3:
            continue
        pts_canvas = [transform(p[0], p[1]) for p in zona]
        pts_array = np.array(pts_canvas, np.int32)
        if pts_array.shape[0] >= 3:
            cv2.fillPoly(overlay, [pts_array], (0, 0, 255))
            cv2.polylines(canvas, [pts_array], isClosed=True, color=(0, 0, 150), thickness=2)
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

def create_birds_eye_view(points_arucos, points_objects, zonas_prohibidas=None, canvas_size=(400, 800), margin=50):
    """
    Create a Bird's Eye View of the environment.

    Parameters
    ----------
    points_arucos : list of tuple
        List of detected ArUco markers as `(marker_id, (x, z))`.
    points_objects : list of tuple
        List of detected objects as `(label, (x, z))`.
    zonas_prohibidas : list of tuple, optional
        List of restricted zones, each zone is a list of `(x, z)` coordinates.
    canvas_size : tuple of int, optional
        Height and width of the output canvas in pixels (default is (400, 800)).
    margin : int, optional
        Margin in pixels between the drawing and the canvas border (default is 50).

    Returns
    -------
    numpy.ndarray
        The Bird's Eye View canvas with markers, objects, and restricted zones drawn.

    Notes
    -----
    - All coordinates are assumed to be in world space `(x, z)`.
    - The view is automatically scaled to fit all points within the specified canvas size.
    - Restricted zones are drawn first, followed by markers and objects.
    """
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    all_points = []
    if points_arucos:
        all_points += [p[1] for p in points_arucos]
    if points_objects:
        all_points += [p[1] for p in points_objects]
    if zonas_prohibidas:
        for zona in zonas_prohibidas:
            all_points += zona
    if not all_points:
        cv2.putText(canvas, "No hay puntos para mostrar", (50, canvas_size[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return canvas

    xs = [p[0] for p in all_points]
    zs = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    available_w = canvas_size[1] - 2 * margin
    available_h = canvas_size[0] - 2 * margin
    x_scale = available_w / (x_max - x_min) if x_max != x_min else 1
    z_scale = available_h / (z_max - z_min) if z_max != z_min else 1
    scale = min(x_scale, z_scale)

    def transform(x, z):
        cx = int(margin + (x - x_min) * scale)
        cy = int(canvas_size[0] - (margin + (z - z_min) * scale))
        return (cx, cy)

    if zonas_prohibidas:
        draw_zonas_prohibidas(canvas, zonas_prohibidas, transform)

    if len(points_arucos) > 1:
        pts_canvas = [transform(p[1][0], p[1][1]) for p in points_arucos]
        cv2.polylines(canvas, [np.array(pts_canvas, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        for (marker_id, _), pt in zip(points_arucos, pts_canvas):
            cv2.circle(canvas, pt, 5, (255, 0, 0), -1)
            cv2.putText(canvas, str(marker_id), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for (label, pos) in points_objects:
        pt = transform(pos[0], pos[1])
        cv2.circle(canvas, pt, 6, (0, 180, 0), -1)
        cv2.putText(canvas, label, (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

    return canvas
