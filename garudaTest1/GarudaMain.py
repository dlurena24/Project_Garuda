import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from ArUco_detection_test import MarkerDetectionSystem
except Exception as e:
    print("Error importando MarkerDetectionSystem:", e)
    sys.exit(1)

# Parámetros estéreo
B = 20
FOV = 80

# Modelo YOLO
model = YOLO("best.pt")

def calculate_3d_point(x1, x2, width):
    fPixel = (width * 0.5) / np.tan(FOV * 0.5 * np.pi / 180)
    disparity = abs(x2 - x1)
    if disparity == 0:
        return None
    Z = (B * fPixel) / disparity
    X = ((x1 + x2) / 2.0) / 30.0
    return float(X), float(Z)

def create_birds_eye_view(points_arucos, points_objects, canvas_size=(400, 800), margin=50):
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    all_points = []
    if points_arucos:
        all_points += [p[1] for p in points_arucos]
    if points_objects:
        all_points += [p[1] for p in points_objects]
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

def detect_objects_and_draw(frame):
    detections = []
    results = model(frame)
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), max(10, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append((label, (float(cx), float(cy))))
    return detections

def scan_aruco(frameL, frameR, mdsL, mdsR):
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

def main(cam_left=0, cam_right=1):
    capL = cv2.VideoCapture(cam_left)
    capR = cv2.VideoCapture(cam_right)
    if not capL.isOpened() or not capR.isOpened():
        print("No se pudo abrir alguna cámara.")
        return
    mdsL = MarkerDetectionSystem()
    mdsR = MarkerDetectionSystem()
    fase = 1
    points_arucos_static = []
    print("Teclas: 1=solo video, 2=scan terreno, 3=detección objetos, y=aceptar, r=re-scan, q=salir")
    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break
        bev = None
        if fase == 1:
            bev = create_birds_eye_view(points_arucos_static, [])
        elif fase == 2:
            points_arucos_current = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            bev = create_birds_eye_view(points_arucos_current, [])
            cv2.putText(frameL, "Fase 2: 'y'=aceptar, 'r'=re-scan", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif fase == 3:
            objsL = detect_objects_and_draw(frameL)
            objsR = detect_objects_and_draw(frameR)
            points_objects = []
            height, width, _ = frameL.shape
            for labelL, (cxL, cyL) in objsL:
                for labelR, (cxR, cyR) in objsR:
                    if labelL == labelR:
                        pos3d = calculate_3d_point(cxL, cxR, width)
                        if pos3d:
                            points_objects.append((labelL, pos3d))
                        break
            bev = create_birds_eye_view(points_arucos_static, points_objects)
        combined_top = np.hstack((frameL, frameR))
        bev_resized = cv2.resize(bev, (combined_top.shape[1], bev.shape[0]))
        final_view = np.vstack((combined_top, bev_resized))
        cv2.imshow("Sistema", final_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            fase = 1
        elif key == ord('2'):
            fase = 2
        elif key == ord('3'):
            if points_arucos_static:
                fase = 3
            else:
                print("Debes escanear el terreno antes (fase 2).")
        elif fase == 2 and key == ord('y'):
            points_arucos_static = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            print("Mapa aceptado. Pasando a detección de objetos.")
            fase = 3
        elif fase == 2 and key == ord('r'):
            print("Re-escaneando terreno.")
            points_arucos_static = []
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
