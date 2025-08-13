import sys
import time
import os
import cv2
import numpy as np
import datetime
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

# ====================== FUNCIONES AUXILIARES ======================

def calculate_3d_point(x1, x2, width):
    fPixel = (width * 0.5) / np.tan(FOV * 0.5 * np.pi / 180)
    disparity = abs(x2 - x1)
    if disparity == 0:
        return None
    Z = (B * fPixel) / disparity
    X = ((x1 + x2) / 2.0) / 30.0
    return float(X), float(Z)

def punto_en_zona(punto, zona):
    """Verifica si un punto (x,z) está dentro del polígono zona."""
    if not zona or len(zona) < 3:
        return False
    pts = np.array(zona, np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, punto, False)
    return result >= 0

def guardar_log_txt(mensaje):
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "eventos.txt"), "a", encoding="utf-8") as f:
        f.write(mensaje + "\n")

def guardar_captura(frame, objeto, zona_idx):
    os.makedirs("capturas", exist_ok=True)
    hora_archivo = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_archivo = os.path.join("capturas", f"captura_{objeto}_zona{zona_idx + 1}_{hora_archivo}.png")
    cv2.imwrite(nombre_archivo, frame)

def registrar_evento(log_eventos, objeto, zona_idx, frame_final=None):
    hora = datetime.datetime.now().strftime("%H:%M:%S")
    mensaje = f"{hora} - {objeto} - Zona {zona_idx+1}"
    log_eventos.append(mensaje)
    print(mensaje)
    guardar_log_txt(mensaje)
    if frame_final is not None:
        guardar_captura(frame_final, objeto, zona_idx)
    if len(log_eventos) > 20:
        log_eventos.pop(0)

def mostrar_log_en_ventana(frame, log_eventos):
    if not log_eventos:
        return frame
    h, w = frame.shape[:2]
    x0 = w - 320
    y0 = h - 200  # Más arriba
    for i, msg in enumerate(reversed(log_eventos[-5:])):
        y = y0 + i * 20
        cv2.putText(frame, msg, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
        cv2.putText(frame, msg, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def draw_zonas_prohibidas(canvas, zonas, transform):
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

# ====================== MAIN ======================

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
    zonas_prohibidas = []
    zona_actual = []
    log_eventos = []
    intrusion_activa = {}

    print("Teclas: 1=solo video, 2=scan terreno, 3=zonas prohibidas, 4=detección objetos")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break
        bev = None

        if fase == 1:
            bev = create_birds_eye_view(points_arucos_static, [], zonas_prohibidas)

        elif fase == 2:
            points_arucos_current = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            bev = create_birds_eye_view(points_arucos_current, [], zonas_prohibidas)
            cv2.putText(frameL, "Fase 2: 'y'=aceptar mapa", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif fase == 3:
            zona_actual = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            zona_temp = [p[1] for p in zona_actual] if zona_actual and len(zona_actual) >= 3 else []
            bev = create_birds_eye_view(points_arucos_static, [], zonas_prohibidas + ([zona_temp] if zona_temp else []))
            cv2.putText(frameL, "Fase 3: 'y'=guardar zona, 'n'=siguiente", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif fase == 4:
            objsL = detect_objects_and_draw(frameL)
            objsR = detect_objects_and_draw(frameR)
            points_objects = []
            height, width, _ = frameL.shape

            intrusiones_a_guardar = []
            intrusiones_actuales = set()

            for labelL, (cxL, cyL) in objsL:
                for labelR, (cxR, cyR) in objsR:
                    if labelL == labelR:
                        pos3d = calculate_3d_point(cxL, cxR, width)
                        if pos3d:
                            points_objects.append((labelL, pos3d))
                            for idx, zona in enumerate(zonas_prohibidas):
                                if punto_en_zona(pos3d, zona):
                                    intrusiones_actuales.add((labelL, idx))
                                    if (labelL, idx) not in intrusion_activa:
                                        intrusiones_a_guardar.append((labelL, idx))
                                        intrusion_activa[(labelL, idx)] = True
                        break

            intrusion_activa = {k: v for k, v in intrusion_activa.items() if k in intrusiones_actuales}

            bev = create_birds_eye_view(points_arucos_static, points_objects, zonas_prohibidas)
            combined_top = np.hstack((frameL, frameR))
            bev_resized = cv2.resize(bev, (combined_top.shape[1], bev.shape[0]))
            final_view = np.vstack((combined_top, bev_resized))
            final_view = mostrar_log_en_ventana(final_view, log_eventos)

            for label, idx in intrusiones_a_guardar:
                registrar_evento(log_eventos, label, idx, final_view)

            cv2.imshow("Sistema", final_view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        combined_top = np.hstack((frameL, frameR))
        bev_resized = cv2.resize(bev, (combined_top.shape[1], bev.shape[0]))
        final_view = np.vstack((combined_top, bev_resized))
        final_view = mostrar_log_en_ventana(final_view, log_eventos)
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
                print("Primero debes escanear el terreno en fase 2.")
        elif key == ord('4'):
            if points_arucos_static:
                fase = 4
            else:
                print("Primero debes escanear el terreno en fase 2.")
        elif fase == 2 and key == ord('y'):
            points_arucos_static = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            fase = 3
        elif fase == 3 and key == ord('y'):
            if zona_actual and len(zona_actual) >= 3:
                zonas_prohibidas.append([p[1] for p in zona_actual])
                print(f"Zona prohibida agregada. Total: {len(zonas_prohibidas)}")
        elif fase == 3 and key == ord('n'):
            fase = 4

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
