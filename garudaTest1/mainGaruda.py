"""
mainGaruda.py

Main real-time detection and monitoring system using stereo vision,
object detection (YOLO), and ArUco markers for mapping and zone supervision.

Workflow
--------
1. Connects to two cameras (stereo setup).
2. Detects ArUco markers to map the environment.
3. Creates a Bird's Eye View of the scene.
4. Allows the definition of restricted zones.
5. Detects objects and logs intrusions in restricted areas.

Requirements
------------
- OpenCV
- Internal modules in the `core` package
- YOLO model configured in `config.YOLO_MODEL_PATH`
"""
import cv2
from config import YOLO_MODEL_PATH
from core.detection import ObjectDetector, scan_aruco
from core.mapView import create_birds_eye_view
from core.loggingSystem import EventLogger
from core.geometry import punto_en_zona

try:
    from core.ArUco_DetectionAux import MarkerDetectionSystem
except Exception as e:
    print("Error importando MarkerDetectionSystem:", e)
    exit()

def main(cam_left=0, cam_right=1):
    """
    Main entry point for the detection system.

    Parameters
    ----------
    cam_left : int, optional
        Index of the left camera (default is 0).
    cam_right : int, optional
        Index of the right camera (default is 1).

    Phases
    ------
    1 : Initial terrain view (no active detection).
    2 : Terrain scanning with ArUco markers to create the map.
    3 : Restricted zone definition.
    4 : Object detection and intrusion monitoring.

    Keyboard Controls
    -----------------
    '1' : Switch to phase 1.
    '2' : Switch to phase 2.
    '3' : Switch to phase 3 (requires map from phase 2).
    '4' : Switch to phase 4 (requires map from phase 2).
    'y' : Confirm action in phases 2 and 3.
    'n' : Move from phase 3 to phase 4.
    'q' : Quit program.

    Notes
    -----
    - Requires YOLO model path to be defined in the `config` module.
    - All restricted zones are defined using 3 or more detected ArUco markers.
    """
    print(".::::::::Welcome to project Garuda::::::::.")
    capL = cv2.VideoCapture(cam_left)
    capR = cv2.VideoCapture(cam_right)
    if not capL.isOpened() or not capR.isOpened():
        print("No se pudo abrir alguna cámara.")
        return

    mdsL = MarkerDetectionSystem()
    mdsR = MarkerDetectionSystem()
    detector = ObjectDetector(YOLO_MODEL_PATH)
    logger = EventLogger()

    fase = 1
    points_arucos_static = []
    zonas_prohibidas = []
    zona_actual = []
    intrusion_activa = {}

    while True:
        print("Loading camera feeds...")
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        bev = None

        if fase == 1:
            bev = create_birds_eye_view(points_arucos_static, [], zonas_prohibidas)
            print("Phase 1 loaded!")

        elif fase == 2:
            points_arucos_current = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            bev = create_birds_eye_view(points_arucos_current, [], zonas_prohibidas)
            cv2.putText(frameL, "Fase 2: 'y'=aceptar mapa", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Phase 2 loaded!")

        elif fase == 3:
            zona_actual = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            zona_temp = [p[1] for p in zona_actual] if zona_actual and len(zona_actual) >= 3 else []
            bev = create_birds_eye_view(points_arucos_static, [], zonas_prohibidas + ([zona_temp] if zona_temp else []))
            cv2.putText(frameL, "Fase 3: 'y'=guardar zona", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Phase 3 loaded!")

        elif fase == 4:
            objsL = detector.detect_and_draw(frameL)
            objsR = detector.detect_and_draw(frameR)
            points_objects = []
            height, width, _ = frameL.shape

            intrusiones_a_guardar = []
            intrusiones_actuales = set()

            print("Phase 4 loaded!")

            for labelL, (cxL, cyL) in objsL:
                for labelR, (cxR, cyR) in objsR:
                    if labelL == labelR:
                        from core.geometry import calculate_3d_point
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
            combined_top = cv2.hconcat([frameL, frameR])
            bev_resized = cv2.resize(bev, (combined_top.shape[1], bev.shape[0]))
            final_view = cv2.vconcat([combined_top, bev_resized])
            final_view = logger.draw_on_frame(final_view)

            for label, idx in intrusiones_a_guardar:
                logger.log_event(label, idx, final_view)

            cv2.imshow("Garuda", final_view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        combined_top = cv2.hconcat([frameL, frameR])
        bev_resized = cv2.resize(bev, (combined_top.shape[1], bev.shape[0]))
        final_view = cv2.vconcat([combined_top, bev_resized])
        final_view = logger.draw_on_frame(final_view)
        cv2.imshow("Garuda", final_view)

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
                print("First you have to scan the terrain on phase 2.")
        elif key == ord('4'):
            if points_arucos_static:
                fase = 4
            else:
                print("First you have to scan the terrain on phase 2.")
        elif fase == 2 and key == ord('y'):
            points_arucos_static = scan_aruco(frameL.copy(), frameR.copy(), mdsL, mdsR)
            fase = 3
        elif fase == 3 and key == ord('y'):
            if zona_actual and len(zona_actual) >= 3:
                zonas_prohibidas.append([p[1] for p in zona_actual])
                print(f"Forbidden zone added. Total: {len(zonas_prohibidas)}")
        #elif fase == 3 and key == ord('n'):
        #    fase = 4

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
