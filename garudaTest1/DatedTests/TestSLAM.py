import cv2
import numpy as np

# Crear el detector ORB
orb = cv2.ORB_create()

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Cargar las imágenes en secuencia (puedes usar tu propio conjunto de imágenes)
imagenes = ['terreno1.jpg', 'terreno2.jpg', 'terreno3.jpg']  # Lista de imágenes secuenciales
frames = [cv2.imread(imagen) for imagen in imagenes]

# Crear una máscara para dibujar las trayectorias
mask = np.zeros_like(frames[0])

# Definir los límites para detectar el color amarillo en espacio HSV
lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

# Iterar sobre las imágenes
for i in range(len(frames)):
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)

    # Crear una máscara para el color amarillo
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar la máscara a la imagen original
    yellow_output = cv2.bitwise_and(frames[i], frames[i], mask=yellow_mask)

    # Convertir la imagen filtrada a escala de grises
    gray = cv2.cvtColor(yellow_output, cv2.COLOR_BGR2GRAY)

    # Aplicar la detección de bordes (usando Canny)
    edges = cv2.Canny(gray, 50, 150)

    # Detección de líneas usando la Transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Dibujar las líneas detectadas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frames[i], (x1, y1), (x2, y2), (0, 255, 0), 5)  # Líneas verdes sobre el amarillo detectado

    # Mostrar la imagen con las líneas detectadas
    cv2.imshow('Líneas amarillas detectadas', frames[i])

    # Romper el loop si se presiona la tecla 'q'
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
