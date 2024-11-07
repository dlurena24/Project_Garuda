import cv2
import numpy as np


# Función para detectar contornos del color rojo y encontrar esquinas
def detectar_contornos_y_esquinas(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color rojo
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Crear una máscara para el color rojo
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    # Encontrar los contornos
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    if contours:
        # Tomar el contorno más grande (asumiendo que es nuestra forma)
        largest_contour = max(contours, key=cv2.contourArea)

        # Aproximar el contorno a un polígono
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Extraer las esquinas
        corners = [point[0] for point in approx]

    return contours, red_mask, corners


# Main code
img1 = cv2.imread('../Imgs/izq.jpg')
img2 = cv2.imread('../Imgs/der.jpg')

contours1, mask1, corners1 = detectar_contornos_y_esquinas(img1)
contours2, mask2, corners2 = detectar_contornos_y_esquinas(img2)

# Visualize the contours and corners
for idx, (img, contours, corners) in enumerate([(img1, contours1, corners1), (img2, contours2, corners2)]):
    img_result = img.copy()

    # Dibujar contornos
    cv2.drawContours(img_result, contours, -1, (0, 255, 0), 2)

    # Dibujar y numerar las esquinas
    for i, corner in enumerate(corners):
        cv2.circle(img_result, tuple(corner), 5, (0, 0, 255), -1)  # Círculo rojo
        cv2.putText(img_result, str(i), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    imgResize = cv2.resize(img_result, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow(f'Image {idx + 1} with Contours and Corners', imgResize)

cv2.waitKey(0)
cv2.destroyAllWindows()