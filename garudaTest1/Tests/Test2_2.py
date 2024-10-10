import cv2
import numpy as np


# Función para detectar contornos del color amarillo
def detectar_contornos(image):
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

    return contours, red_mask




# Main code::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
img1 = cv2.imread('../Imgs/izq.jpg')
img2 = cv2.imread('../Imgs/der.jpg')

contours1, mask1 = detectar_contornos(img1)
contours2, mask2 = detectar_contornos(img2)


#
# try:
#     points_3D = triangulacion_3_camaras(points1, points2, points3, camera_matrix, cam_positions)
#     print("Coordenadas 3D estimadas del terreno:")
#     print(points_3D)
# except Exception as e:
#     print(f"Error during triangulation: {e}")

# Visualize the yellow contours and keypoints
for idx, (img, contours) in enumerate([(img1, contours1), (img2, contours2)]):
    img_contours = img.copy()
    img = cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    imgResize = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow(f'Image {idx+1} with Contours and Keypoints', imgResize)


# cv2.imshow('x', mask2)



cv2.waitKey(0)
cv2.destroyAllWindows()
