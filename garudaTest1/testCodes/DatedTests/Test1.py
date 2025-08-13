import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('../Imgs/terreno.jpg')
resized_image = cv2.resize(image, (500, 500))


# Convertir la imagen de BGR a HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir los rangos para el color amarillo en HSV
# Estos valores pueden ajustarse según la tonalidad del amarillo en tu imagen
lower_yellow = np.array([20, 100, 100])  # Rango inferior
upper_yellow = np.array([30, 255, 255])  # Rango superior

# Crear una máscara para los píxeles dentro del rango del color amarillo
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# resized_mask = cv2.resize(mask, (100, 100))

# Aplicar la máscara a la imagen original para extraer solo las áreas amarillas
result = cv2.bitwise_and(image, image, mask=mask)
resized_result = cv2.resize(result, (500, 500))

# Mostrar la imagen original, la máscara y el resultado
# cv2.imshow('Imagen original', resized_image)
# cv2.imshow('Mascara', resized_mask)
# cv2.imshow('Resultado (color amarillo extraído)', resized_result)

# Encontrar los contornos de las áreas amarillas
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original (opcional para visualización)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos dibujados
cv2.imshow('Contornos detectados', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Obtener coordenadas de los contornos (opcional para exportación)
# for contour in contours:
#     for point in contour:
#         print(point)  # Estas son las coordenadas de los puntos en el contorno