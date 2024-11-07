import cv2
import numpy as np


# Función para detectar contornos del color amarillo
def detectar_contornos_amarillos(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color amarillo
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])

    # Crear una máscara para el color amarillo
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Encontrar los contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


# Función para extraer puntos clave usando ORB
def extraer_puntos_clave(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


# Función para hacer coincidir puntos clave entre dos imágenes
def emparejar_puntos(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:min(len(matches), 50)]  # Limit to 50 best matches

def triangulacion_3_camaras(p1, p2, p3, camera_matrix, cam_positions):
    # Ensure all point arrays have the same length
    min_points = min(len(p1), len(p2), len(p3))
    p1 = p1[:min_points]
    p2 = p2[:min_points]
    p3 = p3[:min_points]

    P1 = np.hstack((camera_matrix, np.array([[0], [0], [0]])))
    P2 = np.hstack((camera_matrix, np.array([[cam_positions[0]], [0], [0]])))
    P3 = np.hstack((camera_matrix, np.array([[cam_positions[1]], [cam_positions[2]], [0]])))

    points_4D_12 = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    points_4D_13 = cv2.triangulatePoints(P1, P3, p1.T, p3.T)
    points_4D_23 = cv2.triangulatePoints(P2, P3, p2.T, p3.T)

    points_4D_12 /= points_4D_12[3]
    points_4D_13 /= points_4D_13[3]
    points_4D_23 /= points_4D_23[3]

    points_3D = (points_4D_12[:3] + points_4D_13[:3] + points_4D_23[:3]) / 3.0

    return points_3D.T



# Main code::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
img1 = cv2.imread('../Imgs/Floor1.jpg')
img2 = cv2.imread('../Imgs/floor2.jpg')
img3 = cv2.imread('../Imgs/floor3.jpg')

contours1, mask1 = detectar_contornos_amarillos(img1)
contours2, mask2 = detectar_contornos_amarillos(img2)
contours3, mask3 = detectar_contornos_amarillos(img3)

kp1, des1 = extraer_puntos_clave(mask1)
kp2, des2 = extraer_puntos_clave(mask2)
kp3, des3 = extraer_puntos_clave(mask3)

matches12 = emparejar_puntos(des1, des2)
matches13 = emparejar_puntos(des1, des3)
matches23 = emparejar_puntos(des2, des3)

points1 = np.float32([kp1[m.queryIdx].pt for m in matches12])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches12])
points3 = np.float32([kp3[m.trainIdx].pt for m in matches13])

print(f"Number of matched points: {len(points1)}, {len(points2)}, {len(points3)}")

camera_matrix = np.array([[800, 0, 250], [0, 800, 250], [0, 0, 1]])
cam_positions = np.array([1.0, -0.5, np.sqrt(3) / 2])

try:
    points_3D = triangulacion_3_camaras(points1, points2, points3, camera_matrix, cam_positions)
    print("Coordenadas 3D estimadas del terreno:")
    print(points_3D)
except Exception as e:
    print(f"Error during triangulation: {e}")

# Visualize the yellow contours and keypoints
for idx, (img, contours, kp) in enumerate([(img1, contours1, kp1), (img2, contours2, kp2), (img3, contours3, kp3)]):
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    img_keypoints = cv2.drawKeypoints(img_contours, kp, None, color=(0, 0, 255), flags=0)
    cv2.imshow(f'Image {idx+1} with Contours and Keypoints', img_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
