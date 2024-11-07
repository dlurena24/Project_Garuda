import cv2
import numpy as np
import matplotlib.pyplot as plt

#Detectar contornos
def detectar_contornos_amarillos(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def estimate_transform(src_points, dst_points):
    M, _ = cv2.findHomography(src_points, dst_points)
    return M


def transform_contour(contour, M):
    contour_float = contour.astype(np.float32)
    contour_reshaped = contour_float.reshape(-1, 1, 2)
    transformed_contour = cv2.perspectiveTransform(contour_reshaped, M)
    return transformed_contour.reshape(-1, 2)


def create_unified_topdown_map(images, src_points_list, dst_points):
    all_contours = []
    for idx, (image, src_points) in enumerate(zip(images, src_points_list)):
        contours = detectar_contornos_amarillos(image)
        if not contours:
            print(f"No contours found in image {idx + 1}")
            continue
        try:
            M = estimate_transform(src_points, dst_points)
            for contour in contours:
                if contour.shape[0] < 4:  # Skip contours with too few points
                    continue
                transformed_contour = transform_contour(contour, M)
                all_contours.append(transformed_contour)
        except Exception as e:
            print(f"Error processing image {idx + 1}: {str(e)}")

    # Combine all contours
    combined_contour = np.vstack(all_contours)

    # Plot the combined contour
    plt.figure(figsize=(10, 10))
    plt.plot(combined_contour[:, 0], combined_contour[:, 1], 'b-')
    plt.title("Unified Top-Down 2D Map of Terrain Contours")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


# Load images
img1 = cv2.imread('../Imgs/terreno1.jpg')
img2 = cv2.imread('../Imgs/terreno2.jpg')
img3 = cv2.imread('../Imgs/terreno3.jpg')

# Check if images are loaded correctly
if img1 is None or img2 is None or img3 is None:
    print("Error: One or more images could not be loaded.")
    exit()

# Define source points (corners of the yellow contour in each image)
# These points should be manually selected for each image
src_points1 = np.array([[100, 100], [400, 100], [400, 300], [100, 300]], dtype=np.float32)
src_points2 = np.array([[150, 150], [450, 150], [450, 350], [150, 350]], dtype=np.float32)
src_points3 = np.array([[200, 200], [500, 200], [500, 400], [200, 400]], dtype=np.float32)

# Define destination points (where these corners should be in the top-down view)
dst_points = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)

# Create the unified top-down map
create_unified_topdown_map([img1, img2, img3], [src_points1, src_points2, src_points3], dst_points)

# Visualize the original images with detected contours
for idx, img in enumerate([img1, img2, img3]):
    contours = detectar_contornos_amarillos(img)
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv2.imshow(f'Image {idx + 1} with Contours', img_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()