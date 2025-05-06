import cv2
import numpy as np
import time


def list_available_cameras():
    """
    Lists all available camera devices by attempting to open each index
    from 0 to 9 (typical range for most systems).
    """
    available_cameras = []

    print("Searching for connected cameras...")
    for i in range(10):  # Check indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera index {i}: Available")
            cap.release()
        else:
            print(f"Camera index {i}: Not available")

    return available_cameras


def test_camera(camera_index, display_time=30):
    """
    Tests a specific camera by displaying its feed for a few seconds.

    Args:
        camera_index: Index of the camera to test
        display_time: Time in seconds to display the camera feed
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nTesting Camera {camera_index}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Displaying feed for {display_time} seconds. Press 'q' to exit early.")

    start_time = time.time()

    while (time.time() - start_time) < display_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Add text showing camera index
        text = f"Camera Index: {camera_index}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(f'Camera {camera_index} Test', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

def testBothCameras():
    """
    Opens and displays feeds from two cameras (indices 1 and 2) side by side.
    Both camera images are mirrored horizontally.
    Press 'q' to exit.
    """
    # Open both cameras
    capR = cv2.VideoCapture(1)
    capL = cv2.VideoCapture(2)

    # Check if cameras opened successfully
    if not capR.isOpened() or not capL.isOpened():
        print("Error: Could not open one or both cameras.")
        # Release any camera that did open
        if capR.isOpened():
            capR.release()
        if capL.isOpened():
            capL.release()
        return

    print("Displaying both camera feeds (mirrored). Press 'q' to exit.")

    while True:
        # Capture frames from both cameras
        retR, frameR = capR.read()
        retL, frameL = capL.read()

        # Check if frames were successfully captured
        if not retR or not retL:
            print("Error: Failed to capture image from one or both cameras")
            break

        # Mirror the images horizontally (flip around y-axis)
        frameR = cv2.flip(frameR, 1)
        frameL = cv2.flip(frameL, 1)

        # Add labels to identify cameras
        cv2.putText(frameR, "Camera 1 (Right) - Mirrored", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frameL, "Camera 2 (Left) - Mirrored", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Option 1: Display in separate windows
        #cv2.imshow('Camera Right', frameR)
        #cv2.imshow('Camera Left', frameL)

        # Option 2: Display side by side in one window
        # Resize if needed to make them the same height
        height_R = frameR.shape[0]
        height_L = frameL.shape[0]

        # Use the smaller height for both
        target_height = min(height_R, height_L)

        # Resize maintaining aspect ratio
        aspect_R = frameR.shape[1] / frameR.shape[0]
        aspect_L = frameL.shape[1] / frameL.shape[0]

        width_R = int(target_height * aspect_R)
        width_L = int(target_height * aspect_L)

        resized_R = cv2.resize(frameR, (width_R, target_height))
        resized_L = cv2.resize(frameL, (width_L, target_height))

        # Concatenate horizontally
        combined_frame = np.hstack((resized_L, resized_R))
        cv2.imshow('Both Cameras (Mirrored)', combined_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the captures and close windows
    capR.release()
    capL.release()
    cv2.destroyAllWindows()

def garudaTest1():
    """
    Lists all available webcams and allows the user to test them.
    """
    print("Project Garuda - Webcam Detection Utility")
    print("----------------------------------------")

    #available_cameras = list_available_cameras()

    #if not available_cameras:
        #print("No cameras detected. Please check your connections and try again.")
        #return

    #print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")

    # Test each camera
    #for cam_index in available_cameras:
    #    test_camera(cam_index)

    # Test each camera
    #for cam_index in [1,2]:
    #    test_camera(cam_index)

    print("\nTesting both cameras side by side, please stand by...")
    testBothCameras()

    print("\nWebcam testing complete.")


if __name__ == "__main__":
    garudaTest1()
