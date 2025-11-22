# This script is an entry point to the Aruco marker detection and pose estimation.
# It uses the camera calibration values to estimate the pose of the markers.
# The script will display the original image and the image with the detected markers and their pose.
# It is using the OpenCV library 4.10+ which has the latest Aruco functions

# This is the vision library OpenCV
import cv2
# This is the Aruco library from OpenCV
import cv2.aruco as aruco
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities. We will use this to ensure a steady processing rate
import time 

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') #from Jupyter notebook
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef'] #distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 40 # SIZE OF THE MARKER IN mm (HAVE TO MEASURE IF PRINTED)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define a processing rate
processing_period = 0.25

# Create three OpenCV named windows
window_width = 768 #keep 1920:1080 ratio
window_height = 432
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gray", window_width, window_height)
cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canny", window_width, window_height)

# Position the windows next to each other
cv2.moveWindow("Gray", 0, 0)
cv2.moveWindow("Frame", 780, 0)
cv2.moveWindow("Canny", 0, 510)

# Start capturing video
cap = cv2.VideoCapture(1) #1 is the external camera

# Set the width and heigth of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)

# Set the starting time
start_time = time.time()
fps = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Example of additional image processing: Gaussian Blur
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    # Example of additional image processing: Canny Edge Detection
    canny = cv2.Canny(gauss, 75, 100)#change the thresholds if needed

    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw detected markers
        gray = aruco.drawDetectedMarkers(gray, corners, ids)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs,_objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)

        # Find the topmost marker (smallest average Y coordinate)
        topmost_ind = 0
        topmost_y = float('inf')
        
        for ind, marker_id in enumerate(ids.flatten()):
            # Calculate average Y position of marker corners
            marker_corners = corners[ind][0]
            avg_y = marker_corners[:, 1].mean()
            
            if avg_y < topmost_y:
                topmost_y = avg_y
                topmost_ind = ind
        
        # Process only the topmost marker
        print("\n--- TOPMOST MARKER DETECTED ---\n")
        marker_id = ids.flatten()[topmost_ind]
        rvec = rvecs[topmost_ind]
        tvec = tvecs[topmost_ind]
        
        # Export pose information for topmost marker
        tvec_flat = tvec.flatten()
        rvec_flat = rvec.flatten()
        
        print(f"MARKER ID: {marker_id}\n")
        print(f"    Index: {topmost_ind}\n")
        print(f"    Translation Vector (tvec): {tvec_flat}")
        print(f"        X: {tvec_flat[0]:.2f} mm, Y: {tvec_flat[1]:.2f} mm, Z: {tvec_flat[2]:.2f} mm\n")
        print(f"    Rotation Vector (rvec): {rvec_flat}")
        print(f"        Rx: {rvec_flat[0]:.4f}, Ry: {rvec_flat[1]:.4f}, Rz: {rvec_flat[2]:.4f}\n")
        
        # Draw axis only for the topmost marker (check if endpoints are in frame)
        axis_length = 50  # Use smaller axis length to stay in frame
        
        # Project axis endpoint to check if it's within bounds
        axis_point, _ = cv2.projectPoints(np.float32([[0, 0, axis_length]]), rvec, tvec, CM, dist_coef)
        axis_x = axis_point[0, 0, 0]
        axis_y = axis_point[0, 0, 1]
        
        # Only draw if projected endpoint is within frame boundaries
        if 0 <= axis_x < frame.shape[1] and 0 <= axis_y < frame.shape[0]:
            gray = cv2.drawFrameAxes(gray, CM, dist_coef, rvec, tvec, axis_length)
            frame = cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, axis_length)
        
        # Add text label to topmost marker on frame
        cv2.putText(frame, f"TOPMOST: ID {marker_id}", (int(topmost_y), 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    # Add the frame rate to the images
    cv2.putText(gray, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(gray, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canny, f"CAMERA FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canny, f"PROCESSING FPS: {1/processing_period:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Gray', gray)
    cv2.imshow('Frame', frame)
    cv2.imshow('Canny', canny)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ensure a steady processing rate
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()