import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import math

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') #from Jupyter notebook
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef'] #distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 40 # SIZE OF THE MARKER IN mm (HAVE TO MEASURE IF PRINTED)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Input specific ArUco IDs to detect (leave empty to detect all)
aruco_ids_input = input("Enter specific ArUco IDs to detect (comma-separated, e.g., 0,1,2). Leave empty to detect all: ").strip()
if aruco_ids_input:
    try:
        specified_ids = set(int(id.strip()) for id in aruco_ids_input.split(','))
        print(f"Detecting markers with IDs: {sorted(specified_ids)}")
    except ValueError:
        print("Invalid input. Detecting all markers.")
        specified_ids = None
else:
    specified_ids = None

# Define a processing rate
processing_period = 0.1 #--------------------------------------------------------------------------------------------------select processing rate here

# Create OpenCV named windows
window_width = 768 #keep 1920:1080 ratio
window_height = 432 #keep 1920:1080 ratio
print("Creating windows...")
window_names = ["Frame"] #, "Gray", "Canny"
for name in window_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)

# Position the windows next to each other
#cv2.moveWindow("Gray", 0, 0)
cv2.moveWindow("Frame", 780, 0)
#cv2.moveWindow("Canny", 0, 510)
print("Windows created. Starting camera feed...\n")

# Start capturing video
cap = cv2.VideoCapture(1) #----------------------------------------------------------------------------------------------------------select camera here

# Set the width and heigth of the camera to 1920x1080
cap.set(3,1920) #------------------------------------------------------------------------------------------------------------set camera resolution here
cap.set(4,1080)

# Set the starting time
start_time = time.time()
fps = 0

# Initialise print timer
last_print_time = time.time()

while True:
    loop_start_timestamp = time.time()
    
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
        # Filter markers by specified IDs if applicable
        if specified_ids is not None:
            filtered_indices = [i for i, marker_id in enumerate(ids.flatten()) if marker_id in specified_ids]
            if not filtered_indices:
                # No markers match the specified IDs, skip this frame
                ids = None
            else:
                # Keep only the filtered markers
                corners = [corners[i] for i in filtered_indices]
                ids = ids[filtered_indices]
        
    # If markers are detected (after filtering)
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
        marker_id = ids.flatten()[topmost_ind]
        rvec = rvecs[topmost_ind]
        tvec = tvecs[topmost_ind]
        
        # Export pose information for topmost marker
        tvec_flat = tvec.flatten()
        rvec_flat = rvec.flatten()
        
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            
            # Convert Rodrigues vector (rvec) to Rotation Matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler Angles (Pitch, Yaw, Roll) from Matrix
            sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            # Convert radians to degrees
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)
            
            print("\n--- TOPMOST MARKER DETECTED ---")
            print(f"MARKER ID: {marker_id}")
            print(f"    Index: {topmost_ind}")
            #print(f"    Translation Vector (tvec): {tvec_flat}")
            #print(f"        X: {tvec_flat[0]:.2f} mm, Y: {tvec_flat[1]:.2f} mm, Z: {tvec_flat[2]:.2f} mm")
            #print(f"    Rotation Vector (rvec): {rvec_flat}")
            print( f"       ROTATION (deg): Roll={roll:.1f}") #Pitch={pitch:.1f}, Yaw={yaw:.1f}, 
            
            last_print_time = current_time
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
    fps_label = f"CAMERA FPS: {fps:.2f}"
    proc_label = f"PROCESSING FPS: {1/processing_period:.2f}"
    
    images_to_label = [gray, frame, canny]
    
    for img in images_to_label:
        cv2.putText(img, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, proc_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the resulting frame
    show_list = [("Frame", frame)] #, ("Gray", gray), ("Canny", canny)
    for name, img in show_list:
        cv2.imshow(name, img)
    
    # Break the loop on 'q' key press (use longer wait time to allow window responsiveness)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    
    # Calculate how long the processing took
    processing_time = time.time() - loop_start_timestamp
    
    # Calculate FPS based on actual processing speed
    if processing_time > 0:
        fps = 1.0 / processing_time
    
    # Sleep only if we have time left in our 0.1s window
    if processing_time < processing_period:
        time.sleep(processing_period - processing_time)

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()