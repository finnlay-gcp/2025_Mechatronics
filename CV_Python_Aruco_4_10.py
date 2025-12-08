import cv2
import cv2.aruco as aruco
import numpy as np
import time 

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') #from Jupyter notebook
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef'] #distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 98 # SIZE OF THE MARKER IN mm (HAVE TO MEASURE IF PRINTED)
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
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.moveWindow("Frame", 780, 0)
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
    if ids is not None and specified_ids is not None:
        filtered_indices = [i for i, marker_id in enumerate(ids.flatten()) if marker_id in specified_ids]
        if not filtered_indices:
            ids = None
        else:
            corners = [corners[i] for i in filtered_indices]
            ids = ids[filtered_indices]
        
    # If markers are detected (after filtering)
    if ids is not None:
        # Draw detected markers
        gray = aruco.drawDetectedMarkers(gray, corners, ids)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose of each marker
        rvecs, tvecs,_objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)
        
        # Loop through ALL detected markers to draw axes
        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            
            # Visualization: Draw Axis
            axis_length = 50
            # Project axis point to ensure it is in frame before drawing
            axis_point, _ = cv2.projectPoints(np.float32([[0, 0, axis_length]]), rvec, tvec, CM, dist_coef)
            axis_x = axis_point[0, 0, 0]
            axis_y = axis_point[0, 0, 1]
            
            if 0 <= axis_x < frame.shape[1] and 0 <= axis_y < frame.shape[0]:
                cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, axis_length)

        # --- DISTANCE CALCULATION ---
        # We need at least 2 markers to calculate a distance
        if len(ids) >= 2:
            min_dist = float('inf')
            closest_pair = None # Will store indices (i, j)

            # Compare every marker with every other marker
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    # Extract 3D position vectors (x, y, z)
                    pos1 = tvecs[i][0]
                    pos2 = tvecs[j][0]
                    
                    # Calculate Euclidean distance in 3D space
                    dist = np.linalg.norm(pos1 - pos2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (i, j)
            
            # If we found a pair (which we always should if len >= 2)
            if closest_pair:
                idx1, idx2 = closest_pair
                
                # Get the center points of the markers on the 2D image for drawing the line
                # corners shape is (N, 1, 4, 2)
                c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                c2 = np.mean(corners[idx2][0], axis=0).astype(int)
                
                # Draw a yellow line between the closest pair
                cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                
                # Calculate midpoint on screen to place the text
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                
                # Display distance text
                dist_text = f"{min_dist:.1f} mm"
                cv2.putText(frame, dist_text, tuple(midpoint), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Print to console periodically
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    id1_num = ids[idx1][0]
                    id2_num = ids[idx2][0]
                    print(f"Closest Pair: ID {id1_num} & ID {id2_num} | Distance: {min_dist:.2f} mm")
                    last_print_time = current_time

    # Add the frame rate to the images
    fps_label = f"CAMERA FPS: {fps:.2f}"
    proc_label = f"PROCESSING FPS: {1/processing_period:.2f}"
    
    cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, proc_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("Frame", frame)
    
    # Break the loop on 'q' key press (use longer wait time to allow window responsiveness)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    
    # FPS Calculation and Sleep
    processing_time = time.time() - loop_start_timestamp
    if processing_time > 0:
        fps = 1.0 / processing_time
    
    if processing_time < processing_period:
        time.sleep(processing_period - processing_time)

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()