import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import math

# --- NEW HELPER FUNCTION ---
def calculate_bearing(src_idx, dst_idx, rvecs, tvecs):
    """Calculates angle of dst relative to src's local orientation"""
    t_src = tvecs[src_idx][0]
    t_dst = tvecs[dst_idx][0]
    r_src = rvecs[src_idx]
    
    # Vector from Source to Target in Camera Space
    vec_cam = t_dst - t_src
    
    # Get Rotation Matrix of Source and project vector to local space
    R, _ = cv2.Rodrigues(r_src)
    vec_local = np.dot(R.T, vec_cam)
    
    # Calculate Angle in Source's Planar (XY) System
    angle_rad = math.atan2(vec_local[1], vec_local[0])
    return np.degrees(angle_rad)

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') #from Jupyter notebook
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef'] #distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 84 #----------------------------------------------------------------------------SIZE OF THE MARKER IN mm (HAVE TO MEASURE IF PRINTED)
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
window_width = 960 #keep 1920:1080 ratio
window_height = 540 #keep 1920:1080 ratio
print("Creating windows...")
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.moveWindow("Frame", 540, 0)
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
            cv2.drawFrameAxes(frame, CM, dist_coef, rvecs[i], tvecs[i], 50)

        # ---------------- NEW SECTION: SHORTEST DISTANCE & RELATIVE YAW ----------------
        if len(ids) >= 2:
            min_dist = float('inf')
            closest_pair = None # Will store indices (i, j)

            # Find closest pair
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    dist = np.linalg.norm(tvecs[i][0] - tvecs[j][0])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (i, j)
            
            if closest_pair:
                idx1, idx2 = closest_pair
                id1_num = ids[idx1][0]
                id2_num = ids[idx2][0]

                # Determine which marker is the "Source" (Lesser ID)
                if id1_num < id2_num:
                    src_idx, dst_idx = idx1, idx2
                    src_id = id1_num
                    dst_id = id2_num
                else:
                    src_idx, dst_idx = idx2, idx1
                    src_id = id2_num
                    dst_id = id1_num

                # Calculate Bearing (Yaw of dst relative to src)
                bearing = calculate_bearing(src_idx, dst_idx, rvecs, tvecs)

                # Visuals: Draw line and text
                c1 = np.mean(corners[src_idx][0], axis=0).astype(int)
                c2 = np.mean(corners[dst_idx][0], axis=0).astype(int)
                cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(frame, f"Dist: {min_dist:.0f}mm", tuple(midpoint), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Print Info Periodically
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"\n--- PAIR ANALYSIS ---")
                    print(f"Source: ID {src_id} -> Target: ID {dst_id}")
                    print(f"Distance: {min_dist:.2f} mm")
                    print(f"Bearing: {bearing:.2f} deg (Target is at this angle relative to Source ID {src_id})")
                    last_print_time = current_time
        # -------------------------------------------------------------------------------
    
    
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