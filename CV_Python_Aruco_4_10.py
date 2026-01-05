import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket

# --- NETWORK CONFIGURATION (UDP) ---
RPI_IP = "169.254.1.105" # <-------------------------------------------------------------CHANGE THIS to the Raspberry Pi's IP address
RPI_PORT = 50002 # -------------------------------------------------------------------------------------The port the Pi will listen on
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.1

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') #from Jupyter notebook
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef'] #distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 98 # -------------------------------------------------------------------------------SIZE OF THE MARKER IN mm (HAVE TO MEASURE IF PRINTED)
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
processing_period = 0.05 #--------------------------------------------------------------------------------------------------select processing rate here

# Create OpenCV named windows
window_scale = 0.75 #--------------------------------------------------------------------------------------------------------scale the window size here
window_width = int(1920 * window_scale)
window_height = int(1080 * window_scale)
print("Creating windows...")
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.moveWindow("Frame", 0, 0)
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
    
    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # If markers are detected
    if ids is not None:
        # Dictionary to store the best candidate for each ID: {marker_id: (index, perimeter)}
        best_markers = {}

        for i, marker_id in enumerate(ids.flatten()):
            # 1. Filter by specified IDs (if user set any)
            if specified_ids is not None and marker_id not in specified_ids:
                continue

            # 2. Calculate perimeter (proxy for size)
            perimeter = cv2.arcLength(corners[i], True)

            # 3. Keep only the largest marker for this ID
            if marker_id not in best_markers:
                best_markers[marker_id] = (i, perimeter)
            else:
                # If this new detection is larger than the stored one, replace it
                if perimeter > best_markers[marker_id][1]:
                    best_markers[marker_id] = (i, perimeter)

        # Reconstruct the lists using only the best indices
        if best_markers:
            # Extract the original indices of the largest markers
            valid_indices = [val[0] for val in best_markers.values()]
            
            # Rebuild corners and ids arrays
            corners = tuple([corners[i] for i in valid_indices])
            ids = np.array([ids[i] for i in valid_indices])
        else:
            ids = None
    
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
                
                # --- SORT IDS FOR STABILITY ---
                # Ensure we always measure FROM the lower ID TO the higher ID
                # This prevents the angle from flipping if detection order changes
                if ids[idx1][0] > ids[idx2][0]:
                    idx1, idx2 = idx2, idx1
                
                # Get the center points of the markers on the 2D image for drawing the line
                c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                c2 = np.mean(corners[idx2][0], axis=0).astype(int)
                
                # Draw a yellow line between the closest pair (The "Target Line")
                cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                
                # --- ACCURATE ANGLE CALCULATION ---
                # 1. Get Rotation Matrix for the "Source" Marker (idx1)
                # This matrix converts local marker coordinates (X, Y, Z) to camera coordinates
                rmat, _ = cv2.Rodrigues(rvecs[idx1][0])
                
                # 2. Extract the Orientation Vector
                # By default, ArUco 'Up' is the Y-axis (Green). 
                # Column 0 = X (Right), Column 1 = Y (Up/Top), Column 2 = Z (Forward/Normal)
                orientation_vec = rmat[:, 1] 
                
                # 3. Create the Line Vector (From Source -> Target)
                # It is crucial this vector points FROM idx1 TO idx2
                line_vec = tvecs[idx2][0] - tvecs[idx1][0]
                
                # 4. Calculate Signed Angle (-180 to +180)
                unit_orient = orientation_vec / np.linalg.norm(orientation_vec)
                unit_line = line_vec / np.linalg.norm(line_vec)
                
                # Get the Z-axis (Normal vector) of the source marker to define "Up"
                normal_vec = rmat[:, 2] 
                
                # Calculate components
                dot_prod = np.dot(unit_orient, unit_line)       # Cosine component
                cross_prod = np.cross(unit_orient, unit_line)   # Vector perpendicular to turn
                
                # Project the cross product onto the normal vector to get the Sine component
                sine_component = np.dot(cross_prod, normal_vec)
                
                # Use arctan2 to calculate the full signed angle
                angle_rad = np.arctan2(sine_component, dot_prod)
                angle_deg = np.degrees(angle_rad)
                
                # --- VISUAL DEBUGGING ---
                # Draw the "Orientation" vector on screen in pink so you can see what is being compared.
                # If the pink line overlaps the yellow line, Angle is 0.
                projected_orientation_end = tvecs[idx1][0] + (orientation_vec * (min_dist * 0.5)) # Scale line to half distance
                
                # Project this 3D point to 2D image
                p_end, _ = cv2.projectPoints(projected_orientation_end.reshape(1, 1, 3), np.zeros((3,1)), np.zeros((3,1)), CM, dist_coef)
                p_end_2d = tuple(p_end[0][0].astype(int))
                
                # Draw pink "Heading" Line
                cv2.line(frame, tuple(c1), p_end_2d, (255, 105, 180), 3) 
                
                # Display text
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                info_text = f"Dist: {min_dist:.0f}mm | Ang: {angle_deg:.0f}"
                cv2.putText(frame, info_text, tuple(midpoint), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                angle_tolerance = 7 #-------------------------------------------------------------------------------------------tolerance for straight
                
                # Print to console
                current_time = time.time()
                if current_time - last_print_time >= 1: #--------------------------------------------------------------------------------print speed
                    id1_num = ids[idx1][0]
                    id2_num = ids[idx2][0]
                    print(f"ID {id1_num}->{id2_num} | Dist: {min_dist:.1f}mm | Angle: {angle_deg:.1f} deg | Turn: {'LEFT' if angle_deg > angle_tolerance else 'RIGHT' if angle_deg < -angle_tolerance else 'STRAIGHT'}")
                    last_print_time = current_time
            
            # --- ALERTS ---
            if angle_deg > angle_tolerance:
                cv2.putText(frame, "TURN LEFT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
            elif angle_deg < -angle_tolerance:
                cv2.putText(frame, "TURN RIGHT",(100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
            else:
                cv2.putText(frame, "STRAIGHT AHEAD",(100, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            
            # --- UDP COMMUNICATION ---
            # Runs every single frame, sending 2 or 1 or 0
            current_time = time.time()
            
            if (current_time - last_udp_send_time) >= UDP_INTERVAL:
                
                if angle_deg > angle_tolerance:
                    udp_message = b'\x01'
                elif angle_deg < -angle_tolerance:
                    udp_message = b'\x02'
                else:
                    udp_message = b'\x00'
                
                try:
                    sock.sendto(udp_message, (RPI_IP, RPI_PORT))
                    last_udp_send_time = current_time 
                except Exception as e:
                    print(f"UDP Error: {e}")
    
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