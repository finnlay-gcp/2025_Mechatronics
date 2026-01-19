import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket
import struct
import errno

# Constants - Receive Configuration
UDP_RECEIVE_IP = "172.26.236.13"  # Your PC's WiFi IP
UDP_RECEIVE_PORT = 50003  # Different port for PC to receive responses
BUFFER_SIZE = 1024

# Constants - Send Configuration
UDP_SEND_IP = "138.38.228.74"  # Raspberry Pi IP
UDP_SEND_PORT = 50002  # Pi is listening on this port

# --- CHANGED: SETUP RECEIVE SOCKET ONCE HERE ---
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock_receive.bind((UDP_RECEIVE_IP, UDP_RECEIVE_PORT))
    sock_receive.setblocking(False) # <--- CRITICAL FIX: Don't wait for data
    print(f"Listening for UDP on {UDP_RECEIVE_IP}:{UDP_RECEIVE_PORT}")
except Exception as e:
    print(f"Socket Bind Error: {e}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.5 # ------------------------------------------------------------------------------------------------udp send interval

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
processing_period = 0.1 #--------------------------------------------------------------------------------------------------select processing rate here

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

# Flag to stop sending after completion
task_completed = False

# This controls which marker index we are currently targeting
# 1 = Second lowest (Target 1), 2 = Third lowest (Target 2), etc.
current_target_rank = 1 
SEQUENCE_SWITCH_DIST = 300 # ------------------------------------------------------------------------Distance in mm to trigger switch to next marker
switching_cooldown = 0     # Timer to prevent double-switching


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
    
    # Variables for UDP sending (reset every frame)
    angle_deg = None
    min_dist = None
    
    # If markers are detected (after filtering)
    if ids is not None:
        # Draw detected markers and axes
        gray = aruco.drawDetectedMarkers(gray, corners, ids)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs,_objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)
        
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, CM, dist_coef, rvecs[i], tvecs[i], 50)
        
        # ------------------ SEQUENCE LOGIC ------------------
        # We need at least 2 markers to calculate a distance
        if len(ids) >= 2:
            
            # 1. Zip IDs with their index so we can sort them but keep track of where they are in rvecs/tvecs
            # Format: [(ID_Value, Index_in_Array), ...]
            id_map = []
            for i, id_val in enumerate(ids.flatten()):
                id_map.append((id_val, i))
            
            # 2. Sort by ID Value (Lowest to Highest)
            id_map.sort(key=lambda x: x[0])
            
            # 3. Identify Source (Lowest ID)
            source_id, source_idx = id_map[0]
            
            # 4. Identify Target (Based on current rank)
            # If current_target_rank is 1, we want the item at index 1 in the sorted list (2nd lowest)
            if current_target_rank < len(id_map):
                target_id, target_idx = id_map[current_target_rank]
                
                # --- CALCULATE DISTANCE & ANGLE (Source -> Target) ---
                pos1 = tvecs[source_idx][0]
                pos2 = tvecs[target_idx][0]
                
                # Euclidean distance
                min_dist = np.linalg.norm(pos1 - pos2)
                
                # --- CHECK THRESHOLD TO SWITCH TARGET ---
                if min_dist < SEQUENCE_SWITCH_DIST and (time.time() - switching_cooldown > 3.0):
                    print(f"\n[!!!] THRESHOLD REACHED ({min_dist:.0f}mm). Switching sequence from Target {current_target_rank} to {current_target_rank + 1}...\n")
                    current_target_rank += 1
                    switching_cooldown = time.time()
                    # We continue processing this frame, but next frame will look for the new target
                
                # --- VISUALIZATION & ANGLE CALC (Same as before) ---
                c1 = np.mean(corners[source_idx][0], axis=0).astype(int)
                c2 = np.mean(corners[target_idx][0], axis=0).astype(int)
                
                cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                
                rmat, _ = cv2.Rodrigues(rvecs[source_idx][0])
                orientation_vec = rmat[:, 1] 
                line_vec = tvecs[target_idx][0] - tvecs[source_idx][0]
                
                unit_orient = orientation_vec / np.linalg.norm(orientation_vec)
                unit_line = line_vec / np.linalg.norm(line_vec)
                normal_vec = rmat[:, 2] 
                
                dot_prod = np.dot(unit_orient, unit_line)
                cross_prod = np.cross(unit_orient, unit_line)
                sine_component = np.dot(cross_prod, normal_vec)
                
                angle_rad = np.arctan2(sine_component, dot_prod)
                angle_deg = np.degrees(angle_rad)
                
                # Draw heading for debug
                projected_end = tvecs[source_idx][0] + (orientation_vec * (min_dist * 0.5))
                p_end, _ = cv2.projectPoints(projected_end.reshape(1, 1, 3), np.zeros((3,1)), np.zeros((3,1)), CM, dist_coef)
                cv2.line(frame, tuple(c1), tuple(p_end[0][0].astype(int)), (255, 105, 180), 3)
                
                # Text Info
                info_text = f"SEQ: {current_target_rank} | ID {source_id}->{target_id} | Dist: {min_dist:.0f}"
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(frame, info_text, tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Tolerance checks
                angle_tolerance = 7 #------------------------------------------------------------------------------------------------angle tolerance in degrees
                dist_tolerance = 200 #--------------------------------------------------------------------------------------------------distance tolerance in mm
                
                if angle_deg > angle_tolerance:
                    cv2.putText(frame, "TURN LEFT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                elif angle_deg < -angle_tolerance:
                    cv2.putText(frame, "TURN RIGHT",(100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "STRAIGHT",(100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            else:
                # Target rank is higher than available markers
                status = f"SEQ: {current_target_rank} | WAITING FOR MARKER..."
                cv2.putText(frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # =================== UDP COMMUNICATION ===================
    # Only send if we successfully calculated a distance/angle this frame
    if angle_deg is not None and min_dist is not None:
        current_time = time.time()
        angle_rounded = 5 * round(angle_deg / 5)
        dist_rounded = 5 * round(min_dist / 5)
        turn_or_move = 0
        
        if (current_time - last_udp_send_time) >= UDP_INTERVAL and not task_completed:
            try:
                if abs(angle_rounded) <= angle_tolerance:
                    turn_or_move = 1
                if dist_rounded <= dist_tolerance:
                    turn_or_move = 2
                
                # Send
                udp_message = struct.pack('<ddd', float(angle_rounded), float(dist_rounded), float(turn_or_move))
                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                sock_send_temp.close()
                last_udp_send_time = current_time
                
                # Receive (Non-blocking)
                try:
                    data, addr = sock_receive.recvfrom(BUFFER_SIZE)
                    if len(data) >= 8:
                        received_value = struct.unpack('<d', data[:8])[0]
                        if received_value == 1.0:
                            print("Task complete signal received.")
                            task_completed = True
                except socket.error as e:
                    pass # No data
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