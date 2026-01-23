import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket
import struct
import errno

# Constants - Send Configuration
UDP_SEND_IP = "172.26.236.13"  # ------------------------------------------------------------Raspberry Pi IP
UDP_SEND_PORT = 50002 #--------------------------------------------------Pi is listening on this port

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.25 #------------------------------------------------------------------------------------------------udp send interval

# Load the camera calibration values
try:
    camera_calibration = np.load('workdir/Calibration.npz') 
    CM=camera_calibration['CM'] 
    dist_coef=camera_calibration['dist_coef'] 
except:
    CM = np.eye(3); dist_coef = np.zeros(5)

# Define the ArUco dictionary and parameters
marker_size = 105 #------------------------------------------------------------------------------------------------Marker size in mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Input specific ArUco IDs to detect
aruco_ids_input = input("Enter specific ArUco IDs to detect (comma-separated). Leave empty to detect all: ").strip()
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
processing_period = 0.1 #--------------------------------------------------------------------------------------------------seconds

# Create OpenCV named windows
window_scale = 1 #------------------------------------------------------------------------------------------------Window scale
window_width = int(1920 * window_scale)
window_height = int(1080 * window_scale)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.moveWindow("Frame", 0, 0)

# Start capturing video
cap = cv2.VideoCapture(1) #------------------------------------------------------------------------------------------------Camera index
cap.set(3,1920) 
cap.set(4,1080)

# Set the starting time
start_time = time.time()
fps = 0

# Initialise print timer
last_print_time = time.time()

# Flag to stop sending after completion
task_completed = False

# --- TARGET SEQUENCE STATE ---
min_valid_id = 1 
HOME_ID = 9 # <---------------------------------------------------------------- CHANGE THIS TO YOUR HOME MARKER ID
TARGET_REACH_THRESHOLD = 200 #----------------------------------------------------------------------------------in mm 

# --- WAIT MODE STATE VARIABLES ---
wait_mode = False          # Are we currently waiting at a target?
wait_start_time = 0        # When did the wait start?
WAIT_DURATION = 5.0        # --------------------------------------------------------------------------How long to wait in seconds
frozen_target_id = -1      # Which ID are we waiting at?

# --- NEW STATE VARIABLE ---
return_to_home_mode = False # False = Going to Next ID, True = Going back to Home

print(f"Starting tracking. Sequence: ID {min_valid_id} -> Wait -> Home ({HOME_ID}) -> Wait -> Next ID...")

while True:
    loop_start_timestamp = time.time()
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Standard frame processing logic (always runs so we can see the camera)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # --- VISUALIZATION LOGIC ---
    if ids is not None:
        gray = aruco.drawDetectedMarkers(gray, corners, ids)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, CM, dist_coef, rvecs[i], tvecs[i], 50)
    
    # ================= LOGIC CONTROL =================
    
    # 1. IF WE ARE IN WAIT MODE (We just reached a target OR Home)
    if wait_mode:
        
        # Calculate time remaining
        elapsed = time.time() - wait_start_time
        remaining = WAIT_DURATION - elapsed
        
        # Display Status
        if return_to_home_mode:
            # We are waiting at HOME before starting next sequence
            msg = f"WAITING AT HOME. NEXT TARGET {frozen_target_id + 1} IN: {remaining:.1f}s"
        else:
            # We are waiting at a TARGET before going home
            msg = f"WAITING AT TARGET {frozen_target_id}. GOING HOME IN: {remaining:.1f}s"

        cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        # UDP Logic during Wait Mode (Sending Stop/Hold)
        current_time = time.time()
        if (current_time - last_udp_send_time) >= UDP_INTERVAL and not task_completed:
            turn_or_move = 2 # Default Hold
            
            # Special logic from original code for ID 17
            if frozen_target_id == 17 and not return_to_home_mode: 
                turn_or_move = 3
            
            try:
                # Send 0,0,2 to indicate "Hold Position"
                udp_message = struct.pack('<ddd', 0.0, 0.0, float(turn_or_move))
                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                sock_send_temp.close()
                last_udp_send_time = current_time
            except Exception as e:
                print(f"UDP Error: {e}")
        
        # Check if wait is over
        if elapsed >= WAIT_DURATION:
            wait_mode = False # Stop waiting
            
            if return_to_home_mode:
                # We finished waiting at Home. Time to switch to the NEXT Sequence ID.
                min_valid_id = frozen_target_id + 1
                return_to_home_mode = False 
                print(f"Home wait done. Switching to Sequence ID >= {min_valid_id}")
            else:
                # We finished waiting at a Target. Time to switch to HOME mode.
                return_to_home_mode = True
                print(f"Target wait done. Returning to Home ID {HOME_ID}")
    
    # 2. IF WE ARE NOT WAITING (Normal Tracking)
    else:
        # Need ids and ID 0 to proceed
        if ids is not None:
            flat_ids = ids.flatten()
            if 0 in flat_ids:
                
                # --- DETERMINE THE TARGET ID ---
                target_id = None
                
                if return_to_home_mode:
                    # We are trying to find Home
                    if HOME_ID in flat_ids:
                        target_id = HOME_ID
                else:
                    # We are trying to find the next sequence ID
                    # Filter IDs >= min_valid_id AND exclude Home ID and ID 0
                    candidates = [i for i in flat_ids if i != 0 and i != HOME_ID and i >= min_valid_id]
                    if len(candidates) > 0:
                        target_id = min(candidates)
                
                # --- IF WE FOUND A VALID TARGET ---
                if target_id is not None:
                    
                    idx1 = np.where(flat_ids == 0)[0][0]       # Source (ID 0)
                    idx2 = np.where(flat_ids == target_id)[0][0] # Target (Home or Sequence)
                    
                    # --- CALCULATION ---
                    pos1 = tvecs[idx1][0]
                    pos2 = tvecs[idx2][0]
                    min_dist = np.linalg.norm(pos1 - pos2)
                    
                    # --- CHECK IF REACHED ---
                    if min_dist < TARGET_REACH_THRESHOLD:
                        print(f"!!! REACHED TARGET ID {target_id} (Dist: {min_dist:.0f}mm) !!!")
                        print(f"!!! Entering WAIT MODE. !!!")
                        
                        # Trigger Wait Mode
                        wait_mode = True
                        wait_start_time = time.time()
                        
                        # IMPORTANT: Record which ID we just hit so we know what to do next
                        frozen_target_id = target_id
                        
                        # If we just hit Home, we do NOT change min_valid_id yet.
                        # We change it after the wait is done.
                    
                    else:
                        # --- NORMAL TRACKING VISUALS & UDP ---
                        c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                        c2 = np.mean(corners[idx2][0], axis=0).astype(int)
                        cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                        
                        rmat, _ = cv2.Rodrigues(rvecs[idx1][0])
                        orientation_vec = rmat[:, 1] 
                        line_vec = tvecs[idx2][0] - tvecs[idx1][0]
                        
                        unit_orient = orientation_vec / np.linalg.norm(orientation_vec)
                        unit_line = line_vec / np.linalg.norm(line_vec)
                        normal_vec = rmat[:, 2] 
                        
                        dot_prod = np.dot(unit_orient, unit_line)       
                        cross_prod = np.cross(unit_orient, unit_line)   
                        sine_component = np.dot(cross_prod, normal_vec)
                        
                        angle_deg = np.degrees(np.arctan2(sine_component, dot_prod))
                        
                        # Draw Heading Line
                        projected_end = tvecs[idx1][0] + (orientation_vec * (min_dist * 0.5))
                        p_end, _ = cv2.projectPoints(projected_end.reshape(1, 1, 3), np.zeros((3,1)), np.zeros((3,1)), CM, dist_coef)
                        cv2.line(frame, tuple(c1), tuple(p_end[0][0].astype(int)), (255, 105, 180), 3) 
                        
                        midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                        info_text = f"Tgt: {target_id} | Dist: {min_dist:.0f}mm | Ang: {angle_deg:.0f}"
                        cv2.putText(frame, info_text, tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        if time.time() - last_print_time >= 1: 
                            mode_str = "HOME" if return_to_home_mode else "SEQ"
                            print(f"[{mode_str}] ID 0 -> {target_id} | Dist: {min_dist:.0f}mm")
                            last_print_time = time.time()
                        
                        # --- UDP SENDING (NORMAL) ---
                        angle_rounded = round(angle_deg)
                        dist_rounded = round(min_dist)
                        turn_or_move = 0
                        angle_tolerance = 7 
                        dist_tolerance = 200 
                        
                        if abs(angle_rounded) > angle_tolerance:
                            turn_or_move = 1
                        if abs(angle_rounded) <= angle_tolerance:
                            turn_or_move = 2
                        if dist_rounded <= dist_tolerance:
                            turn_or_move = 3
                        
                        # Special Case ID 17 (Only applies if we are targeting 17, not Home)
                        if target_id == 17 and dist_rounded <= dist_tolerance:
                            turn_or_move = 4
                        
                        current_time = time.time()
                        if (current_time - last_udp_send_time) >= UDP_INTERVAL and not task_completed:
                            try:
                                udp_message = struct.pack('<ddd', float(angle_rounded), float(dist_rounded), float(turn_or_move))
                                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                                sock_send_temp.close()
                                last_udp_send_time = current_time
                            except Exception as e:
                                print(f"UDP Error: {e}")
                else:
                    # No valid target found
                    if return_to_home_mode:
                        cv2.putText(frame, f"SEARCHING FOR HOME ID {HOME_ID}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.putText(frame, f"WAITING FOR ID >= {min_valid_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    proc_time = time.time() - loop_start_timestamp
    if proc_time < processing_period:
        time.sleep(processing_period - proc_time)

cap.release()
cv2.destroyAllWindows()