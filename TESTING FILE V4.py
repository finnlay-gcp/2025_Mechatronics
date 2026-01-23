import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket
import struct

# Constants - Send Configuration
UDP_SEND_IP = "172.26.236.13" 
UDP_SEND_PORT = 50002 

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.25 

# Load calibration
try:
    camera_calibration = np.load('workdir/Calibration.npz') 
    CM = camera_calibration['CM'] 
    dist_coef = camera_calibration['dist_coef'] 
except:
    CM = np.eye(3); dist_coef = np.zeros(5)

# ArUco Setup
marker_size = 105 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Window Setup
window_width = 1920
window_height = 1080
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)

cap = cv2.VideoCapture(1) 
cap.set(3,1920) 
cap.set(4,1080)

# State Variables
min_valid_id = 1 
HOME_ID = 10          # <--- SET YOUR HOME ID HERE
TARGET_REACH_THRESHOLD = 200 

wait_mode = False          
wait_start_time = 0        
WAIT_DURATION = 5.0        
frozen_target_id = -1      

return_to_home_mode = False 

# --- FLICKER FIX: PERSISTENT UDP VARIABLES ---
# We store the values we want to send here, so they persist across frames
# even if detection is lost momentarily.
active_udp_payload = None # Tuple: (angle, dist, command)

print(f"Starting tracking. Sequence -> Wait -> Home ({HOME_ID}) -> Wait...")

while True:
    loop_start_timestamp = time.time()
    
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Visualization
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, CM, dist_coef, rvecs[i], tvecs[i], 50)
    
    # ================= LOGIC CONTROL =================
    
    # 1. WAIT MODE (Output is fixed to HOLD)
    if wait_mode:
        elapsed = time.time() - wait_start_time
        remaining = WAIT_DURATION - elapsed
        
        # Force payload to HOLD (0,0,2) or (0,0,3) for ID 17
        cmd = 2
        if frozen_target_id == 17 and not return_to_home_mode: cmd = 3
        active_udp_payload = (0.0, 0.0, float(cmd))
        
        msg = f"WAITING... NEXT IN: {remaining:.1f}s"
        cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        if elapsed >= WAIT_DURATION:
            wait_mode = False 
            active_udp_payload = None # Clear payload to allow new lock
            
            if return_to_home_mode:
                min_valid_id = frozen_target_id + 1
                return_to_home_mode = False 
                print(f"Wait done. Target: Sequence ID >= {min_valid_id}")
            else:
                return_to_home_mode = True
                print(f"Wait done. Target: Home ({HOME_ID})")
    
    # 2. TRACKING MODE
    else:
        # Check if we need to FIND a target (Only if we don't have a locked payload)
        if ids is not None:
            flat_ids = ids.flatten()
            if 0 in flat_ids:
                
                # Determine Target
                target_id = None
                if return_to_home_mode:
                    if HOME_ID in flat_ids: target_id = HOME_ID
                else:
                    candidates = [i for i in flat_ids if i != 0 and i != HOME_ID and i >= min_valid_id]
                    if len(candidates) > 0: target_id = min(candidates)
                
                if target_id is not None:
                    # Calculate Live Data
                    idx1 = np.where(flat_ids == 0)[0][0]       
                    idx2 = np.where(flat_ids == target_id)[0][0] 
                    
                    pos1 = tvecs[idx1][0]
                    pos2 = tvecs[idx2][0]
                    min_dist = np.linalg.norm(pos1 - pos2)
                    
                    # Check Reached Condition (Always uses LIVE distance)
                    if min_dist < TARGET_REACH_THRESHOLD:
                        print(f"!!! REACHED TARGET {target_id} !!!")
                        wait_mode = True
                        wait_start_time = time.time()
                        frozen_target_id = target_id
                        active_udp_payload = None # Clear immediately
                    
                    else:
                        # --- LOCKING LOGIC ---
                        # If we don't have a payload yet, calculate and lock it NOW.
                        # Once locked, we DO NOT update it until reach threshold is met.
                        if active_udp_payload is None:
                            # Calculate Angle
                            c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                            c2 = np.mean(corners[idx2][0], axis=0).astype(int)
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
                            
                            # Determine Command
                            angle_rounded = round(angle_deg)
                            dist_rounded = round(min_dist)
                            turn_or_move = 0
                            if abs(angle_rounded) > 7: turn_or_move = 1
                            else: turn_or_move = 2
                            if dist_rounded <= 200: turn_or_move = 3
                            if target_id == 17 and dist_rounded <= 200: turn_or_move = 4
                            
                            # LOCK IT
                            active_udp_payload = (float(angle_rounded), float(dist_rounded), float(turn_or_move))
                            print(f"--> LOCKED VALUES: {active_udp_payload}")
                        
                        # Visuals (Just for user, doesn't affect UDP)
                        c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                        c2 = np.mean(corners[idx2][0], axis=0).astype(int)
                        cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                        mid = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
                        cv2.putText(frame, f"LIVE: {min_dist:.0f}mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # ================= UDP SEND BLOCK (MOVED OUTSIDE) =================
    # This now runs EVERY loop iteration. 
    # If we have a locked payload, we send it. Even if camera is blocked.
    current_time = time.time()
    if (current_time - last_udp_send_time) >= UDP_INTERVAL:
        if active_udp_payload is not None:
            try:
                # Unpack the locked values
                p_angle, p_dist, p_cmd = active_udp_payload
                
                # Send
                udp_message = struct.pack('<ddd', p_angle, p_dist, p_cmd)
                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                sock_send_temp.close()
                last_udp_send_time = current_time
                
                # Debug print
                # print(f"UDP Sent: {p_angle}, {p_dist}, {p_cmd}")
            except Exception as e:
                print(f"UDP Error: {e}")
        else:
            # Optional: Send zeros if searching? Or just send nothing.
            # Sending nothing is safer to keep Simulink "holding" the last value if configured.
            # If you want to force zeros when searching, uncomment below:
            # active_udp_payload = (0.0, 0.0, 0.0) 
            pass

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    # Loop timing
    proc_time = time.time() - loop_start_timestamp
    if proc_time < 0.1: time.sleep(0.1 - proc_time)

cap.release()
cv2.destroyAllWindows()