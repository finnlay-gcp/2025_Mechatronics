import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket
import struct
import errno

# Constants - Receive Configuration
UDP_RECEIVE_IP = "172.26.236.13"  # ----------------------------------------------------------------Your PC's WiFi IP
UDP_RECEIVE_PORT = 50003 #---------------------------------------------------------------Listening Port
BUFFER_SIZE = 1024

# Constants - Send Configuration
UDP_SEND_IP = "172.26.236.13"  # ------------------------------------------------------------Raspberry Pi IP
UDP_SEND_PORT = 50002 #--------------------------------------------------Pi is listening on this port

# --- SETUP RECEIVE SOCKET ONCE ---
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock_receive.bind((UDP_RECEIVE_IP, UDP_RECEIVE_PORT))
    sock_receive.setblocking(False) 
    print(f"Listening for UDP on {UDP_RECEIVE_IP}:{UDP_RECEIVE_PORT}")
except Exception as e:
    print(f"Socket Bind Error: {e}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.25 #------------------------------------------------------------------------------------------------udp send interval

# Load the camera calibration values
camera_calibration = np.load('workdir/Calibration.npz') 
CM=camera_calibration['CM'] 
dist_coef=camera_calibration['dist_coef'] 

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
TARGET_REACH_THRESHOLD = 200 #----------------------------------------------------------------------------------in mm 

# --- WAIT MODE STATE VARIABLES ---
wait_mode = False          # Are we currently waiting at a target?
wait_start_time = 0        # When did the wait start?
WAIT_DURATION = 10.0        # --------------------------------------------------------------------------How long to wait in seconds
frozen_target_id = -1      # Which ID are we waiting at?

print(f"Starting tracking. Target sequence starts at ID {min_valid_id}...")

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
    
    # 1. IF WE ARE IN WAIT MODE (We just reached a target)
    if wait_mode:
        elapsed = time.time() - wait_start_time
        remaining = WAIT_DURATION - elapsed
        
        # Display Waiting Status
        msg = f"REACHED ID {frozen_target_id}! HOLDING: {remaining:.1f}s"
        cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
        
        # UDP Logic during Wait Mode
        current_time = time.time()
        if (current_time - last_udp_send_time) >= UDP_INTERVAL and not task_completed:
            # Force value to 2 (stopped/reached)
            turn_or_move = 2
            # Special case for ID 17 -> 3
            if frozen_target_id == 17: #-------------------------------------------------------Special target ID
                turn_or_move = 3
            
            # Values are irrelevant during wait, but we send 0
            angle_rounded = 0.0
            dist_rounded = 0.0
            
            try:
                udp_message = struct.pack('<ddd', float(angle_rounded), float(dist_rounded), float(turn_or_move))
                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                sock_send_temp.close()
                last_udp_send_time = current_time
            except Exception as e:
                print(f"UDP Error: {e}")
        
        # Check if wait is over
        if elapsed >= WAIT_DURATION:
            print(f"Wait complete. Moving sequence to ID >= {frozen_target_id + 1}")
            min_valid_id = frozen_target_id + 1
            wait_mode = False # Resume normal tracking
    
    # 2. IF WE ARE NOT WAITING (Normal Tracking)
    else:
        # Need ids and ID 0 to proceed
        if ids is not None:
            flat_ids = ids.flatten()
            if 0 in flat_ids:
                # Filter IDs >= min_valid_id
                candidates = [i for i in flat_ids if i != 0 and i >= min_valid_id]
                
                if len(candidates) > 0:
                    target_id = min(candidates)
                    
                    idx1 = np.where(flat_ids == 0)[0][0]       # Source (ID 0)
                    idx2 = np.where(flat_ids == target_id)[0][0] # Target
                    
                    # --- CALCULATION ---
                    pos1 = tvecs[idx1][0]
                    pos2 = tvecs[idx2][0]
                    min_dist = np.linalg.norm(pos1 - pos2)
                    
                    # --- CHECK IF REACHED ---
                    if min_dist < TARGET_REACH_THRESHOLD:
                        print(f"!!! REACHED TARGET ID {target_id} (Dist: {min_dist:.0f}mm) !!!")
                        print(f"!!! Entering WAIT MODE for {WAIT_DURATION} seconds !!!")
                        
                        # Trigger Wait Mode
                        wait_mode = True
                        wait_start_time = time.time()
                        frozen_target_id = target_id
                    
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
                            print(f"ID 0 -> {target_id} | Dist: {min_dist:.0f}mm | Seq Floor: {min_valid_id}")
                            last_print_time = time.time()
                        
                        # --- UDP RECEIVING ---
                        def receive_udp_continuous():
                            """Continuously listen for UDP data until non-zero value received"""
                            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            sock.bind((UDP_RECEIVE_IP, UDP_RECEIVE_PORT))
                            sock.settimeout(2)
                            
                            print("Waiting for confirmation from Raspberry Pi", end="", flush=True)
                            
                            while True:
                                try:
                                    data, addr = sock.recvfrom(BUFFER_SIZE)
                                    
                                    # Decode and check if non-zero
                                    if len(data) >= 8:
                                        received_value = struct.unpack('<d', data[:8])[0]
                                        
                                        # Only return if value is non-zero (ignore zeros)
                                        if received_value != 0.0:
                                            sock.close()
                                            print(f"\nReceived value: {received_value}")
                                            return received_value
                                        # If zero, continue waiting without printing
                                        
                                except socket.timeout:
                                    print(".", end="", flush=True)
                                    continue
                        
                        # --- UDP SENDING (NORMAL) ---
                        angle_rounded = round(angle_deg)
                        dist_rounded = round(min_dist)
                        turn_or_move = 0
                        angle_tolerance = 7 #------------------------------------------------------------------------------------------------degrees
                        dist_tolerance = 200 #--------------------------------------------------------------------------------------------------mm
                        
                        if abs(angle_rounded) > angle_tolerance:
                            turn_or_move = 1
                        if abs(angle_rounded) <= angle_tolerance:
                            turn_or_move = 2
                        if dist_rounded <= dist_tolerance:
                            turn_or_move = 3
                        
                        # Special Case ID 7
                        if target_id == 7 and dist_rounded <= dist_tolerance:
                            turn_or_move = 4
                        
                        current_time = time.time()
                        if (current_time - last_udp_send_time) >= UDP_INTERVAL and not task_completed:
                            try:
                                udp_message = struct.pack('<ddd', float(angle_rounded), float(dist_rounded), float(turn_or_move))
                                sock_send_temp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                sock_send_temp.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
                                sock_send_temp.close()
                                last_udp_send_time = current_time
                                
                                # Receive Check
                                try:
                                    data, addr = sock_receive.recvfrom(BUFFER_SIZE)
                                    if len(data) >= 8:
                                        val = struct.unpack('<d', data[:8])[0]
                                        if val == 1.0:
                                            print("Task complete signal received.")
                                            task_completed = True
                                except socket.error:
                                    pass 
                            except Exception as e:
                                print(f"UDP Error: {e}")
                else:
                    cv2.putText(frame, f"WAITING FOR ID >= {min_valid_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Wait for non-zero response from Raspberry Pi
        received_value = receive_udp_continuous()
        
        # Check if received value is 1
        if received_value == 1.0:
            print("Task complete! Sending stop signal [0, 0, 0]...")
            time.sleep(0.1)
            angle_rounded = 0.0
            dist_rounded = 0.0
            turn_or_move = 0
            udp_message = struct.pack('<ddd', float(angle_rounded), float(dist_rounded), float(turn_or_move))
            time.sleep(0.3)
        else:
            print(f"Unexpected value {received_value}, exiting...")
            break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    proc_time = time.time() - loop_start_timestamp
    if proc_time < processing_period:
        time.sleep(processing_period - proc_time)

cap.release()
cv2.destroyAllWindows()