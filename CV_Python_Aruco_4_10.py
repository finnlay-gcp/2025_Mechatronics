import cv2
import cv2.aruco as aruco
import numpy as np
import time 
import socket
import struct

# --- NETWORK CONFIGURATION ---

# 1. SENDING CONFIGURATION (Target: Raspberry Pi)
RPI_IP = "138.38.228.74"     # <--- CHANGE THIS to the Raspberry Pi's IP address
RPI_PORT = 50002             # The port the Pi will listen on
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. RECEIVING CONFIGURATION (Host: Your PC)
UDP_RECEIVE_IP = "172.26.118.176"  # <--- CHANGE THIS to Your PC's WiFi IP
UDP_RECEIVE_PORT = 50003           # Port for PC to receive responses
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Setup Receive Socket options
try:
    sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_receive.bind((UDP_RECEIVE_IP, UDP_RECEIVE_PORT))
    sock_receive.setblocking(0) # <--- CRITICAL: Set to Non-Blocking so video doesn't freeze
    print(f"Listening for UDP responses on {UDP_RECEIVE_IP}:{UDP_RECEIVE_PORT}")
except Exception as e:
    print(f"Error binding receive socket: {e}")

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.25 
stop_signal_received = False # Flag to track if Pi said "Done"

# Load the camera calibration values
try:
    camera_calibration = np.load('workdir/Calibration.npz') # Ensure this path is correct
    CM = camera_calibration['CM']
    dist_coef = camera_calibration['dist_coef']
except Exception as e:
    print(f"Warning: Could not load calibration file: {e}")
    # Fallback to dummy values if file missing (prevents crash, but pose will be wrong)
    CM = np.eye(3)
    dist_coef = np.zeros((5,1))

# Define the ArUco dictionary and parameters
marker_size = 98 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Input specific ArUco IDs to detect
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
processing_period = 0.1 

# Create OpenCV named windows
window_scale = 0.75 
window_width = int(1920 * window_scale)
window_height = int(1080 * window_scale)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)
cv2.moveWindow("Frame", 0, 0)

# Start capturing video
cap = cv2.VideoCapture(1) 
cap.set(3, 1920) 
cap.set(4, 1080)

# Timers
start_time = time.time()
fps = 0
last_print_time = time.time()

print("System Ready. Starting Loop...")

while True:
    loop_start_timestamp = time.time()
    
    # 1. CHECK FOR INCOMING UDP MESSAGES (Non-blocking)
    try:
        data, addr = sock_receive.recvfrom(1024)
        if len(data) >= 8:
            received_value = struct.unpack('<d', data[:8])[0]
            if received_value == 1.0:
                print("\n[UDP RX] Received 'Task Complete' signal (1.0) from Pi!")
                stop_signal_received = True
            elif received_value != 0:
                print(f"\n[UDP RX] Received value: {received_value}")
    except BlockingIOError:
        # No data received, continue normally
        pass
    except Exception as e:
        print(f"UDP Receive Error: {e}")

    # 2. CAPTURE & PROCESS FRAME
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Variables to hold navigation data (Default: Stop)
    angle_to_send = 0.0
    dist_to_send = 0.0
    tom_to_send = 0.0
    valid_navigation_data = False

    # If markers are detected
    if ids is not None:
        # --- Marker Filtering Logic (Best Marker per ID) ---
        best_markers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if specified_ids is not None and marker_id not in specified_ids:
                continue
            perimeter = cv2.arcLength(corners[i], True)
            if marker_id not in best_markers or perimeter > best_markers[marker_id][1]:
                best_markers[marker_id] = (i, perimeter)
        
        if best_markers:
            valid_indices = [val[0] for val in best_markers.values()]
            corners = tuple([corners[i] for i in valid_indices])
            ids = np.array([ids[i] for i in valid_indices])
        else:
            ids = None

    # If we still have valid markers after filtering
    if ids is not None:
        # If we lost tracking previously, assume new task, reset stop signal
        # (Optional logic: remove if you want manual reset only)
        # stop_signal_received = False 

        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, CM, dist_coef)
        
        # Draw axes
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, CM, dist_coef, rvecs[i], tvecs[i], 50)
        
        # --- DISTANCE & ANGLE CALCULATION ---
        if len(ids) >= 2:
            min_dist = float('inf')
            closest_pair = None
            
            # Find closest pair
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    dist = np.linalg.norm(tvecs[i][0] - tvecs[j][0])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (i, j)
            
            if closest_pair:
                idx1, idx2 = closest_pair
                # Sort IDs for consistency (Low -> High)
                if ids[idx1][0] > ids[idx2][0]:
                    idx1, idx2 = idx2, idx1

                # Visuals
                c1 = np.mean(corners[idx1][0], axis=0).astype(int)
                c2 = np.mean(corners[idx2][0], axis=0).astype(int)
                cv2.line(frame, tuple(c1), tuple(c2), (0, 255, 255), 2)
                
                # Math: Calculate Angle
                rmat, _ = cv2.Rodrigues(rvecs[idx1][0])
                orientation_vec = rmat[:, 1] # Y-axis
                line_vec = tvecs[idx2][0] - tvecs[idx1][0]
                
                unit_orient = orientation_vec / np.linalg.norm(orientation_vec)
                unit_line = line_vec / np.linalg.norm(line_vec)
                normal_vec = rmat[:, 2] # Z-axis
                
                dot_prod = np.dot(unit_orient, unit_line)
                cross_prod = np.cross(unit_orient, unit_line)
                sine_component = np.dot(cross_prod, normal_vec)
                
                angle_rad = np.arctan2(sine_component, dot_prod)
                angle_deg = np.degrees(angle_rad)
                
                # Update data to send
                angle_to_send = angle_deg
                dist_to_send = min_dist
                valid_navigation_data = True
                
                # Tolerances
                angle_tolerance = 7
                dist_tolerance = 200

                # Determine TOM (Turn Or Move)
                # 0 = Stop/Wait, 1 = Turn, 2 = Move
                if abs(angle_deg) <= angle_tolerance:
                    tom_to_send = 1.0 # Aligned, ready to rotate? (Or logic from script 1 was 1=turn?)
                    # Script 1 logic: if angle small -> turn_or_move = 1. 
                    # Actually usually: if angle BIG -> Turn (1). If angle SMALL -> Move (2).
                    # Let's stick to Script 1 exact logic:
                    # "if abs(angle) <= tolerance: tom=1" (This implies 1 is Move? logic seems inverted in original comments but code ruled)
                    # Let's use strict logic:
                    tom_to_send = 1.0 # "Aligned" state
                
                if min_dist <= dist_tolerance:
                    tom_to_send = 2.0 # "Close enough" state

                # Display Info
                midpoint = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(frame, f"Dist: {min_dist:.0f} | Ang: {angle_deg:.0f}", tuple(midpoint), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Console Print
                if time.time() - last_print_time >= 1:
                    status = "DONE/STOP" if stop_signal_received else "ACTIVE"
                    print(f"[{status}] Dist: {min_dist:.1f}mm | Angle: {angle_deg:.1f} | TOM: {tom_to_send}")
                    last_print_time = time.time()
    
    else:
        # No markers found - Reset logic? 
        # If you want the robot to reset when you hide markers, uncomment below:
        if stop_signal_received:
             print("Tracking lost - Resetting Stop Signal")
             stop_signal_received = False
        pass

    # 3. UDP SENDING LOGIC
    current_time = time.time()
    if (current_time - last_udp_send_time) >= UDP_INTERVAL:
        try:
            # Logic: If we received "1.0" from Pi, we FORCE send 0,0,0 (Stop)
            # If we haven't received "1.0", we send the calculated camera values
            
            if stop_signal_received:
                # Send STOP command
                msg = struct.pack('<ddd', 0.0, 0.0, 0.0)
                # Visual Feedback on screen
                cv2.putText(frame, "TASK COMPLETE", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif valid_navigation_data:
                # Send NAVIGATION command
                angle_rnd = round(angle_to_send)
                dist_rnd = round(dist_to_send)
                tom_rnd = round(tom_to_send)
                msg = struct.pack('<ddd', angle_rnd, dist_rnd, tom_rnd)
            else:
                # No markers seen, send 0,0,0 or Keep Alive?
                # Usually safer to send 0s if nothing is seen
                msg = struct.pack('<ddd', 0.0, 0.0, 0.0)

            sock_send.sendto(msg, (RPI_IP, RPI_PORT))
            last_udp_send_time = current_time
            
        except Exception as e:
            print(f"UDP Send Error: {e}")

    # 4. GUI UPDATE
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'): # Manual Reset option
        stop_signal_received = False
        print("Manual Reset: sending commands again.")

    # FPS Calculation
    processing_time = time.time() - loop_start_timestamp
    if processing_time > 0:
        fps = 1.0 / processing_time
    
    if processing_time < processing_period:
        time.sleep(processing_period - processing_time)

# Cleanup
cap.release()
cv2.destroyAllWindows()
sock_send.close()
sock_receive.close()