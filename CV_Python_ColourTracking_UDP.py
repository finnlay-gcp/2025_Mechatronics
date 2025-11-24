# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python
import numpy as np
# This is a library to get access to time-related functionalities
import time 
# This is a library to handle file paths
import os
# Library for Networking
import socket

# --- NETWORK CONFIGURATION ---
RPI_IP_ADDRESS = "192.168.1.15" # <--------------------------------------------------------------------------------- CHANGE THIS TO YOUR PI'S IP ADDRESS
RPI_PORT = 5005 # ------------------------------------------------------------------------------------------------ CHANGE THIS TO MATCH THE PORT IN laser_server.py
# -----------------------------

# Setup UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define a processing rate
processing_period = 0.25

# --- CONFIGURATION ---
# Define the size of the center detection box (in pixels)
scan_band_height = 300 
snapshot_folder = "Snapshots" 
enable_snapshots = True 

color_definitions = {
    "Red": {
        "ranges": [
            (np.array([0, 70, 50]), np.array([15, 255, 255])),     
            (np.array([165, 70, 50]), np.array([180, 255, 255]))   
        ],
        "draw_color": (0, 0, 255) 
    },
    "Green": {
        "ranges": [ (np.array([35, 50, 50]), np.array([90, 255, 255])) ],
        "draw_color": (0, 255, 0) 
    },
    "Blue": {
        "ranges": [ (np.array([95, 70, 50]), np.array([155, 255, 255])) ],
        "draw_color": (255, 0, 0) 
    }
}

# --- USER INPUT SELECTION ---
print("\n" + "="*40)
print("      COLOR SCANNER CONFIGURATION")
print("="*40)
available_colors = list(color_definitions.keys())
print(f"Available colors: {', '.join(available_colors)}")
print("Type 'All' to scan for everything.")
print("-" * 40)

while True:
    user_input = input(">> Enter color to scan: ").strip().capitalize()
    if user_input == "All":
        break
    elif user_input in color_definitions:
        color_definitions = {user_input: color_definitions[user_input]}
        break
    else:
        print(f"Error: '{user_input}' is not a valid option.")
print("="*40 + "\n")

# Create OpenCV named windows
window_width = 768 
window_height = 432 
window_names = ["Colour Detection"]
for name in window_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)
cv2.moveWindow("Colour Detection", 0, 0)

cap = cv2.VideoCapture(1) 
cap.set(3,1920)
cap.set(4,1080)

start_time = time.time()
fps = 0

if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

def take_snapshot(frame, color_name):
    timestamp = int(time.time() * 1000)
    filename = os.path.join(snapshot_folder, f"detected_{color_name}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"*** SNAPSHOT SAVED: {filename} ***")

active_colors = set() 

# State tracking for Network to avoid spamming packets
last_sent_state = "OFF"

while True:
    ret, frame = cap.read()
    if not ret: break

    gauss = cv2.GaussianBlur(frame, (5, 5), 0)    
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)

    height, width, _ = frame.shape
    cy = height // 2
    band_half_height = scan_band_height // 2
    band_top_y = cy - band_half_height
    band_bottom_y = cy + band_half_height

    cv2.line(frame, (0, band_top_y), (width, band_top_y), (200, 200, 200), 2)
    cv2.line(frame, (0, band_bottom_y), (width, band_bottom_y), (200, 200, 200), 2)
    cv2.putText(frame, "SCAN BAND", (10, band_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    current_frame_colors = set()

    for color_name, params in color_definitions.items():
        final_mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for (lower, upper) in params["ranges"]:
            temp_mask = cv2.inRange(hsv, lower, upper)
            final_mask = cv2.bitwise_or(final_mask, temp_mask)

        final_mask = cv2.erode(final_mask, None, iterations=2)
        final_mask = cv2.dilate(final_mask, None, iterations=2)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000: 
                x, y, w, h = cv2.boundingRect(c)
                obj_cy = y + (h // 2)

                if band_top_y < obj_cy < band_bottom_y:
                    current_frame_colors.add(color_name)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), params["draw_color"], 2)
                    cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, params["draw_color"], 2)
                    cv2.circle(frame, (x + (w // 2), obj_cy), 5, params["draw_color"], -1)
    
    # --- NETWORK TRIGGER LOGIC ---
    # Send "ON" if objects are found, "OFF" if empty
    # We only send data if the state CHANGES to avoid flooding the network
    target_state = "ON" if len(current_frame_colors) > 0 else "OFF"
    
    if target_state != last_sent_state:
        try:
            sock.sendto(target_state.encode('utf-8'), (RPI_IP_ADDRESS, RPI_PORT))
            last_sent_state = target_state
            # Visual feedback on PC screen
            print(f"Sent UDP Command: {target_state}")
        except Exception as e:
            print(f"Network Error: {e}")

    # Visual Indicator on PC Screen
    if last_sent_state == "ON":
        cv2.putText(frame, "LASER ACTIVE", (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # -----------------------------

    # Snapshot Logic
    for col in current_frame_colors:
        if col not in active_colors:
            if enable_snapshots:
                take_snapshot(frame, col)
    
    active_colors = current_frame_colors

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Colour Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()

cap.release()
cv2.destroyAllWindows()