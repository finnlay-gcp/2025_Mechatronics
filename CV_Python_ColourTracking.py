# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities. We will use this to ensure a steady processing rate
import time 
# This is a library to handle file paths
import os
# Library for networking (UDP)
import socket

# --- NETWORK CONFIGURATION (UDP) ---
RPI_IP = "138.38.226.136"  # <-------------------------------------------------------------CHANGE THIS to the Raspberry Pi's IP address
RPI_PORT = 50002 # -------------------------------------------------------------------------------------The port the Pi will listen on
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- CONFIGURATION ---
scan_band_height = 60 #----------------------------------------------------------------------------------------------------Band height
scan_band_width = 300 #-----------------------------------------------------------------------------------------------------Band width
snapshot_folder = "Snapshots" # Folder name
enable_snapshots = False  # ----------------------------------------------------------Set True to save images, False to disable saving

# --- UDP TIMING CONFIGURATION ---
last_udp_send_time = 0
UDP_INTERVAL = 0.1

# --- color DEFINITIONS (HSV RANGES) ---
S_MIN = 50 
V_MIN = 50

color_definitions = {
    "Red": {
        "ranges": [
            # Lower Red
            (np.array([0, S_MIN, V_MIN]), np.array([20, 255, 255])), 
            # Upper Red
            (np.array([160, S_MIN, V_MIN]), np.array([180, 255, 255])) 
        ],
        "draw_color": (0, 0, 255)
    },
    "Green": {
        "ranges": [
            # Green Range
            (np.array([30, 25, 25]), np.array([95, 255, 255])) 
        ],
        "draw_color": (0, 255, 0)
    },
    "Blue": {
        "ranges": [
            # Blue Range
            (np.array([100, S_MIN, V_MIN]), np.array([150, 255, 255])) 
        ],
        "draw_color": (255, 0, 0)
    },
}

# --- USER INPUT SELECTION ---
print("\n" + "="*40)
print("      color SCANNER CONFIGURATION")
print("="*40)
available_colors = list(color_definitions.keys())
print(f"Available colors: {', '.join(available_colors)}")
print("Type 'All' to scan for everything.")
print("-" * 40)

while True:
    raw_input = input(">> Enter color(s) to scan (comma-separated): ")
    
    selected_inputs = [item.strip().capitalize() for item in raw_input.split(',')]

    if "All" in selected_inputs:
        print("Confirmed: Scanning for ALL colors.")
        break

    invalid_choices = [c for c in selected_inputs if c not in color_definitions]

    if not invalid_choices and selected_inputs:     
        color_definitions = {k: v for k, v in color_definitions.items() if k in selected_inputs}
        
        formatted_list = ", ".join(selected_inputs)
        print(f"Confirmed: Scanning for: {formatted_list}")
        break
    else:
        error_msg = ", ".join(invalid_choices) if invalid_choices else "Empty input"
        print(f"Error: The following are not valid options: {error_msg}")
        print(f"Available options: {', '.join(available_colors)} or 'All'.")

# --- WINDOW SETUP ---
window_width = 768*2 #Keep 1920:1080 ratio
window_height = 432*2 #Keep 1920:1080 ratio
print("Creating windows...")
window_names = ["color Detection"]
for name in window_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)

cv2.moveWindow("color Detection", 0, 0)
print("Windows created. Starting camera feed...\n")

cap = cv2.VideoCapture(1) #-----------------------------------------------------------------------------------------Select camera here

cap.set(3,640)
cap.set(4,360)

# --- SNAPSHOT CONFIGURATION ---
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)
    print(f"Created directory: {snapshot_folder}")

def take_snapshot(frame, color_name):
    """Saves the current frame to the subfolder."""
    timestamp = int(time.time() * 1000)
    filename = os.path.join(snapshot_folder, f"detected_{color_name}_{timestamp}.jpg")
    
    cv2.imwrite(filename, frame)
    print(f"*** SNAPSHOT SAVED: {filename} ***")

# --- INITIALISATION ---
active_colors = set() 
prev_frame_time = time.time()

# --- MAIN LOOP ---
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Additional image processing: Gaussian Blur
    gauss = cv2.GaussianBlur(frame, (5, 5), 0)    

    # Convert BGR to HSV
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)

    # Calculate scan band limits
    height, width, _ = frame.shape
    cy = height // 2
    cx = width // 2
    # Calculate the top and bottom Y-coordinates
    band_half_height = scan_band_height // 2
    band_top_y = cy - band_half_height
    band_bottom_y = cy + band_half_height
    # Calculate the left and right X-coordinates
    band_half_width = scan_band_width // 2
    band_left_x = cx - band_half_width
    band_right_x = cx + band_half_width

    # --- CREATE ZONE MASK ---
    # Create a completely black image the size of the frame
    zone_mask = np.zeros((height, width), dtype="uint8")
    # Draw a solid white rectangle where the scan band is
    cv2.rectangle(zone_mask, (band_left_x, band_top_y), (band_right_x, band_bottom_y), 255, -1)
    # Visually draw the scan band on the frame
    cv2.rectangle(frame, (band_left_x, band_top_y), (band_right_x, band_bottom_y), (200, 200, 200), 2)
    cv2.putText(frame, "SCAN ZONE", (band_left_x + 5, band_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    best_detection = None 
    
    current_frame_colors = set()

    # Loop through every color selected
    for color_name, params in color_definitions.items():
        final_mask = np.zeros(hsv.shape[:2], dtype="uint8")
        
        for (lower, upper) in params["ranges"]:
            temp_mask = cv2.inRange(hsv, lower, upper)
            final_mask = cv2.bitwise_or(final_mask, temp_mask)

        # Apply the zone mask
        final_mask = cv2.bitwise_and(final_mask, zone_mask)
        # Clean up noise
        final_mask = cv2.erode(final_mask, None, iterations=2)
        final_mask = cv2.dilate(final_mask, None, iterations=2)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Find the largest blob of this color
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 1000: 
                x, y, w, h = cv2.boundingRect(c)
                obj_cy = y + (h // 2)
                obj_cx = x + (w // 2)

                # Determine best detection across all colors
                if best_detection is None or area > best_detection['area']:
                    best_detection = {
                        'area': area,
                        'rect': (x, y, w, h),
                        'color_name': color_name,
                        'draw_color': params["draw_color"],
                        'center': (obj_cx, obj_cy)
                    }

    # --- DRAWING AND ALERTS ---
    # If found a valid "Champion" blob
    if best_detection:
        # Unpack data
        x, y, w, h = best_detection['rect']
        color_name = best_detection['color_name']
        draw_color = best_detection['draw_color']
        cx, cy = best_detection['center']

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), draw_color, 2)
        label = f"{color_name}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
        cv2.circle(frame, (cx, cy), 5, draw_color, -1)

        # Add to current colors set (for snapshot logic)
        current_frame_colors.add(color_name)

        # Display "LASER ON" text on screen
        cv2.putText(frame, "LASER ON", (width - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    # --- UDP COMMUNICATION ---
    # Runs every single frame, sending 1 or 0
    current_time = time.time()
    
    if (current_time - last_udp_send_time) >= UDP_INTERVAL:
        
        if best_detection:
            udp_message = b'\x01'
        else:
            udp_message = b'\x00'

        try:
            sock.sendto(udp_message, (RPI_IP, RPI_PORT))
            last_udp_send_time = current_time 
        except Exception as e:
            print(f"UDP Error: {e}")

    # --- SNAPSHOT LOGIC ---
    # Check which colors are NEW in the band
    for col in current_frame_colors:
        if col not in active_colors:
            if enable_snapshots:
                take_snapshot(frame, col)
    # Update memory
    active_colors = current_frame_colors

    # --- FPS CALCULATION AND DISPLAY ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("color Detection", frame)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break

# When everything is done, release capture and close windows
cap.release()
cv2.destroyAllWindows()
sock.close() # Close UDP socket cleanly