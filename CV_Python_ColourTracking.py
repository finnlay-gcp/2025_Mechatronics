# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities. We will use this to ensure a steady processing rate
import time 

# Define a processing rate
processing_period = 0.25

# --- CONFIGURATION ---
# Define the size of the center detection box (in pixels)
scan_band_height = 300 

# Dictionary structure: "Color Name": { "ranges": [ (lower, upper) ], "draw_color": (B, G, R) }
color_definitions = {
    "Red": {
        "ranges": [
            (np.array([0, 70, 50]), np.array([15, 255, 255])),     # Lower Red (0-10)
            (np.array([165, 70, 50]), np.array([180, 255, 255]))   # Upper Red (170-180)
        ],
        "draw_color": (0, 0, 255) # Red in BGR
    },
    "Green": {
        "ranges": [
            (np.array([35, 50, 50]), np.array([90, 255, 255]))    # Green range
        ],
        "draw_color": (0, 255, 0) # Green in BGR
    },
    "Blue": {
        "ranges": [
            (np.array([95, 70, 50]), np.array([155, 255, 255]))   # Blue range
        ],
        "draw_color": (255, 0, 0) # Blue in BGR
    }
}

# Create OpenCV named windows
window_width = 768 #keep 1920:1080 ratio
window_height = 432 #keep 1920:1080 ratio
print("Creating windows...")
window_names = ["Colour Detection"]
for name in window_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)

# Position the windows next to each other
cv2.moveWindow("Colour Detection", 0, 0)
print("Windows created. Starting camera feed...\n")

# Start capturing video
cap = cv2.VideoCapture(1) #----------------------------------------------------------------------------------select camera here

# Set the width and heigth of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)

# Set the starting time
start_time = time.time()
fps = 0

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

    # --- 1. CALCULATE SCAN BAND LIMITS ---
    height, width, _ = frame.shape
    cy = height // 2
    
    # Calculate the top and bottom Y-coordinates of the band
    band_half_height = scan_band_height // 2
    band_top_y = cy - band_half_height
    band_bottom_y = cy + band_half_height

    # Draw the "Scan Band" lines across the whole screen
    # Line 1: Top limit
    cv2.line(frame, (0, band_top_y), (width, band_top_y), (200, 200, 200), 2)
    # Line 2: Bottom limit
    cv2.line(frame, (0, band_bottom_y), (width, band_bottom_y), (200, 200, 200), 2)
    
    cv2.putText(frame, "SCAN BAND", (10, band_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # -------------------------------------------


    # --- LOOP THROUGH EACH DEFINED COLOR ---
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
            area = cv2.contourArea(c)

            if area > 1000: # Increased area to reduce noise
                x, y, w, h = cv2.boundingRect(c)
                
                # --- 2. CHECK VERTICAL POSITION ONLY ---
                # Calculate center Y of the detected object
                obj_cy = y + (h // 2)

                # Check if object center Y is within the top and bottom band limits
                # We REMOVED the X-axis (horizontal) check here
                if band_top_y < obj_cy < band_bottom_y:
                    
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x+w, y+h), params["draw_color"], 2)
                    label = f"{color_name}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, params["draw_color"], 2)
                    
                    # Draw a dot at the object center
                    obj_cx = x + (w // 2)
                    cv2.circle(frame, (obj_cx, obj_cy), 5, params["draw_color"], -1)

    # Add the frame rate to the images
    fps_label = f"CAMERA FPS: {fps:.2f}"
    proc_label = f"PROCESSING FPS: {1/processing_period:.2f}"

    fps_label = f"CAMERA FPS: {fps:.2f}"
    cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Colour Detection", frame)

    # Break the loop on 'q' key press (use longer wait time to allow window responsiveness)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break

    # Ensure a steady processing rate
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    if elapsed_time < processing_period:
        time.sleep(processing_period - elapsed_time)
    start_time = time.time()

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()