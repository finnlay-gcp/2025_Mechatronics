# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities. We will use this to ensure a steady processing rate
import time 

# Define a processing rate
processing_period = 0.25

# --- CONFIGURATION FOR COLORS ---
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

    # --- DEBUGGING TOOL: CENTER PIXEL PROBE ---
    # Get the center of the screen
    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2
    
    # Get the HSV value of the center pixel
    pixel_center = hsv[cy, cx]
    hue_value = pixel_center[0]
    sat_value = pixel_center[1]
    val_value = pixel_center[2]
    
    # Draw a circle in the center so you know where to aim
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 2)
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), params["draw_color"], 2)
                label = f"{color_name}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, params["draw_color"], 2)

    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        
        # Calculate the area size
        area = cv2.contourArea(c)
        
        # Only react if the object is big enough (ignores tiny background noise)
        if area > 500: 
            
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw a rectangle around the object on the main frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), params["draw_color"], 2)
            
            # Put text near the object
            label = f"{color_name} DETECTED"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, params["draw_color"], 2)
            
            # Print to console (Optional - can be spammy)
            print(color_name + " in frame!")
    
    # Add the frame rate to the images
    fps_label = f"CAMERA FPS: {fps:.2f}"
    proc_label = f"PROCESSING FPS: {1/processing_period:.2f}"

    images_to_label = [frame]

    for img in images_to_label:
        cv2.putText(img, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, proc_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    show_list = [("Colour Detection", frame)]
    for name, img in show_list:
        cv2.imshow(name, img)

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