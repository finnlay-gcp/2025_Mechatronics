# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities. We will use this to ensure a steady processing rate
import time 

# Define a processing rate
processing_period = 0.25

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

    # define range of blue color in HSV
    # H=hue, S=saturation, V=value
    # Less saturation = more white, less value = more black
    lower_blue = np.array([90,100,100])
    upper_blue = np.array([150,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
 
    # Object recognition logic
    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected = False

    # If any contours are found
    if len(contours) > 0:
        # Find the largest contour (the biggest blue blob)
        c = max(contours, key=cv2.contourArea)
        
        # Calculate the area size
        area = cv2.contourArea(c)
        
        # Only react if the object is big enough (ignores tiny background noise)
        if area > 500: 
            object_detected = True
            
            # Get the bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            
            # Draw a rectangle around the object on the main frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Put text near the object
            cv2.putText(frame, "TARGET DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Print to console (Optional - can be spammy)
            print("Object in frame!")
    
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