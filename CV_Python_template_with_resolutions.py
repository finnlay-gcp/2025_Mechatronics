# This is the vision library OpenCV
import cv2
# This is a library for mathematical functions for python (used later)
import numpy as np
# This is a library to get access to time-related functionalities
import time

def probe_resolutions(cap, candidates=None, wait=0.1):
    """Try setting each candidate resolution and read back what the camera accepted."""
    if candidates is None:
        candidates = [
            (1920,1080),(1600,1200),(1280,1024),(1280,960),
            (1280,720),(1024,768),(800,600),(640,480),
            (640,360),(320,240)
        ]
    supported = []
    for w,h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(wait)  # give camera/driver a moment
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            supported.append((w,h))
        else:
            # some cameras report scaled/nearest size; treat equals as support
            # you can include near matches if desired
            pass
    return supported

# Select the first camera (0) that is connected to the machine
# in Laptops should be the build-in camera
cap = cv2.VideoCapture(1)

# Probe before forcing other settings
print("Probing common resolutions...")
supported = probe_resolutions(cap)
print("Supported resolutions (exact matches):", supported)

# Set the width and heigth of the camera to 1920x1080
cap.set(3,1920)
cap.set(4,1080)

# Create three opencv named windows with half the screen size
window_width = 768
window_height = 432
cv2.namedWindow("frame-image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame-image", window_width, window_height)
cv2.namedWindow("gray-image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("gray-image", window_width, window_height)
cv2.namedWindow("canny-image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("canny-image", window_width, window_height)

# Position the windows in quadrants of the screen
cv2.moveWindow("frame-image", 0, 0)
cv2.moveWindow("gray-image", 780, 0)
cv2.moveWindow("canny-image", 0, 510)

# Execute this continuously
while(True):
    
    # Start the performance clock
    start = time.perf_counter()
    
    # Capture current frame from the camera
    ret, frame = cap.read()
    
    # Convert the image from the camera to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Example of additional image processing: Gaussian Blur
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    # Example of additional image processing: Canny Edge Detection
    canny = cv2.Canny(gauss, 75, 100)

    # Display the original frame in a window
    cv2.imshow('frame-image',frame)

    # Display the grey image in another window
    cv2.imshow('gray-image',gray)
    
     # Display the canny image in another window
    cv2.imshow('canny-image',canny)

    # Stop the performance counter
    end = time.perf_counter()
    
    # Print to console the exucution time in FPS (frames per second)
    print ('{:4.1f}'.format(1/(end - start)))

    # If the button q is pressed in one of the windows 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        # Exit the While loop
        break
    

# When everything done, release the capture
cap.release()
# close all windows
cv2.destroyAllWindows()
# exit the kernel
exit(0)