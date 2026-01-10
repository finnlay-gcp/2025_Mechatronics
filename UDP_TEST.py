import socket
import struct
import time

# --- CONFIGURATION ---
RPI_IP = "138.38.228.74"  # <--- Ensure this matches your Pi's IP
RPI_PORT = 50002          # <--- Ensure this matches the listening port
SEND_INTERVAL = 10       # Send message every 10 second

# --- CONSTANT VALUES TO SEND ---
# Change these numbers to test different scenarios on your Pi
TEST_ANGLE = 90.0       # Represents angle_rounded
TEST_DIST = 0.0       # Represents dist_rounded
TEST_MODE = 1.0         # Represents turn_or_move

# --- NETWORK SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"UDP Sender Started.")
print(f"Target: {RPI_IP}:{RPI_PORT}")
print(f"Sending values -> Angle: {TEST_ANGLE}, Dist: {TEST_DIST}, Mode: {TEST_MODE}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        # --- PACKET CREATION ---
        # Matches your original struct format: '<ddd' (Little Endian, 3 doubles)
        udp_message = struct.pack('<ddd', TEST_ANGLE, TEST_DIST, TEST_MODE)
        
        # --- SEND MESSAGE ---
        sock.sendto(udp_message, (RPI_IP, RPI_PORT))
        
        print(f"Message sent at {time.strftime('%H:%M:%S')}")
        
        # --- WAIT ---
        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    print("\nStopping UDP Sender.")
    sock.close()