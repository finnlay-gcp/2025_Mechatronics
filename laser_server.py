import socket
import RPi.GPIO as GPIO

# --- CONFIGURATION ---
LASER_PIN = 17
LISTEN_PORT = 5005
# ---------------------

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(LASER_PIN, GPIO.OUT)
GPIO.output(LASER_PIN, GPIO.LOW)

# Networking Setup (UDP)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 0.0.0.0 means "Listen for connections from anyone on the network"
sock.bind(("0.0.0.0", LISTEN_PORT))

print(f"Listening for laser commands on Port {LISTEN_PORT}...")

try:
    while True:
        # data = the message (e.g., b'ON'), addr = IP address of your PC
        data, addr = sock.recvfrom(1024) 
        message = data.decode('utf-8')

        if message == "ON":
            GPIO.output(LASER_PIN, GPIO.HIGH)
            print(f"Command from {addr}: LASER ON")
        elif message == "OFF":
            GPIO.output(LASER_PIN, GPIO.LOW)
            print(f"Command from {addr}: LASER OFF")

except KeyboardInterrupt:
    print("\nStopping server...")
finally:
    GPIO.cleanup()
    sock.close()