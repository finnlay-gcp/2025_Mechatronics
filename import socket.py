import socket
import RPi.GPIO as GPIO

# Configuration
UDP_IP = "0.0.0.0" # Listen on all available interfaces
UDP_PORT = 5005    # -----------------------------------------------------------------------------------------Must match the sender port
LASER_PIN = 17     # -----------------------------------------------------------------------------------------The GPIO pin your laser is connected to

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(LASER_PIN, GPIO.OUT)
GPIO.output(LASER_PIN, GPIO.LOW)

# Network Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP commands on port {UDP_PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
        message = data.decode('utf-8')
        
        if message == "ON":
            GPIO.output(LASER_PIN, GPIO.HIGH)
            print("Received: ON -> Laser High")
        elif message == "OFF":
            GPIO.output(LASER_PIN, GPIO.LOW)
            print("Received: OFF -> Laser Low")

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    GPIO.cleanup()