#!/usr/bin/env python3
import socket

# Bind to 0.0.0.0. This tells Python: "Listen on ALL available network cards."
# This is much safer than hardcoding a specific IP.
UDP_IP = "0.0.0.0" 
UDP_PORT = 50002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening on all interfaces (0.0.0.0) at Port: {UDP_PORT}")
print("Press Ctrl+C to stop...")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        
        # 1. Print who sent it
        print(f"Received {len(data)} bytes from {addr}")
        
        # 2. Print raw bytes (Best for debugging Simulink)
        # This shows you exactly what Simulink sent (e.g., b'\x00\x00...')
        print(f"Raw hex: {data.hex()}")
        
        # 3. Try decoding ONLY if you are sure it's text
        try:
            print(f"As text: {data.decode('utf-8')}")
        except UnicodeDecodeError:
            print("As text: (Data is not valid UTF-8 text)")
        
        print("-" * 20)
        
    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print(f"Error: {e}")