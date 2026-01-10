# Import libraries
import socket
import struct
import time

# Constants - Receive Configuration
UDP_RECEIVE_IP = "172.26.118.176"  # Your PC's WiFi IP
UDP_RECEIVE_PORT = 50003  # Different port for PC to receive responses
BUFFER_SIZE = 1024

# Constants - Send Configuration
UDP_SEND_IP = "138.38.228.74"  # Raspberry Pi IP
UDP_SEND_PORT = 50002  # Pi is listening on this port

# ==================== SEND FUNCTION ====================
def send_udp(angle, dist, tom):
    """Send 3 doubles (64-bit floats) as UDP data: angle, dist, TOM"""
    udp_message = struct.pack('<ddd', float(angle), float(dist), float(tom))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(udp_message, (UDP_SEND_IP, UDP_SEND_PORT))
    print(f"Sent: angle={angle}, dist={dist}, TOM={tom}")

# ==================== RECEIVE FUNCTION ====================
def receive_udp_continuous():
    """Continuously listen for UDP data until non-zero value received"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_RECEIVE_IP, UDP_RECEIVE_PORT))
    sock.settimeout(2)
    
    print("Waiting for confirmation from Raspberry Pi", end="", flush=True)
    
    while True:
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            
            # Decode and check if non-zero
            if len(data) >= 8:
                received_value = struct.unpack('<d', data[:8])[0]
                
                # Only return if value is non-zero (ignore zeros)
                if received_value != 0.0:
                    sock.close()
                    print(f"\nReceived value: {received_value}")
                    return received_value
                # If zero, continue waiting without printing
                
        except socket.timeout:
            print(".", end="", flush=True)
            continue

# ==================== MAIN ====================
if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("Enter values to send to Raspberry Pi:")
        
        angle_input = input("Angle [default: 90]: ")
        angle = float(angle_input) if angle_input else 90.0
        
        dist_input = input("Distance [default: 0]: ")
        dist = float(dist_input) if dist_input else 0.0
        
        tom_input = input("TOM [default: 0]: ")
        tom = float(tom_input) if tom_input else 0.0
        
        # Send data
        send_udp(angle, dist, tom)
        
        # Wait for non-zero response from Raspberry Pi
        received_value = receive_udp_continuous()
        
        # Check if received value is 1
        if received_value == 1.0:
            print("Task complete! Sending stop signal [0, 0, 0]...")
            time.sleep(0.1)
            send_udp(0.0, 0.0, 0.0)
            time.sleep(0.3)
        else:
            print(f"Unexpected value {received_value}, exiting...")
            break
