import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import struct

# ============================================================
# EDIT THESE SETTINGS
# ============================================================

# The ID on the robot's ArUco tag:
ROBOT_ID = 0

# Your arena markers:
HOME_ID = 1
DOOR_IDS = [2, 3, 4]
SPECIAL_DOOR_ID = 4
QUARANTINE_ID = 5

# UDP
LAPTOP_BIND_IP = "0.0.0.0"
LAPTOP_BIND_PORT = 50003

#PI_IP = "138.38.228.230"   # <-- CHANGE to your Pi IP
PI_IP = "172.26.236.13"
PI_PORT = 50002

# Behavior
WAIT_DURATION_SEC = 10.0
TARGET_REACH_THRESHOLD_MM = 200
DIST_STOP_MM = 200
ANGLE_TOLERANCE_DEG = 7
MARKER_SIZE_MM = 105

WAIT_REQUIRES_UDP_SIGNAL = False
USE_ESCAPE_MODE_CODE = False
ESCAPE_MODE_CODE = 5.0

# Only consider these IDs
ALLOWED_IDS = set([ROBOT_ID, HOME_ID, QUARANTINE_ID] + DOOR_IDS)

# ============================================================
# UDP SETUP
# ============================================================
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_receive.bind((LAPTOP_BIND_IP, LAPTOP_BIND_PORT))
sock_receive.setblocking(False)
print(f"Listening for UDP on {LAPTOP_BIND_IP}:{LAPTOP_BIND_PORT}")
print(f"Sending control UDP to {PI_IP}:{PI_PORT}")

# --- NEW VARIABLES FOR SPACEBAR LOGIC ---
# These hold the calculated values from the camera (updated constantly)
current_angle = 0.0
current_dist = 0.0
current_code = 3.0 # Default to STOP

# These hold what was actually sent last time we pressed space
sent_angle = 0.0
sent_dist = 0.0
sent_code = 3.0

def send_packet_now(angle_deg: float, dist_mm: float, mode_code: float):
    """Sends the packet immediately. Only called when Spacebar is pressed."""
    try:
        msg = struct.pack("<ddd", float(angle_deg), float(dist_mm), float(mode_code))
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(msg, (PI_IP, PI_PORT))
        s.close()
        print(f"SENT UDP -> Ang: {angle_deg:.0f}, Dist: {dist_mm:.0f}, Code: {mode_code:.0f}")
    except Exception as e:
        print(f"UDP Error: {e}")

def try_read_udp_signal_one() -> bool:
    try:
        data, _addr = sock_receive.recvfrom(1024)
        if len(data) >= 8:
            val = struct.unpack("<d", data[:8])[0]
            return (val == 1.0)
    except Exception:
        pass
    return False

# ============================================================
# UI
# ============================================================
def draw_status(frame, text, y=60):
    cv2.putText(frame, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

def draw_udp_values(frame, curr_a, curr_d, curr_c, last_a, last_d, last_c):
    # Yellow: What the camera sees (Pending)
    txt_curr = f"PENDING (Press Space): Ang {curr_a:.0f} | Dist {curr_d:.0f} | Code {curr_c:.0f}"
    cv2.putText(frame, txt_curr, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Pink: What was actually sent last
    txt_sent = f"LAST SENT:             Ang {last_a:.0f} | Dist {last_d:.0f} | Code {last_c:.0f}"
    cv2.putText(frame, txt_sent, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

def draw_battery_ui(frame, progress_0_to_1: float, title="SCREAM ENERGY"):
    progress = float(np.clip(progress_0_to_1, 0.0, 1.0))
    x, y = 40, 220
    bw, bh = 420, 80

    cv2.putText(frame, title, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 3)
    cv2.rectangle(frame, (x + bw + 8, y + 20), (x + bw + 28, y + bh - 20), (255, 255, 255), 3)

    fill_w = int((bw - 10) * progress)
    cv2.rectangle(frame, (x + 5, y + 5), (x + 5 + fill_w, y + bh - 5), (0, 255, 0), -1)

    pct = int(progress * 100)
    cv2.putText(frame, f"{pct}%", (x + bw + 45, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

def draw_visible_ids(frame, ids_list):
    txt = "Visible IDs: " + (", ".join(map(str, ids_list)) if ids_list else "none")
    cv2.putText(frame, txt, (40, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# ============================================================
# CAMERA
# ============================================================
def open_camera():
    for idx in [1]:
        cap_try = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap_try.isOpened():
            cap_try.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap_try.read()
            if ret and frame is not None:
                print(f"Camera opened: index={idx} backend=CAP_DSHOW")
                return cap_try
            cap_try.release()
    raise RuntimeError("Could not open camera.")

cap = open_camera()

# ============================================================
# Calibration
# ============================================================
CM = None
dist_coef = np.zeros((5, 1), dtype=np.float32)
try:
    calib = np.load("workdir/Calibration.npz")
    CM = calib["CM"].astype(np.float32)
    dist_coef = calib["dist_coef"].astype(np.float32)
except Exception:
    pass

# ============================================================
# ArUco detect
# ============================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
aruco_detector = aruco.ArucoDetector(aruco_dict, parameters) if hasattr(aruco, "ArucoDetector") else None

def detect_markers(gray_img):
    if hasattr(aruco, "detectMarkers"):
        return aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
    else:
        return aruco_detector.detectMarkers(gray_img)

# ============================================================
# Pose
# ============================================================
half = (MARKER_SIZE_MM / 2.0)
objp = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

def marker_pose_from_corners(corners_2d, CM_use):
    imgp = corners_2d.astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, CM_use, dist_coef, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else None

# ============================================================
# NAV
# ============================================================
STATE_GO_TO_GOAL = "GO_TO_GOAL"
STATE_WAITING = "WAITING"

state = STATE_GO_TO_GOAL
door_index = 0
goal_id = DOOR_IDS[door_index]

escape_active = False
escape_path = []
escape_step = 0

waiting_at_id = None
wait_start = 0.0
wait_started = False
wait_udp_ok = False

# ============================================================
# Control helpers
# ============================================================
def decide_mode_code(angle_deg, dist_mm, escape_mode=False):
    angle_rounded = float(round(angle_deg))
    dist_rounded = float(round(dist_mm))
    if escape_mode and USE_ESCAPE_MODE_CODE: return angle_rounded, dist_rounded, ESCAPE_MODE_CODE
    code = 1.0 if abs(angle_rounded) > ANGLE_TOLERANCE_DEG else 2.0
    if dist_rounded <= DIST_STOP_MM: code = 3.0
    return angle_rounded, dist_rounded, code

def compute_angle_and_distance(pose_by_id, target_id):
    if ROBOT_ID not in pose_by_id or target_id not in pose_by_id: return None
    rvec_r, tvec_r, center_r = pose_by_id[ROBOT_ID]
    rvec_t, tvec_t, center_t = pose_by_id[target_id]
    pos_robot = tvec_r.reshape(3)
    pos_target = tvec_t.reshape(3)
    dist_mm = float(np.linalg.norm(pos_robot - pos_target))
    rmat, _ = cv2.Rodrigues(rvec_r)
    orientation_vec = rmat[:, 1]
    line_vec = (pos_target - pos_robot)
    unit_orient = orientation_vec / (np.linalg.norm(orientation_vec) + 1e-9)
    unit_line = line_vec / (np.linalg.norm(line_vec) + 1e-9)
    normal_vec = rmat[:, 2]
    dot_prod = np.dot(unit_orient, unit_line)
    cross_prod = np.cross(unit_orient, unit_line)
    sine_component = np.dot(cross_prod, normal_vec)
    angle_deg = float(np.degrees(np.arctan2(sine_component, dot_prod)))
    return angle_deg, dist_mm, center_r, center_t

# ============================================================
# MAIN LOOP
# ============================================================
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 1920, 1080)

print("=== START ===")
print("PRESS SPACEBAR TO SEND UDP COMMANDS")
print(f"Initial goal: {goal_id}")

while True:
    ret, frame = cap.read()
    if not ret or frame is None: continue

    if CM is None:
        h, w = frame.shape[:2]
        fx = fy = 0.9 * w
        cx, cy = w / 2.0, h / 2.0
        CM = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detect_markers(gray)

    pose_by_id = {}
    visible_ids = []

    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten().tolist()
        keep = []
        keep_ids = []
        for i, mid in enumerate(ids_flat):
            if int(mid) in ALLOWED_IDS:
                keep.append(corners_list[i])
                keep_ids.append(int(mid))
        if keep_ids:
            frame = aruco.drawDetectedMarkers(frame, keep, np.array(keep_ids, dtype=np.int32).reshape(-1, 1))
            visible_ids = keep_ids
            for i, marker_id in enumerate(keep_ids):
                c = keep[i][0]
                center = tuple(np.mean(c, axis=0).astype(int))
                pose = marker_pose_from_corners(c, CM)
                if pose:
                    rvec, tvec = pose
                    pose_by_id[int(marker_id)] = (rvec, tvec, center)
                    cv2.drawFrameAxes(frame, CM, dist_coef, rvec, tvec, 20)

    draw_visible_ids(frame, visible_ids)

    # -------------------------------------------------------------------------
    # LOGIC UPDATE (Calculates "current_" vars, but does NOT send UDP)
    # -------------------------------------------------------------------------
    if state == STATE_WAITING:
        # We prepare a STOP command while waiting
        current_angle, current_dist, current_code = 0.0, 0.0, 3.0

        if WAIT_REQUIRES_UDP_SIGNAL and not wait_udp_ok:
            if try_read_udp_signal_one():
                wait_udp_ok = True
                wait_start = time.time()
                wait_started = True
            draw_status(frame, f"AT {waiting_at_id}: WAITING FOR UDP '1.0' ...", y=60)
            draw_battery_ui(frame, 0.0)
        else:
            if not wait_started:
                wait_start = time.time()
                wait_started = True
            elapsed = time.time() - wait_start
            progress = min(1.0, elapsed / WAIT_DURATION_SEC)
            remaining = max(0.0, WAIT_DURATION_SEC - elapsed)
            draw_status(frame, f"AT {waiting_at_id}: CHARGING {remaining:0.1f}s", y=60)
            draw_battery_ui(frame, progress)

            if elapsed >= WAIT_DURATION_SEC:
                if waiting_at_id == SPECIAL_DOOR_ID:
                    escape_active = True
                    escape_path = [HOME_ID, QUARANTINE_ID]
                    escape_step = 0
                    goal_id = escape_path[escape_step]
                elif waiting_at_id == QUARANTINE_ID:
                    goal_id = HOME_ID
                else:
                    goal_id = HOME_ID
                state = STATE_GO_TO_GOAL
                waiting_at_id = None
                wait_started = False
                wait_udp_ok = False

    else:
        # GO TO GOAL
        mode_txt = "ESCAPE" if escape_active else "NORMAL"
        draw_status(frame, f"MODE: {mode_txt} | GOAL: {goal_id}", y=60)

        res = compute_angle_and_distance(pose_by_id, goal_id)
        if res is None:
            draw_status(frame, f"Robot/goal not visible — STOP", y=110)
            # Logic says STOP if lost
            current_angle, current_dist, current_code = 0.0, 0.0, 3.0
        else:
            angle_deg, dist_mm, center_r, center_t = res
            cv2.line(frame, center_r, center_t, (255, 0, 0), 3)
            cv2.putText(frame, f"Dist {dist_mm:.0f}mm  Ang {angle_deg:.0f}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if dist_mm < TARGET_REACH_THRESHOLD_MM:
                # Target reached -> prepare STOP
                current_angle, current_dist, current_code = 0.0, 0.0, 3.0
                
                # Logic to switch goals (happens automatically in memory, but movement relies on spacebar)
                if goal_id in DOOR_IDS or goal_id == QUARANTINE_ID:
                    state = STATE_WAITING
                    waiting_at_id = goal_id
                    wait_started = False
                    wait_udp_ok = (not WAIT_REQUIRES_UDP_SIGNAL)
                elif goal_id == HOME_ID:
                    if escape_active:
                        if escape_step < len(escape_path) - 1:
                            escape_step += 1
                            goal_id = escape_path[escape_step]
                        else:
                            escape_active = False
                            escape_path = []
                            escape_step = 0
                            door_index = (door_index + 1) % len(DOOR_IDS)
                            goal_id = DOOR_IDS[door_index]
                    else:
                        door_index = (door_index + 1) % len(DOOR_IDS)
                        goal_id = DOOR_IDS[door_index]
            else:
                # Normal Navigation
                ang_r, dist_r, code = decide_mode_code(angle_deg, dist_mm, escape_mode=escape_active)
                current_angle, current_dist, current_code = ang_r, dist_r, code

    # -------------------------------------------------------------------------
    # UI AND INPUT
    # -------------------------------------------------------------------------
    draw_udp_values(frame, current_angle, current_dist, current_code, sent_angle, sent_dist, sent_code)
    
    cv2.imshow("Frame", frame)
    
    # Check Keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == 32: # SPACEBAR (ASCII 32)
        # Send current values NOW
        send_packet_now(current_angle, current_dist, current_code)
        # Update "last sent" memory
        sent_angle, sent_dist, sent_code = current_angle, current_dist, current_code

cap.release()
cv2.destroyAllWindows()