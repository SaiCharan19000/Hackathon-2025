import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import platform

# -------- Mediapipe FaceMesh Setup --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- Beep Control --------
beeping = False
stop_beep_flag = False

def continuous_beep():
    global beeping, stop_beep_flag
    while not stop_beep_flag:
        if beeping:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(2000, 500)
            else:
                print('\a')
        time.sleep(0.5)

threading.Thread(target=continuous_beep, daemon=True).start()

def start_beep():
    global beeping
    beeping = True

def stop_beep():
    global beeping
    beeping = False

# -------- Drowsiness Detection (Eye Closure Ratio) --------
def is_eye_closed(landmarks, eye_indices, w, h, threshold=0.25):
    # vertical distance between upper and lower eyelids
    top_y = (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2
    bottom_y = (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2
    vertical_dist = (bottom_y - top_y) * h

    # horizontal distance between eye corners
    left_x = landmarks[eye_indices[0]].x
    right_x = landmarks[eye_indices[3]].x
    horizontal_dist = (right_x - left_x) * w

    ratio = vertical_dist / horizontal_dist
    return ratio < threshold

# -------- Distraction Detection (Head Turn Angle) --------
def get_head_turn_angle(landmarks, w, h):
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])
    left_cheek = np.array([landmarks[234].x * w, landmarks[234].y * h])
    right_cheek = np.array([landmarks[454].x * w, landmarks[454].y * h])

    face_width = np.linalg.norm(left_cheek - right_cheek)
    nose_to_left = np.linalg.norm(nose_tip - left_cheek)
    ratio = nose_to_left / face_width

    return abs(0.5 - ratio) * 200  # pseudo-degrees for deviation from center

# -------- Thresholds and Counters --------
EYE_CLOSED_THRESHOLD = 0.25
EYE_CLOSED_CONSEC_FRAMES = 10

DISTRACTION_ANGLE_THRESHOLD = 10  # degrees
DISTRACTION_CONSEC_FRAMES = 10

drowsy_counter = 0
distract_counter = 0

# -------- Main Loop --------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "Focused"
    stop_beep()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eye landmark indices for approximate upper/lower/lateral points
        left_eye_indices = [33, 159, 160, 133, 144, 153]
        right_eye_indices = [362, 385, 386, 263, 374, 380]

        left_eye_closed = is_eye_closed(landmarks, left_eye_indices, w, h)
        right_eye_closed = is_eye_closed(landmarks, right_eye_indices, w, h)

        # Drowsiness check
        if left_eye_closed and right_eye_closed:
            drowsy_counter += 1
        else:
            if drowsy_counter > 0:
                drowsy_counter -= 1

        # Distraction check
        head_angle = get_head_turn_angle(landmarks, w, h)
        if head_angle > DISTRACTION_ANGLE_THRESHOLD:
            distract_counter += 1
        else:
            distract_counter = 0

        # Determine status and beep alerts
        if drowsy_counter >= EYE_CLOSED_CONSEC_FRAMES:
            status = "Drowsy"
            start_beep()
        elif distract_counter >= DISTRACTION_CONSEC_FRAMES:
            status = "Distracted"
            start_beep()
        else:
            stop_beep()

        # Display debugging info for counters and angle
        cv2.putText(frame, f"Drowsy Count: {drowsy_counter}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Head Angle: {head_angle:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Distract Count: {distract_counter}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Display status with color
    color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Driver Monitoring: Drowsiness & Distraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_beep_flag = True
cap.release()
cv2.destroyAllWindows()