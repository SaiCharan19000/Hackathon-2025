import cv2
import threading
import platform
import time
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

beeping = False
stop_beep_flag = False

def continuous_beep():
    global beeping, stop_beep_flag
    beeping = True
    while not stop_beep_flag:
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(1000, 500)  
        else:
            print("\a", end='', flush=True)
            time.sleep(0.5)
    beeping = False

def enhance_frame(frame):
    # Convert to YUV for brightness adjustment
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Equalize brightness channel
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Gamma correction for low light
    gamma = 1.5  # Higher = brighter
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)

    return frame

# Start webcam
cap = cv2.VideoCapture(0)

# Try to set auto exposure (if supported)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

closed_eyes_frames = 0
ALARM_THRESHOLD = 20  # number of frames to trigger alarm

print("✅ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Enhance for better detection in low light
    frame = enhance_frame(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_detected = len(faces) > 0
    eyes_detected = False

    if face_detected:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) > 0:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            break 

    if not face_detected:
        alert_reason = "😵 Face Not Detected!"
        if not beeping:
            stop_beep_flag = False
            threading.Thread(target=continuous_beep, daemon=True).start()

    elif not eyes_detected:
        closed_eyes_frames += 1

        if closed_eyes_frames >= ALARM_THRESHOLD:
            alert_reason = "😴 Eyes Closed!"
            if not beeping:
                stop_beep_flag = False
                threading.Thread(target=continuous_beep, daemon=True).start()
        else:
            alert_reason = None

    else:
        closed_eyes_frames = 0
        alert_reason = None
        if beeping:
            stop_beep_flag = True  

    
    if alert_reason:
        cv2.putText(frame, alert_reason, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "✅ All Good", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Drowsiness + Distraction Detection (Low Light Enhanced)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


stop_beep_flag = True
cap.release()
cv2.destroyAllWindows()