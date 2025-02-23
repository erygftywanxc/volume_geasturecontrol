import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Get system audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[:2]
current_vol = volume.GetMasterVolumeLevel()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            index_tip = (landmarks[8].x * frame.shape[1], landmarks[8].y * frame.shape[0])
            thumb_tip = (landmarks[4].x * frame.shape[1], landmarks[4].y * frame.shape[0])
            wrist = (landmarks[0].x * frame.shape[1], landmarks[0].y * frame.shape[0])
            
            angle = calculate_angle(thumb_tip, wrist, index_tip)
            
            # Map angle to volume range
            new_vol = np.interp(angle, [30, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(new_vol, None)
            
            # Display volume percentage
            vol_percent = np.interp(new_vol, [min_vol, max_vol], [0, 100])
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("genxC - Gesture Volume Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
