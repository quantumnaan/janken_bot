import cv2
import mediapipe as mp

# from hand_detector import HandDetector
# import pickle as pk
# import numpy as np

# detector = HandDetector()
# detector.load_model()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # カメラのキャプチャ
# OpenCVのウィンドウを作成
# cv2.namedWindow('Hand Game', cv2.WINDOW_NORMAL)

def capture_hand_one_frame():
  ret, frame = cap.read() # カメラから1フレームを取得
  if not ret:
    print("Error: Could not read frame from camera.")
    return "Error"

  # 手のジェスチャーを検出
  frame = cv2.flip(frame,1)
  gesture = detect_hand_gesture(frame)
  
  # cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  # cv2.imshow("Hand Game", frame)
  
  return gesture

def detect_hand_gesture(frame):
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)

  gesture = "Unknown"
  if results.multi_hand_landmarks:
    if len(results.multi_hand_landmarks) > 1:
      print("Multiple hands detected, only processing the first one.")
    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    gesture = classify_hand_gesture(hand_landmarks)
  else:
    print("No hands detected.")
  
  return gesture
    
def classify_hand_gesture(hand_landmarks):
  oya_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
  oya_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
  hito_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
  hito_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
  naka_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
  naka_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
  kusuri_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
  kusuri_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
  ko_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
  ko_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
  wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
  
  # hand_vect = np.array([
    

  
  THRESHOLD = 0.05
  if hito_tip.y - hito_base.y > -THRESHOLD \
      and naka_tip.y - naka_base.y > -THRESHOLD \
      and kusuri_tip.y - kusuri_base.y > -THRESHOLD \
      and ko_tip.y - ko_base.y > -THRESHOLD:
    gesture = "グー"
  elif hito_tip.y - hito_base.y < -THRESHOLD \
        and naka_tip.y - naka_base.y < -THRESHOLD \
        and kusuri_tip.y - kusuri_base.y < -THRESHOLD \
        and ko_tip.y - ko_base.y < -THRESHOLD:
    gesture = "パー"
  elif hito_tip.y - hito_base.y < -THRESHOLD \
        and naka_tip.y - naka_base.y < -THRESHOLD \
        and kusuri_tip.y - kusuri_base.y > -THRESHOLD \
        and ko_tip.y - ko_base.y > -THRESHOLD:
    gesture = "チョキ"
  else:
    gesture = "Unknown"
    
  return gesture
    
def release_resources():
  cap.release()
  cv2.destroyAllWindows()