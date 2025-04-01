import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0) # カメラのキャプチャ

def capture_hand_one_frame():
  ret, frame = cap.read() # カメラから1フレームを取得
  if not ret:
    print("Error: Could not read frame from camera.")
    return "Error"

  # 手のジェスチャーを検出
  gesture = detect_hand_gesture(frame)
  
  return gesture

def detect_hand_gesture(frame):
  hands = mp_hands.Hands()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)

  gesture = "Unknown"
  if results.multi_hand_landmarks:
    if len(results.multi_hand_landmarks) > 1:
      print("Multiple hands detected, only processing the first one.")
    hand_landmarks = results.multi_hand_landmarks[0]

    gesture = classify_hand_gesture(hand_landmarks)
  
  return gesture
    
def classify_hand_gesture(hand_landmarks):
  hito_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
  hito_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
  naka_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
  naka_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
  kusuri_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
  kusuri_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
  ko_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
  ko_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
  
  THRESHOLD = 0.02
  if hito_tip.y - hito_base.y < THRESHOLD \
      and naka_tip.y - naka_base.y < THRESHOLD \
      and kusuri_tip.y - kusuri_base.y < THRESHOLD \
      and ko_tip.y - ko_base.y < THRESHOLD:
    gesture = "グー"
  elif hito_tip.x - hito_base.x > THRESHOLD \
        and naka_tip.x - naka_base.x > THRESHOLD \
        and kusuri_tip.x - kusuri_base.x > THRESHOLD \
        and ko_tip.x - ko_base.x > THRESHOLD:
    gesture = "パー"
  elif hito_tip.y - hito_base.y < THRESHOLD \
        and naka_tip.y - naka_base.y < THRESHOLD \
        and kusuri_tip.y - kusuri_base.y > THRESHOLD \
        and ko_tip.y - ko_base.y > THRESHOLD:
    gesture = "チョキ"
  else:
    gesture = "Unknown"