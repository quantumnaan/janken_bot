import cv2
import base64
import mediapipe as mp
import numpy as np

# import pickle as pk
# import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # カメラのキャプチャ
# OpenCVのウィンドウを作成

def capture_hand_one_frame():
  ret, frame = cap.read() # カメラから1フレームを取得
  if not ret:
    print("Error: Could not read frame from camera.")
    return "Error"

  # 手のジェスチャーを検出
  frame = cv2.flip(frame,1)
  gesture = detect_hand_gesture(frame)
  
  return gesture

def get_picture():
  ret, frame = cap.read() # カメラから1フレームを取得
  if not ret:
    print("Error: Could not read frame from camera.")
    return "Error"

  _, buffer = cv2.imencode('.jpg', frame)  # JPEG形式にエンコード
  return buffer.tobytes()  # バイナリ形式で返す

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

def calculate_angle(a, b, c):
  """
  Args:
    a: (x, y, z) 
    b: (x, y, z) 
    c: (x, y, z) 
  Returns:
    angle: (1,) 角度
  """
  ba = (a.x - b.x, a.y - b.y, a.z - b.z)
  bc = (c.x - b.x, c.y - b.y, c.z - b.z)
  
  dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
  norm_ba = (ba[0]**2 + ba[1]**2 + ba[2]**2) ** 0.5
  norm_bc = (bc[0]**2 + bc[1]**2 + bc[2]**2) ** 0.5
  
  angle = np.arccos(dot_product / (norm_ba * norm_bc)) * 180 / np.pi
  
  return angle
    
def classify_hand_gesture(hand_landmarks):
  wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
  
  hito_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
  hito_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
  
  naka_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
  naka_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
  
  kusuri_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
  kusuri_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
  
  hito_angle = calculate_angle(hito_tip, hito_base, wrist)
  naka_angle = calculate_angle(naka_tip, naka_base, wrist)
  kusuri_angle = calculate_angle(kusuri_tip, kusuri_base, wrist)

  th_angle = 120
  if hito_angle > th_angle and naka_angle > th_angle and kusuri_angle > th_angle:
    gesture = "パー"
  elif hito_angle < th_angle and naka_angle < th_angle and kusuri_angle < th_angle:
    gesture = "グー"
  elif hito_angle > th_angle and naka_angle > th_angle and kusuri_angle < th_angle:
    gesture = "チョキ"
  else:
    gesture = "Unknown"
    
  return gesture
    
def release_resources():
  cap.release()
  cv2.destroyAllWindows()
  
def check_cam():
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # BGR → RGBに変換（MediaPipeはRGBで処理）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手を検出
    results = hands.process(rgb_frame)

    # 手が検出されていれば描画
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 画面に表示
    cv2.imshow("MediaPipe Hands Test", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

def generate_frames():
  while True:
    ret, frame = cap.read()  # カメラからフレームを取得
    if not ret:
      print("Error: Could not read frame from camera.")
      break

    # フレームをJPEG形式にエンコード
    _, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    # フレームをHTTPレスポンスとして送信
    yield (b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
if __name__ == "__main__":
  check_cam()
  release_resources()