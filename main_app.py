import numpy as np
import pickle as pk
from flask import Flask, render_template, url_for
from flask_socketio import SocketIO, send
import cv2
import base64

import os
import sys
sys.path.append(os.path.abspath("src"))

from vae import VAE
from camera_stream import detect_hand_gesture
from utils import *

file_data = "./data/data.pkl"

app = Flask(__name__)
socketio = SocketIO(app)
vae = VAE()
vae.load_model()
cnt = 0

ones_data = []


@app.route("/")
def index():
  return render_template("index.html")

@socketio.on("choose")
def choose(choices):
  """
  Args:
    choice: 人が出した手
    cp_choice: CPが出した手
  """
  choice = choices["var1"]
  cp_choice = choices["var2"]
  print("いい手を選ぶぞ!")
  ones_data.append((choice, cp_choice))
  data_mat = make_data_mat(np.array(ones_data)).view(-1, 3*NS)
  prob_mat = vae(data_mat)[0].view(-1, 3, NS)
  state = make_state(choice, cp_choice)
  prob_next = prob_mat[:, :, state].detach().numpy().flatten()
  
  print(f"相手の1つ前の手が {choice} で次出す手の確率が {prob_next} だから..")
  choise = (np.argmax(prob_next) + 2 )%3
  print(f"{choise} を選びました")
  return int(choise)

@socketio.on("reset")
def reset():
  ones_data.clear()
  cnt += 1
  if(cnt%10==0):{
    vae.load_model()
  }
  
@socketio.on("save_data")
def save_data():
  with open(file_data, 'ab') as f:
    pk.dump(ones_data, f)
  return "saved"

@socketio.on("video_frame")
def handle_video(data):
  # 画像データをデコード
  img_data = base64.b64decode(data['image'])
  np_arr = np.frombuffer(img_data, np.uint8)
  frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  
  # 手のジェスチャーを検出
  gesture = detect_hand_gesture(frame)
  
  # 結果をクライアントに送信
  socketio.emit("gesture", {"gesture": gesture})

if __name__ == "__main__":
  socketio.run(app, debug=True)