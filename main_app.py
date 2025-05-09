import numpy as np
import csv
from flask import Flask, render_template, url_for, jsonify
from flask_socketio import SocketIO, send
import time

import os
import sys
sys.path.append(os.path.abspath("src"))
# sys.path.append(os.path.abspath(".."))
from vae import VAE
from camera_stream import capture_hand_one_frame, get_picture
from utils import *

# TODO: ゲージたまった少し後，手の認識詰める

file_data = "./data/data.csv"

app = Flask(__name__)
socketio = SocketIO(app)
vae = VAE()
vae.load_model()
cnt_play = 0 # プレイした人の数

ones_data = []

from flask import Response
from camera_stream import generate_frames

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
  return render_template("index.html")

@socketio.on("update_picture")
def update_picture():
  img_binary = get_picture()
  socketio.emit("updated_picture", img_binary)

@socketio.on("choose")
def choose(choices):
  """
  Args:
    choices['var1']: 人が出した手
    choices['var2']: CPが出した手
  """
  global ones_data
  choice = choices["var1"]
  cp_choice = choices["var2"]
  print("いい手を選ぶぞ!")
  ones_data.append((choice, cp_choice))
  data_mat = make_data_mat(np.array(ones_data))
  z_star = vae.map_z(data_mat)
  prob_mat = vae.decoder(z_star).view(-1, 3, NS)
  state = make_state(choice, cp_choice)
  prob_next = prob_mat[:, :, state].detach().numpy().flatten()
  
  print(f"相手の1つ前の手が {choice} で次出す手の確率が {prob_next} だから..")
  choise = (np.argmax(prob_next) + 2 )%3
  print(f"{choise} を選びました")
  return int(choise)

@socketio.on("reset")
def reset():
  global cnt_play, ones_data
  if(len(ones_data) > 0):
    ones_data.clear()
    cnt_play = cnt_play + 1
    if(cnt_play%10==0):
      vae.load_model()
      
@socketio.on("save_data")
def save_data():
  global ones_data
  if (len(ones_data) > 0):
    with open(file_data, 'ab') as f:    
      writer = csv.writer(f)
      # データを書き込む
      row = np.array(ones_data).astype(np.int8)
      writer.writerow(row.flatten())
    print(f"{len(ones_data)}ターン分のデータを保存しました")
    data_mat = make_data_mat(np.array(ones_data)).view(-1, 3*NS)
    prob_mat = vae(data_mat)[0].view(-1, 3, NS).detach().numpy()
    print(f"data_mat:\n {data_mat.view(3, NS).detach().numpy()}\n")
    print(f"prob_mat:\n {prob_mat}")
  
    socketio.emit("save_done")

@socketio.on("capture_hand")
def capture_hand():
  cap_time = 5 # cap_time回撮影し，多数決
  
  gestures = []
  for i in range(cap_time):
    gestures.append(capture_hand_one_frame())
    time.sleep(0.01) # 10ms待つ
  
  if gestures.count("グー") > cap_time / 2:
    gesture = "グー"
  elif gestures.count("パー") > cap_time / 2:
    gesture = "パー"
  elif gestures.count("チョキ") > cap_time / 2:
    gesture = "チョキ"
  else:
    gesture = "Unknown"
  
  # 結果をクライアントに送信
  socketio.emit("capture_done", {"gesture": gesture})

if __name__ == "__main__":
  socketio.run(app, debug=True)