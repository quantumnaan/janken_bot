import numpy as np
import pickle as pk
import os
from flask import Flask, render_template, url_for
import base64
from flask_socketio import SocketIO, send

import os
import sys
sys.path.append(os.path.abspath("src"))

from vae import VAE
from utils import *

file_data = "./data/data.pkl"

app = Flask(__name__)
socketio = SocketIO(app)
vae = VAE()
vae.load_model()

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
  
@socketio.on("save_data")
def save_data():
  # TODO: 上書きしない
  with open(file_data, 'wb') as f:
    pk.dump(ones_data, f)
  return "saved"

if __name__ == "__main__":
  socketio.run(app, debug=True)