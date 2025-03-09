import numpy as np
import pickle as pk
import os
from flask import Flask, render_template, request, jsonify
import base64
from flask_socketio import SocketIO, send

from src.bayes_estimation import BayesEstimation

file_data = "./data/data.pkl"

app = Flask(__name__)
socketio = SocketIO(app)
bayes_estimation = BayesEstimation()

ones_data = []

def data2indices(datas):
  ret = []
  for i in range(len(datas)-1):
    ret.append(datas[i+1] + datas[i]*3)
  return np.array(ret)  

@app.route("/")
def index():
  return render_template("index.html")

@socketio.on("choose")
def choose(data):
  print("いい手を選ぶぞ!")
  ones_data.append(data)
  states = data2indices(ones_data)
  probs = bayes_estimation.estimate(states)
  print(f"相手の1つ前の手が {data} で {probs} だから..")
  choise = (np.argmax(probs[3*data:3*data+3]) + 2 )%3
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