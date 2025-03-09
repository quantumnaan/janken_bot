import numpy as np
import pickle as pk
import os
from flask import Flask
from flask_socketio import SocketIO

from bayes_estimation import BayesEstimation

file_data = "./data/data.pkl"

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on("estimate")
def estimate(data):
  x = np.array(data["x"])
  est = bayes_estimation.estimate(x)
  return est.tolist()