import numpy as np
import pickle as pk
import os
from flask import Flask, render_template
from flask_socketio import SocketIO

from src.bayes_estimation import BayesEstimation

file_data = "./data/data.pkl"

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def index():
  return render_template("index.html")

@socketio.on("estimate")
def estimate(data):
  x = np.array(data["x"])
  
  return x

if __name__ == "__main__":
  bayes_estimation = BayesEstimation()
  socketio.run(app, debug=True)