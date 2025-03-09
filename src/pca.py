import pickle as pk
import os
import numpy as np


file_param = "./data/params.pkl"
file_mat = "./data/transmat.pkl"
file_data = "./data/data.pkl"

with open(file_param, 'wb') as f:
  pk.dump({
    "Sig": np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]]), 
    "mu": np.array([0.33, 0.33, 0.33])
    }, f)

with open(file_mat, 'wb') as f:
  pk.dump({
    "transmat": np.array([[0.9, 0.05, 0.05], 
                          [0.05, 0.9, 0.05], 
                          [0.05, 0.05, 0.9]])
    }, f)