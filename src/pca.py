import pickle as pk
import os
import numpy as np
from sklearn.decomposition import PCA


file_param = "./data/params.pkl"
file_mat = "./data/transmat.pkl"
file_data = "./data/data.pkl"

def make_data(file_data):
  with open(file_data, "rb") as f:
    # dataは(前の手,前の勝敗,次の手)のリスト
    data = pk.load(f)
  
  

def data_pca(data):
  pca = PCA(n_components=3)
  pca.fit(data)
  return pca

with open(file_param, 'wb') as f:
  pk.dump({
    "Sig": np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]]), 
    "mu": np.array([0.33, 0.33, 0.33])
    }, f)

with open(file_mat, 'wb') as f:
  pk.dump({
    "transmat": np.array([
      [0.9, 0.05, 0.05], 
      [0.05, 0.9, 0.05], 
      [0.05, 0.05, 0.9],
      [0.9, 0.05, 0.05], 
      [0.05, 0.9, 0.05], 
      [0.05, 0.05, 0.9],
      [0.9, 0.05, 0.05], 
      [0.05, 0.9, 0.05], 
      [0.05, 0.05, 0.9],
    ])
    }, f)