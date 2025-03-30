import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os

from constatants import *
from utils import *


BATCH = 20
EPOCH = 20
Z_DIM = 4

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.fc1 = nn.Linear(3*NS, 128)
    self.fc21 = nn.Linear(128, Z_DIM)  # mean
    self.fc22 = nn.Linear(128, Z_DIM)  # log variance
    self.relu = nn.ReLU()

  def forward(self, x):
    h1 = self.fc1(x)
    h1 = self.relu(h1)
    mu = self.fc21(h1)
    logvar = self.fc22(h1)
    
    ep = torch.randn_like(mu)
    z = mu + torch.exp(logvar / 2) * ep
    return z, mu, logvar
        
class Decoder(nn.Module):
  def __init__(self):   
    super(Decoder, self).__init__()
    self.fc1 = nn.Linear(Z_DIM, 128)
    self.fc2 = nn.Linear(128, 3*NS)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, z):
    h1 = self.fc1(z)
    h1 = self.relu(h1)
    h1 = self.fc2(h1).view(-1, 3, NS)
    x_recon = self.softmax(h1)
    return x_recon
  
class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    z, mu, logvar = self.encoder(x)
    x_recon = self.decoder(z)
    return x_recon, mu, logvar
  
def criterion(pred_mat, data_mat, mu, logvar):
  """
  Args:
    pred_mat: (B, 3, NS) 予測値
    data_mat: (B, 3, NS) 各状態において各手を何回選んだかの行列
    mu: (B, Z_DIM) 平均
    logvar: (B, Z_DIM) 対数分散
    m: (B,) 一人当たりのターン数
  Returns:
    loss: (1,) 誤差
  """
  BCE = - sum(data_mat.flatten() * torch.log(pred_mat.flatten() + np.spacing(1)))
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE + KLD

def load_data():
  """
  Returns:
    data: (人数, 3, NS) 各状態において各手を何回選んだかの行列
  """
  with open(file_data, 'rb') as f:
    data = pk.load(f)
  return data

def load_param():
  """
  Returns:
    data: (人数, 3, NS) 各人の手の出し方の行列
  """
  with open(file_data_param, 'rb') as f:
    data = pk.load(f)
  return data

def make_data_mat(data):
  """
  Args:
    data: 人i:[ターンt:(人iが出した手, cpが出した手)]
  Returns:
    data_mat: (3, NS) 各状態において各手を何回選んだかの行列
  """
  data_mat = torch.zeros((3, NS))
  m = data.shape[0]
  for t in range(m-1):
    state = make_state(data[t,0], data[t,1])
    data_mat[data[t+1,0]][state] += 1
  return data_mat



if __name__ == "__main__":
  # データの読み込み
  data = load_data()
  data_mats = []
  for i in range(len(data)):
    data_mats.append(make_data_mat(data[i]))
  data_mats = np.array(data_mats) # (人数, 3, NS)
  data_mats = torch.tensor(data_mats, dtype=torch.float32).view(-1, 3*NS)
  
  # DataLoaderの作成
  dataset = torch.utils.data.TensorDataset(data_mats)
  dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

  # VAEの初期化
  vae = VAE()
  optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

  # 学習
  for epoch in range(EPOCH):
    for i, (data,) in enumerate(dataloader):
      optimizer.zero_grad()
      recon_batch, mu, logvar = vae(data)
      loss = criterion(recon_batch, data, mu, logvar)
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
  
  data_param = load_param()
  true_mat = data_param["sample_data"]
  human = 0
  print(f"true_mat: \n{true_mat[:,:,human]}")
  pred_mat = vae(data_mats[human].view(1, -1))[0][0].detach().numpy()
  print(f"pred_mat: \n{pred_mat}")