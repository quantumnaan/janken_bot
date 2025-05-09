import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import csv
import os
from tqdm import tqdm

from constatants import *
from utils import *


BATCH = 2
EPOCH = 100
Z_DIM = 4

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.fc1 = nn.Linear(3*NS, 32)
    self.fc21 = nn.Linear(32, Z_DIM)  # mean
    self.fc22 = nn.Linear(32, Z_DIM)  # log variance
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
    self.fc1 = nn.Linear(Z_DIM, 32)
    self.fc2 = nn.Linear(32, 3*NS)
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
    
    self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4)

  def forward(self, x):
    z, mu, logvar = self.encoder(x)
    x_recon = self.decoder(z)
    return x_recon, mu, logvar
    
  # def train_onedata(self, onedatamat, epoch = EPOCH):
  #   """
  #   Args:
  #     onedata: (1, 3, NS) 各人の手の出し方の行列
  #   """

  #   # 学習
  #   for epoch in tqdm(range(epoch)):
  #     self.optimizer.zero_grad()
  #     recon_batch, mu, logvar = self(onedatamat)
  #     loss = criterion(recon_batch, onedatamat, mu, logvar)
  #     loss.backward()
  #     self.optimizer.step()
  
  def train(self, data_mats):
    """
    Args:
      data: (人数, 3, NS) 各人の手の出し方の行列
    """
    # DataLoaderの作成
    dataset = torch.utils.data.TensorDataset(data_mats)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    losses = []
    # 学習
    with tqdm(range(EPOCH)) as pbar:
      for epoch in pbar:
        for i, (data,) in enumerate(dataloader):
          self.optimizer.zero_grad()
          recon_batch, mu, logvar = self(data)
          loss = criterion(recon_batch, data, mu, logvar)
          loss.backward()
          self.optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({"loss":loss.item()})
    return losses
        
  def map_z(self, data_mat, get_loss = False):
    """
    Args:
      data_mat: (3, NS) 各人の手の出し方の行列
    Returns:
      z: (Z_DIM) 潜在変数
    """
    z = torch.randn(Z_DIM).view(1, -1)
    z.requires_grad = True
    optimizer_z = torch.optim.Adam([z], lr=1e-2)
    loss_prev = 1e10
    loss = 1e9
    losses = []
    # 最適化
    while abs(loss - loss_prev) > 1e-4:
      loss_prev = loss
      optimizer_z.zero_grad()
      loss = self.loss_map(z, data_mat)
      loss.backward()
      optimizer_z.step()
      losses.append(loss.item())
    
    if get_loss:
      return z.detach(), losses
    else:
      return z.detach()
    
      
  def loss_map(self, z, data_mat):
    """
    Args:
      z: (Z_DIM) 潜在変数
      data_mat: (3, NS) その人の手の出し方の行列
    Returns:
      loss: (1) その最小化がMAPに同値になるような目的関数(- log P(x|z) - log P(z))
    """
    loss = 0.5* torch.square(z).sum()
    choice_mat = self.decoder(z).view(3, NS)
    for i in range(3):
      for j in range(NS):
        if data_mat[i][j] != 0:
          loss += -data_mat[i][j] * torch.log(choice_mat[i][j] + np.spacing(1))
    return loss
    
  
  def load_model(self):
    if os.path.exists(file_model):
      self.load_state_dict(torch.load(file_model))
      print("model loaded")
    else:
      print("model not found")
      
  def save_model(self):
    if os.path.exists(file_model):
      os.remove(file_model)
    torch.save(self.state_dict(), file_model)
  
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
  reader = csv.reader(open(file_data, 'r'))
  
  data = []
  for row in reader:
    data.append([])
    for i in range(0, len(row), 2):
      data[-1].append((int(row[i]), int(row[i+1])))
      
  return data

def load_param():
  """
  Returns:
    data: (人数, 3, NS) 各人の手の出し方の行列
  """
  with open(file_data_param, 'rb') as f:
    data = pk.load(f)
  return data


def train_vae(vae):
  
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

  optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

  # 学習
  for epoch in tqdm(range(EPOCH)):
    for i, (data,) in enumerate(dataloader):
      optimizer.zero_grad()
      recon_batch, mu, logvar = vae(data)
      loss = criterion(recon_batch, data, mu, logvar)
      loss.backward()
      optimizer.step()


if __name__ == "__main__":
  
  # データの読み込み
  data = load_data()
  print(f"data length: {len(data)}")
  data_mats = []
  for i in range(len(data)):
    data_mats.append(make_data_mat(data[i]))
  data_mats = np.array(data_mats) # (人数, 3, NS)
  data_mats = torch.tensor(data_mats, dtype=torch.float32).view(-1, 3*NS)
  
  # VAEの初期化
  vae = VAE()
  losses = vae.train(data_mats)
  vae.save_model()
  
  human = len(data) -2
  
  # data_param = load_param()
  # true_mat = data_param["sample_data"]
  # print(f"true_mat: \n{true_mat[:,:,human]}")
  
  print(f"data_mats: \n{data_mats[human].view(3, NS).detach().numpy()}")
  
  z_star, losses_map = vae.map_z(data_mats[human].view(3,NS), get_loss=True)
  pred_mat = vae.decoder(z_star).view(3, NS).detach().numpy()
  
  print(f"pred_mat: \n{pred_mat}")
  
  zs = np.zeros((len(data), Z_DIM))
  for i in range(len(data)):
    zs[i] = vae.encoder(data_mats[i].view(1, -1))[0][0].detach().numpy()
    
  
  
  ax1 = plt.subplot(131)
  ax1.set_title("latent space")
  ax1.set_xlabel("z1")
  ax1.set_ylabel("z2")
  ax1.scatter(zs[:,0], zs[:,1])
  
  ax2 = plt.subplot(132)
  ax2.set_title("loss")
  ax2.set_xlabel("epoch")
  ax2.set_ylabel("loss")
  ax2.plot(losses)
  
  ax3 = plt.subplot(133)
  ax3.set_title("loss")
  ax3.set_xlabel("epoch")
  ax3.set_ylabel("loss")
  ax3.plot(losses_map)
  
  plt.show()