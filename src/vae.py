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
import time

from constatants import *
from utils import *


BATCH = 20
# CVの結果から
MONTE = 3
EPOCH = 125
Z_DIM = 4
H_DIM = 26
LR = 0.0088908714321722

class Encoder(nn.Module):
  def __init__(self, z_dim=Z_DIM, h_dim=H_DIM):
    super(Encoder, self).__init__()
    self.fc1 = nn.Linear(3*NS, h_dim)
    self.fcmu = nn.Linear(h_dim, z_dim)  # mean
    self.fcsig = nn.Linear(h_dim, z_dim)  # log variance
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    h1 = self.fc1(x)
    # h1 = self.relu(h1)
    # h1 = self.fc2(h1)
    # h1 = self.relu(h1)
    # h1 = self.fc3(h1)
    h1 = self.sigmoid(h1)
    mu = self.fcmu(h1)
    logvar = self.fcsig(h1)

    ep = torch.randn_like(mu)
    z = mu + torch.exp(logvar / 2) * ep
    return z, mu, logvar
        
class Decoder(nn.Module):
  def __init__(self, z_dim=Z_DIM, h_dim=H_DIM):   
    super(Decoder, self).__init__()
    self.fc1 = nn.Linear(z_dim, h_dim)
    self.fc2 = nn.Linear(h_dim, 3*NS)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, z):
    h1 = self.fc1(z)
    h1 = self.relu(h1)
    h1 = self.fc2(h1).view(-1, 3, NS)
    x_recon = self.softmax(h1)
    return x_recon
  
class VAE(nn.Module):
  def __init__(self, z_dim=Z_DIM, h_dim=H_DIM, lr=LR, epoch=EPOCH, monte=MONTE):
    super(VAE, self).__init__()
    self.z_dim = z_dim
    self.epoch = epoch
    self.monte = monte
    self.encoder = Encoder(z_dim, h_dim)
    self.decoder = Decoder(z_dim, h_dim)

    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    z, mu, logvar = self.encoder(x)
    x_recon = self.decoder(z)
    return x_recon, mu, logvar
  
  def train_model(self, data_mats):
    """
    Args:
      data_mats: (人数, 3, NS) 各人の手の出し方の行列
    """
    # DataLoaderの作成
    dataset = torch.utils.data.TensorDataset(data_mats)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
    losses = []
    # 学習
    with tqdm(range(self.epoch)) as pbar:
      for epoch in pbar:
        for i, (data,) in enumerate(dataloader):
          self.optimizer.zero_grad()
          loss = 0
          for j in range(self.monte):
            recon_batch, mu, logvar = self(data)
            loss += criterion(recon_batch, data, mu, logvar) / MONTE
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
    self.z = torch.zeros(1,self.z_dim, requires_grad=True) #torch.randn(1, Z_DIM, requires_grad=True)
    self.data_mat = data_mat
    self.optimizer_z = torch.optim.LBFGS([self.z], lr=1e-1)
    loss_prev = 1e10
    lossv = 1e9
    losses = []
    iters = 0
    # 最適化
    while abs(loss_prev - lossv) > 1e-3:
      loss_prev = lossv
      loss = self.optimizer_z.step(self.closure) # LBFGSを使うときはclosureを使う
      lossv = loss.item()
      losses.append(lossv)
      iters += 1
      if iters > 100:
        break

    if get_loss:
      return self.z.detach(), losses
    else:
      return self.z.detach()
    
      
  def loss_map(self, z, data_mat):
    """
    Args:
      z: (1, Z_DIM) 潜在変数
      data_mat: (3, NS) その人の手の出し方の行列
    Returns:
      loss: (1) その最小化がMAPに同値になるような目的関数(- log P(x|z) - log P(z))
    """
    loss = 0.5* torch.sum(torch.square(z))
    choice_mat = self.decoder(z).view(3, NS)
    loss += - torch.sum(data_mat.flatten() * torch.log(choice_mat.flatten() + np.spacing(1)))
    return loss
    
  def closure(self):
    self.optimizer_z.zero_grad()
    loss = self.loss_map(self.z, self.data_mat)
    loss.backward()
    return loss
  
  def load_model(self):
    try:
      if os.path.exists(file_model):
        self.load_state_dict(torch.load(file_model))
        print("model loaded")
    except:
      print("model not found")
      
  def save_model(self):
    if os.path.exists(file_model):
      os.remove(file_model)
    torch.save(self.state_dict(), file_model)
    
  def evaluate(self, data_mats):
    """
    Args:
      data_mats: (人数, 3*NS) 各人の手の出し方の行列
    Returns:
      loss: (1) その人の手の出し方の行列に対する誤差
    """
    losses = []
    for i in range(len(data_mats)):
      _, losses_map = self.map_z(data_mats[i].view(3, NS), get_loss=True)
      loss = losses_map[-1]
      losses.append(loss)
    return np.mean(losses)
      
  
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
  BCE = - torch.sum(data_mat.flatten() * torch.log(pred_mat.flatten() + np.spacing(1)))
  KLD = - 0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
  return (BCE + KLD) / data_mat.shape[0]

def load_data():
  reader = csv.reader(open(file_data, 'r'))

  data = []
  for row in reader:
    data_raw = []
    for i in range(0, len(row), 2):
      data_raw.append((int(row[i]), int(row[i+1])))
    data.append(data_raw)
    
  return data

def load_param():
  """
  Returns:
    data: (人数, 3, NS) 各人の手の出し方の行列
  """
  with open(file_data_param, 'rb') as f:
    data = pk.load(f)
  return data

def data_to_mat(data):
  """
  Args:
    data: (人数, 20) (プレイヤーの手, CPUの手)各人の手の出し方の行列
  Returns:
    data_mat: (人数, 3*NS) 各人の手の出し方の行列を1次元に変換
  """
  data_mats = []
  for i in range(len(data)):
    data_mats.append(make_data_mat(data[i]))
  return np.array(data_mats)



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
  vae.load_model()
  losses = vae.train_model(data_mats)
  vae.save_model()
  
  np.random.seed()
  human = np.random.randint(0, len(data))

  # with open(file_data_param, 'r') as f:
  #   reader = csv.reader(f)
  #   true_mats = []
  #   for row in reader:
  #     true_mats.append(np.array(row, dtype=np.float32).reshape(3, NS))
  # print(f"true_mats: \n{true_mats[human]}")
  
  print(f"data_mats: \n{data_mats[human].view(3, NS).detach().numpy()}")

  st = time.time()
  z_star, losses_map = vae.map_z(data_mats[human].view(3,NS), get_loss=True)
  pred_mat = vae.decoder(z_star).view(3, NS).detach().numpy()
  print(f"elapsed time for MAP: {time.time() - st}")
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
  print(f"loss: {losses_map[-1]}")
  
  plt.show()